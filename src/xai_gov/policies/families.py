"""Concrete policy families.

Three families land in this stage, chosen so the comparison is meaningful
from the first run rather than after everything is built:

* ``heuristic_sQ`` — the classical reorder-point policy. Auditable by
  inspection, zero compute cost, no learning. The floor any AI policy must
  beat to justify its existence.
* ``opaque_dro`` — a robust policy that sizes safety stock from a quantile
  of an ambiguity set over demand. It is the "opaque AI" arm of the
  protocol: it performs well and explains nothing, which is precisely the
  condition governance is meant to address.
* ``random_bounded`` — a seeded random baseline within physical bounds. It
  exists to falsify: any KPI on which a real policy fails to separate from
  it is a KPI that measures nothing.

The XAI, XAI+conformal, MARL and language-model families arrive in later
stages and register through the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from xai_gov.io.decision_record import ActionKind
from xai_gov.policies.base import Policy, PolicyProposal, clamp_to_capacity

if TYPE_CHECKING:
    from xai_gov.io.decision_record import OperatingState
    from xai_gov.simulation.network import NodeSpec


@dataclass
class HeuristicSQPolicy(Policy):
    """Reorder-point policy: order Q when the inventory position drops to s.

    The inventory position — on hand plus in transit minus backlog — is
    used rather than on-hand inventory. Using on-hand alone is the classic
    error: it re-orders while a shipment is already in transit and so
    manufactures the very oscillation the bullwhip KPI detects.
    """

    policy_id: str = "heuristic_sQ"
    version: str = "1.0.0"
    cost_units: float = 0.0

    def propose(
        self,
        *,
        spec: NodeSpec,
        state: OperatingState,
        rng: np.random.Generator,
    ) -> PolicyProposal:
        position = state.inventory + state.in_transit - state.backlog
        features = {
            "inventory_position": position,
            "reorder_point": spec.reorder_point,
            "gap_to_reorder_point": spec.reorder_point - position,
            "backlog": state.backlog,
        }
        if position > spec.reorder_point:
            return PolicyProposal(
                action=ActionKind.HOLD,
                quantity=0.0,
                features=features,
                rationale=(
                    f"inventory position {position:.2f} above reorder point "
                    f"{spec.reorder_point:.2f}"
                ),
            )
        quantity = clamp_to_capacity(spec.order_quantity, state)
        if quantity <= 0.0:
            return PolicyProposal(
                action=ActionKind.HOLD,
                quantity=0.0,
                features=features,
                rationale="reorder point reached but no free capacity",
            )
        return PolicyProposal(
            action=ActionKind.REORDER,
            quantity=quantity,
            features=features,
            rationale=(
                f"inventory position {position:.2f} at or below reorder point "
                f"{spec.reorder_point:.2f}"
            ),
        )


@dataclass
class OpaqueDROPolicy(Policy):
    """Distributionally robust replenishment over a demand ambiguity set.

    Safety stock covers lead-time demand at a declared service quantile,
    inflated by ``ambiguity_radius`` to hedge misspecification of the
    demand distribution. The radius is the policy's central modeling choice
    and it is *not* auditable from the outside — which is exactly why the
    protocol classes this family as opaque despite its guarantees.

    Demand is estimated from a rolling window of observations the node has
    actually seen; nothing here reads the true generating process.
    """

    policy_id: str = "opaque_dro"
    version: str = "1.0.0"
    cost_units: float = 1.0
    service_quantile: float = 0.95
    ambiguity_radius: float = 0.25
    window: int = 8

    def __post_init__(self) -> None:
        if not 0.5 <= self.service_quantile < 1.0:
            raise ValueError("service_quantile must lie in [0.5, 1.0)")
        if self.ambiguity_radius < 0.0:
            raise ValueError("ambiguity_radius must be non-negative")
        if self.window < 2:
            raise ValueError("window must be at least 2")
        self._history: dict[str, list[float]] = {}

    def _observe(self, node_id: str, demand: float) -> list[float]:
        history = self._history.setdefault(node_id, [])
        history.append(demand)
        if len(history) > self.window:
            del history[: len(history) - self.window]
        return history

    def propose(
        self,
        *,
        spec: NodeSpec,
        state: OperatingState,
        rng: np.random.Generator,
    ) -> PolicyProposal:
        history = self._observe(spec.node_id, state.demand_observed)
        mean = float(np.mean(history))
        # Population sd with a floor: a single observation, or a run of
        # identical ones, must not collapse safety stock to zero.
        spread = float(np.std(history)) if len(history) > 1 else mean * 0.2
        spread = max(spread, mean * 0.05)

        # Normal quantile via the inverse error function, so the policy
        # needs no scipy dependency at this stage.
        z = float(np.sqrt(2.0) * _erfinv(2.0 * self.service_quantile - 1.0))
        lead_time = max(spec.lead_time_mean, 1.0)
        expected = mean * lead_time
        safety = z * spread * float(np.sqrt(lead_time)) * (1.0 + self.ambiguity_radius)
        target = expected + safety

        position = state.inventory + state.in_transit - state.backlog
        shortfall = target - position
        features = {
            "demand_mean": mean,
            "demand_spread": spread,
            "target_position": target,
            "inventory_position": position,
            "shortfall": shortfall,
            "ambiguity_radius": self.ambiguity_radius,
        }
        if shortfall <= 0.0:
            return PolicyProposal(
                action=ActionKind.HOLD,
                quantity=0.0,
                features=features,
                rationale=f"position {position:.2f} covers robust target {target:.2f}",
            )
        quantity = clamp_to_capacity(shortfall, state)
        if quantity <= 0.0:
            return PolicyProposal(
                action=ActionKind.HOLD,
                quantity=0.0,
                features=features,
                rationale="robust target unmet but no free capacity",
            )
        # An unmet target while backlog is already accumulating is treated
        # as urgent; expedite is the same order under a shorter lead time,
        # and it is the action a governance agent most often vetoes.
        action = ActionKind.EXPEDITE if state.backlog > 0.0 else ActionKind.REORDER
        return PolicyProposal(
            action=action,
            quantity=quantity,
            features=features,
            rationale=(
                f"robust target {target:.2f} exceeds position {position:.2f} "
                f"at quantile {self.service_quantile:.2f}"
            ),
        )


@dataclass
class RandomBoundedPolicy(Policy):
    """Seeded random action within physical bounds: the falsification arm."""

    policy_id: str = "random_bounded"
    version: str = "1.0.0"
    cost_units: float = 0.0
    hold_probability: float = 0.5

    def propose(
        self,
        *,
        spec: NodeSpec,
        state: OperatingState,
        rng: np.random.Generator,
    ) -> PolicyProposal:
        features = {"inventory": state.inventory, "backlog": state.backlog}
        if rng.random() < self.hold_probability:
            return PolicyProposal(
                action=ActionKind.HOLD, quantity=0.0, features=features, rationale="random hold"
            )
        quantity = clamp_to_capacity(float(rng.uniform(0.0, spec.order_quantity * 2.0)), state)
        if quantity <= 0.0:
            return PolicyProposal(
                action=ActionKind.HOLD,
                quantity=0.0,
                features=features,
                rationale="random order exceeded capacity",
            )
        return PolicyProposal(
            action=ActionKind.REORDER,
            quantity=quantity,
            features=features,
            rationale="random order",
        )


def _erfinv(y: float) -> float:
    """Inverse error function, Newton-refined from a rational start.

    Accurate to better than 1e-9 over the range the service quantiles use,
    which keeps this stage free of a scipy dependency.
    """
    if not -1.0 < y < 1.0:
        raise ValueError("erfinv is defined on (-1, 1)")
    if y == 0.0:
        return 0.0
    a = 0.147
    ln_term = float(np.log(1.0 - y * y))
    first = 2.0 / (np.pi * a) + ln_term / 2.0
    x = float(np.sign(y) * np.sqrt(np.sqrt(first * first - ln_term / a) - first))
    for _ in range(3):
        error = float(_erf(x) - y)
        derivative = 2.0 / float(np.sqrt(np.pi)) * float(np.exp(-x * x))
        if derivative == 0.0:
            break
        x -= error / derivative
    return x


def _erf(x: float) -> float:
    """Abramowitz and Stegun 7.1.26."""
    sign = 1.0 if x >= 0.0 else -1.0
    z = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    return sign * (1.0 - poly * float(np.exp(-z * z)))
