"""Tabular Q-learning replenishment policy.

The first learned policy in the project. It exists so the comparison between
a fallback shield and a projection shield can be made over a policy that
*adapts* to the mediation it runs under, not only over fixed ones. A fixed
policy keeps proposing the action a fallback shield blocks; a learning one
may discover which proposals get blocked and route around them. Whether the
harm of a fallback survives that adaptation is an empirical question, and
this policy is the instrument that asks it.

Design choices, each stated because a reviewer will ask:

* **Tabular, not deep.** The state is the inventory-position band and the
  backlog band, the same discretization family the shield's abstraction uses.
  A tabular learner has no function-approximation error to confound the
  comparison, and its Q-table can be inspected entry by entry.
* **Pre-trained on a surrogate, then learned online.** Before the first
  decision for a node, the policy trains on a single-node inventory surrogate
  built from that node's own parameters. The surrogate contains no oversight,
  so the policy enters the twin unmediated in its training and meets the
  shield only at deployment. Online updates then continue every period.
* **Learns from what it proposed.** The policy is never told whether its
  proposal was executed or substituted. Each update credits the proposed
  action with the consequence observed next period. This is the
  post-posed-shield convention, and it is the setting in which a learner can
  adapt to a fallback: proposals that keep getting blocked keep looking bad.
* **Deterministic.** Pre-training is seeded from the declared seed and the
  node id; online exploration draws from the policy's own stream. Two runs
  with the same seed produce the same Q-tables and the same decisions.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from xai_gov.io.decision_record import ActionKind
from xai_gov.policies.base import Policy, PolicyProposal, clamp_to_capacity

if TYPE_CHECKING:
    from xai_gov.io.decision_record import OperatingState
    from xai_gov.simulation.network import NodeSpec

#: Upper edges of the inventory-position bands; a final open band lies above.
POSITION_EDGES: tuple[float, ...] = (-20.0, 0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0)
#: Upper edges of the backlog bands; a final open band lies above.
BACKLOG_EDGES: tuple[float, ...] = (0.0, 10.0, 50.0)
#: Actions, as multiples of the node's nominal order quantity.
ACTION_FRACTIONS: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0)
#: Rewards are costs divided by this, so Q-values stay in a readable range.
REWARD_SCALE = 10.0

_PRETRAINED: dict[tuple[Any, ...], np.ndarray] = {}


def band(value: float, edges: tuple[float, ...]) -> int:
    """Index of the first band whose upper edge is not below ``value``."""
    for index, edge in enumerate(edges):
        if value <= edge:
            return index
    return len(edges)


def state_key(inventory: float, in_transit: float, backlog: float) -> tuple[int, int]:
    """Discretized state: (inventory-position band, backlog band)."""
    position = inventory + in_transit - backlog
    return band(position, POSITION_EDGES), band(backlog, BACKLOG_EDGES)


def period_cost(spec: NodeSpec, inventory: float, backlog: float, quantity: float) -> float:
    """One period's cost, the negative of the reward before scaling."""
    return (
        spec.holding_cost * max(inventory, 0.0)
        + spec.backlog_cost * max(backlog, 0.0)
        + spec.order_cost * max(quantity, 0.0)
    )


@dataclass(slots=True)
class _NodeLearner:
    q: np.ndarray
    last_state: tuple[int, int] | None = None
    last_action: int | None = None
    last_quantity: float = 0.0
    updates: int = 0


@dataclass
class QLearningPolicy(Policy):
    """Tabular Q-learning over inventory-position and backlog bands."""

    policy_id: str = "qlearning_tabular"
    version: str = "1.0.0"
    cost_units: float = 0.0
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.05
    pretrain_episodes: int = 400
    pretrain_periods: int = 60
    pretrain_epsilon: float = 0.2
    demand_mean: float = 20.0
    demand_sd: float = 6.0
    seed: int = 20261019
    online: bool = True
    _learners: dict[str, _NodeLearner] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must lie in (0, 1]")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must lie in [0, 1)")
        rates = (("epsilon", self.epsilon), ("pretrain_epsilon", self.pretrain_epsilon))
        for name, value in rates:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.pretrain_episodes < 0 or self.pretrain_periods < 1:
            raise ValueError("pretrain_episodes must be >= 0 and pretrain_periods >= 1")
        if self.demand_sd < 0.0:
            raise ValueError("demand_sd must be non-negative")

    # -- learning ---------------------------------------------------------
    def _shape(self) -> tuple[int, int, int]:
        return (len(POSITION_EDGES) + 1, len(BACKLOG_EDGES) + 1, len(ACTION_FRACTIONS))

    def _update(
        self, q: np.ndarray, s: tuple[int, int], a: int, reward: float, s2: tuple[int, int]
    ) -> None:
        target = reward + self.gamma * float(q[s2].max())
        q[(*s, a)] += self.alpha * (target - float(q[(*s, a)]))

    def _greedy(self, q: np.ndarray, s: tuple[int, int]) -> int:
        # argmax returns the lowest index on ties, so a tie resolves to the
        # smaller order and the choice is deterministic.
        return int(np.argmax(q[s]))

    def pretrain(self, spec: NodeSpec) -> np.ndarray:
        """Train on a single-node surrogate of this node, without oversight."""
        key = (
            spec.node_id, spec.initial_inventory, spec.capacity, spec.order_quantity,
            spec.lead_time_mean, spec.holding_cost, spec.backlog_cost, spec.order_cost,
            self.alpha, self.gamma, self.pretrain_episodes, self.pretrain_periods,
            self.pretrain_epsilon, self.demand_mean, self.demand_sd, self.seed,
        )
        cached = _PRETRAINED.get(key)
        if cached is not None:
            return cached.copy()

        q = np.zeros(self._shape())
        rng = np.random.default_rng(self.seed + zlib.crc32(spec.node_id.encode()))
        lead = max(round(spec.lead_time_mean), 1)
        for _ in range(self.pretrain_episodes):
            inventory, backlog = spec.initial_inventory, 0.0
            pipeline = [0.0] * lead
            for _ in range(self.pretrain_periods):
                s = state_key(inventory, sum(pipeline), backlog)
                if rng.random() < self.pretrain_epsilon:
                    a = int(rng.integers(len(ACTION_FRACTIONS)))
                else:
                    a = self._greedy(q, s)
                headroom = max(spec.capacity - inventory - sum(pipeline), 0.0)
                quantity = min(ACTION_FRACTIONS[a] * spec.order_quantity, headroom)
                pipeline.append(quantity)
                inventory += pipeline.pop(0)
                need = max(float(rng.normal(self.demand_mean, self.demand_sd)), 0.0) + backlog
                shipped = min(inventory, need)
                inventory -= shipped
                backlog = need - shipped
                reward = -period_cost(spec, inventory, backlog, quantity) / REWARD_SCALE
                self._update(q, s, a, reward, state_key(inventory, sum(pipeline), backlog))
        _PRETRAINED[key] = q.copy()
        return q

    def _learner(self, spec: NodeSpec) -> _NodeLearner:
        learner = self._learners.get(spec.node_id)
        if learner is None:
            learner = _NodeLearner(q=self.pretrain(spec))
            self._learners[spec.node_id] = learner
        return learner

    # -- policy -----------------------------------------------------------
    def propose(
        self,
        *,
        spec: NodeSpec,
        state: OperatingState,
        rng: np.random.Generator,
    ) -> PolicyProposal:
        learner = self._learner(spec)
        s = state_key(state.inventory, state.in_transit, state.backlog)

        if self.online and learner.last_state is not None and learner.last_action is not None:
            # Credit the *proposed* action with what happened next. The policy
            # is not told whether the proposal was executed or substituted.
            cost = period_cost(spec, state.inventory, state.backlog, learner.last_quantity)
            reward = -cost / REWARD_SCALE
            self._update(learner.q, learner.last_state, learner.last_action, reward, s)
            learner.updates += 1

        if rng.random() < self.epsilon:
            a = int(rng.integers(len(ACTION_FRACTIONS)))
            mode = "explore"
        else:
            a = self._greedy(learner.q, s)
            mode = "greedy"
        quantity = clamp_to_capacity(ACTION_FRACTIONS[a] * spec.order_quantity, state)

        learner.last_state, learner.last_action, learner.last_quantity = s, a, quantity
        position = state.inventory + state.in_transit - state.backlog
        features = {
            "inventory_position": position,
            "backlog": state.backlog,
            "position_band": float(s[0]),
            "backlog_band": float(s[1]),
            "q_value": float(learner.q[(*s, a)]),
        }
        if quantity <= 0.0:
            return PolicyProposal(
                action=ActionKind.HOLD,
                quantity=0.0,
                features=features,
                rationale=f"q-learning {mode}: hold in state {s}",
            )
        return PolicyProposal(
            action=ActionKind.REORDER,
            quantity=quantity,
            features=features,
            rationale=f"q-learning {mode}: order {ACTION_FRACTIONS[a]:g}xQ in state {s}",
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            **super().to_payload(),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "pretrain_episodes": self.pretrain_episodes,
            "pretrain_periods": self.pretrain_periods,
            "demand_mean": self.demand_mean,
            "demand_sd": self.demand_sd,
            "seed": self.seed,
            "online": self.online,
            "online_updates": {k: v.updates for k, v in sorted(self._learners.items())},
        }
