"""The decision record.

One record per decision, carrying every field the protocol's traceability
contract requires. The record is deliberately flat and self-describing:
an auditor reading a single line must be able to reconstruct why the
system acted as it did, without consulting the code.

Fields are grouped as follows.

* what was seen      period, operating state
* who proposed       policy id and version, proposed action and quantity
* why it was risky   causal descriptor and its fidelity, nonconformity
                     score, threshold in force, belief over the regime
* what governance did governance action, justification, escalation,
                     safe mode, imputed cost
* what the shield did whether it intervened and with which substitution
* what happened      final action executed, resulting cost
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ActionKind(StrEnum):
    """Operational actions the policy may propose."""

    HOLD = "hold"
    REORDER = "reorder"
    EXPEDITE = "expedite"
    CANCEL = "cancel"


class GovernanceAction(StrEnum):
    """The five actions of the governance agent (protocol section 6.4)."""

    APPROVE = "approve"
    VETO = "veto"
    ESCALATE = "escalate"
    REDUCE_AUTONOMY = "reduce_autonomy"
    ACTIVATE_SAFE_MODE = "activate_safe_mode"


class RiskRegime(StrEnum):
    """The latent regime the governance agent holds a belief over."""

    NOMINAL = "nominal"
    DRIFT = "drift"
    DISRUPTION = "disruption"


@dataclass(frozen=True, slots=True)
class OperatingState:
    """The observable operating state x_t."""

    period: int
    inventory: float
    backlog: float
    in_transit: float
    capacity: float
    demand_observed: float
    data_delay: int = 0

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CausalDescriptor:
    """The governance descriptor phi_t (protocol definition 1).

    ``informative`` is False when explanatory fidelity falls below the
    configured floor; the governance agent must then treat the descriptor
    as absent rather than trust a misleading signal.
    """

    actionability_score: float
    feature_effects: dict[str, float] = field(default_factory=dict)
    explanatory_fidelity: float = 0.0
    identifiable: bool = True
    informative: bool = True
    method: str = "none"

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConformalSignal:
    """The calibrated nonconformity signal u_t.

    ``threshold`` is None when no conformal scheme is active. A sentinel such
    as infinity would be a non-number masquerading as a threshold: it is not
    JSON-representable, so it could not be sealed into the hash-chained log,
    and it would silently satisfy every "score below threshold" comparison
    made against it. Absence is recorded as absence.
    """

    score: float
    threshold: float | None
    level: float
    flagged_ood: bool
    scheme: str = "none"
    martingale_value: float = 1.0
    change_declared: bool = False

    def __post_init__(self) -> None:
        if self.scheme == "none":
            if self.flagged_ood or self.change_declared:
                raise ValueError("no conformal scheme is active, yet a flag is raised")
        elif self.threshold is None:
            raise ValueError(f"scheme {self.scheme!r} is active but declares no threshold")

    @property
    def active(self) -> bool:
        return self.scheme != "none"

    @property
    def exceeds_threshold(self) -> bool:
        """False when no scheme is active: absence of a test is not evidence."""
        return self.threshold is not None and self.score > self.threshold

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ShieldOutcome:
    """What the verified shield did, if anything."""

    activated: bool = False
    property_id: str | None = None
    substituted_action: ActionKind | None = None
    substituted_quantity: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "activated": self.activated,
            "property_id": self.property_id,
            "substituted_action": (
                self.substituted_action.value if self.substituted_action else None
            ),
            "substituted_quantity": self.substituted_quantity,
        }


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """The immutable unit of the decision log."""

    period: int
    node_id: str
    state: OperatingState
    policy_id: str
    policy_version: str
    proposed_action: ActionKind
    proposed_quantity: float
    descriptor: CausalDescriptor
    conformal: ConformalSignal
    belief: dict[str, float]
    governance_action: GovernanceAction
    governance_rationale: str
    autonomy_regime: str
    escalated: bool
    safe_mode: bool
    intervention_cost: float
    shield: ShieldOutcome
    final_action: ActionKind
    final_quantity: float
    operating_cost: float = 0.0

    def __post_init__(self) -> None:
        total = sum(self.belief.values())
        if self.belief and abs(total - 1.0) > 1e-6:
            raise ValueError(f"belief must be a distribution; sums to {total:.6f}")

    @property
    def action_was_modified(self) -> bool:
        """True when governance or the shield changed the proposed action."""
        return (
            self.final_action != self.proposed_action
            or self.final_quantity != self.proposed_quantity
        )

    def belief_of(self, regime: RiskRegime) -> float:
        return float(self.belief.get(regime.value, 0.0))

    def to_payload(self) -> dict[str, Any]:
        """Canonical dictionary form, written into the hash-chained log."""
        return {
            "period": self.period,
            "node_id": self.node_id,
            "state": self.state.to_payload(),
            "policy": {"id": self.policy_id, "version": self.policy_version},
            "proposed": {
                "action": self.proposed_action.value,
                "quantity": self.proposed_quantity,
            },
            "descriptor": self.descriptor.to_payload(),
            "conformal": self.conformal.to_payload(),
            "belief": dict(sorted(self.belief.items())),
            "governance": {
                "action": self.governance_action.value,
                "rationale": self.governance_rationale,
                "autonomy_regime": self.autonomy_regime,
                "escalated": self.escalated,
                "safe_mode": self.safe_mode,
                "intervention_cost": self.intervention_cost,
            },
            "shield": self.shield.to_payload(),
            "final": {"action": self.final_action.value, "quantity": self.final_quantity},
            "operating_cost": self.operating_cost,
            "action_was_modified": self.action_was_modified,
        }

    def to_flat_row(self) -> dict[str, Any]:
        """Flat projection for the events CSV and for pandas analysis."""
        return {
            "period": self.period,
            "node_id": self.node_id,
            "inventory": self.state.inventory,
            "backlog": self.state.backlog,
            "in_transit": self.state.in_transit,
            "demand_observed": self.state.demand_observed,
            "policy_id": self.policy_id,
            "proposed_action": self.proposed_action.value,
            "proposed_quantity": self.proposed_quantity,
            "actionability_score": self.descriptor.actionability_score,
            "explanatory_fidelity": self.descriptor.explanatory_fidelity,
            "descriptor_informative": self.descriptor.informative,
            "conformal_score": self.conformal.score,
            "conformal_threshold": self.conformal.threshold,
            "conformal_level": self.conformal.level,
            "flagged_ood": self.conformal.flagged_ood,
            "change_declared": self.conformal.change_declared,
            "belief_disruption": self.belief_of(RiskRegime.DISRUPTION),
            "belief_drift": self.belief_of(RiskRegime.DRIFT),
            "governance_action": self.governance_action.value,
            "autonomy_regime": self.autonomy_regime,
            "escalated": self.escalated,
            "safe_mode": self.safe_mode,
            "shield_activated": self.shield.activated,
            "intervention_cost": self.intervention_cost,
            "final_action": self.final_action.value,
            "final_quantity": self.final_quantity,
            "operating_cost": self.operating_cost,
        }
