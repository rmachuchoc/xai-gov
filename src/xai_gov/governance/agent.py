"""Governance mediation interface, and the ungoverned control arm.

The interface is fixed here so the engine is complete before the
constrained-POMDP agent exists. `NoGovernanceAgent` is not a placeholder:
it is the control condition of every experiment in the protocol — the arm
against which the value of governance is measured. Its belief is honestly
degenerate (all mass on ``nominal``) and its descriptor is marked
uninformative, so no downstream analysis can mistake absence of governance
for confident governance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from xai_gov.io.decision_record import (
    ActionKind,
    CausalDescriptor,
    ConformalSignal,
    GovernanceAction,
    RiskRegime,
    ShieldOutcome,
)

if TYPE_CHECKING:
    from xai_gov.io.decision_record import OperatingState
    from xai_gov.policies.base import PolicyProposal
    from xai_gov.simulation.network import NodeSpec

NEUTRAL_DESCRIPTOR = CausalDescriptor(
    actionability_score=0.0,
    explanatory_fidelity=0.0,
    identifiable=False,
    informative=False,
    method="none",
)

NEUTRAL_CONFORMAL = ConformalSignal(
    score=0.0, threshold=None, level=0.0, flagged_ood=False, scheme="none"
)

NOMINAL_BELIEF: dict[str, float] = {
    RiskRegime.NOMINAL.value: 1.0,
    RiskRegime.DRIFT.value: 0.0,
    RiskRegime.DISRUPTION.value: 0.0,
}


@dataclass(frozen=True, slots=True)
class GovernanceVerdict:
    """The mediated outcome of one proposal."""

    action: GovernanceAction
    final_action: ActionKind
    final_quantity: float
    rationale: str
    descriptor: CausalDescriptor
    conformal: ConformalSignal
    belief: dict[str, float]
    intervention_cost: float
    escalated: bool
    safe_mode: bool
    shield: ShieldOutcome


class GovernanceAgent(ABC):
    """Mediates between a proposal and its execution."""

    agent_id: str = "abstract"
    version: str = "0.0.0"
    autonomy_regime: str = "H3"

    @abstractmethod
    def mediate(
        self,
        *,
        spec: NodeSpec,
        state: OperatingState,
        proposal: PolicyProposal,
    ) -> GovernanceVerdict:
        """Decide what actually happens to ``proposal``."""

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.agent_id,
            "version": self.version,
            "autonomy_regime": self.autonomy_regime,
        }


@dataclass
class NoGovernanceAgent(GovernanceAgent):
    """The control arm: every proposal executes unchanged, at zero cost."""

    agent_id: str = "no_governance"
    version: str = "1.0.0"
    autonomy_regime: str = "H3"

    def mediate(
        self,
        *,
        spec: NodeSpec,
        state: OperatingState,
        proposal: PolicyProposal,
    ) -> GovernanceVerdict:
        return GovernanceVerdict(
            action=GovernanceAction.APPROVE,
            final_action=proposal.action,
            final_quantity=proposal.quantity,
            rationale="no governance layer active; proposal executed as issued",
            descriptor=NEUTRAL_DESCRIPTOR,
            conformal=NEUTRAL_CONFORMAL,
            belief=dict(NOMINAL_BELIEF),
            intervention_cost=0.0,
            escalated=False,
            safe_mode=False,
            shield=ShieldOutcome(),
        )
