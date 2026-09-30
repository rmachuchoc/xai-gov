"""Governance: mediation between proposal and execution.

`CpomdpGovernanceAgent` is resolved lazily: it depends on the conformal
layer, and importing it eagerly would make every consumer of the ungoverned
control arm pay for that dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xai_gov.governance.agent import (
    NEUTRAL_CONFORMAL,
    NEUTRAL_DESCRIPTOR,
    GovernanceAgent,
    GovernanceVerdict,
    NoGovernanceAgent,
)
from xai_gov.governance.autonomy import AutonomyRegime, AutonomyState, rights_for
from xai_gov.governance.belief import BeliefFilter, RegimeModel
from xai_gov.governance.budget import GovernanceBudget
from xai_gov.governance.thresholds import (
    ThresholdEconomics,
    myopic_threshold,
    solve_threshold,
)

if TYPE_CHECKING:
    from xai_gov.governance.cpomdp import CpomdpGovernanceAgent

__all__ = [
    "NEUTRAL_CONFORMAL",
    "NEUTRAL_DESCRIPTOR",
    "AutonomyRegime",
    "AutonomyState",
    "BeliefFilter",
    "CpomdpGovernanceAgent",
    "GovernanceAgent",
    "GovernanceBudget",
    "GovernanceVerdict",
    "NoGovernanceAgent",
    "RegimeModel",
    "ThresholdEconomics",
    "myopic_threshold",
    "rights_for",
    "solve_threshold",
]


def __getattr__(name: str) -> Any:
    if name == "CpomdpGovernanceAgent":
        from xai_gov.governance.cpomdp import CpomdpGovernanceAgent as agent

        return agent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
