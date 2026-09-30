"""Scalable oversight: learned delegation and the simulated supervisor."""

from __future__ import annotations

from xai_gov.oversight.delegation import (
    ComplementarityLedger,
    DeferralDecision,
    DeferralPolicy,
    LearnedDeferral,
    NoDeferral,
    ThresholdDeferral,
    available_deferral_policies,
    build_deferral_policy,
)
from xai_gov.oversight.supervisor import SimulatedSupervisor, SupervisorJudgement

__all__ = [
    "ComplementarityLedger",
    "DeferralDecision",
    "DeferralPolicy",
    "LearnedDeferral",
    "NoDeferral",
    "SimulatedSupervisor",
    "SupervisorJudgement",
    "ThresholdDeferral",
    "available_deferral_policies",
    "build_deferral_policy",
]
