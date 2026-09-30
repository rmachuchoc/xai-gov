"""Causal layer: structural model, recourse, fidelity, descriptors."""

from __future__ import annotations

from xai_gov.causal.descriptor import (
    CausalRecourseDescriptor,
    DescriptorBuilder,
    NoDescriptor,
    PostHocDescriptor,
    available_descriptors,
    build_descriptor,
)
from xai_gov.causal.fidelity import FidelityReport, explanatory_fidelity
from xai_gov.causal.recourse import (
    Counterfactual,
    RecourseResult,
    RecourseSearch,
    build_recourse_search,
)
from xai_gov.causal.scm import (
    CausalWorld,
    StructuralCausalModel,
    Variable,
    reorder_rule,
    world_from_state,
)

__all__ = [
    "CausalRecourseDescriptor",
    "CausalWorld",
    "Counterfactual",
    "DescriptorBuilder",
    "FidelityReport",
    "NoDescriptor",
    "PostHocDescriptor",
    "RecourseResult",
    "RecourseSearch",
    "StructuralCausalModel",
    "Variable",
    "available_descriptors",
    "build_descriptor",
    "build_recourse_search",
    "explanatory_fidelity",
    "reorder_rule",
    "world_from_state",
]
