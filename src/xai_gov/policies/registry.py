"""Policy registry.

Policies are named in configuration and constructed here. The registry is
the single place that knows which families exist, so adding the XAI, MARL
or language-model arms in a later stage touches one file and no call site.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from xai_gov.policies.base import Policy
from xai_gov.policies.families import HeuristicSQPolicy, OpaqueDROPolicy, RandomBoundedPolicy

_BUILDERS: dict[str, Callable[..., Policy]] = {
    "heuristic_sQ": HeuristicSQPolicy,
    "opaque_dro": OpaqueDROPolicy,
    "random_bounded": RandomBoundedPolicy,
}

# Families declared in the protocol that do not exist yet. Naming one in a
# configuration fails with the stage that will deliver it, rather than with
# an opaque KeyError.
_PLANNED: dict[str, str] = {
    "xai": "stage 4 (structural causal model and causal recourse)",
    "xai_conformal": "stage 4",
    "marl_heterogeneous": "stage 5 (heterogeneous MARL baseline)",
    "llm_agent": "stage 5 (local language-model agent)",
}


def available_policies() -> tuple[str, ...]:
    return tuple(sorted(_BUILDERS))


def build_policy(config: dict[str, Any]) -> Policy:
    """Construct a policy from its configuration block."""
    name = str(config.get("name", "heuristic_sQ"))
    if name in _PLANNED:
        raise NotImplementedError(
            f"policy family {name!r} is scheduled for {_PLANNED[name]}; "
            f"available now: {', '.join(available_policies())}"
        )
    builder = _BUILDERS.get(name)
    if builder is None:
        raise ValueError(
            f"unknown policy {name!r}; available: {', '.join(available_policies())}"
        )
    params = config.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"policy {name!r}: 'params' must be a mapping")
    try:
        return builder(**params)
    except TypeError as error:
        raise ValueError(f"policy {name!r} rejected its parameters: {error}") from error
