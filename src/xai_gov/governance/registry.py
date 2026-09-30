"""Governance architecture registry.

Naming an architecture in configuration resolves it here. The registry is
also where a not-yet-built architecture reports the stage that will deliver
it, so a configuration referring to `federated` fails with a sentence rather
than a KeyError.
"""

from __future__ import annotations

from typing import Any

from xai_gov.causal.descriptor import build_descriptor
from xai_gov.governance.agent import GovernanceAgent, NoGovernanceAgent
from xai_gov.governance.belief import build_regime_model
from xai_gov.governance.budget import build_budget
from xai_gov.governance.thresholds import build_economics
from xai_gov.oversight.delegation import build_deferral_policy
from xai_gov.oversight.supervisor import SimulatedSupervisor

_PLANNED: dict[str, str] = {
    "multi_role": "stage 6 (role-dependent observations across echelons)",
    "federated": "stage 6 (federated governance across echelons)",
}

_DEFAULT_CONFORMAL: dict[str, Any] = {
    "score": "studentized_residual",
    "level": {"scheme": "aci", "params": {"target_level": 0.1, "gamma": 0.02}},
    "calibrator": {"name": "weighted", "params": {"window": 200, "decay": 0.97}},
    "martingale": {"enabled": True, "params": {"delta": 0.01}},
    "warmup": 10,
}


def _build_shield(config: dict[str, Any]) -> Any:
    """Construct the shield, or the control arm when it is disabled."""
    from xai_gov.verification.shield import NoShield, Shield
    from xai_gov.verification.specification import build_specification

    if not config or not config.get("shield", False):
        return NoShield()
    try:
        return Shield(
            specification=build_specification(config.get("specification")),
            horizon_lookahead=int(config.get("horizon_lookahead", 1)),
        )
    except TypeError as error:
        raise ValueError(f"shield rejected its parameters: {error}") from error


def _build_monitor(config: dict[str, Any]) -> Any:
    """Construct the runtime monitor.

    Monitors default to on whenever a specification is present, including in
    the unshielded control arm. Instrumenting only the shielded arm would make
    the two incomparable, and the price of verified safety is a difference
    between them.
    """
    from xai_gov.verification.model_checker import Abstraction
    from xai_gov.verification.monitors import NoMonitor, RuntimeMonitor
    from xai_gov.verification.specification import build_specification

    if not config or not config.get("monitors", False):
        return NoMonitor()
    abstraction_config = config.get("abstraction", {})
    if not isinstance(abstraction_config, dict):
        raise ValueError("verification 'abstraction' must be a mapping")
    kwargs: dict[str, Any] = {}
    for key in ("inventory_bands", "backlog_bands"):
        if key in abstraction_config:
            kwargs[key] = tuple(float(v) for v in abstraction_config[key])
    try:
        return RuntimeMonitor(
            specification=build_specification(config.get("specification")),
            abstraction=Abstraction(**kwargs),
        )
    except TypeError as error:
        raise ValueError(f"runtime monitor rejected its parameters: {error}") from error


def _build_supervisor(config: dict[str, Any], *, seed: int) -> SimulatedSupervisor | None:
    """Construct the simulated supervisor, when oversight is active.

    None when the deferral policy is 'none': a supervisor who is never consulted
    is not a neutral default but a claim that oversight exists, and the run
    record must not carry one.
    """
    if not config or str(config.get("policy", "none")) == "none":
        return None
    params = dict(config.get("supervisor", {}))
    accuracy = params.pop("accuracy", None)
    kwargs: dict[str, Any] = {"seed": seed}
    if accuracy is not None:
        if not isinstance(accuracy, dict):
            raise ValueError("supervisor 'accuracy' must be a mapping")
        kwargs["accuracy"] = {str(k): float(v) for k, v in accuracy.items()}
    kwargs.update(params)
    try:
        return SimulatedSupervisor(**kwargs)
    except TypeError as error:
        raise ValueError(f"supervisor rejected its parameters: {error}") from error


def available_architectures() -> tuple[str, ...]:
    return ("no_governance", "cpomdp", "centralized")


def build_governance(config: dict[str, Any], *, seed: int = 0) -> GovernanceAgent:
    """Construct the governance arm named in configuration."""
    name = str(config.get("architecture", "no_governance"))
    if name in _PLANNED:
        raise NotImplementedError(
            f"governance architecture {name!r} is scheduled for {_PLANNED[name]}; "
            f"available now: {', '.join(available_architectures())}"
        )

    regime = str(config.get("autonomy_regime", "H3"))

    if name == "no_governance":
        return NoGovernanceAgent(autonomy_regime=regime)

    if name in ("cpomdp", "centralized"):
        # 'centralized' is the protocol's name for one agent mediating every
        # node; that is what this implementation does, so the two names build
        # the same object. They stay distinct in configuration because the
        # factorial design contrasts centralized against multi-role.
        from xai_gov.conformal.detector import DetectorBank
        from xai_gov.governance.cpomdp import CpomdpGovernanceAgent

        conformal = config.get("conformal")
        if conformal is not None and not isinstance(conformal, dict):
            raise ValueError("governance 'conformal' must be a mapping")

        levels = config.get("levels", {})
        if not isinstance(levels, dict):
            raise ValueError("governance 'levels' must be a mapping")
        # Named explicitly rather than splatted: a **kwargs splat here could
        # silently overwrite autonomy_regime or a detector, so a typo in the
        # config would rebuild a different agent instead of being refused.
        unknown = set(levels) - {"escalation_belief", "safe_mode_belief", "quantity_bound"}
        if unknown:
            raise ValueError(
                f"unknown governance levels {sorted(unknown)}; expected "
                "escalation_belief, safe_mode_belief, quantity_bound"
            )

        try:
            return CpomdpGovernanceAgent(
                autonomy_regime=regime,
                detectors=DetectorBank(conformal or _DEFAULT_CONFORMAL),
                descriptor_builder=build_descriptor(config.get("explainability", {})),
                shield=_build_shield(config.get("verification", {})),
                monitor=_build_monitor(config.get("verification", {})),
                deferral=build_deferral_policy(config.get("oversight", {})),
                supervisor=_build_supervisor(config.get("oversight", {}), seed=seed),
                regime_model=build_regime_model(config.get("regime_model", {})),
                economics=build_economics(config.get("economics", {})),
                budget=build_budget(config.get("budget", {})),
                escalation_belief=float(levels.get("escalation_belief", 0.6)),
                safe_mode_belief=float(levels.get("safe_mode_belief", 0.9)),
                quantity_bound=float(levels.get("quantity_bound", 60.0)),
                recourse_loss_relief=float(config.get("recourse_loss_relief", 0.0)),
            )
        except TypeError as error:
            raise ValueError(f"governance {name!r} rejected its parameters: {error}") from error

    raise ValueError(
        f"unknown governance architecture {name!r}; "
        f"available: {', '.join(available_architectures())}"
    )
