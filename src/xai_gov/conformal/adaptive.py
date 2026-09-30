"""Adaptive conformal inference.

The classical split scheme fixes the level and lets coverage drift when the
distribution moves. Adaptive conformal inference (Gibbs and Candès, 2021)
inverts that: the level is updated online,

    ε_{t+1} = clip( ε_t + γ (ε_target − err_t) )

where ``err_t`` is 1 when the observation exceeded the threshold in force.
The mechanism is a feedback controller. Flagging more often than the target
drives ε down, which raises the threshold and reduces flagging; flagging too
rarely does the reverse. Long-run average coverage is then attained with no
distributional assumption whatsoever — which is the property the project
needs, because the assumption the classical scheme relies on is exactly the
one a disruption destroys.

Two costs are paid for that, and both are reported rather than glossed:
interval width becomes variable, and the guarantee is on the *average* over
the horizon, not on any single period.

`ClippedAci` bounds ε away from 0 and 1. Unbounded ε is the standard
formulation's known pathology: a long quiet stretch drives ε toward 0, the
threshold becomes unbounded, and the detector goes blind at precisely the
moment a disruption starts. The clip is what keeps the detector responsive,
and its width is an experimental factor rather than a hidden constant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xai_gov.core.variants import build_variant


@dataclass(slots=True)
class AciState:
    """The level controller of adaptive conformal inference."""

    target_level: float = 0.1
    gamma: float = 0.02
    min_level: float = 0.005
    max_level: float = 0.5
    level: float = field(init=False)
    updates: int = 0
    exceedances: int = 0
    clipped_low: int = 0
    clipped_high: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.target_level < 1.0:
            raise ValueError("target_level must lie in (0, 1)")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive; a zero rate is the split scheme")
        if not 0.0 < self.min_level < self.max_level < 1.0:
            raise ValueError("levels must satisfy 0 < min_level < max_level < 1")
        if not self.min_level <= self.target_level <= self.max_level:
            raise ValueError("target_level must lie within [min_level, max_level]")
        self.level = self.target_level

    def update(self, *, exceeded: bool) -> float:
        """Fold one outcome in and return the new level."""
        error = 1.0 if exceeded else 0.0
        proposed = self.level + self.gamma * (self.target_level - error)
        if proposed < self.min_level:
            proposed = self.min_level
            self.clipped_low += 1
        elif proposed > self.max_level:
            proposed = self.max_level
            self.clipped_high += 1
        self.level = proposed
        self.updates += 1
        self.exceedances += int(exceeded)
        return self.level

    @property
    def realized_rate(self) -> float:
        """Empirical exceedance rate, which should track ``target_level``."""
        return self.exceedances / self.updates if self.updates else 0.0

    @property
    def coverage_error(self) -> float:
        """Realized rate minus target: the quantity Theorem 1 bounds.

        Its magnitude, not its sign, is what the theorem constrains. Reported
        per run as CCR in the guarantees KPI layer.
        """
        return self.realized_rate - self.target_level

    @property
    def saturated(self) -> bool:
        """True when the controller spent most of its updates against a bound.

        A saturated controller is not adapting, it is pinned. This is the
        diagnostic that distinguishes a genuinely anomalous environment from
        the detector failure the pilot exhibited, and it belongs in the run
        record either way.
        """
        if self.updates < 10:
            return False
        return (self.clipped_low + self.clipped_high) / self.updates > 0.5

    def to_payload(self) -> dict[str, Any]:
        return {
            "scheme": "aci",
            "target_level": self.target_level,
            "gamma": self.gamma,
            "level": round(self.level, 6),
            "updates": self.updates,
            "exceedances": self.exceedances,
            "realized_rate": round(self.realized_rate, 6),
            "coverage_error": round(self.coverage_error, 6),
            "clipped_low": self.clipped_low,
            "clipped_high": self.clipped_high,
            "saturated": self.saturated,
        }


@dataclass(slots=True)
class FixedLevel:
    """The non-adaptive control arm: the level never moves.

    Present so that RQ6 has something to compare against. Its coverage error
    under shift is the quantity the adaptive scheme is claimed to fix, and a
    claim with no measured baseline is not a result.
    """

    target_level: float = 0.1
    level: float = field(init=False)
    updates: int = 0
    exceedances: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.target_level < 1.0:
            raise ValueError("target_level must lie in (0, 1)")
        self.level = self.target_level

    def update(self, *, exceeded: bool) -> float:
        self.updates += 1
        self.exceedances += int(exceeded)
        return self.level

    @property
    def realized_rate(self) -> float:
        return self.exceedances / self.updates if self.updates else 0.0

    @property
    def coverage_error(self) -> float:
        return self.realized_rate - self.target_level

    @property
    def saturated(self) -> bool:
        return False

    def to_payload(self) -> dict[str, Any]:
        return {
            "scheme": "fixed",
            "target_level": self.target_level,
            "level": self.level,
            "updates": self.updates,
            "exceedances": self.exceedances,
            "realized_rate": round(self.realized_rate, 6),
            "coverage_error": round(self.coverage_error, 6),
        }


LevelController = AciState | FixedLevel


def build_level_controller(config: dict[str, Any]) -> LevelController:
    """Construct the level controller named in configuration.

    Parameters are filtered to the fields the chosen scheme accepts; see
    ``core.variants`` for why that is the correct treatment and where the
    line between a merge artifact and a typo is drawn.
    """
    return build_variant(
        kind="level controller",
        name=str(config.get("scheme", "aci")),
        params=config.get("params", {}),
        variants={"aci": AciState, "fixed": FixedLevel},
    )
