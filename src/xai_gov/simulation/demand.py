"""Demand processes.

Each regime is an explicit, seeded generator of exogenous demand at the
most downstream echelon. Regimes are experimental factors, so they are
constructed from configuration and carry a stable ``name`` that appears in
every artifact; a result can never be traced to an anonymous "demand
model".

The distinction that matters for the project is between regimes that keep
the distribution exchangeable over the horizon (``stable``, ``growing``)
and those that break it (``double_shock``, ``high_volatility``,
``structural_change``). The conformal layer of stage 3 is evaluated against
exactly that contrast, and `DemandProcess.shift_periods` declares where the
breaks are so the analysis need not rediscover them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DemandProcess(ABC):
    """A seeded exogenous demand process over a finite horizon."""

    name: str
    mean: float
    dispersion: float = 0.2

    @abstractmethod
    def intensity(self, period: int) -> float:
        """Expected demand at ``period``, before noise."""

    @property
    def shift_periods(self) -> tuple[int, ...]:
        """Periods at which exchangeability is deliberately broken."""
        return ()

    @property
    def exchangeable(self) -> bool:
        return not self.shift_periods

    def draw(self, period: int, rng: np.random.Generator) -> float:
        """Draw realized demand.

        A gamma is used rather than a normal: demand is non-negative and
        right-skewed, and a truncated normal would quietly distort the mean
        exactly in the tight-capacity regimes where it matters.
        """
        intensity = max(self.intensity(period), 1e-9)
        if self.dispersion <= 0.0:
            return float(intensity)
        shape = 1.0 / (self.dispersion**2)
        scale = intensity / shape
        return float(rng.gamma(shape=shape, scale=scale))

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean": self.mean,
            "dispersion": self.dispersion,
            "shift_periods": list(self.shift_periods),
            "exchangeable": self.exchangeable,
        }


@dataclass(frozen=True)
class StableDemand(DemandProcess):
    """Stationary demand: the in-distribution reference."""

    def intensity(self, period: int) -> float:
        return self.mean


@dataclass(frozen=True)
class GrowingDemand(DemandProcess):
    """Deterministic trend. Non-stationary in mean, but with no break."""

    growth_per_period: float = 0.01

    def intensity(self, period: int) -> float:
        return self.mean * (1.0 + self.growth_per_period) ** period


@dataclass(frozen=True)
class DoubleShockDemand(DemandProcess):
    """Two step changes: the canonical disruption-and-recovery pattern."""

    first_shock_period: int = 10
    second_shock_period: int = 20
    first_multiplier: float = 2.0
    second_multiplier: float = 0.5

    def intensity(self, period: int) -> float:
        if period >= self.second_shock_period:
            return self.mean * self.second_multiplier
        if period >= self.first_shock_period:
            return self.mean * self.first_multiplier
        return self.mean

    @property
    def shift_periods(self) -> tuple[int, ...]:
        return (self.first_shock_period, self.second_shock_period)


@dataclass(frozen=True)
class HighVolatilityDemand(DemandProcess):
    """Stationary in mean, volatile in scale.

    Dispersion itself oscillates, which is the case a detector calibrated
    on a quiet window handles worst: the mean never moves, so a
    mean-shift test sees nothing.
    """

    volatility_multiplier: float = 3.0
    cycle: int = 5

    def intensity(self, period: int) -> float:
        return self.mean

    def draw(self, period: int, rng: np.random.Generator) -> float:
        phase = (period % self.cycle) / max(self.cycle - 1, 1)
        dispersion = self.dispersion * (1.0 + (self.volatility_multiplier - 1.0) * phase)
        shape = 1.0 / (dispersion**2)
        return float(rng.gamma(shape=shape, scale=self.mean / shape))


@dataclass(frozen=True)
class StructuralChangeDemand(DemandProcess):
    """One unannounced regime change, with a new mean and a new dispersion.

    This is the transfer condition of the protocol: nothing signals the
    change in advance, and the post-change process is not a scaling of the
    pre-change one.
    """

    change_period: int = 15
    new_mean_multiplier: float = 1.6
    new_dispersion: float = 0.45

    def intensity(self, period: int) -> float:
        return self.mean * self.new_mean_multiplier if period >= self.change_period else self.mean

    def draw(self, period: int, rng: np.random.Generator) -> float:
        dispersion = self.new_dispersion if period >= self.change_period else self.dispersion
        intensity = max(self.intensity(period), 1e-9)
        shape = 1.0 / (dispersion**2)
        return float(rng.gamma(shape=shape, scale=intensity / shape))

    @property
    def shift_periods(self) -> tuple[int, ...]:
        return (self.change_period,)


_REGISTRY: dict[str, type[DemandProcess]] = {
    "stable": StableDemand,
    "growing": GrowingDemand,
    "double_shock": DoubleShockDemand,
    "high_volatility": HighVolatilityDemand,
    "structural_change": StructuralChangeDemand,
}


def available_regimes() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def build_demand(config: dict[str, Any]) -> DemandProcess:
    """Construct a demand process from its configuration block.

    Only keys the chosen regime actually declares are forwarded; an unknown
    key is an error rather than a silently ignored typo, because a factor
    that fails to apply would corrupt a whole experimental cell.
    """
    regime = str(config.get("regime", "stable"))
    if regime not in _REGISTRY:
        raise ValueError(
            f"unknown demand regime {regime!r}; available: {', '.join(available_regimes())}"
        )
    cls = _REGISTRY[regime]
    declared = set(cls.__dataclass_fields__)
    kwargs: dict[str, Any] = {
        "name": regime,
        "mean": float(config.get("mean", 10.0)),
        "dispersion": float(config.get("dispersion", 0.2)),
    }
    reserved = {"regime", "mean", "dispersion", "name"}
    unknown = [key for key in config if key not in reserved and key not in declared]
    if unknown:
        raise ValueError(
            f"demand regime {regime!r} does not accept {', '.join(sorted(unknown))}; "
            f"accepted keys: {', '.join(sorted(declared - {'name'}))}"
        )
    for key, value in config.items():
        if key not in reserved and key in declared:
            kwargs[key] = value
    return cls(**kwargs)
