"""Test martingales for regime change.

A count of exceedances cannot say *when* the distribution changed without a
threshold chosen after seeing the data — the optional-stopping problem the
protocol commits to avoiding. A test martingale replaces the count with
accumulated evidence: a sequence of bets against the hypothesis of
exchangeability whose expected value under that hypothesis is bounded by its
starting value.

For a nonnegative martingale started at 1, Ville's inequality bounds
P( sup_t M_t ≥ 1/δ ) by δ. **That bound is not claimed here.** It needs the
p-values to be exactly uniform and independent under the null, and these are
not: the calibration set is recency-weighted and updated online, the scores
come from a forecaster that keeps learning, and the demand an upstream node
sees is its customer's orders. So 1/δ is used only as the declaration
threshold, and the false-alarm rate is *measured*, on stationary null runs
(`analysis.detector_null`), instead of being bounded by δ.

The bets are placed on conformal p-values. Under exchangeability those are
uniform on (0, 1); a power bet p^(ε−1) pays off when p-values run small,
which is what a distribution shift produces. Because no single ε is right
for an unknown shift magnitude, the implementation mixes over a grid of ε —
a convex mixture of martingales is itself a martingale, so mixing costs no
validity while removing an arbitrary tuning choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

_LOG_CAP = 700.0  # exp() overflows near 709


def conformal_p_value(score: float, calibration: list[float]) -> float:
    """The smoothed conformal p-value of ``score``.

    Defined as (1 + #{s ≥ score}) / (1 + n): the +1 in both places is the
    finite-sample correction, and it guarantees the p-value is never exactly
    zero. A zero p-value would make the martingale infinite on a single
    observation, which is not evidence but arithmetic.
    """
    n = len(calibration)
    if n == 0:
        return 1.0
    at_least = sum(1 for value in calibration if value >= score)
    return (1.0 + at_least) / (1.0 + n)


@dataclass(slots=True)
class MixturePowerMartingale:
    """Mixture of power martingales over a grid of betting exponents.

    Tracked in log space: the martingale grows multiplicatively and a shift
    of any duration overflows a float within a few dozen periods otherwise.
    """

    epsilons: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 0.95)
    delta: float = 0.01
    _log_components: list[float] = field(default_factory=list, repr=False)
    observations: int = 0
    declared_at: int | None = None
    peak_log_value: float = 0.0

    def __post_init__(self) -> None:
        if not self.epsilons:
            raise ValueError("at least one betting exponent is required")
        for epsilon in self.epsilons:
            if not 0.0 < epsilon < 1.0:
                raise ValueError(f"betting exponent {epsilon} must lie in (0, 1)")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")
        self._log_components = [0.0] * len(self.epsilons)

    # -- state ------------------------------------------------------------
    @property
    def threshold(self) -> float:
        """The declaration threshold 1/δ (a threshold, not a guarantee)."""
        return 1.0 / self.delta

    @property
    def log_threshold(self) -> float:
        return -math.log(self.delta)

    @property
    def log_value(self) -> float:
        """log of the mixture, computed stably."""
        if not self._log_components:
            return 0.0
        largest = max(self._log_components)
        total = sum(math.exp(component - largest) for component in self._log_components)
        return largest + math.log(total / len(self._log_components))

    @property
    def value(self) -> float:
        """The martingale value, capped so it stays a finite float.

        The cap is cosmetic: once the value is astronomically past 1/δ the
        exact magnitude carries no additional decision content, and the log
        value is retained for anyone who wants it.
        """
        return math.exp(min(self.log_value, _LOG_CAP))

    @property
    def change_declared(self) -> bool:
        return self.declared_at is not None

    @property
    def evidence_ratio(self) -> float:
        """How far past the stopping threshold the evidence has run, in logs."""
        return self.log_value - self.log_threshold

    # -- updating ---------------------------------------------------------
    def update(self, p_value: float) -> bool:
        """Place one round of bets. Returns True the period a change is declared."""
        if not 0.0 < p_value <= 1.0:
            raise ValueError(f"p-value {p_value} must lie in (0, 1]")
        log_p = math.log(p_value)
        for index, epsilon in enumerate(self.epsilons):
            # log of ε · p^(ε−1)
            self._log_components[index] += math.log(epsilon) + (epsilon - 1.0) * log_p
        self.observations += 1
        self.peak_log_value = max(self.peak_log_value, self.log_value)

        if self.declared_at is None and self.log_value >= self.log_threshold:
            self.declared_at = self.observations
            return True
        return False

    def reset(self) -> None:
        """Restart the bets after a declared change has been acted upon.

        Resetting is a modeling choice: the next betting episode starts from
        1 again. No false-alarm bound is claimed for either episode; the null
        line measures the rate.
        """
        self._log_components = [0.0] * len(self.epsilons)
        self.declared_at = None
        self.observations = 0

    def to_payload(self) -> dict[str, Any]:
        return {
            "delta": self.delta,
            "declaration_threshold": round(self.threshold, 6),
            # Decision 6 of the revision: no Ville bound is claimed.
            "false_alarm_guarantee": None,
            "false_alarm_rate": "measured on the stationary null line",
            "epsilons": list(self.epsilons),
            "observations": self.observations,
            "log_value": round(self.log_value, 6),
            "log_threshold": round(self.log_threshold, 6),
            "peak_log_value": round(self.peak_log_value, 6),
            "change_declared": self.change_declared,
            "declared_at": self.declared_at,
        }


@dataclass(slots=True)
class NoMartingale:
    """Control arm: no change detection at all."""

    observations: int = 0
    declared_at: int | None = None

    @property
    def value(self) -> float:
        return 1.0

    @property
    def log_value(self) -> float:
        return 0.0

    @property
    def change_declared(self) -> bool:
        return False

    def update(self, p_value: float) -> bool:
        self.observations += 1
        return False

    def reset(self) -> None:
        self.observations = 0

    def to_payload(self) -> dict[str, Any]:
        return {"delta": None, "observations": self.observations, "change_declared": False}


ChangeDetector = MixturePowerMartingale | NoMartingale


def build_martingale(config: dict[str, Any]) -> ChangeDetector:
    """Construct the change detector named in configuration."""
    if not config or not config.get("enabled", True):
        return NoMartingale()
    params = dict(config.get("params", {}))
    epsilons = params.pop("epsilons", None)
    if epsilons is not None:
        params["epsilons"] = tuple(float(value) for value in epsilons)
    try:
        return MixturePowerMartingale(**params)
    except TypeError as error:
        raise ValueError(f"martingale rejected its parameters: {error}") from error
