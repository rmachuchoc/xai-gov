"""Conformal risk control.

Coverage is the wrong target for a governance layer. A prediction interval
that covers the demand 90% of the time says nothing about the quantity that
actually matters: the probability that an *unsafe action* passes the filter
undetected. Those two differ whenever unsafe actions are concentrated in the
tail — which is exactly the case here.

This module therefore calibrates the threshold against a declared loss
function rather than against coverage (Angelopoulos et al., 2022). Given a
calibration set of (score, loss) pairs, it selects the most permissive
threshold λ whose risk — the expected loss of actions the filter lets
through — is provably below the tolerance α:

    R̂(λ) + bound(n, δ) ≤ α

The upper confidence bound is what makes the statement a guarantee rather
than an observation: without it, λ is fitted to the calibration set and its
risk on new data is unknown. A Hoeffding bound is used, which is
distribution free for a loss in [0, 1] and conservative — the conservatism is
the price of assuming nothing about the loss distribution, and it is stated
in the run record so nobody mistakes the resulting λ for the tightest one.

The monotonicity that makes the search valid: risk is non-increasing in
strictness. A lower λ flags more, so fewer unsafe actions survive. The
search therefore returns the largest admissible λ — flagging as little as
the risk tolerance permits, since every flag costs governance budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def hoeffding_bound(n: int, delta: float) -> float:
    """One-sided Hoeffding deviation for a loss bounded in [0, 1]."""
    if n <= 0:
        return 1.0
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1)")
    return math.sqrt(math.log(1.0 / delta) / (2.0 * n))


@dataclass(frozen=True, slots=True)
class RiskCalibration:
    """The outcome of one calibration, with everything needed to audit it."""

    threshold: float | None
    empirical_risk: float
    bound: float
    tolerance: float
    sample_size: int
    admissible: bool
    reason: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "empirical_risk": round(self.empirical_risk, 6),
            "bound": round(self.bound, 6),
            "tolerance": self.tolerance,
            "sample_size": self.sample_size,
            "admissible": self.admissible,
            "reason": self.reason,
        }


@dataclass(slots=True)
class ConformalRiskController:
    """Calibrates a threshold against a safety loss, not against coverage."""

    tolerance: float = 0.05
    delta: float = 0.1
    window: int = 200
    min_samples: int = 20
    _scores: list[float] = field(default_factory=list, repr=False)
    _losses: list[float] = field(default_factory=list, repr=False)
    calibrations: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.tolerance < 1.0:
            raise ValueError("tolerance must lie in (0, 1)")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must lie in (0, 1)")
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")

    def observe(self, score: float, loss: float) -> None:
        """Record one (score, realized safety loss) pair.

        The loss must lie in [0, 1]; the Hoeffding bound is only valid for a
        bounded loss, and silently clipping an out-of-range value would
        invalidate the guarantee while appearing to work.
        """
        if not 0.0 <= loss <= 1.0:
            raise ValueError(f"safety loss {loss} must lie in [0, 1]")
        self._scores.append(float(score))
        self._losses.append(float(loss))
        if len(self._scores) > self.window:
            del self._scores[: len(self._scores) - self.window]
            del self._losses[: len(self._losses) - self.window]

    @property
    def size(self) -> int:
        return len(self._scores)

    def risk_at(self, threshold: float) -> float:
        """Empirical risk of the actions a threshold lets through.

        Only unflagged observations contribute: an action the filter caught
        was never executed, so its loss is not the filter's risk. The
        denominator is the full sample, because a filter that flags almost
        everything must not be credited with low risk merely for having a
        small pass-through set.
        """
        if not self._scores:
            return 0.0
        survived = sum(
            loss for score, loss in zip(self._scores, self._losses, strict=True)
            if score <= threshold
        )
        return survived / len(self._scores)

    def calibrate(self) -> RiskCalibration:
        """Select the most permissive threshold whose risk bound holds."""
        n = self.size
        if n < self.min_samples:
            return RiskCalibration(
                threshold=None,
                empirical_risk=0.0,
                bound=1.0,
                tolerance=self.tolerance,
                sample_size=n,
                admissible=False,
                reason=f"needs {self.min_samples} calibration pairs, has {n}",
            )

        bound = hoeffding_bound(n, self.delta)
        if bound >= self.tolerance:
            # No threshold can be certified at this tolerance with this much
            # data. Reporting the tightest observed threshold anyway would be
            # presenting an uncertified number as a guarantee.
            return RiskCalibration(
                threshold=None,
                empirical_risk=self.risk_at(min(self._scores)),
                bound=bound,
                tolerance=self.tolerance,
                sample_size=n,
                admissible=False,
                reason=(
                    f"confidence bound {bound:.4f} already exceeds tolerance "
                    f"{self.tolerance:.4f}; more calibration data is required"
                ),
            )

        best: float | None = None
        best_risk = 0.0
        # Candidates are the observed scores: risk only changes there.
        for candidate in sorted(set(self._scores)):
            risk = self.risk_at(candidate)
            if risk + bound <= self.tolerance:
                best, best_risk = candidate, risk
            else:
                break  # risk is non-decreasing in the threshold
        self.calibrations += 1

        if best is None:
            return RiskCalibration(
                threshold=None,
                empirical_risk=self.risk_at(min(self._scores)),
                bound=bound,
                tolerance=self.tolerance,
                sample_size=n,
                admissible=False,
                reason="even the strictest threshold exceeds the risk tolerance",
            )
        return RiskCalibration(
            threshold=best,
            empirical_risk=best_risk,
            bound=bound,
            tolerance=self.tolerance,
            sample_size=n,
            admissible=True,
            reason="risk bound satisfied",
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "tolerance": self.tolerance,
            "delta": self.delta,
            "window": self.window,
            "size": self.size,
            "calibrations": self.calibrations,
        }


def build_risk_controller(config: dict[str, Any]) -> ConformalRiskController | None:
    """Construct the risk controller, or None when risk control is off."""
    if not config or not config.get("enabled", False):
        return None
    params = config.get("params", {})
    if not isinstance(params, dict):
        raise ValueError("risk control 'params' must be a mapping")
    try:
        return ConformalRiskController(**params)
    except TypeError as error:
        raise ValueError(f"risk controller rejected its parameters: {error}") from error
