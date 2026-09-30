"""Conformal calibration: the classical scheme, and its weighted repair.

`SplitCalibrator` is the textbook split-conformal quantile. It carries a
marginal coverage guarantee **under exchangeability**, and the project keeps
it precisely so that the guarantee can be shown to fail: it is the control
arm for RQ6.

`WeightedCalibrator` implements the non-exchangeable repair — calibration
residuals are reweighted so recent observations dominate. It recovers
approximate coverage with an explicit penalty in the distance between the
calibration and test distributions, rather than assuming that distance away.

Neither class adapts its *level*; that is the job of the adaptive scheme.
Keeping quantile estimation and level adaptation in separate objects is what
makes the two mechanisms separately testable — and the protocol's claim is
about the combination, so the parts must be measurable alone.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from xai_gov.core.variants import build_variant


class Calibrator(ABC):
    """Turns a window of past scores into a threshold at a given level."""

    name: str = "abstract"

    @abstractmethod
    def observe(self, score: float) -> None:
        """Add one score to the calibration set."""

    @abstractmethod
    def threshold(self, level: float) -> float | None:
        """Return the (1 - level) quantile, or None while uncalibrated."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of scores currently held."""

    @abstractmethod
    def scores(self) -> list[float]:
        """The calibration scores, oldest first.

        Exposed as part of the interface because the conformal p-value needs
        the same set the threshold was drawn from. Reaching into a private
        attribute for it would break silently the day a calibrator changes
        its storage.
        """

    def to_payload(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size}


def conformal_quantile(sorted_scores: list[float], level: float) -> float:
    """The finite-sample conformal quantile.

    The index is ceil((n + 1)(1 - level)) rather than the empirical
    quantile's ceil(n(1 - level)). The +1 is the whole finite-sample
    correction: it accounts for the test point itself joining the ranking,
    and dropping it is what turns a valid guarantee into an approximate one.
    When the index exceeds n the quantile is unbounded — with too few points
    at this level no finite threshold is justified, and the caller must be
    told so rather than handed the maximum observed score.
    """
    n = len(sorted_scores)
    if n == 0:
        raise ValueError("cannot compute a quantile of an empty calibration set")
    index = math.ceil((n + 1) * (1.0 - level))
    if index > n:
        return float("nan")  # signals "no finite threshold at this level"
    return sorted_scores[max(index - 1, 0)]


@dataclass(slots=True)
class SplitCalibrator(Calibrator):
    """Classical split conformal over a rolling window.

    Valid under exchangeability. The rolling window is itself a mild
    violation of that assumption, and it is used because an unbounded window
    would never respond to a regime change at all; the resulting guarantee is
    approximate and the project measures the gap rather than asserting it.
    """

    window: int = 100
    name: str = "split"
    _scores: deque[float] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must hold at least two scores")
        self._scores = deque(maxlen=self.window)

    def observe(self, score: float) -> None:
        self._scores.append(float(score))

    @property
    def size(self) -> int:
        return len(self._scores)

    def scores(self) -> list[float]:
        return list(self._scores)

    def threshold(self, level: float) -> float | None:
        """None means no finite threshold is justified — from too few scores
        or from a level too extreme for the sample size. Both cases mean the
        same thing operationally: nothing may be flagged."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        if self.size < 2:
            return None
        quantile = conformal_quantile(sorted(self._scores), level)
        return None if quantile != quantile else quantile  # NaN: unbounded


@dataclass(slots=True)
class WeightedCalibrator(Calibrator):
    """Conformal prediction beyond exchangeability, by recency weighting.

    Each score carries weight ``decay ** age``. The weighted quantile is the
    smallest score whose cumulative normalized weight reaches 1 - level.

    The coverage penalty is not hidden: `total_variation_budget` reports the
    weight mass that a distribution shift can move, which is the quantity
    Theorem 1 bounds. A decay close to 1 approaches the exchangeable case
    (large effective sample, no adaptivity); a small decay adapts fast at the
    cost of an effective sample size that may be too small to justify any
    finite threshold at the requested level.
    """

    window: int = 200
    decay: float = 0.97
    name: str = "weighted"
    _scores: deque[float] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        if self.window < 2:
            raise ValueError("window must hold at least two scores")
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must lie in (0, 1]")
        self._scores = deque(maxlen=self.window)

    def observe(self, score: float) -> None:
        self._scores.append(float(score))

    @property
    def size(self) -> int:
        return len(self._scores)

    def scores(self) -> list[float]:
        return list(self._scores)

    def weights(self) -> list[float]:
        """Weights aligned with insertion order, newest last."""
        n = self.size
        return [self.decay ** (n - 1 - index) for index in range(n)]

    @property
    def effective_sample_size(self) -> float:
        """Kish effective sample size: (Σw)² / Σw².

        The honest denominator for a weighted guarantee, and the quantity that
        bounds the level a window can support: the conformal quantile at level
        ε needs the score at index ceil((n+1)(1-ε)), so at most a
        (1 - ceil((n+1)(1-ε))/n) fraction of the set can ever exceed the
        threshold. When that ceiling falls below ε the detector cannot reach its
        own target however the level adapts.

        For a truncated geometric window this is close to (1+decay)/(1-decay),
        not 1/(1-decay) — the second moment of the weights carries a factor of
        two that is easy to drop and doubles the apparent memory.
        """
        weights = self.weights()
        if not weights:
            return 0.0
        total = sum(weights)
        return (total * total) / sum(w * w for w in weights)

    @property
    def total_variation_budget(self) -> float:
        """Weight mass in the oldest half of the window.

        A proxy for how much of the calibration set predates a possible
        change point, and therefore for the coverage penalty a shift induces.
        """
        weights = self.weights()
        if not weights:
            return 0.0
        half = len(weights) // 2
        return sum(weights[:half]) / sum(weights)

    def threshold(self, level: float) -> float | None:
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        if self.size < 2:
            return None
        # The test point is given the newest weight, mirroring the +1 of the
        # unweighted conformal quantile.
        weights = self.weights()
        test_weight = 1.0
        total = sum(weights) + test_weight
        paired = sorted(zip(self._scores, weights, strict=True), key=lambda item: item[0])

        cumulative = 0.0
        for score, weight in paired:
            cumulative += weight
            if cumulative / total >= 1.0 - level:
                return float(score)
        # The test point's own weight is all that remains: no finite
        # threshold is justified at this level.
        return None

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "decay": self.decay,
            "effective_sample_size": round(self.effective_sample_size, 4),
            "total_variation_budget": round(self.total_variation_budget, 4),
        }


def build_calibrator(config: dict[str, Any]) -> Calibrator:
    """Construct a calibrator from configuration."""
    # Annotated explicitly: inferring T from a dict of two ABC subclasses
    # collapses to Never, because a shared base is not a common return type
    # unless it is named.
    variants: dict[str, type[Calibrator]] = {
        "split": SplitCalibrator,
        "weighted": WeightedCalibrator,
    }
    return build_variant(
        kind="calibrator",
        name=str(config.get("name", "weighted")),
        params=config.get("params", {}),
        variants=variants,
    )
