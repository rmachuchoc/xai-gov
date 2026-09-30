"""Nonconformity scores and the forecaster they are measured against.

A nonconformity score answers one question: how atypical is this
observation relative to what was expected. The protocol needs the score to
be *calibrated*, so it is always reported together with the predictor that
produced the expectation — a raw residual whose predictor is unstated is
not auditable.

The forecaster here is deliberately simple: an exponentially weighted level
with an exponentially weighted mean absolute deviation. It is not the
project's forecasting contribution, it is the reference against which
nonconformity is defined. Two properties matter and both are satisfied:
it is causal (period t uses only observations up to t-1, so no target leaks
into its own score) and it is deterministic given the seed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

_MIN_SCALE = 1e-6


@dataclass(slots=True)
class EwmaForecaster:
    """Level and dispersion tracked by exponential weighting.

    ``alpha`` is the level rate and ``beta`` the dispersion rate. Dispersion
    adapts more slowly than level by default: a volatility regime that has
    genuinely changed should widen the scale, but a single outlier should not.

    ``scale_mode`` decides what the studentized score divides by, and it is the
    difference between a detector that sees a volatility shift and one that
    cannot. Under ``adaptive`` the scale co-adapts with the series, so when
    volatility rises the numerator and the denominator rise together and the
    score stays flat — the detector normalizes the shift away and under-flags.
    Under ``reference`` the scale is frozen once ``reference_after``
    observations have been seen, so a later change in dispersion registers as
    nonconformity, which is what a distribution-shift detector is for.

    The default stays ``adaptive`` because that is what the first campaign ran
    and changing it silently would make two studies incomparable.
    """

    alpha: float = 0.3
    beta: float = 0.1
    scale_mode: str = "adaptive"
    reference_after: int = 15
    level: float | None = None
    scale: float = 0.0
    observations: int = 0
    _reference_scale: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name, value in (("alpha", self.alpha), ("beta", self.beta)):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.scale_mode not in ("adaptive", "reference"):
            raise ValueError(
                f"scale_mode must be 'adaptive' or 'reference', not {self.scale_mode!r}"
            )
        if self.reference_after < 2:
            raise ValueError(
                "reference_after must be at least 2; a scale frozen from one "
                "observation is not an estimate of dispersion"
            )

    @property
    def initialized(self) -> bool:
        return self.level is not None

    def predict(self) -> float | None:
        """The one-step forecast, or None before the first observation."""
        return self.level

    def dispersion(self) -> float:
        """Current scale estimate, floored so a score never divides by zero.

        Returns the frozen reference once one exists. Before it is established
        the adaptive scale is used in both modes, because a detector has to
        divide by something while it is warming up.
        """
        if self.scale_mode == "reference" and self._reference_scale is not None:
            return max(self._reference_scale, _MIN_SCALE)
        return max(self.scale, _MIN_SCALE)

    @property
    def reference_scale(self) -> float | None:
        """The frozen scale, once the reference window has closed."""
        return self._reference_scale

    def update(self, observation: float) -> None:
        """Fold one observation in. Call *after* scoring it."""
        if self.level is None:
            self.level = observation
            # A first observation carries no information about spread; a
            # fraction of the level is used as a prior rather than zero,
            # which would make the very first score infinite.
            self.scale = abs(observation) * 0.2
        else:
            deviation = abs(observation - self.level)
            self.scale = (1.0 - self.beta) * self.scale + self.beta * deviation
            self.level = (1.0 - self.alpha) * self.level + self.alpha * observation
        self.observations += 1

        if (
            self.scale_mode == "reference"
            and self._reference_scale is None
            and self.observations >= self.reference_after
        ):
            self._reference_scale = self.scale

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "ewma",
            "alpha": self.alpha,
            "beta": self.beta,
            "scale_mode": self.scale_mode,
            "level": self.level,
            "scale": self.scale,
            "reference_scale": self._reference_scale,
            "observations": self.observations,
        }


@dataclass(slots=True)
class RidgeForecaster:
    """A learned one-step forecaster over recent lags.

    Present to demonstrate the property the conformal layer exists for: the
    coverage guarantee holds whatever the underlying predictor is. Showing it
    only for a hand-written exponentially weighted level is the weakest
    possible demonstration of a model-agnostic claim — the guarantee reads as a
    property of that forecaster rather than of the calibration.

    Fitted by ridge-regularized least squares on the history so far, refitted
    every ``refit_every`` periods. Two constraints keep it honest: the design
    matrix holds only lags strictly before the target, so no observation
    informs its own forecast; and the fit is deterministic given the data, so a
    run stays reproducible from its seed.
    """

    lags: int = 4
    ridge: float = 1.0
    refit_every: int = 8
    minimum_history: int = 12
    level: float | None = None
    scale: float = 0.0
    observations: int = 0
    _history: list[float] = field(default_factory=list, repr=False)
    _weights: list[float] = field(default_factory=list, repr=False)
    _fits: int = 0

    def __post_init__(self) -> None:
        if self.lags < 1:
            raise ValueError("lags must be at least 1")
        if self.ridge <= 0.0:
            raise ValueError(
                "ridge must be positive; an unregularized fit on a short history "
                "is singular whenever two lags are collinear"
            )
        if self.refit_every < 1:
            raise ValueError("refit_every must be at least 1")
        if self.minimum_history <= self.lags:
            raise ValueError("minimum_history must exceed the number of lags")

    @property
    def initialized(self) -> bool:
        return self.level is not None

    def predict(self) -> float | None:
        return self.level

    def dispersion(self) -> float:
        return max(self.scale, _MIN_SCALE)

    def update(self, observation: float) -> None:
        """Fold one observation in, refitting when due.

        Dispersion tracks the *forecast* residual rather than deviation from a
        level, because that is what the studentized score divides by. A learned
        forecaster with a level-based scale would report shrinking
        nonconformity as it improved, blinding the detector in exactly the
        regime where the forecaster works well.
        """
        if self.level is not None:
            residual = abs(observation - self.level)
            self.scale = (
                residual if self.scale == 0.0 else 0.9 * self.scale + 0.1 * residual
            )
        self._history.append(float(observation))
        self.observations += 1
        if (
            len(self._history) >= self.minimum_history
            and self.observations % self.refit_every == 0
        ):
            self._fit()
        self.level = self._forecast()

    def _fit(self) -> None:
        rows: list[list[float]] = []
        targets: list[float] = []
        for index in range(self.lags, len(self._history)):
            rows.append([1.0, *self._history[index - self.lags : index]])
            targets.append(self._history[index])
        if len(rows) < 2:
            return

        width = self.lags + 1
        gram = [[0.0] * width for _ in range(width)]
        moment = [0.0] * width
        for row, target in zip(rows, targets, strict=True):
            for i in range(width):
                moment[i] += row[i] * target
                for j in range(width):
                    gram[i][j] += row[i] * row[j]
        # The intercept is left unpenalized: shrinking it toward zero would
        # bias the forecast toward the origin rather than toward the mean.
        for i in range(1, width):
            gram[i][i] += self.ridge

        solved = _solve(gram, moment)
        if solved is not None:
            self._weights = solved
            self._fits += 1

    def _forecast(self) -> float:
        if not self._weights or len(self._history) < self.lags:
            window = self._history[-self.lags :] or self._history
            return sum(window) / len(window) if window else 0.0
        design = [1.0, *self._history[-self.lags :]]
        return sum(w * x for w, x in zip(self._weights, design, strict=True))

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": "ridge",
            "lags": self.lags,
            "ridge": self.ridge,
            "fits": self._fits,
            "level": self.level,
            "scale": self.scale,
            "observations": self.observations,
        }


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting.

    Returns None on a singular system rather than raising: a forecaster that
    cannot fit this window should keep its previous weights, not abort the run.
    """
    size = len(rhs)
    augmented = [[*row, rhs[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda r: abs(augmented[r][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            return None
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        for row in range(column + 1, size):
            factor = augmented[row][column] / augmented[column][column]
            for col in range(column, size + 1):
                augmented[row][col] -= factor * augmented[column][col]

    solution = [0.0] * size
    for row in reversed(range(size)):
        total = augmented[row][size] - sum(
            augmented[row][col] * solution[col] for col in range(row + 1, size)
        )
        solution[row] = total / augmented[row][row]
    return solution


Forecaster = EwmaForecaster | RidgeForecaster


def build_forecaster(config: dict[str, Any]) -> Forecaster:
    """Construct the forecaster named in configuration.

    Defaults to the exponentially weighted one, so every existing experiment
    behaves as it did. The learned forecaster is opt-in because swapping it
    silently would change every conformal arm at once and make the two
    demonstrations indistinguishable.
    """
    params = dict(config or {})
    kind = str(params.pop("kind", "ewma"))
    builders: dict[str, type[EwmaForecaster] | type[RidgeForecaster]] = {
        "ewma": EwmaForecaster,
        "ridge": RidgeForecaster,
    }
    builder = builders.get(kind)
    if builder is None:
        raise ValueError(
            f"unknown forecaster {kind!r}; available: {', '.join(sorted(builders))}"
        )
    try:
        return builder(**params)
    except TypeError as error:
        raise ValueError(
            f"forecaster {kind!r} rejected its parameters: {error}"
        ) from error


class NonconformityScore(ABC):
    """Maps an observation and a forecast to a nonconformity value."""

    name: str = "abstract"

    @abstractmethod
    def score(self, observation: float, forecaster: Any) -> float:
        """Return a non-negative nonconformity value.

        The forecaster is duck-typed on predict and dispersion rather than
        pinned to a class. That is the point of the conformal layer: the
        coverage guarantee holds whatever the predictor is, so the score must
        not be able to tell one predictor from another.
        """


@dataclass(frozen=True, slots=True)
class AbsoluteResidual(NonconformityScore):
    """|y - ŷ|. Interpretable in the units of the series."""

    name: str = "absolute_residual"

    def score(self, observation: float, forecaster: Any) -> float:
        prediction = forecaster.predict()
        if prediction is None:
            return 0.0
        # Cast at the boundary: duck typing buys the model-agnostic interface
        # and costs the static guarantee that predict returns a number, so the
        # conversion is where that guarantee is restored.
        return abs(observation - float(prediction))


@dataclass(frozen=True, slots=True)
class StudentizedResidual(NonconformityScore):
    """|y - ŷ| / σ̂: scale free, and the default.

    Studentizing matters for the protocol's central contrast. Under a
    volatility regime the absolute residual grows even when the process is
    behaving exactly as its own scale predicts, so an absolute-residual
    detector saturates and flags everything — which is the failure mode the
    pilot's OOD rate of 0.98 exhibited. Dividing by the tracked scale makes
    the score answer "atypical for this regime" rather than "large".
    """

    name: str = "studentized_residual"

    def score(self, observation: float, forecaster: Any) -> float:
        prediction = forecaster.predict()
        if prediction is None:
            return 0.0
        return abs(observation - float(prediction)) / float(forecaster.dispersion())


_SCORES: dict[str, NonconformityScore] = {
    AbsoluteResidual().name: AbsoluteResidual(),
    StudentizedResidual().name: StudentizedResidual(),
}


def available_scores() -> tuple[str, ...]:
    return tuple(sorted(_SCORES))


def build_score(name: str) -> NonconformityScore:
    score = _SCORES.get(name)
    if score is None:
        raise ValueError(
            f"unknown nonconformity score {name!r}; available: {', '.join(available_scores())}"
        )
    return score


@dataclass(slots=True)
class ScoreStream:
    """Per-series forecaster plus score, kept causal by construction.

    ``observe`` scores first and updates second. Reversing the two would let
    the observation influence its own expectation, which inflates apparent
    calibration and is the most common silent error in online conformal
    implementations.
    """

    score_fn: NonconformityScore
    forecaster: Forecaster = field(default_factory=EwmaForecaster)

    def observe(self, observation: float) -> float:
        value = self.score_fn.score(observation, self.forecaster)
        self.forecaster.update(observation)
        return float(value)

    @property
    def warm(self) -> bool:
        """True once the forecaster has seen at least one observation."""
        return self.forecaster.initialized
