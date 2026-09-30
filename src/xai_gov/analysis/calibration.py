"""Simulation-based calibration, and the sim-to-real gap.

Fitting a simulator by searching for the parameters that best match some data
gives one number per parameter and no idea how well determined it was. That
matters here because the whole project's external validity rests on how close
the twin is to a real logistics network, and a point estimate cannot express
"this parameter is pinned by the data and that one is barely constrained".

So the target is the *posterior distribution* over parameters, estimated
without a likelihood — which the simulator does not have in closed form. Two
methods are provided:

* `abc_rejection` — approximate Bayesian computation. Simple, exact in the
  limit, wasteful. It is the control reference precisely because there is
  nothing in it to get wrong.
* `sequential_abc` — the same idea with the tolerance tightened over rounds,
  which reaches the same posterior at a fraction of the simulations.

The posterior alone is not enough, and this is where most simulation studies
stop. A model can fit the statistics it was calibrated on and be wrong about
everything else, so `posterior_predictive_check` scores the fit on statistics
*not used in calibration*. The distance between predicted and observed on those
held-out statistics is the sim-to-real gap, and the protocol reports it as a
number attached to every conclusion rather than as a paragraph of caveats.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Simulator = Callable[[dict[str, float], int], list[float]]
SummaryFn = Callable[[list[float]], dict[str, float]]


@dataclass(frozen=True, slots=True)
class Prior:
    """A uniform prior over one simulator parameter."""

    name: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(f"{self.name}: low must be strictly below high")

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.uniform(self.low, self.high, size=size)

    def to_payload(self) -> dict[str, Any]:
        return {"name": self.name, "low": self.low, "high": self.high}


@dataclass(frozen=True, slots=True)
class Posterior:
    """The accepted parameter draws, with what they imply."""

    names: tuple[str, ...]
    draws: tuple[tuple[float, ...], ...]
    proposals: int
    tolerance: float

    @property
    def accepted(self) -> int:
        return len(self.draws)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposals if self.proposals else 0.0

    def mean(self) -> dict[str, float]:
        if not self.draws:
            return {}
        columns = list(zip(*self.draws, strict=True))
        return {
            name: float(np.mean(column))
            for name, column in zip(self.names, columns, strict=True)
        }

    def credible_interval(self, level: float = 0.9) -> dict[str, tuple[float, float]]:
        """Equal-tailed credible interval per parameter."""
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        if len(self.draws) < 2:
            return {}
        tail = (1.0 - level) / 2.0
        columns = list(zip(*self.draws, strict=True))
        return {
            name: (
                float(np.quantile(column, tail)),
                float(np.quantile(column, 1.0 - tail)),
            )
            for name, column in zip(self.names, columns, strict=True)
        }

    def identified(self, prior_widths: dict[str, float], *, ratio: float = 0.5) -> dict[str, bool]:
        """Which parameters the data actually constrained.

        A posterior nearly as wide as its prior means the data said almost
        nothing about that parameter. Reporting its mean as a calibrated value
        would present a prior as a finding, so each parameter carries a verdict
        rather than only a number.
        """
        intervals = self.credible_interval()
        return {
            name: (upper - lower) < ratio * prior_widths.get(name, float("inf"))
            for name, (lower, upper) in intervals.items()
        }

    def to_payload(self) -> dict[str, Any]:
        intervals = self.credible_interval()
        return {
            "accepted": self.accepted,
            "proposals": self.proposals,
            "acceptance_rate": round(self.acceptance_rate, 6),
            "tolerance": round(self.tolerance, 6),
            "mean": {k: round(v, 6) for k, v in self.mean().items()},
            "credible_interval_90": {
                k: [round(lo, 6), round(hi, 6)] for k, (lo, hi) in intervals.items()
            },
        }


def bootstrap_scales(
    series: list[float],
    summarize: SummaryFn,
    *,
    replicates: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """Each statistic's own sampling variability, for use as its scale.

    This function exists because of a defect worth recording. Normalizing a
    summary distance by the *magnitude* of the observed statistic — the
    obvious choice, and the one this project shipped first — divides by a
    number that can be arbitrarily close to zero. First-order autocorrelation
    of an independent series has expectation about −1/n and sampling standard
    deviation about 1/√n: at n = 120 that is roughly −0.008 ± 0.09. Dividing a
    difference of 0.09 by an observed 0.008 contributes a term of order 10 to
    a distance whose informative terms are of order 0.1, so the sampler spent
    its entire budget matching noise. Both simulator parameters came back
    weakly identified for exactly this reason, and more proposals could not
    have fixed it.

    Scaling by sampling variability instead makes every statistic contribute
    in units of its own noise: an uninformative statistic gets a large scale
    and therefore stops dominating. The estimate is a nonparametric bootstrap
    of the observed series, so it needs no distributional assumption.
    """
    if replicates < 20:
        raise ValueError("bootstrap needs at least 20 replicates to estimate a scale")
    if len(series) < 20:
        raise ValueError("cannot bootstrap a scale from fewer than 20 observations")

    rng = np.random.default_rng(seed)
    values = np.asarray(series, dtype=float)
    drawn: dict[str, list[float]] = {}
    for _ in range(replicates):
        resampled = values[rng.integers(0, len(values), size=len(values))]
        for key, value in summarize(list(resampled)).items():
            drawn.setdefault(key, []).append(value)

    scales: dict[str, float] = {}
    for key, sampled in drawn.items():
        spread = float(np.std(sampled, ddof=1))
        # Floored so a statistic that is constant under resampling does not
        # divide by zero. The floor is deliberately absolute rather than
        # relative: a statistic with no sampling variability carries no
        # information about which parameters generated the data either.
        scales[key] = max(spread, 1e-6)
    return scales


def summary_distance(
    left: dict[str, float], right: dict[str, float], *, scales: dict[str, float] | None = None
) -> float:
    """Normalized distance between two sets of summary statistics.

    Each statistic is divided by its scale before comparison. Without that,
    a statistic measured in units of thousands would dominate one measured in
    fractions, and the calibration would silently be fitting only the former.
    """
    shared = sorted(set(left) & set(right))
    if not shared:
        raise ValueError("no shared summary statistics to compare")
    scales = scales or {}
    total = 0.0
    for key in shared:
        # Falls back to magnitude only when no sampling scale was supplied.
        # See bootstrap_scales for why magnitude is the wrong denominator and
        # what it cost this project.
        scale = scales.get(key) or max(abs(right[key]), 1e-9)
        total += ((left[key] - right[key]) / scale) ** 2
    return float(np.sqrt(total / len(shared)))


def abc_rejection(
    *,
    simulator: Simulator,
    summarize: SummaryFn,
    observed: dict[str, float],
    priors: Sequence[Prior],
    proposals: int = 500,
    quantile: float = 0.1,
    seed: int = 0,
    scales: dict[str, float] | None = None,
) -> Posterior:
    """Rejection ABC: keep the draws whose summaries land closest.

    The tolerance is set as a quantile of the realized distances rather than as
    an absolute number. An absolute tolerance chosen in advance either accepts
    everything or nothing, depending on a scale nobody knows before running the
    simulator once.
    """
    if not priors:
        raise ValueError("calibration needs at least one prior")
    if proposals < 10:
        raise ValueError("proposals must be at least 10 for a usable posterior")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must lie in (0, 1]")

    rng = np.random.default_rng(seed)
    names = tuple(prior.name for prior in priors)
    candidates = np.column_stack([prior.sample(rng, proposals) for prior in priors])

    distances: list[float] = []
    for index, row in enumerate(candidates):
        parameters = dict(zip(names, row, strict=True))
        simulated = simulator(parameters, int(rng.integers(0, 2**31 - 1)))
        distances.append(summary_distance(summarize(simulated), observed, scales=scales))
        del index

    tolerance = float(np.quantile(distances, quantile))
    accepted = tuple(
        tuple(float(value) for value in row)
        for row, distance in zip(candidates, distances, strict=True)
        if distance <= tolerance
    )
    return Posterior(
        names=names, draws=accepted, proposals=proposals, tolerance=tolerance
    )


def sequential_abc(
    *,
    simulator: Simulator,
    summarize: SummaryFn,
    observed: dict[str, float],
    priors: Sequence[Prior],
    rounds: int = 3,
    proposals_per_round: int = 200,
    quantile: float = 0.3,
    seed: int = 0,
) -> Posterior:
    """ABC with the tolerance tightened over rounds.

    Each round proposes around the previous round's accepted draws, so the
    simulator is not spent exploring regions already ruled out. The proposal
    spread shrinks with the accepted spread, which is what makes the sequence
    converge rather than merely narrow.
    """
    if rounds < 1:
        raise ValueError("at least one round is required")

    rng = np.random.default_rng(seed)
    names = tuple(prior.name for prior in priors)
    current = np.column_stack([prior.sample(rng, proposals_per_round) for prior in priors])
    tolerance = float("inf")
    total_proposals = 0

    for _ in range(rounds):
        distances: list[float] = []
        for row in current:
            parameters = dict(zip(names, row, strict=True))
            simulated = simulator(parameters, int(rng.integers(0, 2**31 - 1)))
            distances.append(summary_distance(summarize(simulated), observed))
        total_proposals += len(current)

        tolerance = float(np.quantile(distances, quantile))
        kept = current[np.array(distances) <= tolerance]
        if len(kept) < 2:
            break

        # Perturb around what survived, with the spread the survivors show.
        spread = np.std(kept, axis=0)
        spread = np.where(spread <= 0.0, 1e-6, spread)
        picks = rng.integers(0, len(kept), size=proposals_per_round)
        perturbation = rng.normal(
            0.0, spread * 0.5, size=(proposals_per_round, len(names))
        )
        proposed = kept[picks] + perturbation
        for column, prior in enumerate(priors):
            proposed[:, column] = np.clip(proposed[:, column], prior.low, prior.high)
        current = proposed

    final_draws = tuple(tuple(float(v) for v in row) for row in current)
    return Posterior(
        names=names, draws=final_draws, proposals=total_proposals, tolerance=tolerance
    )


@dataclass(frozen=True, slots=True)
class PredictiveCheck:
    """How the calibrated simulator does on statistics it was not fitted to."""

    calibrated_statistics: tuple[str, ...]
    held_out_statistics: tuple[str, ...]
    calibrated_distance: float
    held_out_distance: float
    per_statistic: dict[str, float]

    @property
    def sim_to_real_gap(self) -> float:
        """The number every conclusion is qualified by.

        The held-out distance, not the calibrated one. A model fits what it was
        fitted to by construction; the question external validity turns on is
        what it gets right that it was never shown.
        """
        return self.held_out_distance

    @property
    def overfitted(self) -> bool:
        """Whether the fit is much better on calibration than on held-out data.

        A model that matches its calibration statistics far better than
        anything else has learned those statistics, not the process.
        """
        return self.held_out_distance > 2.0 * max(self.calibrated_distance, 1e-9)

    def to_payload(self) -> dict[str, Any]:
        return {
            "calibrated_statistics": list(self.calibrated_statistics),
            "held_out_statistics": list(self.held_out_statistics),
            "calibrated_distance": round(self.calibrated_distance, 6),
            "held_out_distance": round(self.held_out_distance, 6),
            "sim_to_real_gap": round(self.sim_to_real_gap, 6),
            "overfitted": self.overfitted,
            "per_statistic": {k: round(v, 6) for k, v in sorted(self.per_statistic.items())},
        }


def posterior_predictive_check(
    *,
    simulator: Simulator,
    summarize: SummaryFn,
    observed: dict[str, float],
    posterior: Posterior,
    calibrated_statistics: Sequence[str],
    draws: int = 50,
    seed: int = 0,
    scales: dict[str, float] | None = None,
) -> PredictiveCheck:
    """Score the calibrated simulator, separating fitted from held-out."""
    if not posterior.draws:
        raise ValueError("cannot check a posterior with no accepted draws")

    rng = np.random.default_rng(seed)
    calibrated = tuple(sorted(set(calibrated_statistics) & set(observed)))
    held_out = tuple(sorted(set(observed) - set(calibrated)))
    if not held_out:
        raise ValueError(
            "every observed statistic was used in calibration, so no predictive "
            "check is possible; hold at least one out"
        )

    predictions: list[dict[str, float]] = []
    for _ in range(draws):
        row = posterior.draws[int(rng.integers(0, posterior.accepted))]
        parameters = dict(zip(posterior.names, row, strict=True))
        predictions.append(summarize(simulator(parameters, int(rng.integers(0, 2**31 - 1)))))

    averaged = {
        key: float(np.mean([p[key] for p in predictions if key in p]))
        for key in observed
        if any(key in p for p in predictions)
    }
    per_statistic = {
        key: abs(averaged[key] - observed[key]) / max(abs(observed[key]), 1e-9)
        for key in averaged
    }
    return PredictiveCheck(
        calibrated_statistics=calibrated,
        held_out_statistics=held_out,
        calibrated_distance=summary_distance(
            {k: averaged[k] for k in calibrated if k in averaged},
            {k: observed[k] for k in calibrated},
            scales=scales,
        ),
        held_out_distance=summary_distance(
            {k: averaged[k] for k in held_out if k in averaged},
            {k: observed[k] for k in held_out},
            scales=scales,
        ),
        per_statistic=per_statistic,
    )


@dataclass(slots=True)
class DemandSummary:
    """Summary statistics of a demand series.

    Two groups. Moments and quantiles carry location and dispersion, and the
    quantiles are included because a real series is not normal — intermittent,
    spiky demand makes a mean and a standard deviation a poor description of
    the same distribution the quantiles pin down well.

    The shape statistics are held out of calibration deliberately: a simulator
    can match a mean and a variance while getting the *dynamics* wrong
    entirely, and autocorrelation is where that shows. They are what the
    predictive check scores, so no conclusion rests on a fit the model was
    handed.
    """

    def __call__(self, series: list[float]) -> dict[str, float]:
        if len(series) < 3:
            return {
                "mean": 0.0, "std": 0.0, "q10": 0.0, "q50": 0.0, "q90": 0.0,
                "iqr": 0.0, "cv": 0.0, "autocorr_1": 0.0, "max_ratio": 0.0,
            }
        values = np.asarray(series, dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        centred = values - mean
        denominator = float(np.sum(centred**2))
        autocorr = (
            float(np.sum(centred[:-1] * centred[1:]) / denominator)
            if denominator > 1e-12
            else 0.0
        )
        q25, q75 = (float(np.quantile(values, q)) for q in (0.25, 0.75))
        return {
            "mean": mean,
            "std": std,
            "q10": float(np.quantile(values, 0.10)),
            "q50": float(np.quantile(values, 0.50)),
            "q90": float(np.quantile(values, 0.90)),
            "iqr": q75 - q25,
            "cv": std / mean if abs(mean) > 1e-9 else 0.0,
            "autocorr_1": autocorr,
            "max_ratio": float(np.max(values)) / mean if abs(mean) > 1e-9 else 0.0,
        }


#: Statistics the calibration is allowed to fit. Everything else in the summary
#: is held out for the predictive check, which is what makes the reported gap a
#: statement about generalization rather than about the fit.
CALIBRATION_STATISTICS: tuple[str, ...] = (
    "mean", "std", "q10", "q50", "q90", "iqr",
)


def default_priors() -> tuple[Prior, ...]:
    """Priors over the twin's demand parameters."""
    return (
        Prior(name="base_demand", low=5.0, high=40.0),
        Prior(name="noise_scale", low=0.5, high=10.0),
    )


def prior_widths(priors: Sequence[Prior]) -> dict[str, float]:
    return {prior.name: prior.high - prior.low for prior in priors}
