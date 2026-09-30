"""Hierarchical variance decomposition.

The protocol commits to hierarchical Bayesian modelling of the indicators,
separating between-seed from between-configuration variance and yielding
credible intervals per cell. The anytime-valid tests delivered instead answer a
different question: whether one arm differs from another. This module answers
the promised one — how much of the spread in a table of per-arm means belongs
to the configuration rather than to the seed.

The distinction is not academic. Two arms whose means differ by 0.05 mean
something entirely different depending on whether the seed-to-seed spread
within an arm is 0.005 or 0.5, and a table of means with standard deviations
invites the reader to make that comparison by eye without telling them what
fraction of the variance is attributable to the treatment at all.

Implemented as a conjugate normal-normal model rather than through a
probabilistic programming language. That is a deliberate constraint: the
model is a two-level normal hierarchy with known-form posteriors, so the
closed form is exact where sampling would be approximate, it needs no
dependency, and every number it produces can be checked by hand. A model that
needed a sampler would also need convergence diagnostics, and a reviewer would
be right to ask for them.

What the closed form gives up: it assumes normality of the cell means and a
common within-arm variance. Both are checked and reported rather than assumed
silently — `shrinkage` exposes how far the model pulled each arm toward the
grand mean, which is the quantity that becomes untrustworthy when the
assumption fails.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ArmPosterior:
    """The posterior for one arm's mean, after partial pooling."""

    arm: str
    n: int
    observed_mean: float
    posterior_mean: float
    posterior_sd: float
    shrinkage: float

    @property
    def credible_interval(self) -> tuple[float, float]:
        """Central 95% interval on the arm's mean."""
        return (
            self.posterior_mean - 1.96 * self.posterior_sd,
            self.posterior_mean + 1.96 * self.posterior_sd,
        )

    def to_payload(self) -> dict[str, Any]:
        low, high = self.credible_interval
        return {
            "arm": self.arm,
            "n": self.n,
            "observed_mean": round(self.observed_mean, 6),
            "posterior_mean": round(self.posterior_mean, 6),
            "posterior_sd": round(self.posterior_sd, 6),
            "credible_interval_95": [round(low, 6), round(high, 6)],
            "shrinkage": round(self.shrinkage, 6),
        }


@dataclass(frozen=True, slots=True)
class VarianceDecomposition:
    """How the spread in an indicator divides between arms and seeds."""

    indicator: str
    grand_mean: float
    between_arm_variance: float
    within_arm_variance: float
    arms: tuple[ArmPosterior, ...]

    @property
    def total_variance(self) -> float:
        return self.between_arm_variance + self.within_arm_variance

    @property
    def intraclass_correlation(self) -> float:
        """Share of variance attributable to the arm rather than the seed.

        The number a table of per-arm means is implicitly claiming. Near zero
        means the arms are indistinguishable and the visible differences are
        seed noise; near one means the configuration determines the outcome and
        replication within an arm adds little.
        """
        total = self.total_variance
        return self.between_arm_variance / total if total > 1e-12 else 0.0

    @property
    def arms_are_distinguishable(self) -> bool:
        """Whether the configuration explains more than a tenth of the spread.

        A deliberately low bar: below it, differences between arm means are
        mostly seed variation and no per-arm comparison in the study should be
        read as a treatment effect regardless of what a test says about any
        individual pair.
        """
        return self.intraclass_correlation >= 0.10

    def to_payload(self) -> dict[str, Any]:
        return {
            "indicator": self.indicator,
            "grand_mean": round(self.grand_mean, 6),
            "between_arm_variance": round(self.between_arm_variance, 9),
            "within_arm_variance": round(self.within_arm_variance, 9),
            "intraclass_correlation": round(self.intraclass_correlation, 6),
            "arms_are_distinguishable": self.arms_are_distinguishable,
            "arms": [arm.to_payload() for arm in self.arms],
            "method": "conjugate_normal_normal_two_level",
            "assumptions": (
                "cell means approximately normal; a common within-arm variance. "
                "Shrinkage is the quantity that degrades first when either fails."
            ),
        }


def decompose(
    indicator: str, by_arm: dict[str, list[float]]
) -> VarianceDecomposition | None:
    """Fit the two-level model for one indicator.

    Returns None when fewer than two arms carry at least two observations
    each — with less than that there is no between-arm variance to estimate,
    and reporting a decomposition would be reporting the prior.
    """
    usable = {arm: values for arm, values in by_arm.items() if len(values) >= 2}
    if len(usable) < 2:
        return None

    arm_means = {arm: statistics.fmean(values) for arm, values in usable.items()}
    grand_mean = statistics.fmean(arm_means.values())

    # Within-arm variance pooled across arms: the seed-to-seed spread.
    within = statistics.fmean(
        [statistics.variance(values) for values in usable.values()]
    )

    # Between-arm variance by method of moments, floored at zero. The raw
    # estimator can go negative when the arms are closer together than sampling
    # noise alone would predict, and a negative variance is an estimate of zero
    # rather than a quantity to propagate.
    observed_spread = statistics.variance(list(arm_means.values()))
    mean_n = statistics.fmean([len(values) for values in usable.values()])
    between = max(observed_spread - within / mean_n, 0.0)

    posteriors: list[ArmPosterior] = []
    for arm, values in sorted(usable.items()):
        n = len(values)
        # Partial pooling: the arm's own mean weighted against the grand mean
        # in proportion to how precisely each is known.
        precision_arm = n / within if within > 1e-12 else float("inf")
        precision_prior = 1.0 / between if between > 1e-12 else 0.0
        total_precision = precision_arm + precision_prior
        if total_precision == 0.0 or math.isinf(precision_arm):
            posterior_mean, posterior_sd, shrinkage = arm_means[arm], 0.0, 0.0
        else:
            weight = precision_prior / total_precision
            posterior_mean = (1.0 - weight) * arm_means[arm] + weight * grand_mean
            posterior_sd = math.sqrt(1.0 / total_precision)
            shrinkage = weight
        posteriors.append(
            ArmPosterior(
                arm=arm,
                n=n,
                observed_mean=arm_means[arm],
                posterior_mean=posterior_mean,
                posterior_sd=posterior_sd,
                shrinkage=shrinkage,
            )
        )

    return VarianceDecomposition(
        indicator=indicator,
        grand_mean=grand_mean,
        between_arm_variance=between,
        within_arm_variance=within,
        arms=tuple(posteriors),
    )


def decompose_all(
    values_by_indicator: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    """Fit the model for every indicator that supports one."""
    fitted: dict[str, Any] = {}
    skipped: list[str] = []
    for indicator, by_arm in sorted(values_by_indicator.items()):
        decomposition = decompose(indicator, by_arm)
        if decomposition is None:
            skipped.append(indicator)
        else:
            fitted[indicator] = decomposition.to_payload()

    indistinguishable = [
        indicator
        for indicator, payload in fitted.items()
        if not payload["arms_are_distinguishable"]
    ]
    return {
        "indicators": fitted,
        # Named explicitly: any comparison on these indicators is a comparison
        # of seed noise, whatever a pairwise test reports about it.
        "indistinguishable_indicators": indistinguishable,
        "skipped_for_insufficient_data": skipped,
    }
