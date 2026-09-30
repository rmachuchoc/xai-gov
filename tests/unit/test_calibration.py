"""Calibration: a posterior, and the gap it does not close."""

from __future__ import annotations

import numpy as np
import pytest

from xai_gov.analysis.calibration import (
    DemandSummary,
    Prior,
    abc_rejection,
    default_priors,
    posterior_predictive_check,
    prior_widths,
    sequential_abc,
    summary_distance,
)

SUMMARIZE = DemandSummary()
PRIORS = (
    Prior(name="base_demand", low=5.0, high=40.0),
    Prior(name="noise_scale", low=0.5, high=10.0),
)


def simulator(parameters: dict[str, float], seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return [
        float(rng.normal(parameters["base_demand"], parameters["noise_scale"]))
        for _ in range(60)
    ]


TRUTH = {"base_demand": 20.0, "noise_scale": 3.0}
OBSERVED = SUMMARIZE(simulator(TRUTH, 99))


def test_statistics_are_normalized_before_comparison() -> None:
    """Without it, a statistic in thousands would silently dominate one in
    fractions and the calibration would fit only the former."""
    close = summary_distance({"big": 1010.0, "small": 0.5}, {"big": 1000.0, "small": 0.5})
    far = summary_distance({"big": 1000.0, "small": 5.0}, {"big": 1000.0, "small": 0.5})
    assert far > close


def test_no_shared_statistics_is_refused() -> None:
    with pytest.raises(ValueError, match="no shared summary statistics"):
        summary_distance({"a": 1.0}, {"b": 2.0})


def test_abc_recovers_the_parameter_that_generated_the_data() -> None:
    posterior = abc_rejection(
        simulator=simulator,
        summarize=SUMMARIZE,
        observed=OBSERVED,
        priors=PRIORS,
        proposals=400,
        quantile=0.05,
        seed=1,
    )
    assert posterior.accepted > 0
    assert abs(posterior.mean()["base_demand"] - 20.0) < 5.0


def test_the_posterior_is_a_distribution_not_a_point() -> None:
    """A point estimate cannot say whether a parameter was pinned by the data
    or barely constrained."""
    posterior = abc_rejection(
        simulator=simulator,
        summarize=SUMMARIZE,
        observed=OBSERVED,
        priors=PRIORS,
        proposals=300,
        quantile=0.1,
        seed=2,
    )
    intervals = posterior.credible_interval(0.9)
    assert intervals["base_demand"][0] < intervals["base_demand"][1]


def test_identifiability_is_reported_per_parameter() -> None:
    """A posterior as wide as its prior means the data said nothing; reporting
    its mean as calibrated would present a prior as a finding."""
    posterior = abc_rejection(
        simulator=simulator,
        summarize=SUMMARIZE,
        observed=OBSERVED,
        priors=PRIORS,
        proposals=400,
        quantile=0.05,
        seed=3,
    )
    verdicts = posterior.identified(prior_widths(PRIORS))
    assert set(verdicts) == {"base_demand", "noise_scale"}
    assert isinstance(verdicts["base_demand"], bool)


def test_sequential_abc_reaches_a_comparable_posterior_for_fewer_runs() -> None:
    sequential = sequential_abc(
        simulator=simulator,
        summarize=SUMMARIZE,
        observed=OBSERVED,
        priors=PRIORS,
        rounds=3,
        proposals_per_round=100,
        seed=4,
    )
    assert sequential.accepted > 0
    assert abs(sequential.mean()["base_demand"] - 20.0) < 6.0


def test_the_tolerance_is_a_quantile_not_an_absolute_number() -> None:
    """An absolute tolerance chosen in advance accepts everything or nothing,
    depending on a scale nobody knows before running the simulator."""
    strict = abc_rejection(
        simulator=simulator, summarize=SUMMARIZE, observed=OBSERVED,
        priors=PRIORS, proposals=200, quantile=0.05, seed=5,
    )
    loose = abc_rejection(
        simulator=simulator, summarize=SUMMARIZE, observed=OBSERVED,
        priors=PRIORS, proposals=200, quantile=0.5, seed=5,
    )
    assert strict.tolerance < loose.tolerance
    assert strict.accepted < loose.accepted


def test_the_predictive_check_scores_held_out_statistics() -> None:
    """A model fits what it was fitted to by construction; external validity
    turns on what it gets right that it was never shown."""
    posterior = abc_rejection(
        simulator=simulator, summarize=SUMMARIZE, observed=OBSERVED,
        priors=PRIORS, proposals=400, quantile=0.05, seed=6,
    )
    check = posterior_predictive_check(
        simulator=simulator,
        summarize=SUMMARIZE,
        observed=OBSERVED,
        posterior=posterior,
        calibrated_statistics=("mean", "std"),
        draws=30,
        seed=7,
    )
    assert set(check.calibrated_statistics) == {"mean", "std"}
    assert "autocorr_1" in check.held_out_statistics
    assert check.sim_to_real_gap == check.held_out_distance


def test_holding_nothing_out_is_refused() -> None:
    """Otherwise the check would score the model on its own training targets."""
    posterior = abc_rejection(
        simulator=simulator, summarize=SUMMARIZE, observed=OBSERVED,
        priors=PRIORS, proposals=100, quantile=0.2, seed=8,
    )
    with pytest.raises(ValueError, match="no predictive check is possible"):
        posterior_predictive_check(
            simulator=simulator,
            summarize=SUMMARIZE,
            observed=OBSERVED,
            posterior=posterior,
            calibrated_statistics=tuple(OBSERVED),
            draws=5,
            seed=9,
        )


def test_an_empty_posterior_cannot_be_checked() -> None:
    from xai_gov.analysis.calibration import Posterior

    with pytest.raises(ValueError, match="no accepted draws"):
        posterior_predictive_check(
            simulator=simulator,
            summarize=SUMMARIZE,
            observed=OBSERVED,
            posterior=Posterior(names=("a",), draws=(), proposals=0, tolerance=0.0),
            calibrated_statistics=("mean",),
        )


def test_invalid_calibration_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one prior"):
        abc_rejection(
            simulator=simulator, summarize=SUMMARIZE, observed=OBSERVED,
            priors=[], proposals=100,
        )
    with pytest.raises(ValueError, match="proposals"):
        abc_rejection(
            simulator=simulator, summarize=SUMMARIZE, observed=OBSERVED,
            priors=PRIORS, proposals=2,
        )


def test_the_summary_reports_dynamics_not_only_moments() -> None:
    """A simulator can match a mean and a variance while getting the dynamics
    wrong entirely, and autocorrelation is where that shows."""
    stats = SUMMARIZE([1.0, 2.0, 3.0, 4.0, 5.0])
    assert "autocorr_1" in stats
    assert stats["autocorr_1"] > 0.0


def test_default_priors_cover_the_twin_parameters() -> None:
    assert {p.name for p in default_priors()} == {"base_demand", "noise_scale"}
