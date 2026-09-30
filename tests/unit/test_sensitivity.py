"""Sobol indices: separating findings from calibration artifacts."""

from __future__ import annotations

import pytest

from xai_gov.analysis.sensitivity import (
    Parameter,
    build_parameter_space,
    saltelli_sample,
    sobol_analysis,
)

PARAMS = [Parameter(name="a", low=0.0, high=1.0), Parameter(name="b", low=0.0, high=1.0)]


def test_an_inverted_range_is_refused() -> None:
    with pytest.raises(ValueError, match="low must be strictly below high"):
        Parameter(name="x", low=1.0, high=0.0)


def test_an_irrelevant_parameter_gets_a_near_zero_index() -> None:
    """The basic requirement: a parameter the outcome ignores must not appear
    to matter."""
    report = sobol_analysis(
        lambda p: p["a"] * 10.0, PARAMS, base_samples=256, seed=1
    )
    indices = {index.name: index for index in report.indices}
    assert indices["b"].total_effect < 0.1
    assert indices["b"].influential is False


def test_the_driving_parameter_is_identified() -> None:
    report = sobol_analysis(
        lambda p: p["a"] * 10.0, PARAMS, base_samples=256, seed=1
    )
    assert report.dominant == "a"
    assert report.robust is False


def test_interaction_is_visible_where_one_at_a_time_would_miss_it() -> None:
    """A pure product has no first-order effect at all: each parameter matters
    only alongside the other, which varying one at a time cannot see."""
    report = sobol_analysis(
        lambda p: (p["a"] - 0.5) * (p["b"] - 0.5) * 20.0,
        PARAMS,
        base_samples=512,
        seed=2,
    )
    for index in report.indices:
        assert index.total_effect > index.first_order
        assert index.interaction_share > 0.2


def test_equal_contributors_leave_the_result_robust() -> None:
    """Two equal contributors each necessarily explain about half the variance.
    Calling that non-robust would be reporting the parameter count, not
    fragility — so dominance requires carrying substantially more than an
    equal share, not merely a majority."""
    report = sobol_analysis(
        lambda p: p["a"] + p["b"], PARAMS, base_samples=256, seed=3
    )
    assert report.robust is True
    assert report.dominant in ("a", "b")


def test_a_single_driver_makes_the_result_fragile() -> None:
    """The case robustness exists to catch: the outcome is a finding about one
    declared cost rather than about governance."""
    report = sobol_analysis(
        lambda p: p["a"] * 10.0 + p["b"] * 0.01, PARAMS, base_samples=256, seed=7
    )
    assert report.robust is False
    assert report.dominant == "a"


def test_indices_stay_within_bounds() -> None:
    """The estimators are unbiased but not bounded; a negative index at small n
    is sampler noise, not a negative contribution to variance."""
    report = sobol_analysis(lambda p: p["a"], PARAMS, base_samples=16, seed=4)
    for index in report.indices:
        assert 0.0 <= index.first_order <= 1.0
        assert 0.0 <= index.total_effect <= 1.0


def test_a_constant_output_apportions_nothing() -> None:
    report = sobol_analysis(lambda _: 5.0, PARAMS, base_samples=32, seed=5)
    assert all(index.total_effect == 0.0 for index in report.indices)
    assert report.robust is True


def test_the_evaluation_count_is_reported() -> None:
    """Each evaluation is a simulation run, so the budget must be visible."""
    report = sobol_analysis(lambda p: p["a"], PARAMS, base_samples=32, seed=6)
    assert report.evaluations == 32 * (len(PARAMS) + 2)


def test_too_few_samples_are_refused() -> None:
    with pytest.raises(ValueError, match="base_samples"):
        saltelli_sample(PARAMS, base_samples=4, seed=0)


def test_no_parameters_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one parameter"):
        saltelli_sample([], base_samples=16, seed=0)


def test_the_registry_defaults_to_the_declared_economics() -> None:
    space = build_parameter_space(None)
    assert "intervention_cost" in space.names()
    assert "disruption_loss" in space.names()


def test_the_registry_refuses_an_incomplete_parameter() -> None:
    with pytest.raises(ValueError, match="missing key"):
        build_parameter_space({"parameters": [{"name": "x", "low": 0.0}]})
