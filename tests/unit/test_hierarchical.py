"""Variance decomposition: how much of the spread is the treatment."""

from __future__ import annotations

from xai_gov.analysis.hierarchical import decompose, decompose_all


def test_arms_that_differ_show_high_intraclass_correlation() -> None:
    """When the configuration determines the outcome, almost all variance is
    between arms."""
    result = decompose(
        "service_level",
        {"a": [0.90, 0.91, 0.89, 0.90], "b": [0.20, 0.21, 0.19, 0.20]},
    )
    assert result is not None
    assert result.intraclass_correlation > 0.9
    assert result.arms_are_distinguishable is True


def test_arms_that_only_differ_by_noise_are_indistinguishable() -> None:
    """The case a table of per-arm means cannot show: visible differences that
    are seed variation."""
    result = decompose(
        "service_level",
        {"a": [0.5, 0.9, 0.1, 0.7, 0.3], "b": [0.4, 0.8, 0.2, 0.6, 0.5]},
    )
    assert result is not None
    assert result.intraclass_correlation < 0.10
    assert result.arms_are_distinguishable is False


def test_between_arm_variance_is_never_negative() -> None:
    """The method-of-moments estimator can go negative when arms sit closer
    than sampling noise predicts; a negative variance is an estimate of zero."""
    result = decompose("x", {"a": [1.0, 5.0, 9.0], "b": [2.0, 6.0, 8.0]})
    assert result is not None
    assert result.between_arm_variance >= 0.0


def test_partial_pooling_pulls_toward_the_grand_mean() -> None:
    result = decompose(
        "x", {"a": [10.0, 11.0, 9.0], "b": [0.0, 1.0, -1.0], "c": [5.0, 6.0, 4.0]}
    )
    assert result is not None
    for arm in result.arms:
        assert 0.0 <= arm.shrinkage <= 1.0
        low, high = arm.credible_interval
        assert low <= arm.posterior_mean <= high


def test_a_noisier_arm_is_shrunk_further() -> None:
    """Shrinkage is inverse to precision, which is what partial pooling means."""
    result = decompose(
        "x",
        {
            "tight": [5.0, 5.01, 4.99, 5.0, 5.0, 5.0],
            "loose": [5.0, 9.0, 1.0, 7.0, 3.0, 5.0],
        },
    )
    assert result is not None
    by_arm = {arm.arm: arm for arm in result.arms}
    assert by_arm["loose"].shrinkage >= by_arm["tight"].shrinkage


def test_fewer_than_two_arms_yields_no_decomposition() -> None:
    """With one arm there is no between-arm variance to estimate, and reporting
    one would be reporting the prior."""
    assert decompose("x", {"only": [1.0, 2.0, 3.0]}) is None
    assert decompose("x", {"a": [1.0], "b": [2.0]}) is None


def test_indistinguishable_indicators_are_named() -> None:
    payload = decompose_all(
        {
            "clear": {"a": [1.0, 1.1, 0.9], "b": [9.0, 9.1, 8.9]},
            "noise": {"a": [1.0, 9.0, 5.0], "b": [2.0, 8.0, 4.0]},
        }
    )
    assert "noise" in payload["indistinguishable_indicators"]
    assert "clear" not in payload["indistinguishable_indicators"]


def test_indicators_with_too_little_data_are_skipped_not_faked() -> None:
    payload = decompose_all({"thin": {"a": [1.0]}})
    assert payload["skipped_for_insufficient_data"] == ["thin"]
    assert payload["indicators"] == {}


def test_the_payload_declares_its_assumptions() -> None:
    result = decompose("x", {"a": [1.0, 2.0], "b": [5.0, 6.0]})
    assert result is not None
    payload = result.to_payload()
    assert payload["method"] == "conjugate_normal_normal_two_level"
    assert "Shrinkage" in payload["assumptions"]
