"""Calibration: the finite-sample correction, and honest refusals."""

from __future__ import annotations

import pytest

from xai_gov.conformal.calibrator import (
    SplitCalibrator,
    WeightedCalibrator,
    build_calibrator,
    conformal_quantile,
)


def test_conformal_quantile_uses_the_n_plus_one_correction() -> None:
    """The +1 accounts for the test point joining the ranking. Dropping it
    turns a finite-sample guarantee into an approximation."""
    scores = [float(i) for i in range(1, 10)]  # n = 9
    # ceil((9+1) * 0.9) = 9 -> the 9th smallest
    assert conformal_quantile(scores, 0.1) == 9.0
    # ceil((9+1) * 0.5) = 5 -> the 5th smallest
    assert conformal_quantile(scores, 0.5) == 5.0


def test_quantile_is_unbounded_when_the_level_is_too_extreme() -> None:
    """With five points, a 1% level justifies no finite threshold. Returning
    the maximum observed score instead would fabricate a guarantee."""
    result = conformal_quantile([1.0, 2.0, 3.0, 4.0, 5.0], 0.01)
    assert result != result  # NaN


def test_empty_calibration_set_is_refused() -> None:
    with pytest.raises(ValueError, match="empty calibration set"):
        conformal_quantile([], 0.1)


def test_split_calibrator_refuses_before_it_has_data() -> None:
    calibrator = SplitCalibrator(window=50)
    assert calibrator.threshold(0.1) is None
    calibrator.observe(1.0)
    assert calibrator.threshold(0.1) is None


def test_split_calibrator_threshold_rises_with_the_scores() -> None:
    calibrator = SplitCalibrator(window=100)
    for value in range(1, 51):
        calibrator.observe(float(value))
    low = calibrator.threshold(0.5)
    high = calibrator.threshold(0.1)
    assert low is not None and high is not None
    assert high > low


def test_split_window_forgets_the_oldest_scores() -> None:
    calibrator = SplitCalibrator(window=10)
    for value in range(100):
        calibrator.observe(float(value))
    assert calibrator.size == 10
    assert min(calibrator.scores()) == 90.0


def test_weighted_calibrator_favours_recent_scores() -> None:
    """A regime change must move the threshold; an unweighted window drags."""
    weighted = WeightedCalibrator(window=100, decay=0.85)
    unweighted = SplitCalibrator(window=100)
    for _ in range(50):
        weighted.observe(1.0)
        unweighted.observe(1.0)
    for _ in range(10):
        weighted.observe(10.0)
        unweighted.observe(10.0)

    weighted_threshold = weighted.threshold(0.2)
    unweighted_threshold = unweighted.threshold(0.2)
    assert weighted_threshold is not None and unweighted_threshold is not None
    assert weighted_threshold > unweighted_threshold


def test_the_effective_sample_caps_the_attainable_flag_rate() -> None:
    """A ceiling on flagging that has nothing to do with the score or the level.

    The conformal quantile at level e needs the score at index
    ceil((n+1)(1-e)), so at most n - that many calibration points can ever
    exceed the threshold. That fraction is the highest flag rate the window can
    produce, and when it falls below the requested level the detector cannot
    reach its own target however the level controller adapts.

    The effect is real and modest: a recency-weighted window of 200 at decay
    0.97 caps flagging near 7.7 percent against a 10 percent target. It is a
    genuine constraint and it does not on its own account for a realized rate
    near zero, which is worth stating rather than overselling.
    """
    import math

    short = WeightedCalibrator(window=200, decay=0.97)
    long = WeightedCalibrator(window=400, decay=0.999)
    for value in range(400):
        short.observe(float(value % 50))
        long.observe(float(value % 50))

    def ceiling(calibrator: WeightedCalibrator, level: float) -> float:
        n = int(calibrator.effective_sample_size)
        return (n - math.ceil((n + 1) * (1.0 - level))) / n

    assert ceiling(short, 0.1) < 0.10
    assert ceiling(long, 0.1) > ceiling(short, 0.1)
    # The long window's ceiling reaches the target; the short one's does not.
    assert ceiling(long, 0.1) >= 0.09


def test_a_longer_memory_yields_a_larger_effective_sample() -> None:
    """The Kish size of a truncated geometric window is about (1+d)/(1-d), not
    1/(1-d): the second moment of the weights carries a factor of two that is
    easy to drop and doubles the apparent memory.
    """
    short = WeightedCalibrator(window=200, decay=0.97)
    long = WeightedCalibrator(window=400, decay=0.999)
    for _ in range(400):
        short.observe(1.0)
        long.observe(1.0)
    assert 60.0 < short.effective_sample_size < 70.0
    assert long.effective_sample_size > 350.0


def test_effective_sample_size_is_reported_honestly() -> None:
    """Claiming a 0.99 level from an effective sample of ten would be
    arithmetic dressed as a guarantee, so the number is exposed."""
    aggressive = WeightedCalibrator(window=200, decay=0.8)
    conservative = WeightedCalibrator(window=200, decay=0.999)
    for _ in range(200):
        aggressive.observe(1.0)
        conservative.observe(1.0)
    assert aggressive.effective_sample_size < 12.0
    assert conservative.effective_sample_size > 150.0


def test_decay_of_one_reduces_to_the_unweighted_case() -> None:
    weighted = WeightedCalibrator(window=100, decay=1.0)
    split = SplitCalibrator(window=100)
    for value in range(1, 41):
        weighted.observe(float(value))
        split.observe(float(value))
    assert weighted.threshold(0.2) == split.threshold(0.2)


def test_total_variation_budget_shrinks_as_decay_shrinks() -> None:
    fast = WeightedCalibrator(window=100, decay=0.8)
    slow = WeightedCalibrator(window=100, decay=0.99)
    for _ in range(100):
        fast.observe(1.0)
        slow.observe(1.0)
    assert fast.total_variation_budget < slow.total_variation_budget


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="at least two"):
        SplitCalibrator(window=1)
    with pytest.raises(ValueError, match="decay"):
        WeightedCalibrator(decay=0.0)
    with pytest.raises(ValueError, match="level must lie"):
        SplitCalibrator().threshold(1.5)


def test_a_scheme_override_drops_foreign_parameters_but_not_typos() -> None:
    """The split scheme uses no decay, but a merged override legitimately
    carries one from the weighted block it replaced."""
    calibrator = build_calibrator({"name": "split", "params": {"window": 50, "decay": 0.97}})
    assert calibrator.name == "split"
    assert calibrator.window == 50  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="unknown calibrator parameters"):
        build_calibrator({"name": "split", "params": {"windwo": 50}})


def test_registry_builds_and_rejects() -> None:
    assert build_calibrator({"name": "split"}).name == "split"
    assert build_calibrator({}).name == "weighted"
    with pytest.raises(ValueError, match="unknown calibrator"):
        build_calibrator({"name": "psychic"})
    with pytest.raises(ValueError, match="must be a mapping"):
        build_calibrator({"name": "split", "params": "nonsense"})
