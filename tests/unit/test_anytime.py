"""Anytime-valid inference: looking whenever, without paying for it."""

from __future__ import annotations

import math

import numpy as np
import pytest

from xai_gov.analysis.agrapa import run_e_process
from xai_gov.analysis.anytime import (
    MeanShiftMartingale,
    combine_evidence,
    confidence_sequence,
    directional_evidence,
    two_sided_peak_log_wealth,
)


def test_a_null_stream_does_not_accumulate_evidence() -> None:
    """The property everything else rests on: no effect, no evidence, however
    long the analyst watches."""
    rng = np.random.default_rng(3)
    martingale = MeanShiftMartingale(alpha=0.05, scale=1.0)
    for _ in range(2000):
        martingale.update(float(rng.normal(0.0, 0.3)))
    assert martingale.decided is False


def test_a_real_effect_is_detected() -> None:
    rng = np.random.default_rng(3)
    martingale = MeanShiftMartingale(alpha=0.05, scale=1.0)
    detected = any(martingale.update(float(rng.normal(0.5, 0.3))) for _ in range(400))
    assert detected is True
    assert martingale.value >= martingale.threshold


def test_optional_stopping_does_not_inflate_the_error_rate() -> None:
    """The whole point. An analyst who checks after every observation and stops
    at the first crossing must still respect alpha."""
    rng = np.random.default_rng(17)
    alarms = 0
    replications = 200
    for _ in range(replications):
        martingale = MeanShiftMartingale(alpha=0.05, scale=1.0)
        for _ in range(300):
            if martingale.update(float(rng.normal(0.0, 0.3))):
                alarms += 1
                break
    assert alarms / replications <= 0.05 + 0.02


def test_a_campaign_can_be_extended_without_correction() -> None:
    """Continuing a campaign that ran out of compute must not invalidate what
    was already reported."""
    rng = np.random.default_rng(5)
    martingale = MeanShiftMartingale(alpha=0.05, scale=1.0)
    for _ in range(50):
        martingale.update(float(rng.normal(0.4, 0.3)))
    early = martingale.value
    for _ in range(200):
        martingale.update(float(rng.normal(0.4, 0.3)))
    assert martingale.value > early
    assert martingale.observations == 250


def test_below_threshold_means_insufficient_evidence_not_no_effect() -> None:
    """The reading a p-value near 0.06 never gets."""
    martingale = MeanShiftMartingale(alpha=0.05, scale=1.0)
    martingale.update(0.01)
    verdict = martingale.verdict()
    assert verdict.decided is False
    assert "not evidence of no effect" in verdict.interpretation


def test_an_extreme_observation_is_clipped_not_amplified() -> None:
    """An outlier must not manufacture evidence."""
    clipped = MeanShiftMartingale(alpha=0.05, scale=1.0)
    clipped.update(1e6)
    modest = MeanShiftMartingale(alpha=0.05, scale=1.0)
    modest.update(1.0)
    assert clipped.value == pytest.approx(modest.value)


def test_the_value_stays_finite_under_a_long_run() -> None:
    martingale = MeanShiftMartingale(alpha=0.05, scale=1.0)
    for _ in range(3000):
        martingale.update(1.0)
    assert martingale.value < float("inf")
    assert martingale.log_value > 100.0


def test_evidence_from_independent_studies_multiplies() -> None:
    """A replication strengthens the first study rather than restarting it."""
    assert combine_evidence([4.0, 5.0]) == pytest.approx(20.0)
    assert combine_evidence([]) == 1.0


def test_negative_e_values_are_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        combine_evidence([1.0, -2.0])


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="alpha"):
        MeanShiftMartingale(alpha=1.0)
    with pytest.raises(ValueError, match="betting fraction"):
        MeanShiftMartingale(fractions=(1.5,))
    with pytest.raises(ValueError, match="scale"):
        MeanShiftMartingale(scale=0.0)


# -- confidence sequences -------------------------------------------------
def test_a_confidence_sequence_covers_the_true_mean() -> None:
    rng = np.random.default_rng(11)
    values = [float(np.clip(rng.normal(0.3, 0.2), -1, 1)) for _ in range(500)]
    sequence = confidence_sequence(values, alpha=0.05, bound=1.0)
    assert sequence.lower <= 0.3 <= sequence.upper


def test_it_is_wider_than_a_fixed_n_interval() -> None:
    """The width is the honest price of being allowed to look whenever."""
    rng = np.random.default_rng(11)
    values = [float(np.clip(rng.normal(0.0, 0.3), -1, 1)) for _ in range(300)]
    sequence = confidence_sequence(values, alpha=0.05, bound=1.0)
    naive = 1.96 * float(np.std(values, ddof=1)) / np.sqrt(len(values))
    assert sequence.width / 2 > naive


def test_it_adapts_to_the_spread_rather_than_to_the_declared_range() -> None:
    """The reason the empirical-Bernstein interval was replaced: its range term
    3·R·log/n dominated whenever the differences were far tighter than the
    declared range, and an interval touching zero sat beside an e-value of
    thousands. The betting sequence bets on the observed spread; what is left
    of its width comes from the bet cap of 1/2, which the test shares."""
    rng = np.random.default_rng(23)
    values = [float(rng.normal(-0.038, 0.025)) for _ in range(480)]
    sequence = confidence_sequence(values, alpha=0.05, bound=1.0)
    assert sequence.upper < 0.0
    assert sequence.width < 0.04  # the empirical-Bernstein interval was 0.076 wide


def test_it_narrows_as_evidence_accumulates() -> None:
    rng = np.random.default_rng(13)
    values = [float(np.clip(rng.normal(0.5, 0.2), -1, 1)) for _ in range(1000)]
    early = confidence_sequence(values[:50], alpha=0.05, bound=1.0)
    late = confidence_sequence(values, alpha=0.05, bound=1.0)
    assert late.width < early.width


def test_a_clear_effect_excludes_zero() -> None:
    rng = np.random.default_rng(19)
    values = [float(np.clip(rng.normal(0.8, 0.1), -1, 1)) for _ in range(500)]
    assert confidence_sequence(values, alpha=0.05, bound=1.0).excludes_zero is True


def test_at_zero_it_inverts_the_two_sided_test_itself() -> None:
    """The sequence is the test read as an interval: at m = 0 its e-process is
    the two-sided e-process that decides H3 and H4, to the last digit."""
    rng = np.random.default_rng(29)
    values = [float(np.clip(rng.normal(0.03, 0.1), -1, 1)) for _ in range(400)]
    path = run_e_process(values, two_sided=True)
    peak = math.exp(float(two_sided_peak_log_wealth(values, [0.0])[0]))
    assert peak == pytest.approx(path.peak, rel=1e-9)


@pytest.mark.parametrize("seed", [31, 37, 41])
def test_a_supported_hypothesis_has_an_interval_that_excludes_zero(seed: int) -> None:
    """Coherence: a one-sided e-value past 2k/alpha forces the two-sided
    e-value at zero past k/alpha >= 1/alpha, so zero cannot be in the interval."""
    rng = np.random.default_rng(seed)
    values = [float(np.clip(rng.normal(0.02, 0.05), -1, 1)) for _ in range(500)]
    evidence = directional_evidence(values, alpha=0.05, hypotheses=6)
    sequence = confidence_sequence(values, alpha=0.05, bound=1.0)
    if evidence.supported:
        assert sequence.excludes_zero


def test_it_covers_at_every_sample_size_at_once() -> None:
    """Uniform coverage, checked the way it is claimed: the true mean must stay
    inside at every n of the path, not only at the last one."""
    rng = np.random.default_rng(43)
    paths, n, misses = 200, 120, 0
    for _ in range(paths):
        values = [float(v) for v in rng.choice([-1.0, 1.0], size=n) * 0.5]
        log_wealth = two_sided_peak_log_wealth(values, [0.0])
        misses += bool(log_wealth[0] >= math.log(20.0))
    assert misses / paths <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / paths)


def test_a_value_outside_the_declared_range_is_refused() -> None:
    with pytest.raises(ValueError, match="declared range"):
        confidence_sequence([0.1, 1.5], alpha=0.05, bound=1.0)


def test_the_interval_and_the_raw_difference_share_a_sign_convention() -> None:
    """A 'less' hypothesis is tested on negated differences, but the interval
    reported beside the raw difference must be on the raw scale.

    Before this, four rows of the hypothesis table showed a positive raw
    difference beside an entirely negative interval — two adjacent columns on
    opposite sign conventions, which reads as a contradiction rather than as an
    orientation. Orientation is an internal device for the one-sided test and
    must not reach the report.
    """
    raw = [0.06, 0.07, 0.05, 0.065, 0.055, 0.062, 0.058, 0.061]
    forward = confidence_sequence(raw, alpha=0.05, bound=0.1)
    negated = confidence_sequence([-value for value in raw], alpha=0.05, bound=0.1)

    # Mirror images, so picking the wrong one flips every bound in the table.
    assert forward.mean > 0.0 > negated.mean
    assert forward.lower == pytest.approx(-negated.upper, abs=1e-6)
    assert forward.upper == pytest.approx(-negated.lower, abs=1e-6)


def test_an_empty_sample_yields_the_widest_interval() -> None:
    sequence = confidence_sequence([], alpha=0.05, bound=2.0)
    assert (sequence.lower, sequence.upper) == (-2.0, 2.0)
    assert sequence.excludes_zero is False
