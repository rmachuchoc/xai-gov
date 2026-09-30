"""Test martingales: anytime-valid change detection.

The property under test is Ville's inequality in practice: watching the
martingale continuously and stopping the first time it crosses 1/delta must
not inflate the false-alarm rate beyond delta. That is what lets the protocol
declare a change without correcting for how long anyone watched.
"""

from __future__ import annotations

import numpy as np
import pytest

from xai_gov.conformal.martingale import (
    MixturePowerMartingale,
    NoMartingale,
    build_martingale,
    conformal_p_value,
)


def test_p_value_is_never_zero() -> None:
    """A zero p-value would make the martingale infinite on one observation:
    arithmetic, not evidence."""
    calibration = [1.0, 2.0, 3.0]
    assert conformal_p_value(1000.0, calibration) == pytest.approx(0.25)
    assert conformal_p_value(0.0, calibration) == pytest.approx(1.0)


def test_p_value_of_an_empty_calibration_set_is_uninformative() -> None:
    assert conformal_p_value(5.0, []) == 1.0


def test_uniform_p_values_do_not_trigger_a_declaration() -> None:
    """Under exchangeability the martingale must not run away, however long
    it is watched."""
    rng = np.random.default_rng(7)
    martingale = MixturePowerMartingale(delta=0.01)
    for _ in range(2000):
        martingale.update(float(rng.uniform(1e-3, 1.0)))
    assert martingale.change_declared is False


def test_small_p_values_declare_a_change() -> None:
    martingale = MixturePowerMartingale(delta=0.01)
    declared = any(martingale.update(0.01) for _ in range(40))
    assert declared is True
    assert martingale.declared_at is not None
    assert martingale.log_value > martingale.log_threshold


def test_false_alarm_rate_respects_ville_across_replications() -> None:
    """The empirical false-alarm rate over many exchangeable sequences must
    stay at or below delta."""
    rng = np.random.default_rng(23)
    alarms = 0
    replications = 200
    for _ in range(replications):
        martingale = MixturePowerMartingale(delta=0.05)
        for _ in range(300):
            if martingale.update(float(rng.uniform(1e-4, 1.0))):
                alarms += 1
                break
    assert alarms / replications <= 0.05 + 0.02


def test_the_value_stays_finite_under_a_long_shift() -> None:
    """Tracked in log space; a naive product overflows within a few dozen
    periods and the detector would crash mid-run."""
    martingale = MixturePowerMartingale(delta=0.01)
    for _ in range(2000):
        martingale.update(1e-6)
    assert martingale.value == martingale.value  # not NaN
    assert martingale.value < float("inf")
    assert martingale.log_value > 100.0


def test_reset_restarts_the_episode_and_keeps_the_peak() -> None:
    martingale = MixturePowerMartingale(delta=0.01)
    for _ in range(40):
        martingale.update(0.01)
    peak = martingale.peak_log_value
    martingale.reset()
    assert martingale.change_declared is False
    assert martingale.observations == 0
    assert martingale.peak_log_value == peak


def test_out_of_range_p_values_are_refused() -> None:
    martingale = MixturePowerMartingale()
    with pytest.raises(ValueError, match="must lie in"):
        martingale.update(0.0)
    with pytest.raises(ValueError, match="must lie in"):
        martingale.update(1.5)


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="betting exponent"):
        MixturePowerMartingale(epsilons=(1.0,))
    with pytest.raises(ValueError, match="delta"):
        MixturePowerMartingale(delta=0.0)
    with pytest.raises(ValueError, match="at least one"):
        MixturePowerMartingale(epsilons=())


def test_disabled_detector_never_declares() -> None:
    detector = build_martingale({"enabled": False})
    assert isinstance(detector, NoMartingale)
    for _ in range(500):
        assert detector.update(1e-9) is False
    assert detector.change_declared is False
