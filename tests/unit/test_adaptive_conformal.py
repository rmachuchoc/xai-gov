"""Adaptive conformal inference: coverage without distributional assumptions.

These tests are the operational content of Theorem 1. The claim is not that
the adaptive scheme is more sensitive, but that its realized flag rate tracks
the nominal level even when the distribution moves — and that the fixed-level
control arm does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from xai_gov.conformal.adaptive import AciState, FixedLevel, build_level_controller
from xai_gov.conformal.calibrator import WeightedCalibrator
from xai_gov.conformal.scores import EwmaForecaster, ScoreStream, StudentizedResidual


def test_flagging_too_often_lowers_the_level() -> None:
    """The controller is negative feedback: more flags, lower level, higher
    threshold, fewer flags."""
    aci = AciState(target_level=0.1, gamma=0.05)
    start = aci.level
    for _ in range(10):
        aci.update(exceeded=True)
    assert aci.level < start


def test_flagging_too_rarely_raises_the_level() -> None:
    aci = AciState(target_level=0.1, gamma=0.05)
    start = aci.level
    for _ in range(10):
        aci.update(exceeded=False)
    assert aci.level > start


def test_the_realized_rate_tracks_the_target_when_the_loop_is_closed() -> None:
    """Feeding exceedances at the target rate leaves the *rate* on target.

    Note what is not asserted: that the level itself converges. When
    exceedances arrive independently of the level, the update has zero drift
    and the level is a bounded random walk — it wanders and hits its clips.
    ACI's guarantee is on realized coverage over the horizon, not on the level
    settling, and asserting the latter would be testing a property the scheme
    never claimed.
    """
    aci = AciState(target_level=0.2, gamma=0.02)
    rng = np.random.default_rng(4)
    for _ in range(4000):
        aci.update(exceeded=bool(rng.random() < 0.2))
    assert abs(aci.coverage_error) < 0.02


def test_the_level_is_clipped_and_the_clipping_is_counted() -> None:
    """Unbounded epsilon is the standard formulation's pathology: a long quiet
    stretch drives the threshold to infinity and the detector goes blind
    exactly when a disruption starts."""
    aci = AciState(target_level=0.1, gamma=0.5, min_level=0.01, max_level=0.4)
    for _ in range(50):
        aci.update(exceeded=True)
    assert aci.level == pytest.approx(0.01)
    assert aci.clipped_low > 0
    assert aci.saturated is True


def test_saturation_is_not_reported_on_thin_evidence() -> None:
    aci = AciState(target_level=0.1, gamma=0.5, min_level=0.05)
    for _ in range(3):
        aci.update(exceeded=True)
    assert aci.saturated is False


def test_fixed_level_never_moves() -> None:
    fixed = FixedLevel(target_level=0.1)
    for _ in range(100):
        fixed.update(exceeded=True)
    assert fixed.level == 0.1
    # And its coverage error is large, which is the point of keeping it.
    assert fixed.coverage_error == pytest.approx(0.9)


def test_adaptive_tracks_coverage_under_a_shift_where_fixed_does_not() -> None:
    """The central claim of RQ6, in miniature.

    A volatility regime change is fed through both schemes with identical
    calibration machinery. The adaptive scheme's realized flag rate stays near
    the target; the fixed one's does not.
    """
    def run(controller: AciState | FixedLevel) -> float:
        rng = np.random.default_rng(11)
        stream = ScoreStream(
            score_fn=StudentizedResidual(), forecaster=EwmaForecaster(alpha=0.3, beta=0.1)
        )
        calibrator = WeightedCalibrator(window=200, decay=0.97)
        flags = scored = 0
        for period in range(600):
            scale = 1.0 if period < 300 else 6.0
            observation = float(rng.normal(10.0, scale))
            score = stream.observe(observation)
            threshold = calibrator.threshold(controller.level)
            if calibrator.size >= 30 and threshold is not None:
                exceeded = score > threshold
                controller.update(exceeded=exceeded)
                flags += int(exceeded)
                scored += 1
            calibrator.observe(score)
        return flags / scored if scored else 0.0

    adaptive_rate = run(AciState(target_level=0.1, gamma=0.02))
    fixed_rate = run(FixedLevel(target_level=0.1))
    assert abs(adaptive_rate - 0.1) <= abs(fixed_rate - 0.1)


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="gamma must be positive"):
        AciState(gamma=0.0)
    with pytest.raises(ValueError, match="target_level"):
        AciState(target_level=1.0)
    with pytest.raises(ValueError, match="min_level"):
        AciState(min_level=0.6, max_level=0.5)
    with pytest.raises(ValueError, match="within"):
        AciState(target_level=0.4, min_level=0.01, max_level=0.3)


def test_foreign_parameters_from_a_scheme_override_are_dropped() -> None:
    """Switching schemes is done by including an override file and the loader
    deep-merges, so a fixed-level block legitimately arrives carrying gamma
    from the adaptive block it replaced."""
    controller = build_level_controller(
        {"scheme": "fixed", "params": {"target_level": 0.1, "gamma": 0.02, "min_level": 0.005}}
    )
    assert controller.to_payload()["scheme"] == "fixed"
    assert controller.level == 0.1


def test_a_parameter_no_scheme_recognizes_is_still_refused() -> None:
    with pytest.raises(ValueError, match="unknown level controller parameters"):
        build_level_controller({"scheme": "aci", "params": {"gama": 0.02}})


def test_registry_builds_both_schemes() -> None:
    assert build_level_controller({"scheme": "aci"}).to_payload()["scheme"] == "aci"
    assert build_level_controller({"scheme": "fixed"}).to_payload()["scheme"] == "fixed"
    with pytest.raises(ValueError, match="unknown level controller"):
        build_level_controller({"scheme": "telepathy"})
