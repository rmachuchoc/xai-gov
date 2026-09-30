"""The detector: composition, and the causality of its ordering."""

from __future__ import annotations

import numpy as np
import pytest

from xai_gov.conformal.detector import DetectorBank, build_detector

ADAPTIVE = {
    "score": "studentized_residual",
    "warmup": 10,
    "level": {"scheme": "aci", "params": {"target_level": 0.1, "gamma": 0.02}},
    "calibrator": {"name": "weighted", "params": {"window": 100, "decay": 0.97}},
    "martingale": {"enabled": True, "params": {"delta": 0.01}},
}


def test_nothing_is_flagged_during_warmup() -> None:
    """Flagging against an uncalibrated threshold is what produced the pilot's
    saturated detector."""
    detector = build_detector(ADAPTIVE)
    rng = np.random.default_rng(3)
    for _ in range(9):
        signal = detector.observe(float(rng.normal(10.0, 1.0)))
        assert signal.active is False
        assert signal.threshold is None
        assert signal.flagged_ood is False


def test_the_signal_activates_once_calibrated() -> None:
    detector = build_detector(ADAPTIVE)
    rng = np.random.default_rng(3)
    signals = [detector.observe(float(rng.normal(10.0, 1.0))) for _ in range(60)]
    assert any(signal.active for signal in signals)
    active = [s for s in signals if s.active]
    assert all(s.threshold is not None for s in active)


def test_a_stationary_series_is_not_flagged_wholesale() -> None:
    detector = build_detector(ADAPTIVE)
    rng = np.random.default_rng(5)
    for _ in range(400):
        detector.observe(float(rng.normal(10.0, 1.0)))
    assert detector.flag_rate < 0.35


def test_a_shift_raises_the_flag_rate() -> None:
    detector = build_detector(ADAPTIVE)
    rng = np.random.default_rng(5)
    for _ in range(200):
        detector.observe(float(rng.normal(10.0, 1.0)))
    quiet = detector.flag_rate
    for _ in range(60):
        detector.observe(float(rng.normal(60.0, 1.0)))
    assert detector.flag_rate > quiet


def test_risk_control_can_only_tighten_the_threshold() -> None:
    """Safety dominates efficiency: a certified risk threshold overrides the
    coverage-derived one only when it is stricter."""
    config = dict(ADAPTIVE)
    config["risk_control"] = {
        "enabled": True,
        "params": {"tolerance": 0.2, "delta": 0.1, "min_samples": 20},
    }
    detector = build_detector(config)
    rng = np.random.default_rng(9)
    plain = build_detector(ADAPTIVE)

    thresholds: list[tuple[float, float]] = []
    for _ in range(200):
        observation = float(rng.normal(10.0, 1.0))
        with_control = detector.observe(observation)
        detector.record_safety_loss(with_control.score, 1.0 if with_control.score > 2 else 0.0)
        without = plain.observe(observation)
        if with_control.threshold is not None and without.threshold is not None:
            thresholds.append((with_control.threshold, without.threshold))

    assert thresholds
    assert all(controlled <= uncontrolled + 1e-9 for controlled, uncontrolled in thresholds)


def test_an_out_of_range_safety_loss_is_refused() -> None:
    config = dict(ADAPTIVE)
    config["risk_control"] = {"enabled": True, "params": {"tolerance": 0.2}}
    detector = build_detector(config)
    with pytest.raises(ValueError, match="must lie in"):
        detector.record_safety_loss(1.0, 2.0)


def test_each_node_gets_its_own_detector() -> None:
    """A shared calibration set across nodes would mix distributions that are
    not exchangeable with one another."""
    bank = DetectorBank(ADAPTIVE)
    first = bank.for_node("retail")
    second = bank.for_node("plant")
    assert first is not second
    assert bank.for_node("retail") is first
    for _ in range(20):
        first.observe(10.0)
    assert first.periods == 20
    assert second.periods == 0
    assert bank.nodes == ("plant", "retail")


def test_the_payload_carries_every_mechanism() -> None:
    detector = build_detector(ADAPTIVE)
    for value in range(40):
        detector.observe(10.0 + value % 3)
    payload = detector.to_payload()
    for key in ("calibrator", "level_controller", "change_detector", "flag_rate"):
        assert key in payload
    assert payload["level_controller"]["scheme"] == "aci"


def test_bad_configuration_is_refused_with_a_reason() -> None:
    with pytest.raises(ValueError, match="unknown nonconformity score"):
        build_detector({"score": "vibes"})
    with pytest.raises(ValueError, match="rejected its parameters"):
        build_detector({"forecaster": {"nonsense": 1}})
