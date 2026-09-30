"""Calibrating against the model's learning curve.

The decisive observation of the first campaign: the detector fired zero times in
45 scored periods, at every calibration window and every scale mode. The payload
showed the signature. A learned forecaster starts with a scale invented from its
first observation, so its early residuals are large and its scores shrink as it
converges. Admitting those scores into calibration makes the threshold a
quantile of the model's own inexperience, and a later score -- produced by a
forecaster that now works -- cannot reach it.

Neither the scale-mode ablation nor the window ablation moved the coverage
error, because neither touched the contamination.

These tests assert the mechanism: that burn-in scores train the forecaster and
stay out of the calibration set. They deliberately do not assert an effect size.
The effect is conditional on the horizon -- over a long stationary series the
contamination dilutes to nothing -- so its magnitude under the campaign's own
regime is measured by the paired RQ6d arm rather than guessed at here.
"""

from __future__ import annotations

import numpy as np
import pytest

from xai_gov.conformal.detector import build_detector

BASE = {
    "score": "studentized_residual",
    "warmup": 10,
    "level": {"scheme": "aci", "params": {"target_level": 0.1, "gamma": 0.05}},
    "calibrator": {"name": "weighted", "params": {"window": 200, "decay": 0.97}},
    "martingale": {"enabled": True, "params": {"delta": 0.01}},
}


def flag_rate(burn_in: int, *, periods: int = 200, seed: int = 4) -> float:
    detector = build_detector({**BASE, "burn_in": burn_in})
    rng = np.random.default_rng(seed)
    for _ in range(periods):
        detector.observe(float(rng.normal(20.0, 3.0)))
    return detector.flag_rate


def test_admitting_burn_in_scores_does_not_matter_on_a_long_stationary_series() -> None:
    """A boundary on the finding, and the reason it took three ablations to see.

    Over 200 stationary periods both settings land near the nominal level: the
    forecaster converges quickly and the burn-in scores are a small fraction of
    a long calibration set, so the contamination is diluted away. The campaign
    ran 60 periods, where the same twelve scores are a quarter of the set.

    Stated as a test rather than a caveat because it is what makes the effect
    conditional: a reader who reproduces this on a long run and sees nothing has
    not refuted the mechanism, they have diluted it.
    """
    contaminated = flag_rate(burn_in=0, periods=200)
    clean = flag_rate(burn_in=12, periods=200)
    assert abs(contaminated - clean) < 0.05


def test_the_effect_size_is_measured_by_the_campaign_not_asserted_here() -> None:
    """The behavioural claim belongs to RQ6d.

    A unit test can show that burn-in scores never enter the calibration set;
    how much that changes realized coverage under the campaign's own demand
    regime and horizon is an empirical question with a paired arm behind it, and
    asserting a magnitude here would be guessing at the answer that arm exists
    to produce.
    """
    detector = build_detector({**BASE, "burn_in": 12})
    for _ in range(30):
        detector.observe(20.0)
    assert detector.burned == 12
    assert detector.calibrator.size == 30 - 12


def test_burned_scores_are_counted_and_never_calibrated() -> None:
    detector = build_detector({**BASE, "burn_in": 12})
    for value in range(12):
        signal = detector.observe(20.0 + value)
        assert signal.active is False
    assert detector.burned == 12
    assert detector.calibrator.size == 0


def test_the_signal_stays_inactive_through_burn_in() -> None:
    """A discarded score must not reach the belief filter either: a detector
    still learning has nothing to say about the regime."""
    detector = build_detector({**BASE, "burn_in": 5})
    signals = [detector.observe(20.0 + i * 0.1) for i in range(5)]
    assert all(s.threshold is None and s.scheme == "none" for s in signals)


def test_the_payload_reports_what_was_discarded() -> None:
    """A coverage number computed after discarding a third of the run means
    something different from one that used all of it."""
    detector = build_detector({**BASE, "burn_in": 12})
    for _ in range(40):
        detector.observe(20.0)
    payload = detector.to_payload()
    assert payload["burn_in"] == 12
    assert payload["burned"] == 12


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="burn_in"):
        build_detector({**BASE, "burn_in": -1})
    with pytest.raises(ValueError, match="warmup must be at least 2"):
        build_detector({**BASE, "warmup": 1})
