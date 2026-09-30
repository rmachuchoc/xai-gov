"""Self-normalization: when studentizing defeats detection.

The first campaign measured a coverage error of -0.158 against a nominal level
of 0.1, and the adaptive scheme came out *worse* than the classical one it was
meant to beat. The mechanism is the studentized score dividing by a scale that
co-adapts with the series: under a volatility shift the numerator and the
denominator rise together, the score stays flat, and the detector normalizes
away the very shift it exists to find.

These tests are that mechanism, and the fix, stated as executable claims.
"""

from __future__ import annotations

import numpy as np
import pytest

from xai_gov.conformal.calibrator import WeightedCalibrator
from xai_gov.conformal.scores import (
    EwmaForecaster,
    ScoreStream,
    StudentizedResidual,
    build_forecaster,
)


def shifted_series(seed: int, *, quiet: int = 120, loud: int = 120) -> list[float]:
    """A volatility shift with the level held constant.

    The level is deliberately unchanged: a detector that only tracks the mean
    would see nothing, so any flag has to come from dispersion.
    """
    rng = np.random.default_rng(seed)
    return [float(rng.normal(20.0, 1.0)) for _ in range(quiet)] + [
        float(rng.normal(20.0, 8.0)) for _ in range(loud)
    ]


def flag_rate_after_shift(scale_mode: str, *, quiet: int = 120, loud: int = 120) -> float:
    stream = ScoreStream(
        score_fn=StudentizedResidual(),
        forecaster=EwmaForecaster(scale_mode=scale_mode, reference_after=30),
    )
    calibrator = WeightedCalibrator(window=200, decay=0.97)
    flags = scored = 0
    for index, value in enumerate(shifted_series(5, quiet=quiet, loud=loud)):
        score = stream.observe(value)
        threshold = calibrator.threshold(0.1)
        if index >= quiet and calibrator.size >= 40 and threshold is not None:
            flags += int(score > threshold)
            scored += 1
        calibrator.observe(score)
    return flags / scored if scored else 0.0


def test_a_co_adapting_scale_normalizes_the_shift_away() -> None:
    """The defect, reproduced. The scale grows with the volatility it is
    supposed to reveal, so the score stays flat across the change point."""
    assert flag_rate_after_shift("adaptive") < 0.10


def test_a_frozen_scale_sees_the_shift() -> None:
    """The fix. With dispersion held at its reference value, a later change in
    spread registers as nonconformity."""
    assert flag_rate_after_shift("reference") > flag_rate_after_shift("adaptive")


def test_the_reference_is_frozen_only_after_its_window() -> None:
    """A scale frozen from one observation is not an estimate of dispersion, so
    the adaptive scale carries the detector while it warms up."""
    forecaster = EwmaForecaster(scale_mode="reference", reference_after=10)
    for value in (20.0, 21.0, 19.0):
        forecaster.update(value)
    assert forecaster.reference_scale is None
    for value in (20.0,) * 10:
        forecaster.update(value)
    assert forecaster.reference_scale is not None


def test_the_frozen_scale_does_not_move_afterwards() -> None:
    forecaster = EwmaForecaster(scale_mode="reference", reference_after=5)
    for value in (20.0, 21.0, 19.0, 20.5, 19.5, 20.0):
        forecaster.update(value)
    frozen = forecaster.dispersion()
    # An alternating series, not a constant one: a constant series converges,
    # the level catches up to it, the deviation falls to zero and the adaptive
    # scale decays. Only sustained variation keeps the underlying scale moving,
    # which is the case this test is about.
    for index in range(60):
        forecaster.update(20.0 + (40.0 if index % 2 else -40.0))
    assert forecaster.dispersion() == pytest.approx(frozen)
    # The adaptive scale kept moving underneath; only what the score divides by
    # is held, so the diagnostic remains available.
    assert forecaster.scale > frozen


def test_the_adaptive_mode_is_unchanged() -> None:
    """The default must stay adaptive: changing it silently would make the two
    campaigns incomparable."""
    assert EwmaForecaster().scale_mode == "adaptive"
    assert EwmaForecaster().reference_scale is None
    assert build_forecaster({}).scale_mode == "adaptive"


def test_an_unknown_scale_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="scale_mode"):
        EwmaForecaster(scale_mode="magic")
    with pytest.raises(ValueError, match="reference_after"):
        EwmaForecaster(scale_mode="reference", reference_after=1)


def test_the_payload_records_which_mode_ran() -> None:
    """A coverage number means different things under the two modes, so the run
    record has to say which produced it."""
    payload = EwmaForecaster(scale_mode="reference", reference_after=3).to_payload()
    assert payload["scale_mode"] == "reference"
    assert "reference_scale" in payload
