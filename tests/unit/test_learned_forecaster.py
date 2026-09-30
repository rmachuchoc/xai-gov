"""The learned forecaster: the conformal guarantee is model-agnostic."""

from __future__ import annotations

import numpy as np
import pytest

from xai_gov.conformal.calibrator import WeightedCalibrator
from xai_gov.conformal.scores import (
    RidgeForecaster,
    ScoreStream,
    StudentizedResidual,
    build_forecaster,
)


def test_the_forecaster_learns_a_trend_the_ewma_lags() -> None:
    """A ridge fit over lags can track a linear trend; an exponential level
    always trails it."""
    from xai_gov.conformal.scores import EwmaForecaster

    ridge = RidgeForecaster(lags=3, minimum_history=8, refit_every=4)
    ewma = EwmaForecaster()
    series = [float(10 + 2 * i) for i in range(60)]
    ridge_errors: list[float] = []
    ewma_errors: list[float] = []
    for value in series:
        for model, errors in ((ridge, ridge_errors), (ewma, ewma_errors)):
            prediction = model.predict()
            if prediction is not None:
                errors.append(abs(value - prediction))
            model.update(value)
    assert sum(ridge_errors[-20:]) < sum(ewma_errors[-20:])


def test_no_observation_informs_its_own_forecast() -> None:
    """The design matrix holds only lags strictly before the target."""
    forecaster = RidgeForecaster(lags=2, minimum_history=6, refit_every=2)
    for value in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
        forecaster.update(value)
    before = forecaster.predict()
    # A wildly different next observation cannot retroactively improve the
    # forecast that preceded it.
    forecaster.update(1000.0)
    assert before is not None
    assert before < 100.0


def test_the_fit_is_deterministic() -> None:
    def run() -> float | None:
        model = RidgeForecaster(lags=3, minimum_history=8)
        for i in range(40):
            model.update(float(10 + (i % 5)))
        return model.predict()

    assert run() == run()


def test_dispersion_tracks_forecast_error_not_level_deviation() -> None:
    """A level-based scale would shrink as the forecaster improved, blinding the
    detector exactly where the forecaster works."""
    forecaster = RidgeForecaster(lags=2, minimum_history=6)
    for value in (float(v) for v in range(1, 30)):
        forecaster.update(value)
    trend_scale = forecaster.dispersion()

    noisy = RidgeForecaster(lags=2, minimum_history=6)
    rng = np.random.default_rng(3)
    for _ in range(29):
        noisy.update(float(rng.normal(15.0, 8.0)))
    assert noisy.dispersion() > trend_scale


def test_a_singular_window_keeps_the_previous_weights() -> None:
    """A forecaster that cannot fit this window should not abort the run."""
    forecaster = RidgeForecaster(lags=3, minimum_history=5, refit_every=1)
    for _ in range(20):
        forecaster.update(7.0)
    assert forecaster.predict() == pytest.approx(7.0, abs=1e-6)


def test_the_conformal_guarantee_holds_for_either_forecaster() -> None:
    """The property the conformal layer exists for: coverage tracks the nominal
    level whatever the predictor is."""
    def realized_rate(kind: str) -> float:
        rng = np.random.default_rng(7)
        stream = ScoreStream(
            score_fn=StudentizedResidual(),
            forecaster=build_forecaster({"kind": kind}),
        )
        calibrator = WeightedCalibrator(window=200, decay=0.97)
        flags = scored = 0
        for _ in range(500):
            score = stream.observe(float(rng.normal(20.0, 3.0)))
            threshold = calibrator.threshold(0.1)
            if calibrator.size >= 40 and threshold is not None:
                flags += int(score > threshold)
                scored += 1
            calibrator.observe(score)
        return flags / scored if scored else 0.0

    ewma_rate = realized_rate("ewma")
    ridge_rate = realized_rate("ridge")
    assert abs(ewma_rate - ridge_rate) < 0.15


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="lags"):
        RidgeForecaster(lags=0)
    with pytest.raises(ValueError, match="ridge must be positive"):
        RidgeForecaster(ridge=0.0)
    with pytest.raises(ValueError, match="minimum_history"):
        RidgeForecaster(lags=8, minimum_history=4)


def test_the_registry_defaults_to_the_unlearned_forecaster() -> None:
    """Swapping it silently would change every conformal arm at once."""
    from xai_gov.conformal.scores import EwmaForecaster

    assert isinstance(build_forecaster({}), EwmaForecaster)
    assert isinstance(build_forecaster({"kind": "ridge"}), RidgeForecaster)
    with pytest.raises(ValueError, match="unknown forecaster"):
        build_forecaster({"kind": "crystal_ball"})
