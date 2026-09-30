"""Calibration scaling and the real-series seam.

The first campaign reported both simulator parameters as weakly identified and
a sim-to-real gap of 1.114 against a reference series the simulator itself had
produced. Two defects, and neither was a shortage of proposals.

The first is a normalization error. Scaling a summary distance by the
*magnitude* of the observed statistic divides by a number that can be
arbitrarily close to zero, and first-order autocorrelation of a
near-independent series sits close to zero by construction. The noise term
dominated the informative terms and the sampler matched noise.

The second is that a gap measured against simulator output is a sim-to-sim gap.
It says nothing about external validity, and the artifact has to say so.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xai_gov.analysis.calibration import (
    CALIBRATION_STATISTICS,
    DemandSummary,
    Prior,
    abc_rejection,
    bootstrap_scales,
    prior_widths,
    summary_distance,
)
from xai_gov.analysis.data import (
    DemandSeries,
    load_series,
    resolve_series,
    synthetic_series,
)

SUMMARIZE = DemandSummary()
PRIORS = (
    Prior(name="base_demand", low=5.0, high=40.0),
    Prior(name="noise_scale", low=0.5, high=10.0),
)


def series(base: float = 22.0, noise: float = 4.5, seed: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(max(0.0, rng.normal(base, noise))) for _ in range(120)]


def simulator(parameters: dict[str, float], seed: int) -> list[float]:
    return series(parameters["base_demand"], parameters["noise_scale"], seed)


# -- the normalization defect ---------------------------------------------
def test_autocorrelation_of_an_independent_series_sits_near_zero() -> None:
    """The precondition that made magnitude the wrong denominator."""
    observed = SUMMARIZE(series())
    assert abs(observed["autocorr_1"]) < 0.2


def test_a_near_zero_statistic_dominates_a_magnitude_scaled_distance() -> None:
    """The defect, reproduced. A difference of 0.09 divided by an observed
    0.008 contributes a term of order 10 to a distance whose informative terms
    are of order 0.1."""
    observed = {"mean": 22.0, "autocorr_1": 0.008}
    simulated = {"mean": 22.1, "autocorr_1": 0.098}
    assert summary_distance(simulated, observed) > 5.0


def test_sampling_scales_stop_a_noise_statistic_from_dominating() -> None:
    """The fix. Scaled by its own sampling variability, an uninformative
    statistic contributes in units of its noise and stops swamping the rest."""
    reference = series()
    observed = SUMMARIZE(reference)
    scales = bootstrap_scales(reference, SUMMARIZE, seed=3)
    perturbed = {**observed, "autocorr_1": observed["autocorr_1"] + 0.09}
    assert summary_distance(perturbed, observed, scales=scales) < 1.0


def test_an_uninformative_statistic_gets_a_larger_relative_scale() -> None:
    """Autocorrelation varies more across resamples, relative to its own size,
    than the mean does. That ratio is what makes it contribute less."""
    reference = series()
    observed = SUMMARIZE(reference)
    scales = bootstrap_scales(reference, SUMMARIZE, seed=3)
    noise_ratio = scales["autocorr_1"] / max(abs(observed["autocorr_1"]), 1e-9)
    signal_ratio = scales["mean"] / abs(observed["mean"])
    assert noise_ratio > signal_ratio


def test_scales_are_floored_for_a_constant_statistic() -> None:
    """A statistic with no sampling variability carries no information about
    which parameters generated the data either, so the floor is absolute."""
    scales = bootstrap_scales([5.0] * 40, SUMMARIZE, seed=1)
    assert all(value > 0.0 for value in scales.values())


def test_bootstrap_refuses_a_sample_too_small_to_estimate_a_scale() -> None:
    with pytest.raises(ValueError, match="fewer than 20"):
        bootstrap_scales([1.0, 2.0, 3.0], SUMMARIZE)
    with pytest.raises(ValueError, match="at least 20 replicates"):
        bootstrap_scales(series(), SUMMARIZE, replicates=5)


# -- identifiability ------------------------------------------------------
def test_scaled_calibration_identifies_the_parameters() -> None:
    """The outcome the fix exists for: with each statistic in units of its own
    noise, the posterior narrows against the prior."""
    reference = series()
    observed = SUMMARIZE(reference)
    scales = bootstrap_scales(reference, SUMMARIZE, seed=5)
    posterior = abc_rejection(
        simulator=simulator, summarize=SUMMARIZE, observed=observed,
        priors=PRIORS, proposals=400, quantile=0.05, seed=5, scales=scales,
    )
    identified = posterior.identified(prior_widths(PRIORS))
    assert identified["base_demand"] is True


def test_the_posterior_recovers_the_generating_parameter() -> None:
    reference = series(base=22.0, noise=4.5)
    observed = SUMMARIZE(reference)
    scales = bootstrap_scales(reference, SUMMARIZE, seed=5)
    posterior = abc_rejection(
        simulator=simulator, summarize=SUMMARIZE, observed=observed,
        priors=PRIORS, proposals=400, quantile=0.05, seed=5, scales=scales,
    )
    assert abs(posterior.mean()["base_demand"] - 22.0) < 4.0


def test_quantiles_are_available_to_the_calibration() -> None:
    """A real series is not normal: intermittent, spiky demand makes a mean and
    a standard deviation a poor description of a distribution the quantiles pin
    down well."""
    for key in ("q10", "q50", "q90", "iqr"):
        assert key in CALIBRATION_STATISTICS
        assert key in SUMMARIZE(series())


def test_the_dynamics_statistics_stay_held_out() -> None:
    """A simulator can match location and spread while getting the dynamics
    wrong entirely, so autocorrelation must not be fitted."""
    assert "autocorr_1" not in CALIBRATION_STATISTICS
    assert "max_ratio" not in CALIBRATION_STATISTICS


# -- the real-series seam -------------------------------------------------
def test_the_synthetic_fallback_labels_its_own_gap() -> None:
    """A study that calibrates against its own output and reports a
    sim-to-real gap has measured nothing about the world."""
    payload = synthetic_series().to_payload()
    assert payload["synthetic"] is True
    assert "sim-to-sim" in payload["gap_interpretation"]
    assert "no external validity" in payload["gap_interpretation"]


def test_a_loaded_series_is_labelled_sim_to_real(tmp_path: Path) -> None:
    path = tmp_path / "demand.csv"
    path.write_text(
        "date,units\n" + "\n".join(f"2024-01-{i:02d},{20 + i % 7}" for i in range(1, 31)),
        encoding="utf-8",
    )
    loaded = load_series(path)
    assert loaded.synthetic is False
    assert loaded.column == "units"
    assert "sim-to-real" in loaded.to_payload()["gap_interpretation"]


def test_the_loader_records_which_column_it_took(tmp_path: Path) -> None:
    """Guessing silently is how a calibration ends up fitted to a row index."""
    path = tmp_path / "demand.csv"
    path.write_text(
        "period,units\n" + "\n".join(f"{i},{20 + i % 5}" for i in range(1, 31)),
        encoding="utf-8",
    )
    assert load_series(path).column == "units"
    assert load_series(path, column="period").column == "period"


def test_an_unknown_column_is_refused_by_name(tmp_path: Path) -> None:
    path = tmp_path / "demand.csv"
    path.write_text("units\n" + "\n".join(str(20 + i) for i in range(30)), encoding="utf-8")
    with pytest.raises(ValueError, match="no column 'sales'"):
        load_series(path, column="sales")


def test_a_non_numeric_cell_names_its_row(tmp_path: Path) -> None:
    path = tmp_path / "demand.csv"
    rows = [str(20 + i) for i in range(30)]
    rows[4] = "n/a"
    path.write_text("units\n" + "\n".join(rows), encoding="utf-8")
    with pytest.raises(ValueError, match="row 6"):
        load_series(path)


def test_a_series_too_short_to_calibrate_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "demand.csv"
    path.write_text("units\n" + "\n".join(str(i) for i in range(10)), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 20 periods"):
        load_series(path)


def test_intermittent_demand_is_reported_not_silently_accepted(tmp_path: Path) -> None:
    """A mostly-zero series is a different modelling problem than this twin
    represents, and the caller should know before reading the posterior."""
    path = tmp_path / "demand.csv"
    values = ["0"] * 24 + ["15"] * 6
    path.write_text("units\n" + "\n".join(values), encoding="utf-8")
    loaded = load_series(path)
    assert loaded.zero_fraction == pytest.approx(0.8)
    assert loaded.intermittent is True


def test_a_missing_file_names_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="one item-store series"):
        load_series(tmp_path / "absent.csv")


def test_resolution_prefers_the_real_series(tmp_path: Path) -> None:
    assert resolve_series(tmp_path).synthetic is True
    target = tmp_path / "data" / "demand"
    target.mkdir(parents=True)
    (target / "demand.csv").write_text(
        "units\n" + "\n".join(str(20 + i % 4) for i in range(30)), encoding="utf-8"
    )
    assert resolve_series(tmp_path).synthetic is False


def test_the_series_carries_its_provenance() -> None:
    reference = DemandSeries(
        values=tuple(float(i) for i in range(30)), source="test", synthetic=False
    )
    payload = reference.to_payload()
    assert payload["periods"] == 30
    assert payload["source"] == "test"
