"""Parameter sweeps: the crossing, and what it refuses to claim."""

from __future__ import annotations

import pytest

from xai_gov.analysis.sweeps import (
    BREAK_EVEN_COSTS,
    SEVERITY_LEVELS,
    SweepPoint,
    SweepResult,
)


def point(level: float, treatment: float | None, control: float | None) -> SweepPoint:
    return SweepPoint(
        level=level,
        label=f"l={level:g}",
        treatment_value=treatment,
        control_value=control,
        included_cells=20,
    )


def result(points: list[SweepPoint], parameter: str = "intervention_cost") -> SweepResult:
    return SweepResult(
        parameter=parameter,
        indicator="organizational_value.total_value_creation",
        treatment="shielded",
        control="governed",
        points=tuple(points),
        replicates=10,
    )


def test_a_point_records_the_runs_behind_it() -> None:
    """A sweep report that names no runs cannot be audited from a published
    bundle, only re-executed; every point carries its runs by arm."""
    sweep_point = SweepPoint(
        level=1.0,
        label="c=1",
        treatment_value=0.2,
        control_value=0.1,
        included_cells=4,
        runs=(("shielded", ("r1", "r2")), ("governed", ("r3", "r4"))),
    )
    payload = sweep_point.to_payload()
    assert [r["run_name"] for r in payload["runs"]] == ["r1", "r2", "r3", "r4"]
    assert {r["arm"] for r in payload["runs"]} == {"shielded", "governed"}


def test_a_point_without_runs_still_serializes() -> None:
    """Older reports predate the field; reading them must not fail."""
    assert point(1.0, 0.2, 0.1).to_payload()["runs"] == []


def test_a_missing_side_yields_no_difference() -> None:
    """A level whose runs did not produce the indicator has no difference; a
    zero there would be read as no effect."""
    assert point(1.0, 0.5, None).difference is None
    assert point(1.0, None, 0.5).difference is None
    assert point(1.0, 0.5, 0.2).difference == pytest.approx(0.3)


def test_the_crossing_is_interpolated_not_snapped_to_the_grid() -> None:
    """A crossing quoted at grid resolution overstates precision."""
    swept = result([point(0.5, 0.2, 0.0), point(1.0, -0.2, 0.0)])
    crossing = swept.crossing
    assert crossing is not None
    assert 0.5 < crossing < 1.0
    assert crossing == pytest.approx(0.75)


def test_an_empty_sweep_reports_no_evidence_rather_than_a_conclusion() -> None:
    """The defect this guard exists for: the first version reported "the sign is
    constant over everything tested" from zero observations, which reads as a
    result and was built on nothing."""
    swept = result([point(0.5, None, None), point(1.0, None, None)])
    assert swept.has_data is False
    interpretation = swept.to_payload()["interpretation"]
    assert "no evidence" in interpretation
    assert "constant over everything tested" not in interpretation
    assert swept.to_payload()["usable_levels"] == 0


def test_one_usable_level_is_not_enough_for_a_curve() -> None:
    swept = result([point(0.5, 0.2, 0.0), point(1.0, None, None)])
    assert swept.has_data is False


def test_net_value_needs_every_term() -> None:
    """An absent cost is not a free run. A run whose cost the KPI layer did not
    report and a run that spent nothing are different readings, and the
    difference between them is the whole question being swept."""
    from xai_gov.analysis.sweeps import _net_value

    assert _net_value(0.9, 0.6, 0.1, 100.0) == pytest.approx(29.9)
    assert _net_value(None, 0.6, 0.1, 100.0) is None
    assert _net_value(0.9, None, 0.1, 100.0) is None
    assert _net_value(0.9, 0.6, None, 100.0) is None


def test_the_two_terms_must_be_made_commensurable() -> None:
    """A fraction in [0, 1] and a cost in the tens cannot be subtracted. The
    first economic run did exactly that and reported a net value of -161.9 for a
    quantity whose operational term cannot leave [-1, 1]."""
    from xai_gov.analysis.sweeps import _net_value

    with pytest.raises(ValueError, match="service_value must be positive"):
        _net_value(0.9, 0.6, 0.1, 0.0)
    # The price sets the scale, so it decides the sign at a given cost.
    assert _net_value(0.67, 0.61, 10.0, 100.0) is not None
    cheap = _net_value(0.67, 0.61, 10.0, 10.0)
    dear = _net_value(0.67, 0.61, 10.0, 1000.0)
    assert cheap is not None and dear is not None
    assert cheap < 0.0 < dear


def test_net_value_can_be_negative() -> None:
    """The finding the campaign already produced: an arm can gain operationally
    and still not pay for the oversight that produced the gain."""
    assert _net_value_local() < 0.0


def _net_value_local() -> float:
    from xai_gov.analysis.sweeps import _net_value

    value = _net_value(0.67, 0.61, 65.0, 100.0)
    assert value is not None
    return value


def test_the_default_indicator_exists_in_a_single_run() -> None:
    """Total value creation is a paired quantity computed in the analysis layer,
    so it is absent from every run's KPI tree; sweeping on it returned None at
    every level."""
    from dataclasses import replace as _r
    from pathlib import Path as _P

    from xai_gov.analysis.sweeps import ParameterSweep
    from xai_gov.core.paths import Paths
    from xai_gov.core.settings import load_settings

    settings = _r(load_settings(Paths(root=_P("."))), outputs_root=_P("."))
    sweep = ParameterSweep(
        settings=settings, project_root=_P("."),
        treatment=_P("a.yaml"), control=_P("b.yaml"),
    )
    assert sweep.indicator.startswith(("operational.", "governance.", "guarantees."))
    assert "total_value_creation" not in sweep.indicator


def test_no_sign_change_reports_none_as_a_finding() -> None:
    """The sign being constant over everything tested is a result, not a gap."""
    swept = result([point(0.5, 0.3, 0.0), point(1.0, 0.2, 0.0), point(2.0, 0.1, 0.0)])
    assert swept.crossing is None
    assert "sign is constant" in swept.to_payload()["interpretation"]


def test_levels_with_missing_data_do_not_create_a_false_crossing() -> None:
    swept = result([point(0.5, 0.2, 0.0), point(1.0, None, None), point(2.0, -0.2, 0.0)])
    crossing = swept.crossing
    # The bracketing pair is the two usable levels, so the crossing lies
    # between them rather than at the gap.
    assert crossing is not None
    assert 0.5 < crossing < 2.0


def test_monotonicity_needs_three_points() -> None:
    """Two points are monotone by construction, so reporting True there would
    be reporting the sample size."""
    assert result([point(1.0, 0.1, 0.0), point(2.0, -0.1, 0.0)]).monotone is None
    swept = result([point(1.0, 0.3, 0.0), point(2.0, 0.1, 0.0), point(3.0, -0.1, 0.0)])
    assert swept.monotone is True


def test_a_non_monotone_curve_is_flagged_and_withholds_the_threshold() -> None:
    """Theorem 2 predicts monotonicity in severity; a violation means an
    assumption is broken and no single threshold should be quoted."""
    swept = result(
        [point(1.5, 0.2, 0.0), point(3.0, -0.1, 0.0), point(6.0, 0.15, 0.0),
         point(10.0, -0.3, 0.0)],
        parameter="volatility_multiplier",
    )
    assert swept.monotone is False
    interpretation = swept.to_payload()["interpretation"]
    assert "not monotone" in interpretation
    assert "should not be quoted" in interpretation


def test_a_monotone_curve_licenses_the_threshold() -> None:
    swept = result(
        [point(1.5, -0.3, 0.0), point(3.0, -0.1, 0.0), point(6.0, 0.1, 0.0),
         point(10.0, 0.4, 0.0)],
        parameter="volatility_multiplier",
    )
    assert swept.monotone is True
    assert "usable threshold" in swept.to_payload()["interpretation"]


def test_the_payload_names_how_the_crossing_was_estimated() -> None:
    payload = result([point(0.5, 0.2, 0.0), point(1.0, -0.2, 0.0)]).to_payload()
    assert payload["crossing_method"] == "linear interpolation between bracketing levels"
    assert payload["replicates_per_level"] == 10
    assert payload["has_data"] is True


def test_the_declared_levels_are_the_ones_the_manuscript_names() -> None:
    """Declared rather than passed as literals, so code and paper cannot drift."""
    assert BREAK_EVEN_COSTS[0] < 1.0 < BREAK_EVEN_COSTS[-1]
    assert list(SEVERITY_LEVELS) == sorted(SEVERITY_LEVELS)
    assert len(SEVERITY_LEVELS) >= 3


def test_a_sweep_refuses_a_single_replicate(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """One run per level cannot separate the parameter from the seed."""
    from dataclasses import replace

    from xai_gov.analysis.sweeps import ParameterSweep
    from xai_gov.core.paths import Paths
    from xai_gov.core.settings import load_settings

    settings = replace(load_settings(Paths(root=tmp_path)), outputs_root=tmp_path)
    sweep = ParameterSweep(
        settings=settings,
        project_root=tmp_path,
        treatment=tmp_path / "a.yaml",
        control=tmp_path / "b.yaml",
        replicates=1,
    )
    with pytest.raises(ValueError, match="at least two replicates"):
        sweep.run(parameter="x", levels=[1.0], override="a.b")


def test_a_sweep_refuses_no_levels(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from xai_gov.analysis.sweeps import ParameterSweep
    from xai_gov.core.paths import Paths
    from xai_gov.core.settings import load_settings

    settings = replace(load_settings(Paths(root=tmp_path)), outputs_root=tmp_path)
    sweep = ParameterSweep(
        settings=settings, project_root=tmp_path,
        treatment=tmp_path / "a.yaml", control=tmp_path / "b.yaml", replicates=4,
    )
    with pytest.raises(ValueError, match="at least one level"):
        sweep.run(parameter="x", levels=[], override="a.b")


def test_mismatched_labels_are_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from dataclasses import replace

    from xai_gov.analysis.sweeps import ParameterSweep
    from xai_gov.core.paths import Paths
    from xai_gov.core.settings import load_settings

    settings = replace(load_settings(Paths(root=tmp_path)), outputs_root=tmp_path)
    sweep = ParameterSweep(
        settings=settings, project_root=tmp_path,
        treatment=tmp_path / "a.yaml", control=tmp_path / "b.yaml", replicates=4,
    )
    with pytest.raises(ValueError, match="match levels one to one"):
        sweep.run(parameter="x", levels=[1.0, 2.0], override="a.b", labels=["only one"])
