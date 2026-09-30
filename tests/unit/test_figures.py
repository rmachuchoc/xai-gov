"""Figures generated from the bundle, not transcribed from it."""

from __future__ import annotations

from xai_gov.analysis.figures import (
    ArmSeries,
    build_figures,
    evidence_bars,
    grouped_bars,
    paired_dots,
    series_from_aggregates,
    strategic_scatter,
    write_figures,
)

AGGREGATES = {
    "ungoverned": {
        "operational.service_level": {"mean": 0.611, "sd": 0.051, "n": 40},
        "governance.intervention_rate": {"mean": 0.0, "sd": 0.0, "n": 40},
    },
    "governed": {
        "operational.service_level": {"mean": 0.059, "sd": 0.027, "n": 40},
        "governance.intervention_rate": {"mean": 0.416, "sd": 0.031, "n": 40},
        "guarantees.ood_rate": {"mean": 0.0, "sd": 0.0, "n": 40},
        "guarantees.mean_nominal_level": {"mean": 0.158, "sd": 0.004, "n": 40},
    },
    "shielded": {
        "operational.service_level": {"mean": 0.670, "sd": 0.042, "n": 40},
        "governance.intervention_rate": {"mean": 0.654, "sd": 0.035, "n": 40},
        "guarantees.ood_rate": {"mean": 0.002, "sd": 0.001, "n": 40},
        "guarantees.mean_nominal_level": {"mean": 0.138, "sd": 0.006, "n": 40},
    },
}

HYPOTHESES = [
    {
        "id": "RQ8",
        "indicator": "operational.service_level",
        "treatment": "shielded",
        "control": "unshielded",
        "pairs": 40,
        "mean_difference": 0.6108,
        "evidence": {
            "forward_e_value": 3.1e8,
            "reverse_e_value": 1.0,
            "refuted": False,
        },
        "supported": True,
    },
    {
        "id": "RQ4",
        "indicator": "operational.service_level",
        "treatment": "governed",
        "control": "ungoverned",
        "pairs": 40,
        "mean_difference": -0.5525,
        # Forward lost the bet; the reverse test is what establishes that the
        # effect ran opposite to the prediction.
        "evidence": {
            "forward_e_value": 0.044,
            "reverse_e_value": 4.2e6,
            "refuted": True,
        },
        "supported": False,
    },
]

BUNDLE = {
    "study": "test",
    "replicates": 40,
    "reporting_status": "confirmatory",
    "power": {"minimum_detectable_effect": 0.495},
    "campaign": {
        "plan": {"alpha": 0.05},
        "hypotheses": HYPOTHESES,
        "strategic": {
            "status": "available",
            "baseline": "ungoverned",
            "arms": {
                "governed": {
                    "value_of_governance": -0.5525,
                    "tvc": -0.4229,
                    "rgd": -0.05,
                    "governance_pays": False,
                },
                "shielded": {
                    "value_of_governance": 0.0583,
                    "tvc": -0.6539,
                    "rgd": 0.01,
                    "governance_pays": False,
                },
            },
        },
    },
}


def test_a_series_omits_arms_without_the_indicator() -> None:
    """The KPI layers refuse to fabricate a zero; a figure that puts one back
    undoes them."""
    series = series_from_aggregates(AGGREGATES, "guarantees.ood_rate")
    assert set(series.values) == {"governed", "shielded"}
    assert "ungoverned" not in series.values


def test_grouped_bars_carry_both_series() -> None:
    chart = grouped_bars(
        series_from_aggregates(AGGREGATES, "operational.service_level"),
        series_from_aggregates(AGGREGATES, "governance.intervention_rate"),
        arms=["ungoverned", "governed", "shielded"],
        primary_label="service level",
        secondary_label="intervention rate",
    )
    assert "0.611" in chart and "0.059" in chart and "0.670" in chart
    assert "intervention rate" in chart
    assert chart.startswith("<svg") and chart.endswith("</svg>")


def test_a_chart_with_no_usable_arms_is_empty_not_blank() -> None:
    """An absent figure tells a caller the campaign produced nothing; an empty
    one implies it produced zeros."""
    empty = ArmSeries(indicator="x", values={})
    assert grouped_bars(empty, empty, arms=["a"]) == ""
    assert paired_dots(empty, empty, arms=["a"]) == ""


def test_paired_dots_need_both_ends() -> None:
    realized = series_from_aggregates(AGGREGATES, "guarantees.ood_rate")
    nominal = series_from_aggregates(AGGREGATES, "guarantees.mean_nominal_level")
    chart = paired_dots(realized, nominal, arms=sorted(AGGREGATES))
    # ungoverned has neither, so it contributes no row.
    assert chart.count('r="4.5"') >= 4
    assert "coverage error" in chart


def test_a_refuted_hypothesis_plots_its_reverse_e_value() -> None:
    """The value drawn to the left is the reverse test, which is what supports
    the claim. Drawing a sub-unit forward value there would show a lost bet as
    though it were evidence for the opposite side."""
    chart = evidence_bars(HYPOTHESES)
    assert "established opposite" in chart
    assert "(opposite)" in chart
    # The reverse value, not the forward 0.044.
    assert "4.20e+06" in chart
    assert "0.044" not in chart


def test_a_supported_hypothesis_plots_its_forward_e_value() -> None:
    chart = evidence_bars(HYPOTHESES)
    assert "3.10e+08" in chart or "3.1e+08" in chart


def test_the_marked_threshold_is_the_familywise_one() -> None:
    """The familywise threshold is the primary decision rule, so it is the line
    a reader should measure against."""
    chart = evidence_bars(HYPOTHESES, alpha=0.05, familywise_threshold=140.0)
    assert chart.startswith("<svg")
    assert "familywise threshold 140" in chart


def test_a_zero_e_value_is_skipped_not_logged() -> None:
    """log10(0) is undefined; dropping the point is correct and crashing is
    not."""
    chart = evidence_bars(
        [{**HYPOTHESES[0], "evidence": {"forward_e_value": 0.0}}]
    )
    assert chart == ""


def test_the_scatter_always_includes_the_zero_line() -> None:
    """Which side of zero an arm falls on is the finding, so an axis fitted to
    the data's own range would hide it."""
    chart = strategic_scatter(BUNDLE["campaign"]["strategic"])
    assert "VoG = 0" in chart
    assert "total value creation" in chart


def test_the_scatter_needs_both_coordinates() -> None:
    partial = {"status": "available", "arms": {"a": {"value_of_governance": 0.1}}}
    assert strategic_scatter(partial) == ""


def test_every_supported_figure_is_built() -> None:
    figures = build_figures(BUNDLE, AGGREGATES)
    assert "fig1_service_and_intervention.svg" in figures
    assert "fig2_coverage_gap.svg" in figures
    assert "fig3_strategic_value.svg" in figures
    assert "fig4_evidence.svg" in figures


def test_a_figure_without_inputs_is_absent_from_the_result() -> None:
    thin = build_figures({"campaign": {"hypotheses": []}}, {})
    assert thin == {}


def test_figures_are_written_as_text_that_diffs(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """SVG rather than a raster export: a figure diffs against its previous
    version, so a reviewer can see what a re-run changed."""
    written = write_figures(bundle=BUNDLE, aggregates=AGGREGATES, directory=tmp_path)
    assert written
    for path in written.values():
        assert path.suffix == ".svg"
        assert path.read_text(encoding="utf-8").startswith("<svg")


def test_regenerating_yields_identical_output(tmp_path) -> None:
    """The point of generating rather than transcribing: same bundle, same
    figure, byte for byte."""
    first = build_figures(BUNDLE, AGGREGATES)
    second = build_figures(BUNDLE, AGGREGATES)
    assert first == second


def test_markup_is_escaped() -> None:
    """An arm name is data, and data that reaches markup unescaped is a way to
    produce a figure that silently fails to render."""
    aggregates = {
        "a<b>": {"operational.service_level": {"mean": 0.5, "sd": 0.1, "n": 2}}
    }
    chart = grouped_bars(
        series_from_aggregates(aggregates, "operational.service_level"),
        ArmSeries(indicator="x", values={}),
        arms=["a<b>"],
    )
    assert "&lt;b&gt;" in chart
    assert "<b>" not in chart
