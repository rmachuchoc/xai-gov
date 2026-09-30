"""Study artifacts: the failures that complete without erroring."""

from __future__ import annotations

from xai_gov.analysis.artifacts import (
    CELL_INDICATORS,
    arm_aggregates,
    cell_table,
    diagnose,
    digest,
    write_artifacts,
)


def bundle(**overrides: object) -> dict:  # type: ignore[type-arg]
    base = {
        "study": "test",
        "replicates": 40,
        "reporting_status": "confirmatory",
        "power": {
            "adequate": True,
            "planned_replicates": 40,
            "required_anytime_valid": 40,
            "minimum_detectable_effect": 0.49,
        },
        "campaign": {
            "campaign_summary": {"cells": 4, "included": 4, "excluded": 0, "exclusions": []},
            "plan": {
                "hypotheses": [
                    {"id": "H1", "direction": "greater", "indicator": "operational.service_level"}
                ]
            },
            "hypotheses": [
                {
                    "id": "H1",
                    "indicator": "operational.service_level",
                    "treatment": "governed",
                    "control": "ungoverned",
                    "pairs": 40,
                    "mean_difference": 0.12,
                    "evidence": {"forward_e_value": 400.0, "reverse_e_value": 1.0},
                    "supported": True,
                }
            ],
            "frontier": {
                "frontier_arms": ["governed"],
                "excluded_arms": {},
                "theorem_3_consistent": True,
            },
            "cells": [
                {
                    "arm": "governed",
                    "replicate": 0,
                    "seed": 1,
                    "run_name": "r1",
                    "included": True,
                    "exclusion_reason": None,
                },
                {
                    "arm": "ungoverned",
                    "replicate": 0,
                    "seed": 1,
                    "run_name": "r2",
                    "included": True,
                    "exclusion_reason": None,
                },
            ],
        },
        "sensitivity": {
            "indices": [
                {"name": "intervention_cost", "first_order": 0.4, "total_effect": 0.6,
                 "influential": True}
            ]
        },
        "calibration": {
            "identified": {"base_demand": True},
            "predictive_check": {
                "overfitted": False,
                "held_out_distance": 0.2,
                "calibrated_distance": 0.15,
                "sim_to_real_gap": 0.2,
            },
        },
    }
    return {**base, **overrides}


KPIS = {
    "r1": {
        "operational": {"service_level": 0.98},
        "governance": {"intervention_rate": 0.3},
    },
    "r2": {"operational": {"service_level": 0.90}},
}


def test_a_clean_study_flags_nothing() -> None:
    assert diagnose(bundle()) == []


def test_identical_arms_are_blocking_not_a_null_result() -> None:
    """Every paired difference exactly zero means the comparison had nothing to
    measure, so reporting it as a null would be wrong."""
    payload = bundle()
    payload["campaign"]["hypotheses"][0].update(
        {
            "mean_difference": 0.0,
            # Both directions at exactly one: the bet lost either way, which
            # only happens when every paired difference is zero.
            "evidence": {"forward_e_value": 1.0, "reverse_e_value": 1.0},
            "supported": False,
        }
    )
    findings = diagnose(payload)
    assert [f.code for f in findings] == ["identical_arms"]
    assert findings[0].severity == "blocking"
    assert "untestable as configured" in findings[0].implication


def test_an_arm_excluded_wholesale_is_a_finding_about_the_arm() -> None:
    """Every cell gone for the same reason is not a set of bad runs. Reported as
    "0 usable pairs" it invites the conclusion that the comparison was
    misconfigured, when the treatment systematically fails a declared
    criterion."""
    payload = bundle()
    payload["campaign"]["campaign_summary"].update(
        {
            "included_by_arm": {"governed": 40, "reference_scale": 0},
            "exclusions": [
                {"arm": "reference_scale", "replicate": i, "run_name": f"r{i}",
                 "reason": "conformal detector reported saturation"}
                for i in range(40)
            ],
        }
    )
    finding = next(
        f for f in diagnose(payload) if f.code == "arm_systematically_excluded"
    )
    assert finding.severity == "blocking"
    assert "reference_scale" in finding.detail
    assert "answered in the negative rather than left untested" in finding.implication


def test_scattered_exclusions_are_not_a_finding_about_the_arm() -> None:
    payload = bundle()
    payload["campaign"]["campaign_summary"].update(
        {
            "included_by_arm": {"governed": 38},
            "exclusions": [
                {"arm": "governed", "replicate": 0, "run_name": "r0", "reason": "a"},
                {"arm": "governed", "replicate": 1, "run_name": "r1", "reason": "b"},
            ],
        }
    )
    assert not any(
        f.code == "arm_systematically_excluded" for f in diagnose(payload)
    )


def test_a_criteria_bump_is_recorded_as_a_note_not_a_problem() -> None:
    """Re-running stale cells is the campaign staying internally consistent."""
    payload = bundle()
    payload["campaign"]["campaign_summary"]["re_run_for_stale_criteria"] = 400
    finding = next(
        f for f in diagnose(payload) if f.code == "criteria_version_advanced"
    )
    assert finding.severity == "note"
    assert "no action needed" in finding.implication


def test_an_absolute_indicator_is_exempt_from_the_signed_error_check() -> None:
    """The diagnostic recommends the absolute indicator, so flagging it would
    make the fix look like the defect."""
    payload = bundle()
    payload["campaign"]["plan"]["hypotheses"][0]["direction"] = "less"
    payload["campaign"]["hypotheses"][0].update(
        {"indicator": "guarantees.abs_coverage_error", "mean_difference": -0.05}
    )
    assert not any(f.code == "signed_error_direction" for f in diagnose(payload))


def test_a_signed_error_still_trips_the_check() -> None:
    payload = bundle()
    payload["campaign"]["plan"]["hypotheses"][0]["direction"] = "less"
    payload["campaign"]["hypotheses"][0].update(
        {"indicator": "guarantees.coverage_error", "mean_difference": -0.05}
    )
    finding = next(f for f in diagnose(payload) if f.code == "signed_error_direction")
    assert finding.severity == "blocking"


def test_an_inert_indicator_is_told_apart_from_identical_arms() -> None:
    """Two failures wear the same zero and call for opposite fixes: both arms at
    zero needs a different regime, both at the same non-zero value needs the
    treatment wired to the indicator."""
    payload = bundle()
    payload["campaign"]["hypotheses"][0].update(
        {
            "indicator": "governance.intervention_rate",
            "treatment": "a",
            "control": "b",
            "mean_difference": 0.0,
            "evidence": {"forward_e_value": 1.0, "reverse_e_value": 1.0},
            "supported": False,
        }
    )
    payload["campaign"]["frontier"]["points"] = [
        {"arm": "a", "values": {"intervention_rate": 0.0}},
        {"arm": "b", "values": {"intervention_rate": 0.0}},
    ]
    findings = diagnose(payload)
    finding = next(f for f in findings if f.code == "indicator_inert")
    assert "nothing to act on" in finding.implication
    assert "needs a regime where" in finding.implication


def test_a_lost_bet_in_one_direction_only_is_not_identical_arms() -> None:
    """A forward e-value of one with a large reverse value means the effect ran
    the other way, not that the arms were the same. Keying the diagnostic on
    the forward value alone would have conflated a real reversal with an empty
    comparison."""
    payload = bundle()
    payload["campaign"]["hypotheses"][0].update(
        {
            "mean_difference": 0.0,
            "evidence": {"forward_e_value": 1.0, "reverse_e_value": 5000.0},
            "supported": False,
        }
    )
    assert not any(f.code == "identical_arms" for f in diagnose(payload))


def test_a_hypothesis_with_no_pairs_is_untested_not_refuted() -> None:
    payload = bundle()
    payload["campaign"]["hypotheses"][0]["pairs"] = 0
    findings = diagnose(payload)
    assert findings[0].code == "no_pairs"
    assert "untested, not refuted" in findings[0].implication


def test_a_reversed_effect_is_surfaced() -> None:
    """Predicted greater, observed negative: a finding to discuss, not an
    'insufficient evidence' line to bury it in."""
    payload = bundle()
    payload["campaign"]["hypotheses"][0]["mean_difference"] = -0.55
    findings = diagnose(payload)
    assert any(f.code == "effect_reversed" for f in findings)


def test_an_exact_zero_sensitivity_index_is_suspected_wiring() -> None:
    """Far more often the model never reads the parameter than the parameter is
    genuinely irrelevant."""
    payload = bundle()
    payload["sensitivity"]["indices"].append(
        {"name": "allowance", "first_order": 0.0, "total_effect": 0.0, "influential": False}
    )
    findings = diagnose(payload)
    assert any(f.code == "parameter_unread" for f in findings)


def test_a_weakly_identified_posterior_is_flagged() -> None:
    payload = bundle()
    payload["calibration"]["identified"] = {"base_demand": False, "noise_scale": False}
    findings = diagnose(payload)
    finding = next(f for f in findings if f.code == "weakly_identified")
    assert "must not be reported as calibrated" in finding.implication


def test_an_overfitted_calibration_is_flagged() -> None:
    payload = bundle()
    payload["calibration"]["predictive_check"]["overfitted"] = True
    assert any(f.code == "calibration_overfit" for f in diagnose(payload))


def test_an_underpowered_study_is_flagged() -> None:
    payload = bundle()
    payload["power"]["adequate"] = False
    findings = diagnose(payload)
    assert any(f.code == "underpowered" for f in findings)


def test_a_theorem_3_violation_blocks() -> None:
    payload = bundle()
    payload["campaign"]["frontier"]["theorem_3_consistent"] = False
    findings = diagnose(payload)
    finding = next(f for f in findings if f.code == "theorem_3_violated")
    assert finding.severity == "blocking"
    assert "do not report" in finding.implication


def test_blocking_findings_sort_first() -> None:
    payload = bundle()
    payload["power"]["adequate"] = False
    payload["campaign"]["hypotheses"][0]["pairs"] = 0
    findings = diagnose(payload)
    assert findings[0].severity == "blocking"


def test_the_cell_table_leaves_gaps_visible() -> None:
    """A missing indicator must stay a gap; zero-filling it would put a
    fabricated value into whatever reads the table."""
    csv = cell_table(bundle(), KPIS)
    rows = csv.strip().split("\n")
    header = rows[0].split(",")
    ungoverned = rows[2].split(",")
    index = header.index("governance_intervention_rate")
    assert ungoverned[index] == ""


def test_the_cell_table_carries_run_identifiers() -> None:
    csv = cell_table(bundle(), KPIS)
    assert "r1" in csv
    assert "r2" in csv
    assert "operational_service_level" in csv


def test_aggregates_report_spread_not_only_means() -> None:
    """Two arms with the same mean and zero variance are the same arm; with real
    spread they are a real null. The sd is what separates them."""
    aggregates = arm_aggregates(bundle(), KPIS)
    entry = aggregates["governed"]["operational.service_level"]
    assert entry["n"] == 1
    assert entry["sd"] == 0.0
    assert entry["mean"] == 0.98


def test_excluded_cells_are_left_out_of_aggregates() -> None:
    payload = bundle()
    payload["campaign"]["cells"][0]["included"] = False
    aggregates = arm_aggregates(payload, KPIS)
    assert "governed" not in aggregates


def test_the_digest_states_the_status_and_the_verdicts() -> None:
    payload = bundle()
    text = digest(payload, diagnose(payload), arm_aggregates(payload, KPIS))
    assert "confirmatory" in text
    assert "| H1 |" in text
    assert "Nothing flagged." in text


def test_the_digest_lists_findings_with_their_implications() -> None:
    payload = bundle()
    payload["campaign"]["hypotheses"][0].update(
        {"mean_difference": 0.0, "evidence": {"forward_e_value": 1.0, "reverse_e_value": 1.0}}
    )
    text = digest(payload, diagnose(payload), arm_aggregates(payload, KPIS))
    assert "BLOCKING" in text
    assert "identical_arms" in text


def test_every_artifact_is_written(tmp_path) -> None:  # type: ignore[no-untyped-def]
    written = write_artifacts(
        bundle=bundle(),
        phases={"01_power": {"adequate": True}},
        kpis_by_run=KPIS,
        directory=tmp_path / "campaign",
    )
    for key in ("cells.csv", "arm_aggregates.json", "diagnostics.json", "digest.md"):
        assert written[key].exists(), key
    assert written["01_power"].exists()
    assert (tmp_path / "campaign" / "phases" / "01_power.json").exists()


def test_the_diagnostics_file_says_whether_reporting_is_safe(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    payload = bundle()
    payload["campaign"]["frontier"]["theorem_3_consistent"] = False
    written = write_artifacts(
        bundle=payload, phases={}, kpis_by_run=KPIS, directory=tmp_path / "c"
    )
    diagnostics = json.loads(written["diagnostics.json"].read_text(encoding="utf-8"))
    assert diagnostics["safe_to_report"] is False
    assert diagnostics["blocking"] >= 1


def test_the_indicator_list_covers_the_frontier_objectives() -> None:
    """A table that cannot check the frontier cannot be used to audit it."""
    for indicator in (
        "guarantees.coverage_error",
        "governance.total_intervention_cost",
        "operational.service_level",
    ):
        assert indicator in CELL_INDICATORS
