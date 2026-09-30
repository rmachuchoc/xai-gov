"""A campaign end to end: from sealed plan to reportable finding."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from xai_gov.analysis.campaign import CampaignRunner, apply_exclusions, plan_campaign
from xai_gov.analysis.preregistration import Hypothesis, Preregistration
from xai_gov.analysis.report import run_campaign
from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings

pytestmark = pytest.mark.integration

EXPERIMENTS = Path("configs/experiments")


@pytest.fixture
def settings(project_root: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    base = load_settings(Paths(root=project_root))
    outputs = tmp_path / "outputs"
    (outputs / "runs").mkdir(parents=True)
    return replace(base, outputs_root=outputs)


@pytest.fixture
def arms(project_root: Path) -> dict[str, Path]:
    return {
        "ungoverned": project_root / EXPERIMENTS / "high_vol_ungoverned.yaml",
        "governed": project_root / EXPERIMENTS / "high_vol_governed_h2.yaml",
    }


@pytest.fixture
def plan() -> Preregistration:
    return Preregistration(
        campaign="test",
        minimum_replicates=2,
        hypotheses=(
            Hypothesis(
                hypothesis_id="H1",
                question="does governance change service level?",
                indicator="operational.service_level",
                treatment="governed",
                control="ungoverned",
                direction="different",
                predicted_effect=0.05,
            ),
        ),
    )


def test_replicates_share_seeds_across_arms(arms: dict[str, Path]) -> None:
    """Cell k of every arm sees the same demand path, so the comparison is
    paired and the design's variance reduction is not paid for twice."""
    cells = plan_campaign(arms=arms, replicates=3, base_seed=42)
    by_replicate: dict[int, set[int]] = {}
    for cell in cells:
        by_replicate.setdefault(cell.replicate, set()).add(cell.seed)
    assert all(len(seeds) == 1 for seeds in by_replicate.values())
    assert len({next(iter(s)) for s in by_replicate.values()}) == 3


def test_a_campaign_needs_arms_and_replicates(arms: dict[str, Path]) -> None:
    with pytest.raises(ValueError, match="at least one replicate"):
        plan_campaign(arms=arms, replicates=0, base_seed=1)
    with pytest.raises(ValueError, match="at least one arm"):
        plan_campaign(arms={}, replicates=1, base_seed=1)


def test_exclusions_are_mechanical() -> None:
    """A criterion applied by judgement after seeing the outcome is not a
    criterion, it is a choice."""
    healthy = ({"status": "completed", "chain_verified": True}, {})
    assert apply_exclusions(*healthy) == (True, None)

    included, reason = apply_exclusions(
        {"status": "completed", "chain_verified": False}, {}
    )
    assert included is False
    assert "hash-chain" in str(reason)

    included, reason = apply_exclusions(
        {"status": "completed", "chain_verified": True},
        {"guarantees": {"saturated": True}},
    )
    assert included is False
    assert "saturation" in str(reason)


def test_a_campaign_runs_and_reports(settings, arms, plan, project_root) -> None:  # type: ignore[no-untyped-def]
    report = run_campaign(
        plan=plan,
        arms=arms,
        settings=settings,
        project_root=project_root,
        replicates=3,
        base_seed=20260703,
    )
    payload = report.to_payload()
    assert payload["campaign_summary"]["included"] > 0
    assert len(payload["hypotheses"]) == 1
    assert payload["hypotheses"][0]["id"] == "H1"


def test_a_campaign_without_a_seal_is_exploratory(settings, arms, plan, project_root) -> None:  # type: ignore[no-untyped-def]
    """An exploratory campaign is legitimate; presenting one as confirmatory is
    not, so the label is attached rather than left implicit."""
    report = run_campaign(
        plan=plan, arms=arms, settings=settings, project_root=project_root,
        replicates=2, base_seed=1,
    )
    assert report.to_payload()["reporting_status"].startswith("exploratory")


def test_a_matching_seal_is_confirmatory(settings, arms, plan, project_root) -> None:  # type: ignore[no-untyped-def]
    report = run_campaign(
        plan=plan, arms=arms, settings=settings, project_root=project_root,
        replicates=2, base_seed=1, sealed_hash=plan.plan_hash,
    )
    assert report.to_payload()["reporting_status"] == "confirmatory"


def test_an_edited_plan_is_caught_at_analysis_time(settings, arms, project_root, plan) -> None:  # type: ignore[no-untyped-def]
    """The seal's whole purpose: an analysis cannot quietly run against a plan
    different from the one that was committed to."""
    edited = replace(
        plan,
        hypotheses=(replace(plan.hypotheses[0], direction="greater"),),
    )
    report = run_campaign(
        plan=edited, arms=arms, settings=settings, project_root=project_root,
        replicates=2, base_seed=1, sealed_hash=plan.plan_hash,
    )
    payload = report.to_payload()
    assert payload["reporting_status"].startswith("exploratory")
    assert payload["seal"]["matches"] is False


def test_a_plan_naming_an_unprovided_arm_is_refused(settings, arms, project_root) -> None:  # type: ignore[no-untyped-def]
    orphan = Preregistration(
        campaign="orphan",
        minimum_replicates=2,
        hypotheses=(
            Hypothesis(
                hypothesis_id="H1",
                question="?",
                indicator="operational.service_level",
                treatment="nonexistent",
                control="ungoverned",
                direction="greater",
                predicted_effect=0.1,
            ),
        ),
    )
    with pytest.raises(ValueError, match="no experiment file"):
        run_campaign(
            plan=orphan, arms=arms, settings=settings, project_root=project_root,
            replicates=2, base_seed=1,
        )


def test_an_absent_indicator_leaves_it_untested(settings, arms, project_root) -> None:  # type: ignore[no-untyped-def]
    """Untested is not refuted. The ungoverned arm has no guarantees layer, so
    a hypothesis about it has no pairs rather than a difference of zero."""
    plan = Preregistration(
        campaign="unavailable",
        minimum_replicates=2,
        hypotheses=(
            Hypothesis(
                hypothesis_id="H1",
                question="?",
                indicator="guarantees.coverage_error",
                treatment="governed",
                control="ungoverned",
                direction="less",
                predicted_effect=0.1,
            ),
        ),
    )
    report = run_campaign(
        plan=plan, arms=arms, settings=settings, project_root=project_root,
        replicates=2, base_seed=1,
    )
    result = report.to_payload()["hypotheses"][0]
    assert result["pairs"] == 0
    assert "untested rather than refuted" in result["verdict"]


def test_every_excluded_run_is_named_with_its_reason(settings, arms, plan, project_root) -> None:  # type: ignore[no-untyped-def]
    """A campaign that drops runs silently is reporting a selection."""
    report = run_campaign(
        plan=plan, arms=arms, settings=settings, project_root=project_root,
        replicates=2, base_seed=1,
    )
    summary = report.to_payload()["campaign_summary"]
    for exclusion in summary["exclusions"]:
        assert exclusion["reason"]
        assert exclusion["run_name"]


def test_the_report_is_written_and_traceable(settings, arms, plan, project_root, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Every number must lead back to a run identifier."""
    import json

    report = run_campaign(
        plan=plan, arms=arms, settings=settings, project_root=project_root,
        replicates=2, base_seed=1,
    )
    path = report.write(tmp_path / "report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["plan_hash"]
    assert all(cell["run_name"] for cell in payload["cells"])


def test_paired_differences_drop_the_pair_not_the_observation(settings, arms, project_root) -> None:  # type: ignore[no-untyped-def]
    """Substituting a mean for a missing side would manufacture agreement."""
    cells = plan_campaign(arms=arms, replicates=2, base_seed=1)
    runner = CampaignRunner(settings=settings, project_root=project_root)
    runner.run(cells)
    differences = runner.paired("governed", "ungoverned", "guarantees.coverage_error")
    # The ungoverned arm never reports this indicator, so there are no pairs.
    assert differences == []
