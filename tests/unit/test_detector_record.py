"""The detector record: coverage against the target, the ε_t clip, the null line."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.factories import state
from xai_gov.analysis.campaign import CampaignCell, CellResult
from xai_gov.analysis.detector_null import NullLinePlan, summarize_arm
from xai_gov.governance.agent import NEUTRAL_DESCRIPTOR
from xai_gov.io.decision_record import (
    ActionKind,
    ConformalSignal,
    DecisionRecord,
    GovernanceAction,
    ShieldOutcome,
)
from xai_gov.kpis.layers import DetectorTarget, _clip_episodes, guarantees_layer

TARGET = DetectorTarget(target_level=0.1, min_level=0.005, max_level=0.5)


def record(period: int, node: str, *, level: float, flagged: bool = False,
           change: bool = False) -> DecisionRecord:
    return DecisionRecord(
        period=period, node_id=node, state=state(period=period),
        policy_id="p", policy_version="1", proposed_action=ActionKind.HOLD,
        proposed_quantity=0.0, descriptor=NEUTRAL_DESCRIPTOR,
        conformal=ConformalSignal(score=1.0, threshold=2.0, level=level, flagged_ood=flagged,
                                  scheme="aci", change_declared=change),
        belief={}, governance_action=GovernanceAction.APPROVE, governance_rationale="",
        autonomy_regime="H2", escalated=False, safe_mode=False, intervention_cost=0.0,
        shield=ShieldOutcome(), final_action=ActionKind.HOLD, final_quantity=0.0,
    )


def test_coverage_error_is_the_realized_rate_minus_the_target() -> None:
    """Not minus the mean adapted level: 2 flags in 10 against 0.10 is +0.10."""
    records = [record(t, "r", level=0.3, flagged=t < 2) for t in range(10)]
    layer = guarantees_layer(records, target=TARGET)
    assert layer["coverage_error"] == pytest.approx(0.1)
    assert layer["abs_coverage_error"] == pytest.approx(0.1)
    assert layer["coverage_error_vs_mean_level"] == pytest.approx(0.2 - 0.3)
    assert layer["target_level"] == 0.1


def test_clip_episodes_are_maximal_runs() -> None:
    assert _clip_episodes([True, True, False, True, False, False, True, True, True]) == [2, 1, 3]
    assert _clip_episodes([]) == []


def test_the_epsilon_record_has_the_trajectory_and_the_clip_per_node() -> None:
    levels = [0.1, 0.005, 0.005, 0.005, 0.02, 0.5, 0.5, 0.1]
    records = [record(t, "r", level=v) for t, v in enumerate(levels)]
    records += [record(t, "d", level=0.1) for t in range(8)]
    epsilon = guarantees_layer(records, target=TARGET)["epsilon"]

    assert epsilon["trajectory"]["r"] == [[t, v] for t, v in enumerate(levels)]
    retail = epsilon["clipping"]["per_node"]["r"]
    assert (retail["clipped_low"], retail["clipped_high"]) == (3, 2)
    assert retail["episodes"] == 2
    assert retail["max_duration"] == 3
    assert retail["mean_duration"] == pytest.approx(2.5)
    assert epsilon["clipping"]["per_node"]["d"]["episodes"] == 0
    assert epsilon["clipping"]["clipped_frequency"] == pytest.approx(5 / 16)
    assert epsilon["clipping"]["episodes"] == 2


def test_a_level_just_above_the_floor_is_not_clipped() -> None:
    records = [record(t, "r", level=0.0050004) for t in range(5)]
    assert guarantees_layer(records, target=TARGET)["epsilon"]["clipping"]["episodes"] == 0


def test_change_declarations_are_reported_per_node() -> None:
    records = [record(0, "r", level=0.1), record(0, "d", level=0.1, change=True)]
    assert guarantees_layer(records, target=TARGET)["change_declared_by_node"] == {
        "d": True, "r": False,
    }


# -- the null line --------------------------------------------------------------
def cell(replicate: int, by_node: dict[str, bool], clipping: dict | None = None) -> CellResult:  # type: ignore[type-arg]
    epsilon = {"clipping": clipping} if clipping else {}
    return CellResult(
        cell=CampaignCell(arm="a", experiment=Path("a.yaml"), replicate=replicate, seed=1),
        run_name=f"run{replicate}",
        kpis={"guarantees": {"status": "available", "coverage_error": -0.02,
                             "change_declared_by_node": by_node, "epsilon": epsilon}},
        included=True,
    )


def test_a_false_alarm_counts_once_per_run_and_only_at_demand_facing_nodes() -> None:
    results = [
        cell(0, {"r": True, "d": True}),
        cell(1, {"r": False, "d": True}),
        cell(2, {"r": False, "d": False}),
        cell(3, {"r": False, "d": False}),
    ]
    summary = summarize_arm(results, demand_facing=frozenset({"r"}))
    assert summary["false_alarms_demand_facing"]["runs_with_alarm"] == 1
    assert summary["false_alarms_demand_facing"]["frequency"] == pytest.approx(0.25)
    assert summary["declarations_any_node"]["runs_with_alarm"] == 2
    alarm = summary["false_alarms_demand_facing"]
    assert alarm["ci_low"] < 0.25 < alarm["ci_high"]


def test_the_null_summary_aggregates_the_clip() -> None:
    clip = {"clipped_frequency": 0.1, "episodes": 2, "mean_duration": 1.5, "max_duration": 2}
    calm = {"clipped_frequency": 0.0, "episodes": 0, "mean_duration": 0.0, "max_duration": 0}
    summary = summarize_arm([cell(0, {}, clip), cell(1, {}, calm)], demand_facing=frozenset())
    clipping = summary["epsilon_clipping"]
    assert clipping["runs_with_clipping"] == 1
    assert clipping["clipped_frequency"]["mean"] == pytest.approx(0.05)
    assert clipping["mean_duration"]["mean"] == pytest.approx(1.5)
    assert clipping["max_duration"] == 2


def test_the_null_line_must_declare_its_own_seed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="base_seed"):
        NullLinePlan.from_config({"arms": {"a": "x.yaml"}}, root=tmp_path)
    plan = NullLinePlan.from_config(
        {"arms": {"a": "x.yaml"}, "replicates": 5, "base_seed": 9}, root=tmp_path
    )
    assert (plan.replicates, plan.base_seed, plan.arms["a"]) == (5, 9, tmp_path / "x.yaml")


def test_the_declared_null_line_uses_a_seed_no_campaign_uses(project_root: Path) -> None:
    """Campaign seeds come from the scripts' --base-seed defaults, not from YAML."""
    import re

    import yaml

    null = yaml.safe_load((project_root / "configs/analysis/detector_null.yaml").read_text())
    assert null["replicates"] == 200
    defaults = {
        int(seed)
        for script in ("run_study.py", "run_sweeps.py")
        for seed in re.findall(
            r'"--base-seed", type=int, default=(\d+)',
            (project_root / "scripts" / script).read_text(encoding="utf-8"),
        )
    }
    assert defaults, "no --base-seed default found; the check would be vacuous"
    assert null["base_seed"] not in defaults
