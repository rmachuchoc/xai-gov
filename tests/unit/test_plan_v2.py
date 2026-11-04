"""The v2 plan: what sealing checks, and the decision rules it fixes."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from xai_gov.analysis.plan_v2 import check_plan, decision_scenario, failures
from xai_gov.analysis.preregistration import Hypothesis, Preregistration, build_preregistration
from xai_gov.io.yaml_loader import load_config

HORIZONS = {arm: 60 for arm in (
    "projection", "fallback", "backup", "q_projection", "q_fallback", "governed",
    "ungoverned", "five_projection", "five_fallback",
)}


@pytest.fixture
def plan(project_root: Path) -> Preregistration:
    config = load_config(project_root / "configs/campaigns/confirmatory_v2.yaml",
                         root=project_root)
    return build_preregistration({k: v for k, v in config.items() if k != "arms"})


def failed_names(plan: Preregistration, root: Path, horizons: dict[str, int] = HORIZONS) -> set:  # type: ignore[type-arg]
    return {c.name for c in failures(check_plan(plan, root=root, horizons=horizons))}


def test_the_repository_plan_passes_every_check(plan: Preregistration, project_root: Path) -> None:
    assert failed_names(plan, project_root) == set()
    assert (plan.replicates, plan.base_seed) == (500, 20261107)


def test_a_reused_seed_is_caught(plan: Preregistration, project_root: Path) -> None:
    for seed in (20260703, 20261105, 20261106, 20261104):
        assert "base seed is new" in failed_names(replace(plan, base_seed=seed), project_root)


def test_too_few_replicates_for_the_simulated_power_is_caught(
    plan: Preregistration, project_root: Path
) -> None:
    assert "replicates meet the simulated power" in failed_names(
        replace(plan, replicates=300), project_root
    )


def test_a_changed_power_report_is_caught(
    plan: Preregistration, project_root: Path, tmp_path: Path
) -> None:
    for name in ("pilot_v2.json", "power_v2.json"):
        source = project_root / "specs/v2" / name
        (tmp_path / "specs/v2").mkdir(parents=True, exist_ok=True)
        (tmp_path / "specs/v2" / name).write_text(source.read_text(encoding="utf-8"))
    power = json.loads((tmp_path / "specs/v2/power_v2.json").read_text())
    power["plan"]["replicates"] = 100
    (tmp_path / "specs/v2/power_v2.json").write_text(json.dumps(power))
    assert "power report unchanged" in failed_names(plan, tmp_path)


def test_a_wrong_threshold_and_unequal_horizons_are_caught(
    plan: Preregistration, project_root: Path
) -> None:
    design = json.loads(json.dumps(plan.design))
    design["test"]["familywise_threshold"] = 140
    broken = replace(plan, design=design)
    names = failed_names(broken, project_root, {**HORIZONS, "ungoverned": 40})
    assert {"threshold is 2k/alpha", "one horizon for every arm"} <= names


# -- the decision rules -----------------------------------------------------------
def outcome(h1: bool, h3: bool, h3_mean: float = 0.0) -> list[dict]:  # type: ignore[type-arg]
    return [
        {"id": "H1", "supported": h1, "mean_difference": 0.3},
        {"id": "H3", "supported": h3, "mean_difference": h3_mean},
    ]


@pytest.mark.parametrize(
    ("h1", "h3", "mean", "rule"),
    [(True, False, 0.01, 1), (True, True, 0.06, 2), (True, True, -0.06, 3),
     (False, True, 0.06, 4), (False, False, 0.0, 4)],
)
def test_the_decision_rules_select_one_scenario(
    h1: bool, h3: bool, mean: float, rule: int
) -> None:
    assert decision_scenario(outcome(h1, h3, mean))["rule"] == rule


def test_the_first_rule_carries_the_new_title() -> None:
    assert decision_scenario(outcome(True, False))["title"].startswith("Admissible Substitution")


# -- preregistration ---------------------------------------------------------------
def test_a_plan_without_v2_fields_keeps_its_v1_payload() -> None:
    hypothesis = Hypothesis("A", "q", "operational.fill_rate", "t", "c", "greater", 0.1)
    plan = Preregistration(campaign="x", hypotheses=(hypothesis,))
    assert set(plan.to_payload()) == {
        "campaign", "alpha", "minimum_replicates", "hypotheses", "exclusion_criteria", "notes",
    }


def test_v2_fields_enter_the_hash() -> None:
    hypothesis = Hypothesis("A", "q", "operational.fill_rate", "t", "c", "greater", 0.1)
    base = Preregistration(campaign="x", hypotheses=(hypothesis,))
    assert replace(base, replicates=500).plan_hash != base.plan_hash
    assert replace(base, base_seed=1).plan_hash != base.plan_hash
    assert replace(base, design={"a": 1}).plan_hash != base.plan_hash


def test_replicates_below_the_minimum_are_refused() -> None:
    hypothesis = Hypothesis("A", "q", "operational.fill_rate", "t", "c", "greater", 0.1)
    with pytest.raises(ValueError, match="below its own minimum"):
        Preregistration(campaign="x", hypotheses=(hypothesis,), replicates=10)


def test_the_repository_plan_matches_its_seal(plan: Preregistration, project_root: Path) -> None:
    """Editing the sealed plan turns this red: amend it with seal_v2.py --amend."""
    from xai_gov.analysis.preregistration import load_seal

    _, sealed_hash = load_seal(project_root / "specs/v2/confirmatory_v2.seal.json")
    assert sealed_hash == plan.plan_hash
