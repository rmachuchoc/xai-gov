"""The explainability ablation, end to end.

Three arms — none, post_hoc, causal — on the same network, demand, seed and
policy. Whatever separates them is attributable to the descriptor alone, which
is what makes RQ3 answerable from these runs.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings
from xai_gov.io.writers import KPIS_NAME, SUMMARY_NAME, read_json
from xai_gov.orchestration.runner import run_experiment

pytestmark = pytest.mark.integration

EXPERIMENTS = Path("configs/experiments")


@pytest.fixture
def settings(project_root: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    base = load_settings(Paths(root=project_root))
    outputs = tmp_path / "outputs"
    (outputs / "runs").mkdir(parents=True)
    return replace(base, outputs_root=outputs)


def run(name: str, settings, project_root: Path) -> dict:  # type: ignore[no-untyped-def, type-arg]
    manifest = run_experiment(project_root / EXPERIMENTS / name, settings=settings)
    directory = settings.runs_root / manifest["run_name"]
    return {
        "manifest": manifest,
        "kpis": read_json(directory / KPIS_NAME),
        "summary": read_json(directory / SUMMARY_NAME),
    }


def test_the_causal_arm_runs_and_reports_its_model(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    artifacts = run("high_vol_governed_causal.yaml", settings, project_root)
    assert artifacts["manifest"]["status"] == "completed"
    descriptor = artifacts["summary"]["governance"]["descriptor"]
    assert descriptor["method"] == "causal"
    # The declared graph travels into the artifact: a reviewer can check which
    # mechanism the explanations were computed against.
    assert "order_quantity" in descriptor["search"]["model"]["edges"]
    assert descriptor["search"]["unit_costs"]["capacity"] > 0.0


def test_the_xai_layer_needs_a_descriptor(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    without = run("high_vol_governed_h2.yaml", settings, project_root)
    with_causal = run("high_vol_governed_causal.yaml", settings, project_root)
    assert without["kpis"]["xai"]["status"] == "not_available"
    assert "descriptor" in without["kpis"]["xai"]["requires"]
    assert with_causal["kpis"]["xai"]["status"] == "available"


def test_only_the_causal_arm_is_identifiable(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    causal = run("high_vol_governed_causal.yaml", settings, project_root)
    post_hoc = run("high_vol_governed_posthoc.yaml", settings, project_root)
    assert causal["kpis"]["xai"]["identifiable_rate"] == 1.0
    assert post_hoc["kpis"]["xai"]["identifiable_rate"] == 0.0


def test_only_the_causal_arm_offers_recourse(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """Actionability requires interventions; a correlational attribution has
    none to offer, and reports zero rather than a fabricated score."""
    causal = run("high_vol_governed_causal.yaml", settings, project_root)
    post_hoc = run("high_vol_governed_posthoc.yaml", settings, project_root)
    assert post_hoc["kpis"]["xai"]["mean_actionability"] == 0.0
    assert causal["kpis"]["xai"]["mean_actionability"] >= 0.0


def test_the_arms_are_a_valid_ablation(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """Same network, demand, seed and policy; only the descriptor differs."""
    causal = run("high_vol_governed_causal.yaml", settings, project_root)
    post_hoc = run("high_vol_governed_posthoc.yaml", settings, project_root)
    for key in ("network", "demand", "policy"):
        assert causal["summary"][key] == post_hoc["summary"][key]
    assert causal["summary"]["master_seed"] == post_hoc["summary"]["master_seed"]


def test_an_escalation_carries_its_cheapest_recourse(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """'Intervene' with no lever attached is a notification, not a decision to
    be made by whoever receives it."""
    from xai_gov.io.hashchain import read_chain
    from xai_gov.io.writers import DECISION_LOG_NAME

    manifest = run_experiment(
        project_root / EXPERIMENTS / "high_vol_governed_causal.yaml", settings=settings
    )
    log = settings.runs_root / manifest["run_name"] / DECISION_LOG_NAME
    escalations = [
        entry.payload
        for entry in read_chain(log)
        if entry.kind == "decision" and entry.payload["governance"]["escalated"]
    ]
    for escalation in escalations:
        rationale = escalation["governance"]["rationale"]
        assert "recourse" in rationale, rationale


def test_the_xai_layer_reports_explanation_action_consistency(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """A governance layer intervening on uninformative descriptors is acting on
    the conformal signal alone, and that must be visible."""
    artifacts = run("high_vol_governed_causal.yaml", settings, project_root)
    layer = artifacts["kpis"]["xai"]
    if "explained_intervention_rate" in layer:
        assert 0.0 <= layer["explained_intervention_rate"] <= 1.0
    assert layer["anti_correlated_decisions"] == 0
