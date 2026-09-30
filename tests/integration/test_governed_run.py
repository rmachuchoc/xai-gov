"""The governed arm, end to end.

These tests exercise the composition the protocol is about: conformal signal
into belief, belief against a derived threshold, intervention against a
budget. The comparisons are all paired — same network, same demand, same seed,
same policy — because an unpaired comparison of governance arms measures the
seed as much as the governance.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings
from xai_gov.io.hashchain import verify_file
from xai_gov.io.writers import DECISION_LOG_NAME, KPIS_NAME, SUMMARY_NAME, read_json
from xai_gov.orchestration.runner import run_experiment

pytestmark = pytest.mark.integration

EXPERIMENTS = Path("configs/experiments")


@pytest.fixture
def settings(project_root: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    base = load_settings(Paths(root=project_root))
    outputs = tmp_path / "outputs"
    (outputs / "runs").mkdir(parents=True)
    return replace(base, outputs_root=outputs)


def run(name: str, settings, project_root: Path) -> tuple[dict, dict]:  # type: ignore[no-untyped-def, type-arg]
    manifest = run_experiment(project_root / EXPERIMENTS / name, settings=settings)
    directory = settings.runs_root / manifest["run_name"]
    return manifest, {
        "kpis": read_json(directory / KPIS_NAME),
        "summary": read_json(directory / SUMMARY_NAME),
        "log": directory / DECISION_LOG_NAME,
    }


def test_the_governed_arm_runs_and_stays_verifiable(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    manifest, artifacts = run("high_vol_governed_h2.yaml", settings, project_root)
    assert manifest["status"] == "completed"
    assert manifest["chain_verified"] is True
    assert verify_file(artifacts["log"]) == manifest["chain_entries"]
    assert artifacts["summary"]["governance"]["id"] == "cpomdp"


def test_the_threshold_is_derived_and_recorded(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """The central change of this stage: no threshold is read from a file, and
    the derivation travels into the artifact so a reviewer can check it."""
    _, artifacts = run("high_vol_governed_h2.yaml", settings, project_root)
    threshold = artifacts["summary"]["governance"]["threshold"]
    assert threshold["method"] == "value_iteration_on_collapsed_belief"
    assert threshold["converged"] is True
    assert 0.0 < threshold["threshold"] < 1.0
    # Accounting for continuation value acts no later than the myopic rule.
    assert threshold["anticipation_gain"] >= -1e-9
    assert threshold["economics"]["disruption_loss"] > 0.0


def test_the_guarantees_layer_is_available_and_unsaturated(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """The pilot's failure mode, checked directly: an OOD rate near 1.0 is a
    saturated detector, not a perpetually anomalous environment."""
    _, artifacts = run("high_vol_governed_h2.yaml", settings, project_root)
    layer = artifacts["kpis"]["guarantees"]
    assert layer["status"] == "available"
    assert layer["scheme"] == "aci"
    # Both saturation directions are failures. A detector that flags almost
    # everything is the pilot's defect; one that never flags is just as
    # useless, and the KPI layer refuses to call either healthy.
    assert 0.0 < layer["ood_rate"] < 0.5, f"detector saturated: {layer['ood_rate']}"
    assert layer["saturated"] is False
    assert abs(layer["coverage_error"]) < 0.35


def test_governance_intervenes_and_charges_its_budget(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    _, artifacts = run("high_vol_governed_h2.yaml", settings, project_root)
    governance = artifacts["kpis"]["governance"]
    budget = artifacts["summary"]["governance"]["budget"]
    assert governance["traceability"] == 1.0
    if governance["intervention_rate"] > 0.0:
        assert budget["interventions"] > 0
        assert budget["spent"] > 0.0
        assert governance["total_intervention_cost"] > 0.0


def test_the_governed_and_ungoverned_arms_are_a_valid_pair(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """Same seed, same network, same demand, same policy: only governance
    differs, which is what makes RGD and TVC computable from the pair."""
    _, governed = run("high_vol_governed_h2.yaml", settings, project_root)
    _, ungoverned = run("high_vol_ungoverned.yaml", settings, project_root)

    for key in ("network", "demand", "policy"):
        assert governed["summary"][key] == ungoverned["summary"][key]
    assert governed["summary"]["master_seed"] == ungoverned["summary"]["master_seed"]

    assert ungoverned["kpis"]["governance"]["intervention_rate"] == 0.0
    assert ungoverned["kpis"]["guarantees"]["status"] == "not_available"


def test_governance_never_increases_the_executed_quantity(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """The agent may refuse, never amplify. Repair requires the verified
    shield, which is stage 5."""
    from xai_gov.io.hashchain import read_chain

    _, artifacts = run("high_vol_governed_h2.yaml", settings, project_root)
    for entry in read_chain(artifacts["log"]):
        if entry.kind != "decision":
            continue
        assert entry.payload["final"]["quantity"] <= entry.payload["proposed"]["quantity"] + 1e-9


def test_a_tight_budget_produces_a_higher_shadow_price(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """kappa* is near zero when oversight is abundant and positive when it
    binds: that is what makes it a price rather than a tuning constant."""
    _, tight = run("tight_budget_governed.yaml", settings, project_root)
    _, ample = run("high_vol_governed_h2.yaml", settings, project_root)
    tight_budget = tight["summary"]["governance"]["budget"]
    ample_budget = ample["summary"]["governance"]["budget"]
    assert tight_budget["allowance"] < ample_budget["allowance"]
    assert tight_budget["shadow_price"] >= ample_budget["shadow_price"]
    # And the scarce arm turned oversight demand away, which is what the price
    # is measuring.
    if tight_budget["refusals"] > 0:
        assert tight_budget["shadow_price"] > 0.0


def test_the_detector_ablation_isolates_adaptivity(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """Identical governance, different conformal scheme. Whatever separates the
    two arms is attributable to adaptivity alone."""
    _, adaptive = run("high_vol_governed_h2.yaml", settings, project_root)
    _, classical = run("high_vol_governed_split.yaml", settings, project_root)

    assert adaptive["kpis"]["guarantees"]["scheme"] == "aci"
    assert classical["kpis"]["guarantees"]["scheme"] == "fixed"
    assert adaptive["summary"]["governance"]["threshold"]["threshold"] == pytest.approx(
        classical["summary"]["governance"]["threshold"]["threshold"]
    )


def test_the_change_martingale_reports_a_detection_delay(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    _, artifacts = run("structural_change_governed.yaml", settings, project_root)
    layer = artifacts["kpis"]["guarantees"]
    assert layer["first_shock_period"] == 15
    # None is a finding (never detected), not a missing value.
    assert layer["detection_delay"] is None or layer["detection_delay"] >= 0


def test_every_shipped_experiment_still_runs(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    for path in sorted((project_root / EXPERIMENTS).glob("*.yaml")):
        manifest = run_experiment(path, settings=settings)
        assert manifest["status"] == "completed", path.name


def test_a_planned_architecture_names_its_stage(settings, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "federated.yaml"
    path.write_text(
        "_include_:\n"
        "  - configs/network/single_echelon.yaml\n"
        "  - configs/scenarios/demand_stable.yaml\n"
        "  - configs/policies/heuristic.yaml\n"
        "governance:\n  architecture: federated\n"
        "experiment:\n  name: federated\n  periods: 5\n  master_seed: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(NotImplementedError, match="stage 6"):
        run_experiment(path, settings=settings)
