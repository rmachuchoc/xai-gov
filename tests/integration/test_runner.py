"""End to end: a shipped experiment must run and leave verifiable artifacts."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings
from xai_gov.io.hashchain import verify_file
from xai_gov.io.writers import DECISION_LOG_NAME, KPIS_NAME, SUMMARY_NAME, read_json
from xai_gov.orchestration.runner import plan_experiment, run_experiment

pytestmark = pytest.mark.integration


@pytest.fixture
def settings(project_root: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    """Real configs, throwaway outputs."""
    base = load_settings(Paths(root=project_root))
    outputs = tmp_path / "outputs"
    (outputs / "runs").mkdir(parents=True)
    return replace(base, outputs_root=outputs)


def test_pilot_experiment_runs_end_to_end(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    manifest = run_experiment(project_root / "configs/experiments/pilot.yaml", settings=settings)
    assert manifest["status"] == "completed"
    assert manifest["decisions"] == 30
    assert manifest["chain_verified"] is True

    directory = settings.runs_root / manifest["run_name"]
    assert verify_file(directory / DECISION_LOG_NAME) == manifest["chain_entries"]

    kpis = read_json(directory / KPIS_NAME)
    assert kpis["operational"]["status"] == "available"
    assert kpis["governance"]["intervention_rate"] == 0.0
    assert kpis["guarantees"]["status"] == "not_available"

    summary = read_json(directory / SUMMARY_NAME)
    assert summary["policy"]["id"] == "heuristic_sQ"
    assert summary["governance"]["id"] == "no_governance"


def test_three_echelon_experiment_runs(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    manifest = run_experiment(
        project_root / "configs/experiments/high_vol_ungoverned.yaml", settings=settings
    )
    assert manifest["status"] == "completed"
    assert manifest["decisions"] == 40 * 3


def test_same_seed_gives_the_same_chain_head(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """Reproducibility is observable from the artifacts alone.

    Chain entry hashes include timestamps, so the heads differ; the decision
    payloads must not.
    """
    path = project_root / "configs/experiments/pilot.yaml"
    first = run_experiment(path, settings=settings)
    second = run_experiment(path, settings=settings)
    assert first["config_hash"] == second["config_hash"]
    # Two runs started within the same second must not share a directory.
    assert first["run_name"] != second["run_name"]

    def decisions(name: str) -> list[dict[str, object]]:
        from xai_gov.io.hashchain import read_chain

        chain = read_chain(settings.runs_root / name / DECISION_LOG_NAME)
        return [entry.payload for entry in chain if entry.kind == "decision"]

    assert decisions(first["run_name"]) == decisions(second["run_name"])


def test_many_runs_in_the_same_second_get_their_own_directory(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """A campaign executes many cells per second; colliding directories would
    interleave two decision logs into one unverifiable file."""
    path = project_root / "configs/experiments/pilot.yaml"
    names = {run_experiment(path, settings=settings)["run_name"] for _ in range(5)}
    assert len(names) == 5
    for name in names:
        assert verify_file(settings.runs_root / name / DECISION_LOG_NAME) > 0


def test_structural_change_experiment_reports_shock_analysis(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    manifest = run_experiment(
        project_root / "configs/experiments/structural_change_dro.yaml", settings=settings
    )
    kpis = read_json(settings.runs_root / manifest["run_name"] / KPIS_NAME)
    risk = kpis["systemic_risk"]
    assert 15 in risk["shock_periods"]
    assert "fill_rate_post_shock" in risk


def test_experiment_without_a_seed_is_refused(settings, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "seedless.yaml"
    path.write_text(
        "experiment:\n  name: seedless\n  periods: 5\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="master_seed"):
        plan_experiment(path, settings=settings)


def test_the_governed_architecture_is_now_available(settings, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    """The stage-2 placement of this test expected 'cpomdp' to refuse. It now
    builds, so the test asserts the arm exists rather than that it is missing;
    the refusal path is still covered by the federated architecture."""
    path = tmp_path / "governed.yaml"
    path.write_text(
        "_include_:\n"
        "  - configs/network/single_echelon.yaml\n"
        "  - configs/scenarios/demand_stable.yaml\n"
        "  - configs/policies/heuristic.yaml\n"
        "  - configs/governance/cpomdp_h2.yaml\n"
        "experiment:\n  name: governed\n  periods: 20\n  master_seed: 1\n",
        encoding="utf-8",
    )
    manifest = run_experiment(path, settings=settings)
    assert manifest["status"] == "completed"


def test_every_shipped_experiment_plans(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    for path in sorted((project_root / "configs/experiments").glob("*.yaml")):
        plan = plan_experiment(path, settings=settings)
        assert plan.master_seed > 0
        assert plan.periods > 0
