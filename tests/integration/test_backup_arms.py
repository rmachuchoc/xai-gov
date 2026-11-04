"""The F-backup arms: two per topology, fixed policy and Q-learning."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings
from xai_gov.io.writers import KPIS_NAME, SUMMARY_NAME, read_json
from xai_gov.orchestration.runner import plan_experiment, run_experiment

pytestmark = pytest.mark.integration

BACKUP_ARMS = ("high_vol_backup_shield", "q_backup", "five_backup_shield", "five_q_backup")


@pytest.fixture
def settings(project_root: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    base = load_settings(Paths(root=project_root))
    outputs = tmp_path / "outputs"
    (outputs / "runs").mkdir(parents=True)
    return replace(base, outputs_root=outputs)


@pytest.mark.parametrize("arm", BACKUP_ARMS)
def test_a_backup_arm_runs_with_an_admissible_replacement(  # type: ignore[no-untyped-def]
    settings, project_root: Path, arm: str
) -> None:
    path = project_root / "configs/experiments" / f"{arm}.yaml"
    manifest = run_experiment(path, settings=settings)
    assert manifest["status"] == "completed"

    directory = settings.runs_root / manifest["run_name"]
    shield = read_json(directory / SUMMARY_NAME)["governance"]["shield"]
    assert shield["mode"] == "backup"
    assert shield["fallback_inadmissible"] == 0
    beta = read_json(directory / KPIS_NAME)["operational"]["fill_rate"]
    assert 0.0 <= beta <= 1.0


@pytest.mark.parametrize(
    ("backup", "projection"),
    [("high_vol_backup_shield", "high_vol_shielded"), ("q_backup", "q_projection"),
     ("five_backup_shield", "five_shielded")],
)
def test_a_backup_arm_differs_from_its_projection_pair_only_in_the_operator(  # type: ignore[no-untyped-def]
    settings, project_root: Path, backup: str, projection: str
) -> None:
    experiments = project_root / "configs/experiments"
    left = plan_experiment(experiments / f"{backup}.yaml", settings=settings).config
    right = plan_experiment(experiments / f"{projection}.yaml", settings=settings).config
    assert left["verification"]["shield_mode"] == "backup"
    assert right["verification"].get("shield_mode", "projection") == "projection"
    assert set(left) == set(right)
    for section in set(left) - {"verification", "experiment"}:
        assert left[section] == right[section], section
    assert left["experiment"]["periods"] == right["experiment"]["periods"]
    assert left["experiment"]["master_seed"] == right["experiment"]["master_seed"]
    strip = {k: v for k, v in left["verification"].items() if k != "shield_mode"}
    assert strip == {k: v for k, v in right["verification"].items() if k != "shield_mode"}
