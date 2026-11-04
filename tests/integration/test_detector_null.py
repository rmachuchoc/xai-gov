"""The null line end to end: stationary arms, false alarms with an interval."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from xai_gov.analysis.detector_null import NullLinePlan, run_null_line
from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings
from xai_gov.io.yaml_loader import load_config

pytestmark = pytest.mark.integration


def test_the_null_line_runs_and_reports_every_arm(project_root: Path, tmp_path: Path) -> None:
    settings = load_settings(Paths(root=project_root))
    (tmp_path / "outputs" / "runs").mkdir(parents=True)
    settings = replace(settings, outputs_root=tmp_path / "outputs")
    config = load_config(project_root / "configs/analysis/detector_null.yaml", root=project_root)
    config["replicates"] = 2
    plan = NullLinePlan.from_config(config, root=project_root)

    summary = run_null_line(plan, settings=settings, project_root=project_root)
    assert set(summary["arms"]) == {"adaptive", "reference_scale"}
    for entry in summary["arms"].values():
        assert entry["usable"] == 2
        alarm = entry["false_alarms_demand_facing"]
        assert 0.0 <= alarm["ci_low"] <= alarm["ci_high"] <= 1.0
        assert entry["coverage_error"]["n"] == 2
    assert "no Ville bound" in summary["claim"]


def test_the_null_arms_have_stationary_demand_and_no_disruptions(
    project_root: Path,
) -> None:
    for name in ("null_adaptive", "null_reference_scale"):
        config = load_config(project_root / f"configs/experiments/{name}.yaml", root=project_root)
        assert config["demand"]["regime"] == "stable"
        assert config["disruptions"] == []
        assert config["experiment"]["periods"] == 60
