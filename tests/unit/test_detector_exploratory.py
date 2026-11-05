"""The warm-up share of the calibration gap (R1.4, the v1 'one seventh' claim)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module(project_root: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "run_detector_exploratory_v2", project_root / "scripts/run_detector_exploratory_v2.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_share_is_the_relative_reduction_of_the_contaminated_gap(project_root: Path) -> None:
    share = _module(project_root).warmup_share([0.07] * 10, [0.06] * 10, resamples=50)
    assert share["share"] == pytest.approx(1 / 7, abs=1e-6)
    low, high = share["ci95"]
    assert low <= share["share"] <= high


def test_the_detector_campaign_runs_on_the_confirmatory_seed_and_horizon(
    project_root: Path,
) -> None:
    import yaml

    config = yaml.safe_load(
        (project_root / "configs/campaigns/detector_exploratory_v2.yaml").read_text()
    )
    assert (config["replicates"], config["base_seed"]) == (500, 20261107)
    assert {h["id"] for h in config["hypotheses"]} == {"RQ6", "RQ6b", "RQ6c", "RQ6d"}
