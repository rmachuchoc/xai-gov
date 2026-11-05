"""The v2 figure script renders from its inputs and writes a table view."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")


def test_the_power_figure_writes_png_pdf_and_csv(project_root: Path, tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "render_figures_v2", project_root / "scripts/render_figures_v2.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.style()
    power = json.loads((project_root / "specs/v2/power_v2.json").read_text(encoding="utf-8"))
    module.fig_power(power, tmp_path, 500)
    for suffix in ("png", "pdf", "csv"):
        assert (tmp_path / f"fig_v2_power.{suffix}").stat().st_size > 0
    header = (tmp_path / "fig_v2_power.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "hypothesis,n,power"
