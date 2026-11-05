"""The v2 figure script renders from its inputs and writes a table view."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")


def _module(project_root: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "render_figures_v2", project_root / "scripts/render_figures_v2.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scientific_notation_uses_times_ten(project_root: Path) -> None:
    sci = _module(project_root).sci
    assert sci(7.18e52) == r"$7.2\times10^{52}$"
    assert "e+" not in sci(2.0e65) and "e" not in sci(2.0e65).replace("times", "")
    assert sci(240.0) == "240.00"


def test_the_power_figure_writes_a_600_dpi_png_and_csv(project_root: Path, tmp_path: Path) -> None:
    module = _module(project_root)
    module.style()
    power = json.loads((project_root / "specs/v2/power_v2.json").read_text(encoding="utf-8"))
    module.fig_power(power, tmp_path, 500)
    for suffix in ("png", "csv"):
        assert (tmp_path / f"fig_v2_power.{suffix}").stat().st_size > 0
    assert not (tmp_path / "fig_v2_power.pdf").exists()
    from PIL import Image

    dpi = Image.open(tmp_path / "fig_v2_power.png").info["dpi"]
    assert round(dpi[0]) == round(dpi[1]) == 600
    header = (tmp_path / "fig_v2_power.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == "hypothesis,n,power"
