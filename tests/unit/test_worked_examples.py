"""The worked examples are computed by the package and stay consistent with it."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _module(project_root: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "worked_examples_v2", project_root / "scripts/worked_examples_v2.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_example_renders(project_root: Path) -> None:
    module = _module(project_root)
    text = "\n".join(
        line
        for part in (module.serial_example, module.envelope_example, module.agrapa_example,
                     module.threshold_example, module.coverage_example)
        for line in part()
    )
    assert "beta over the six periods = 11 / 12" in text
    assert "Envelope: [5, 120]" in text
    assert "b* = 0.06" in text
