"""The confirmatory runner refuses anything but the sealed plan."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _main(project_root: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "run_confirmatory_v2", project_root / "scripts/run_confirmatory_v2.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def test_without_a_seal_the_campaign_does_not_run(project_root: Path, tmp_path: Path) -> None:
    main = _main(project_root)
    code = main(["--seal", str(tmp_path / "absent.seal.json"),
                 "--outputs-root", str(tmp_path / "out")])
    assert code == 2
    assert not any((tmp_path / "out" / "runs").iterdir())


def test_a_seal_for_another_plan_is_refused(project_root: Path, tmp_path: Path) -> None:
    seal = tmp_path / "other.seal.json"
    seal.write_text(json.dumps({"plan": {}, "plan_hash": "0" * 64}))
    main = _main(project_root)
    code = main(["--seal", str(seal), "--outputs-root", str(tmp_path / "out")])
    assert code == 2
    assert not any((tmp_path / "out" / "runs").iterdir())
