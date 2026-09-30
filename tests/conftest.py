from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # editable install not required for tests
    sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def project_root() -> Path:
    return ROOT


@pytest.fixture
def tmp_runs_root(tmp_path: Path) -> Path:
    runs = tmp_path / "outputs" / "runs"
    runs.mkdir(parents=True)
    return runs
