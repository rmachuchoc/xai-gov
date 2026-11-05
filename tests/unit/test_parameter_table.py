"""The parameter table describes the predicates the code evaluates."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tests.factories import state
from xai_gov.verification.specification import PREDICATES


def test_the_documented_bad_states_match_the_predicates(project_root: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "parameter_table_v2", project_root / "scripts/parameter_table_v2.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(module.PREDICATES) == set(PREDICATES)
    boundaries = {
        "stockout": ("inventory", 0.0, 0.01),
        "backlog_critical": ("backlog", 50.0, 49.99),
        "inventory_critical": ("inventory", 5.0, 5.01),
        "capacity_exhausted": ("capacity", 0.0, 0.01),
    }
    for name, (field, bad, good) in boundaries.items():
        assert field in module.PREDICATES[name]
        assert PREDICATES[name](state(**{field: bad})), name
        assert not PREDICATES[name](state(**{field: good})), name
