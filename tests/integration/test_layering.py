"""Layering: the package must stay acyclic, in every import order.

A circular import is not a stylistic problem here. The cycle that existed
(engine → governance → policies → simulation → engine) encoded a real
architectural error: the simulation reaching into the authorities that are
supposed to constrain it. The protocol separates proposal, mediation and
enforcement precisely so that dependency runs one way, and these tests fail
if that direction is ever reversed.

Each case runs in a subprocess with a cold interpreter, because an import
cycle only manifests on the *first* import of the chain — inside an already
warm process every module is cached and the bug hides.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

# Each of these was, or could become, an entry point into the cycle.
ENTRY_POINTS = [
    "xai_gov.simulation.network",
    "xai_gov.simulation.engine",
    "xai_gov.simulation",
    "xai_gov.policies",
    "xai_gov.policies.base",
    "xai_gov.governance",
    "xai_gov.governance.agent",
    "xai_gov.kpis",
    "xai_gov.orchestration",
    "xai_gov.orchestration.runner",
    "xai_gov.cli.main",
]


def import_in_fresh_interpreter(
    statement: str, project_root: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
        cwd=project_root,
        env={
            "PATH": "/usr/bin:/bin",
            "PYTHONPATH": str(project_root / "src"),
            "PYTHONNOUSERSITE": "1",
        },
        check=False,
        timeout=60,
    )


@pytest.mark.parametrize("module", ENTRY_POINTS)
def test_every_module_imports_first_in_a_cold_interpreter(module: str, project_root: Path) -> None:
    result = import_in_fresh_interpreter(f"import {module}", project_root)
    assert result.returncode == 0, (
        f"importing {module} first fails:\n{result.stderr}"
    )
    assert "circular" not in result.stderr.lower()


def test_the_engine_does_not_depend_on_governance_or_policies_at_runtime(
    project_root: Path,
) -> None:
    """The inversion that broke the cycle must stay inverted.

    The twin receives its policy and governance agent as constructor
    arguments. If either package appears in sys.modules after importing the
    engine alone, the dependency has crept back in.
    """
    statement = (
        "import sys; import xai_gov.simulation.engine; "
        "leaked = [m for m in sys.modules "
        "if m.startswith(('xai_gov.governance', 'xai_gov.policies'))]; "
        "print(','.join(sorted(leaked)))"
    )
    result = import_in_fresh_interpreter(statement, project_root)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", (
        f"the engine now imports higher layers at runtime: {result.stdout.strip()}"
    )


def test_governance_does_not_import_the_simulation_package_at_runtime(
    project_root: Path,
) -> None:
    statement = (
        "import sys; import xai_gov.governance.agent; "
        "leaked = [m for m in sys.modules if m.startswith('xai_gov.simulation')]; "
        "print(','.join(sorted(leaked)))"
    )
    result = import_in_fresh_interpreter(statement, project_root)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


def test_lazy_reexports_still_resolve() -> None:
    """The public API survived the fix."""
    import xai_gov.simulation as sim

    assert sim.DigitalTwin.__name__ == "DigitalTwin"
    assert sim.SimulationResult.__name__ == "SimulationResult"
    assert "DigitalTwin" in dir(sim)
    with pytest.raises(AttributeError, match="has no attribute"):
        _ = sim.NoSuchThing  # type: ignore[attr-defined]
