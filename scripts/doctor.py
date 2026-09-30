#!/usr/bin/env python3
"""Environment audit.

Bit-for-bit reproducibility given the seed is a contract of this platform,
and the contract is only as good as the interpreter it runs on. This script
reports every condition that could make a run on this machine differ from a
run elsewhere. It never modifies anything.

    make doctor

Findings are ranked: ERROR breaks a stated contract, WARN can change
results silently, INFO is context worth recording next to a result.
"""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

ERROR, WARN, INFO, OK = "ERROR", "WARN ", "INFO ", "ok   "
_findings: list[tuple[str, str]] = []


def report(level: str, message: str) -> None:
    _findings.append((level, message))


def check_interpreter() -> None:
    report(INFO, f"python {sys.version.split()[0]} at {sys.executable}")
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        report(OK, "running inside a virtual environment")
    else:
        report(ERROR, "not running inside a virtual environment; activate .venv first")
        return
    expected = ROOT / ".venv"
    if expected.is_dir() and Path(sys.prefix).resolve() != expected.resolve():
        report(WARN, f"active venv is {sys.prefix}, not {expected}")


def check_foreign_paths() -> None:
    """Anything on sys.path outside the venv and the project is a risk."""
    venv = Path(sys.prefix).resolve()
    stdlib = Path(sys.base_prefix).resolve()
    foreign: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry).resolve()
        if path == ROOT or ROOT in path.parents:
            continue
        if venv in path.parents or path == venv:
            continue
        if stdlib in path.parents or path == stdlib:
            continue
        foreign.append(str(path))
    if foreign:
        report(WARN, "foreign entries on sys.path (may shadow project packages):")
        for entry in sorted(set(foreign)):
            report(WARN, f"    {entry}")
    else:
        report(OK, "sys.path contains only the venv, the stdlib and this project")

    pythonpath = os.environ.get("PYTHONPATH", "")
    if pythonpath:
        report(WARN, f"PYTHONPATH is set: {pythonpath}")
        if "/opt/ros" in pythonpath:
            report(
                WARN,
                "    ROS is on PYTHONPATH. It registers a pytest plugin that "
                "breaks collection; `make test` disables plugin autoload to "
                "stay hermetic. Consider not sourcing ROS in this shell.",
            )
    else:
        report(OK, "PYTHONPATH is unset")

    if site.ENABLE_USER_SITE:
        report(WARN, "user site-packages is enabled (PYTHONNOUSERSITE not set)")


def check_pytest_plugins() -> None:
    if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD"):
        report(OK, "pytest plugin autoload is disabled")
        return
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - python < 3.10 only
        return
    external = sorted(
        ep.module.split(".")[0]
        for ep in entry_points(group="pytest11")
        if not ep.module.startswith(("_pytest", "pytest_cov"))
    )
    if external:
        report(WARN, f"third-party pytest plugins visible: {', '.join(sorted(set(external)))}")
        report(WARN, "    run tests through `make test` so autoload is disabled")
    else:
        report(OK, "no unexpected pytest plugins visible")


def check_dependencies() -> None:
    from importlib.metadata import PackageNotFoundError, version

    for package in ("PyYAML", "numpy", "pandas", "pytest", "ruff", "mypy"):
        try:
            report(INFO, f"{package} {version(package)}")
        except PackageNotFoundError:
            report(ERROR, f"{package} is not installed; run `make install`")


def check_shadowing() -> None:
    """Where the numeric stack actually resolves from.

    PYTHONPATH precedes the venv on sys.path, so a system package can
    shadow the pinned one and change results without changing any version
    string. This reports the file that would actually be imported.
    """
    import importlib.util

    venv = Path(sys.prefix).resolve()
    for module in ("numpy", "pandas", "yaml"):
        spec = importlib.util.find_spec(module)
        origin = spec.origin if spec and spec.origin else None
        if origin is None:
            report(ERROR, f"{module} cannot be located")
            continue
        path = Path(origin).resolve()
        if venv in path.parents:
            report(OK, f"{module} resolves inside the venv")
        else:
            report(ERROR, f"{module} resolves OUTSIDE the venv: {path}")


def check_project() -> None:
    from xai_gov import __version__
    from xai_gov.core.paths import get_paths

    paths = get_paths()
    report(INFO, f"xai-gov {__version__}")
    report(OK if paths.root == ROOT else WARN, f"project root resolved to {paths.root}")
    if os.environ.get("XAI_GOV_ROOT"):
        report(INFO, f"XAI_GOV_ROOT is set to {os.environ['XAI_GOV_ROOT']}")
    missing = [str(d) for d in (paths.configs, paths.configs / "app") if not d.is_dir()]
    for entry in missing:
        report(ERROR, f"missing expected directory: {entry}")
    if not missing:
        report(OK, "configuration tree present")


def check_determinism() -> None:
    """The seeding contract must hold on this interpreter."""
    from xai_gov.core.seeds import SeedBundle, derive_seed

    if derive_seed(42, "demand") != derive_seed(42, "demand"):
        report(ERROR, "seed derivation is not deterministic on this interpreter")
        return
    first = SeedBundle(master_seed=7)
    a = first.generator("demand").random()
    second = SeedBundle(master_seed=7)
    second.generator("lead_time").random()
    if second.generator("demand").random() != a:
        report(ERROR, "component streams are not independent of construction order")
    else:
        report(OK, "seeding contract holds")

    if os.environ.get("PYTHONHASHSEED") in (None, ""):
        report(
            INFO,
            "PYTHONHASHSEED is unset. Harmless here (no hash-ordered iteration "
            "reaches a result), but set it to 0 if you ever compare log bytes.",
        )


def main() -> int:
    for check in (
        check_interpreter,
        check_foreign_paths,
        check_pytest_plugins,
        check_dependencies,
        check_shadowing,
        check_project,
        check_determinism,
    ):
        try:
            check()
        except Exception as error:  # a failed check is itself a finding
            report(ERROR, f"{check.__name__} raised {type(error).__name__}: {error}")

    print(f"xai-gov environment audit  ({ROOT})\n")
    for level, message in _findings:
        print(f"  {level}  {message}")

    errors = sum(1 for level, _ in _findings if level == ERROR)
    warnings = sum(1 for level, _ in _findings if level == WARN)
    print(f"\n  {errors} error(s), {warnings} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
