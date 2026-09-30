"""Application settings.

`Settings` is the immutable, validated view of ``configs/app/*.yaml`` that
the rest of the code reads. Anything a module needs to know about the
environment arrives through this object; no module reads YAML on its own.

Defaults live in module-level constants rather than as class attributes
read back through ``cls``: these dataclasses use ``slots=True``, so there
is no class-level descriptor to read a default from at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xai_gov.core.paths import Paths, get_paths
from xai_gov.io.yaml_loader import load_config

_APP_FILES = ("paths.yaml", "logging.yaml", "runtime.yaml", "tracking.yaml")

# -- defaults -------------------------------------------------------------
LOG_LEVEL = "INFO"
LOG_CONSOLE = True
LOG_JSONL = True
LOG_FILENAME = "xai_gov.log"

RUN_STRICT = True
RUN_FAIL_ON_UNVERIFIED_LOG = True
RUN_MAX_WORKERS = 1
RUN_FLOAT_PRECISION = 10
RUN_WRITE_EVENTS_CSV = True
RUN_WRITE_DECISION_LOG = True

TRACK_ENABLED = False
TRACK_BACKEND = "none"
TRACK_EXPERIMENT_NAME = "xai-gov"


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    level: str = LOG_LEVEL
    console: bool = LOG_CONSOLE
    jsonl: bool = LOG_JSONL
    filename: str = LOG_FILENAME

    @classmethod
    def from_config(cls, node: dict[str, Any]) -> LoggingSettings:
        return cls(
            level=str(node.get("level", LOG_LEVEL)).upper(),
            console=bool(node.get("console", LOG_CONSOLE)),
            jsonl=bool(node.get("jsonl", LOG_JSONL)),
            filename=str(node.get("filename", LOG_FILENAME)),
        )


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    """Execution policy for a run: strictness, determinism, budgets."""

    strict: bool = RUN_STRICT
    fail_on_unverified_log: bool = RUN_FAIL_ON_UNVERIFIED_LOG
    max_workers: int = RUN_MAX_WORKERS
    float_precision: int = RUN_FLOAT_PRECISION
    write_events_csv: bool = RUN_WRITE_EVENTS_CSV
    write_decision_log: bool = RUN_WRITE_DECISION_LOG

    @classmethod
    def from_config(cls, node: dict[str, Any]) -> RuntimeSettings:
        return cls(
            strict=bool(node.get("strict", RUN_STRICT)),
            fail_on_unverified_log=bool(
                node.get("fail_on_unverified_log", RUN_FAIL_ON_UNVERIFIED_LOG)
            ),
            max_workers=int(node.get("max_workers", RUN_MAX_WORKERS)),
            float_precision=int(node.get("float_precision", RUN_FLOAT_PRECISION)),
            write_events_csv=bool(node.get("write_events_csv", RUN_WRITE_EVENTS_CSV)),
            write_decision_log=bool(node.get("write_decision_log", RUN_WRITE_DECISION_LOG)),
        )


@dataclass(frozen=True, slots=True)
class TrackingSettings:
    """Experiment tracking. Disabled until the MLflow stage is installed."""

    enabled: bool = TRACK_ENABLED
    backend: str = TRACK_BACKEND
    experiment_name: str = TRACK_EXPERIMENT_NAME
    tracking_uri: str | None = None

    @classmethod
    def from_config(cls, node: dict[str, Any]) -> TrackingSettings:
        uri = node.get("tracking_uri")
        return cls(
            enabled=bool(node.get("enabled", TRACK_ENABLED)),
            backend=str(node.get("backend", TRACK_BACKEND)),
            experiment_name=str(node.get("experiment_name", TRACK_EXPERIMENT_NAME)),
            tracking_uri=None if uri is None else str(uri),
        )


@dataclass(frozen=True, slots=True)
class Settings:
    paths: Paths
    outputs_root: Path
    logging: LoggingSettings
    runtime: RuntimeSettings
    tracking: TrackingSettings
    raw: dict[str, Any]

    @property
    def runs_root(self) -> Path:
        return self.outputs_root / "runs"

    @property
    def campaigns_root(self) -> Path:
        return self.outputs_root / "campaigns"


def _section(merged: dict[str, Any], key: str) -> dict[str, Any]:
    """Return one configuration section, tolerating its absence."""
    node = merged.get(key, {})
    return node if isinstance(node, dict) else {}


def load_settings(paths: Paths | None = None) -> Settings:
    """Load and validate the application configuration."""
    resolved = paths or get_paths()
    merged: dict[str, Any] = {}
    for name in _APP_FILES:
        candidate = resolved.config("app", name)
        if candidate.is_file():
            merged |= load_config(candidate, root=resolved.root)

    declared_outputs = _section(merged, "paths").get("outputs", "outputs")
    outputs_root = resolved.resolve(str(declared_outputs))

    settings = Settings(
        paths=resolved,
        outputs_root=outputs_root,
        logging=LoggingSettings.from_config(_section(merged, "logging")),
        runtime=RuntimeSettings.from_config(_section(merged, "runtime")),
        tracking=TrackingSettings.from_config(_section(merged, "tracking")),
        raw=merged,
    )
    resolved.ensure(settings.runs_root, settings.campaigns_root, resolved.logs)
    return settings
