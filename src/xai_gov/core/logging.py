"""Logging.

Two sinks with different jobs: a terse console stream for the operator and
a JSON-lines file for the record. The JSONL sink is what later analysis
reads, so its schema is stable: ``ts``, ``level``, ``logger``, ``msg`` plus
whatever structured fields the call site passed through ``extra``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_CONFIGURED = False
# The attribute names a LogRecord always carries. Derived from a probe
# record rather than hard-coded, so a future Python release that adds an
# attribute does not leak it into every log line.
_RESERVED: frozenset[str] = frozenset(
    vars(logging.LogRecord("", 0, "", 0, "", None, None))
) | {"message", "asctime", "taskName"}


class JsonLinesFormatter(logging.Formatter):
    """Render each record as one canonical JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


class ConsoleFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s  %(levelname)-7s %(name)-28s %(message)s",
                         datefmt="%H:%M:%S")


def configure_logging(
    *,
    level: str = "INFO",
    console: bool = True,
    jsonl: bool = True,
    log_dir: Path | None = None,
    filename: str = "xai_gov.log",
    force: bool = False,
) -> logging.Logger:
    """Configure the root logger once per process."""
    global _CONFIGURED
    root = logging.getLogger("xai_gov")
    if _CONFIGURED and not force:
        return root

    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.propagate = False

    if console:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(ConsoleFormatter())
        root.addHandler(stream)

    if jsonl and log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f"{Path(filename).stem}.jsonl",
                                           encoding="utf-8")
        file_handler.setFormatter(JsonLinesFormatter())
        root.addHandler(file_handler)

    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced child logger."""
    return logging.getLogger(f"xai_gov.{name}" if not name.startswith("xai_gov") else name)
