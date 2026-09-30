"""Run artifacts.

`RunWriter` owns everything a single run leaves on disk and guarantees the
invariant the protocol requires: no decision is executed without a record,
and the record is sealed into a verifiable chain. The directory layout of a
run is fixed, because analysis scripts and the benchmark depend on it:

    outputs/runs/<YYYYmmdd_HHMMSS>_<experiment>/
        manifest.json         run identity, code hash, chain head
        config_snapshot.json  the fully resolved configuration
        decisions.jsonl       the hash-chained decision log
        events.csv            flat projection for analysis
        summary.json          end-of-run aggregates
        kpis.json             the six indicator layers

`close()` verifies the chain before writing the manifest, so a run whose
log does not verify never produces a manifest and cannot be mistaken for a
valid result.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xai_gov.core.hashing import canonical_json, content_hash, hash_tree
from xai_gov.core.logging import get_logger
from xai_gov.io.decision_record import DecisionRecord
from xai_gov.io.hashchain import HashChain

_LOG = get_logger("io.writers")

MANIFEST_NAME = "manifest.json"
CONFIG_SNAPSHOT_NAME = "config_snapshot.json"
DECISION_LOG_NAME = "decisions.jsonl"
EVENTS_NAME = "events.csv"
SUMMARY_NAME = "summary.json"
KPIS_NAME = "kpis.json"


def run_name(experiment: str, *, now: datetime | None = None) -> str:
    """Build the canonical run directory name."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in experiment)
    return f"{stamp}_{safe}"


def allocate_run_directory(runs_root: Path, experiment: str, *, limit: int = 999) -> Path:
    """Create and return a fresh run directory, never reusing one.

    Second-resolution timestamps collide: a campaign executes many cells per
    second, and two runs sharing a directory would interleave their decision
    logs into one unverifiable file. Rather than widen the timestamp — which
    would make run names unreadable — a numeric suffix is appended on
    collision. Creation is the claim: ``exist_ok=False`` makes the check and
    the claim a single atomic operation, so concurrent workers cannot both
    win the same name.
    """
    runs_root.mkdir(parents=True, exist_ok=True)
    base = run_name(experiment)
    for attempt in range(limit + 1):
        candidate = runs_root / (base if attempt == 0 else f"{base}_{attempt + 1:02d}")
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return candidate
    raise RuntimeError(
        f"could not allocate a run directory under {runs_root} after {limit} attempts"
    )


class RunWriter:
    """Create and populate one run directory."""

    def __init__(
        self,
        *,
        runs_root: Path,
        experiment: str,
        config: dict[str, Any],
        master_seed: int,
        source_root: Path | None = None,
        write_events_csv: bool = True,
        write_decision_log: bool = True,
    ) -> None:
        self.experiment = experiment
        self.directory = allocate_run_directory(runs_root, experiment)
        self.name = self.directory.name

        self._config = config
        self._master_seed = int(master_seed)
        self._source_root = source_root
        self._write_events = write_events_csv
        self._started_at = datetime.now(UTC)
        self._records: list[DecisionRecord] = []
        self._closed = False

        chain_path = self.directory / DECISION_LOG_NAME if write_decision_log else None
        self._chain = HashChain(chain_path)
        self._chain.append(
            "run_opened",
            {
                "run_name": self.name,
                "experiment": experiment,
                "master_seed": self._master_seed,
                "config_hash": content_hash(config),
                "started_at": self._started_at.isoformat(timespec="seconds"),
            },
        )
        self._write_json(CONFIG_SNAPSHOT_NAME, config)
        _LOG.info("run opened", extra={"run": self.name, "experiment": experiment})

    # -- recording -------------------------------------------------------
    def record_decision(self, record: DecisionRecord) -> None:
        """Seal one decision into the chain. Must precede its execution."""
        self._require_open()
        self._records.append(record)
        self._chain.append("decision", record.to_payload())

    def record_event(self, kind: str, payload: dict[str, Any]) -> None:
        """Seal a non-decision event (disruption, calibration, monitor hit)."""
        self._require_open()
        self._chain.append(kind, payload)

    # -- artifacts -------------------------------------------------------
    def write_summary(self, summary: dict[str, Any]) -> None:
        self._require_open()
        self._write_json(SUMMARY_NAME, summary)
        self._chain.append("summary", {"summary_hash": content_hash(summary)})

    def write_kpis(self, kpis: dict[str, Any]) -> None:
        self._require_open()
        self._write_json(KPIS_NAME, kpis)
        self._chain.append("kpis", {"kpis_hash": content_hash(kpis)})

    def write_events_csv(self) -> Path | None:
        if not self._write_events or not self._records:
            return None
        path = self.directory / EVENTS_NAME
        fieldnames = list(self._records[0].to_flat_row())
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in self._records:
                writer.writerow(record.to_flat_row())
        return path

    # -- lifecycle -------------------------------------------------------
    def close(self, *, status: str = "completed", verify: bool = True) -> dict[str, Any]:
        """Finish the run: verify the chain, then write the manifest."""
        self._require_open()
        self.write_events_csv()
        self._chain.append(
            "run_closed",
            {"status": status, "decisions": len(self._records),
             "ended_at": datetime.now(UTC).isoformat(timespec="seconds")},
        )
        if verify:
            self._chain.verify()

        manifest = {
            "run_name": self.name,
            "experiment": self.experiment,
            "status": status,
            "started_at": self._started_at.isoformat(timespec="seconds"),
            "ended_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "master_seed": self._master_seed,
            "decisions": len(self._records),
            "chain_entries": len(self._chain),
            "chain_head": self._chain.head,
            "chain_verified": verify,
            "config_hash": content_hash(self._config),
            "code_hash": hash_tree(self._source_root) if self._source_root else None,
            "artifacts": sorted(p.name for p in self.directory.iterdir()),
        }
        self._write_json(MANIFEST_NAME, manifest)
        self._chain.close()
        self._closed = True
        _LOG.info(
            "run closed",
            extra={"run": self.name, "status": status, "chain_head": self._chain.head[:12]},
        )
        return manifest

    def abort(self, reason: str) -> dict[str, Any]:
        """Close a failed run without claiming validity."""
        return self.close(status=f"aborted: {reason}", verify=False)

    def __enter__(self) -> RunWriter:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 tb: object) -> None:
        if self._closed:
            return
        if exc_type is None:
            self.close()
        else:
            self.abort(f"{exc_type.__name__}: {exc}")

    # -- internals -------------------------------------------------------
    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"run {self.name} is already closed")

    def _write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        path = self.directory / filename
        path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        return path


def read_json(path: Path) -> dict[str, Any]:
    """Read a run artifact, asserting the mapping shape the callers assume."""
    with path.open("r", encoding="utf-8") as handle:
        loaded: object = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a JSON object at {path}, got {type(loaded).__name__}")
    return loaded
