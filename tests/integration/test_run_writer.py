"""A run must not be able to claim validity without a verified chain."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.test_decision_record import make_record
from xai_gov.io.hashchain import ChainVerificationError, verify_file
from xai_gov.io.writers import (
    CONFIG_SNAPSHOT_NAME,
    DECISION_LOG_NAME,
    EVENTS_NAME,
    MANIFEST_NAME,
    RunWriter,
    read_json,
)

pytestmark = pytest.mark.integration


def open_writer(runs_root: Path, **kwargs: object) -> RunWriter:
    return RunWriter(
        runs_root=runs_root,
        experiment="pilot",
        config={"experiment": {"name": "pilot"}, "sim": {"periods": 3}},
        master_seed=42,
        **kwargs,  # type: ignore[arg-type]
    )


def test_run_produces_the_expected_artifacts(tmp_runs_root: Path) -> None:
    with open_writer(tmp_runs_root) as writer:
        for period in range(3):
            writer.record_decision(make_record(period=period))
        writer.write_summary({"periods": 3, "service_level": 1.0})
        writer.write_kpis({"operational": {"service_level": 1.0}})
        directory = writer.directory

    for name in (MANIFEST_NAME, CONFIG_SNAPSHOT_NAME, DECISION_LOG_NAME, EVENTS_NAME):
        assert (directory / name).is_file()

    manifest = read_json(directory / MANIFEST_NAME)
    assert manifest["decisions"] == 3
    assert manifest["chain_verified"] is True
    assert manifest["status"] == "completed"
    assert verify_file(directory / DECISION_LOG_NAME) == manifest["chain_entries"]


def test_events_csv_has_one_row_per_decision(tmp_runs_root: Path) -> None:
    with open_writer(tmp_runs_root) as writer:
        for period in range(4):
            writer.record_decision(make_record(period=period))
        directory = writer.directory
    lines = (directory / EVENTS_NAME).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5  # header + 4


def test_failure_aborts_without_a_completed_manifest(tmp_runs_root: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"), open_writer(tmp_runs_root) as writer:
        writer.record_decision(make_record())
        directory = writer.directory
        raise RuntimeError("boom")
    manifest = read_json(directory / MANIFEST_NAME)
    assert manifest["status"].startswith("aborted")
    assert manifest["chain_verified"] is False


def test_writing_after_close_is_refused(tmp_runs_root: Path) -> None:
    writer = open_writer(tmp_runs_root)
    writer.close()
    with pytest.raises(RuntimeError, match="already closed"):
        writer.record_decision(make_record())


def test_tampering_with_a_closed_run_is_detectable(tmp_runs_root: Path) -> None:
    with open_writer(tmp_runs_root) as writer:
        for period in range(3):
            writer.record_decision(make_record(period=period))
        log_path = writer.directory / DECISION_LOG_NAME

    lines = log_path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[2])
    entry["payload"]["final"]["quantity"] = 0.0
    lines[2] = json.dumps(entry)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ChainVerificationError):
        verify_file(log_path)


def test_manifest_records_the_configuration_hash(tmp_runs_root: Path) -> None:
    from xai_gov.core.hashing import content_hash

    config = {"experiment": {"name": "pilot"}, "sim": {"periods": 3}}
    with open_writer(tmp_runs_root) as writer:
        directory = writer.directory
    assert read_json(directory / MANIFEST_NAME)["config_hash"] == content_hash(config)
