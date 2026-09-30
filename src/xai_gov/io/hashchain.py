"""Append-only hash-chained log.

This module implements the traceability contract of the protocol: each
entry carries the hash of its own payload and the hash of the previous
entry, so the log is verifiable as a chain. Any retroactive edit,
deletion or reordering breaks verification at the first affected index,
and `verify_chain` reports exactly where.

The chain is written as JSON Lines: one canonical JSON object per line,
appended and flushed as it is produced, so a crashed run still leaves a
verifiable prefix rather than a corrupt file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from xai_gov.core.hashing import HASH_NAME, canonical_json, content_hash

GENESIS_HASH = "0" * 64
CHAIN_VERSION = 1


@dataclass(frozen=True, slots=True)
class ChainEntry:
    """One verifiable link of the log."""

    index: int
    ts: str
    kind: str
    payload: dict[str, Any]
    payload_hash: str
    prev_hash: str
    entry_hash: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "ts": self.ts,
            "kind": self.kind,
            "payload": self.payload,
            "payload_hash": self.payload_hash,
            "prev_hash": self.prev_hash,
            "entry_hash": self.entry_hash,
        }

    @classmethod
    def from_payload(cls, data: dict[str, Any]) -> ChainEntry:
        return cls(
            index=int(data["index"]),
            ts=str(data["ts"]),
            kind=str(data["kind"]),
            payload=dict(data["payload"]),
            payload_hash=str(data["payload_hash"]),
            prev_hash=str(data["prev_hash"]),
            entry_hash=str(data["entry_hash"]),
        )


def seal(index: int, ts: str, kind: str, payload_hash: str, prev_hash: str) -> str:
    """Compute the entry hash that binds one link to its predecessor."""
    return content_hash(
        {
            "chain_version": CHAIN_VERSION,
            "hash": HASH_NAME,
            "index": index,
            "ts": ts,
            "kind": kind,
            "payload_hash": payload_hash,
            "prev_hash": prev_hash,
        }
    )


class HashChain:
    """Append-only chain, optionally persisted to a JSONL file.

    Use as a context manager so the file handle is closed even when a run
    fails; the entries written before the failure remain verifiable.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._handle: TextIO | None = None
        self._entries: list[ChainEntry] = []
        self._last_hash = GENESIS_HASH
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("w", encoding="utf-8")

    # -- construction ----------------------------------------------------
    def append(self, kind: str, payload: dict[str, Any]) -> ChainEntry:
        """Seal ``payload`` into the chain and return the new entry."""
        index = len(self._entries)
        ts = datetime.now(UTC).isoformat(timespec="microseconds")
        payload_hash = content_hash(payload)
        entry = ChainEntry(
            index=index,
            ts=ts,
            kind=kind,
            payload=payload,
            payload_hash=payload_hash,
            prev_hash=self._last_hash,
            entry_hash=seal(index, ts, kind, payload_hash, self._last_hash),
        )
        self._entries.append(entry)
        self._last_hash = entry.entry_hash
        if self._handle is not None:
            self._handle.write(canonical_json(entry.to_payload()) + "\n")
            self._handle.flush()
        return entry

    # -- inspection ------------------------------------------------------
    @property
    def entries(self) -> tuple[ChainEntry, ...]:
        return tuple(self._entries)

    @property
    def head(self) -> str:
        """Hash of the last entry: the fingerprint of the whole run."""
        return self._last_hash

    def __len__(self) -> int:
        return len(self._entries)

    def verify(self) -> None:
        verify_chain(self._entries)

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __enter__(self) -> HashChain:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class ChainVerificationError(RuntimeError):
    """Raised when a chain fails verification, naming the broken index."""


def verify_chain(entries: list[ChainEntry] | tuple[ChainEntry, ...]) -> None:
    """Verify indices, payload hashes and the linkage of every entry."""
    expected_prev = GENESIS_HASH
    for position, entry in enumerate(entries):
        if entry.index != position:
            raise ChainVerificationError(
                f"index mismatch at position {position}: entry declares {entry.index}"
            )
        if entry.prev_hash != expected_prev:
            raise ChainVerificationError(
                f"broken link at index {entry.index}: "
                f"prev_hash {entry.prev_hash[:12]}… expected {expected_prev[:12]}…"
            )
        if content_hash(entry.payload) != entry.payload_hash:
            raise ChainVerificationError(f"payload altered at index {entry.index}")
        recomputed = seal(entry.index, entry.ts, entry.kind, entry.payload_hash, entry.prev_hash)
        if recomputed != entry.entry_hash:
            raise ChainVerificationError(f"entry hash altered at index {entry.index}")
        expected_prev = entry.entry_hash


def read_chain(path: Path) -> list[ChainEntry]:
    """Read a JSONL chain from disk without verifying it."""
    import json

    entries: list[ChainEntry] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entries.append(ChainEntry.from_payload(json.loads(stripped)))
            except (KeyError, ValueError) as error:
                raise ChainVerificationError(f"malformed entry at line {line_number}") from error
    return entries


def verify_file(path: Path) -> int:
    """Verify a chain on disk; return the number of entries checked."""
    entries = read_chain(path)
    verify_chain(entries)
    return len(entries)
