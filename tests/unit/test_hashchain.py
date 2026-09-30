"""The append-only guarantee: tampering must be detectable."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

from xai_gov.io.hashchain import (
    GENESIS_HASH,
    ChainVerificationError,
    HashChain,
    read_chain,
    verify_chain,
    verify_file,
)


def build_chain(n: int = 5, path: Path | None = None) -> HashChain:
    chain = HashChain(path)
    for i in range(n):
        chain.append("decision", {"period": i, "action": "reorder", "quantity": 10 + i})
    return chain


def test_first_entry_links_to_genesis() -> None:
    chain = build_chain(1)
    assert chain.entries[0].prev_hash == GENESIS_HASH


def test_entries_are_linked_and_verify() -> None:
    chain = build_chain(6)
    entries = chain.entries
    for previous, current in pairwise(entries):
        assert current.prev_hash == previous.entry_hash
    chain.verify()


def test_head_changes_with_every_append() -> None:
    chain = HashChain()
    heads = set()
    for i in range(4):
        chain.append("decision", {"period": i})
        heads.add(chain.head)
    assert len(heads) == 4


def test_payload_tampering_is_detected() -> None:
    chain = build_chain(4)
    entries = list(chain.entries)
    tampered = entries[2]
    tampered.payload["quantity"] = 9999  # mutate in place: same object identity
    with pytest.raises(ChainVerificationError, match="payload altered at index 2"):
        verify_chain(entries)


def test_deleting_an_entry_is_detected() -> None:
    entries = list(build_chain(5).entries)
    del entries[2]
    with pytest.raises(ChainVerificationError):
        verify_chain(entries)


def test_reordering_is_detected() -> None:
    entries = list(build_chain(5).entries)
    entries[1], entries[3] = entries[3], entries[1]
    with pytest.raises(ChainVerificationError):
        verify_chain(entries)


def test_roundtrip_through_disk_verifies(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    chain = build_chain(7, path)
    chain.close()
    assert verify_file(path) == 7
    assert len(read_chain(path)) == 7


def test_edited_file_fails_verification(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    build_chain(4, path).close()

    lines = path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[1])
    entry["payload"]["quantity"] = 1
    lines[1] = json.dumps(entry)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ChainVerificationError):
        verify_file(path)


def test_truncated_file_still_verifies_as_a_prefix(tmp_path: Path) -> None:
    """A crashed run leaves a valid prefix, not a corrupt log."""
    path = tmp_path / "decisions.jsonl"
    build_chain(6, path).close()
    lines = path.read_text(encoding="utf-8").splitlines()[:3]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert verify_file(path) == 3
