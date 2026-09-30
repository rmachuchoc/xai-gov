"""Canonical serialization and content hashing.

Every hash in the project is computed over the *canonical* JSON form of a
payload: keys sorted, no insignificant whitespace, UTF-8, floats rendered
by ``repr``. Without this, a decision record that is semantically
identical would hash differently depending on dictionary insertion order,
and the append-only guarantee of the decision log would be worthless.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

HASH_NAME = "blake2b-256"
_DIGEST_SIZE = 32


def _default(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set | frozenset):
        return sorted(value)
    if hasattr(value, "to_payload"):
        return value.to_payload()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    raise TypeError(f"object of type {type(value).__name__} is not serializable")


class NonSerializableValueError(ValueError):
    """A payload carries a value that cannot be sealed into the log."""


def _locate_non_finite(payload: Any, trail: str = "") -> str | None:
    """Return the path of the first NaN or infinity found, if any."""
    if isinstance(payload, float):
        if payload != payload or payload in (float("inf"), float("-inf")):
            return trail or "<root>"
        return None
    if isinstance(payload, dict):
        for key, value in payload.items():
            found = _locate_non_finite(value, f"{trail}.{key}" if trail else str(key))
            if found:
                return found
    elif isinstance(payload, list | tuple):
        for index, value in enumerate(payload):
            found = _locate_non_finite(value, f"{trail}[{index}]")
            if found:
                return found
    return None


def canonical_json(payload: Any) -> str:
    """Return the canonical JSON text used for hashing and for the log.

    NaN and infinity are refused rather than emitted. They are not valid
    JSON, so a log containing them could not be re-read and verified; and a
    sentinel infinity in a threshold field would silently satisfy every
    comparison made against it. The error names the exact field, so the
    sentinel can be removed where it was introduced.
    """
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=_default,
        )
    except ValueError as error:
        location = _locate_non_finite(payload)
        if location is None:
            raise
        raise NonSerializableValueError(
            f"non-finite value at {location!r} cannot be sealed into the log; "
            "record an absent quantity as null, never as a sentinel"
        ) from error


def hash_bytes(data: bytes) -> str:
    """Hash of a raw byte string."""
    return hashlib.blake2b(data, digest_size=_DIGEST_SIZE).hexdigest()


def content_hash(payload: Any) -> str:
    """Hash of the canonical JSON form of ``payload``."""
    return hash_bytes(canonical_json(payload).encode("utf-8"))


def hash_file(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root: Path, patterns: Iterable[str] = ("**/*.py",)) -> str:
    """Hash a source tree, so a run records the code that produced it.

    Files are visited in sorted order and both path and content enter the
    digest; a renamed file therefore changes the tree hash.
    """
    digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file() or "__pycache__" in path.parts or path in seen:
                continue
            seen.add(path)
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(hash_file(path).encode("ascii"))
    return digest.hexdigest()
