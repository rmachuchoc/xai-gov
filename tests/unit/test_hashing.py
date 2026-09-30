"""Canonical serialization must be order-independent and total."""

from __future__ import annotations

import math

import pytest

from xai_gov.core.hashing import canonical_json, content_hash


def test_key_order_does_not_change_the_hash() -> None:
    a = {"alpha": 1, "beta": {"x": 1, "y": 2}}
    b = {"beta": {"y": 2, "x": 1}, "alpha": 1}
    assert content_hash(a) == content_hash(b)


def test_value_change_changes_the_hash() -> None:
    assert content_hash({"n": 1}) != content_hash({"n": 1.0000001})


def test_nan_is_refused_rather_than_silently_serialized() -> None:
    with pytest.raises(ValueError):
        canonical_json({"n": math.nan})


def test_enum_and_path_are_serializable() -> None:
    from pathlib import Path

    from xai_gov.io.decision_record import ActionKind

    payload = canonical_json({"a": ActionKind.REORDER, "p": Path("/tmp/x")})
    assert '"reorder"' in payload
    assert "/tmp/x" in payload
