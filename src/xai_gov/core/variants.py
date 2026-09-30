"""Building a dataclass from a configuration mapping.

Every registry in the project resolves a name to a class and then hands it a
mapping of parameters. Two things must happen at that boundary, and getting
either wrong has already cost this project a defect:

A parameter the chosen variant does not accept must be *dropped*, not refused.
Variants are switched by including an override file, and the loader
deep-merges, so a ``split`` calibrator legitimately arrives carrying ``decay``
from the ``weighted`` block it replaced. That leftover is an artifact of
composition, not a mistake by whoever wrote the file.

A parameter no variant recognizes must be *refused by name*. Silently dropping
a typo would build a different object than the one configured and report
success, which is the worst possible outcome: the run completes, the artifacts
look valid, and the experiment measured something nobody asked for.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any


def accepted_fields(cls: type) -> set[str]:
    """Init parameters of a dataclass."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    return {field.name for field in fields(cls) if field.init}


def build_variant[T](
    *,
    kind: str,
    name: str,
    params: Any,
    variants: dict[str, type[T]],
) -> T:
    """Resolve ``name`` in ``variants`` and construct it from ``params``.

    ``kind`` names the thing being built, for error messages a reader can act
    on ("calibrator", "level controller").
    """
    if not isinstance(params, dict):
        raise ValueError(f"{kind} 'params' must be a mapping")

    builder = variants.get(name)
    if builder is None:
        raise ValueError(
            f"unknown {kind} {name!r}; available: {', '.join(sorted(variants))}"
        )

    known_anywhere: set[str] = set()
    for candidate in variants.values():
        known_anywhere |= accepted_fields(candidate)
    unknown = set(params) - known_anywhere
    if unknown:
        raise ValueError(
            f"unknown {kind} parameters {sorted(unknown)}; "
            f"recognized: {', '.join(sorted(known_anywhere))}"
        )

    accepted = accepted_fields(builder)
    try:
        return builder(**{key: value for key, value in params.items() if key in accepted})
    except TypeError as error:
        raise ValueError(f"{kind} {name!r} rejected its parameters: {error}") from error
