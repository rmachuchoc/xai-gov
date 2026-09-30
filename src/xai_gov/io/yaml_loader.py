"""Configuration loading.

YAML is the only configuration language, and configuration never reaches
the code as a bare dictionary that any module may mutate. Files can
include others through the ``_include_`` key, which is how a factorial
experiment composes scenario, resources, policy, governance and conformal
fragments without duplicating them.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

INCLUDE_KEY = "_include_"
_MAX_DEPTH = 8


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing, empty or malformed."""


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a single YAML file into a dictionary."""
    if not path.is_file():
        raise ConfigError(f"configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        raise ConfigError(f"configuration file is empty: {path}")
    if not isinstance(loaded, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    return loaded


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` onto ``base`` without mutating either."""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path, *, root: Path | None = None, _depth: int = 0) -> dict[str, Any]:
    """Load a YAML file, resolving ``_include_`` directives recursively.

    Includes are resolved relative to the project root when ``root`` is
    given, otherwise relative to the including file. Later includes win
    over earlier ones, and keys declared in the file itself win over all
    includes — the same precedence a reader expects from the page order.
    """
    if _depth > _MAX_DEPTH:
        raise ConfigError(f"include depth exceeded at {path}; check for a cycle")

    raw = read_yaml(path)
    includes = raw.pop(INCLUDE_KEY, [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list):
        raise ConfigError(f"{INCLUDE_KEY} must be a string or a list in {path}")

    merged: dict[str, Any] = {}
    for entry in includes:
        base = root if root is not None else path.parent
        included = Path(entry)
        target = included if included.is_absolute() else base / included
        merged = deep_merge(merged, load_config(target, root=root, _depth=_depth + 1))

    return deep_merge(merged, raw)


def require(config: Mapping[str, Any], *keys: str) -> Any:
    """Fetch a nested key, failing loudly with the full path on absence."""
    node: Any = config
    for index, key in enumerate(keys):
        if not isinstance(node, Mapping) or key not in node:
            trail = ".".join(keys[: index + 1])
            raise ConfigError(f"missing required configuration key: {trail}")
        node = node[key]
    return node
