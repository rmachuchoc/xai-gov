"""Filesystem layout resolution.

The project root is resolved once, in this order:

1. the ``XAI_GOV_ROOT`` environment variable, when set;
2. the first ancestor of this file that contains ``pyproject.toml``
   (the normal case for an editable install);
3. the current working directory, as a last resort.

Nothing else in the codebase is allowed to build paths by string
concatenation: every directory used at runtime comes from `Paths`, so a
run can be relocated by changing one environment variable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_MARKER = "pyproject.toml"


def _discover_root() -> Path:
    env = os.environ.get("XAI_GOV_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / _MARKER).is_file():
            return parent
    return Path.cwd().resolve()


@dataclass(frozen=True, slots=True)
class Paths:
    """Canonical directories of the project."""

    root: Path

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def data_raw(self) -> Path:
        return self.data / "raw"

    @property
    def data_interim(self) -> Path:
        return self.data / "interim"

    @property
    def data_processed(self) -> Path:
        return self.data / "processed"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def runs(self) -> Path:
        return self.outputs / "runs"

    @property
    def campaigns(self) -> Path:
        return self.outputs / "campaigns"

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def specs(self) -> Path:
        """PCTL / PRISM specifications and synthesized shields."""
        return self.root / "specs"

    def config(self, *parts: str) -> Path:
        return self.configs.joinpath(*parts)

    def resolve(self, candidate: str | Path) -> Path:
        """Resolve a config-declared path relative to the project root.

        Absolute paths are returned untouched, so a campaign can point at
        a scratch volume without editing the code.
        """
        path = Path(candidate).expanduser()
        return path if path.is_absolute() else (self.root / path).resolve()

    def ensure(self, *dirs: Path) -> None:
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def ensure_runtime_tree(self) -> None:
        self.ensure(self.runs, self.campaigns, self.figures, self.logs)


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    """Return the process-wide `Paths` instance."""
    return Paths(root=_discover_root())
