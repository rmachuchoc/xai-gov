"""Deterministic seeding.

Reproducibility bit-for-bit given the seed is a contract of the platform,
not a convenience. A single master seed is declared per run; every
component derives its own stream from it by hashing the master seed
together with a stable component label. Two consequences matter:

* adding a component never shifts the random stream of existing ones, so
  results stay comparable across code versions;
* the same component in the same run always draws the same numbers,
  regardless of the order in which components are constructed.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

import numpy as np

_DERIVE_BITS = 64
_MASK = (1 << _DERIVE_BITS) - 1


def derive_seed(master_seed: int, label: str) -> int:
    """Derive a stable child seed from ``master_seed`` and ``label``."""
    material = f"{int(master_seed)}::{label}".encode()
    digest = hashlib.blake2b(material, digest_size=_DERIVE_BITS // 8).digest()
    return int.from_bytes(digest, "big") & _MASK


@dataclass(slots=True)
class SeedBundle:
    """Owner of every random stream in a run."""

    master_seed: int
    _generators: dict[str, np.random.Generator] = field(default_factory=dict, repr=False)

    def generator(self, label: str) -> np.random.Generator:
        """Return the NumPy generator for ``label``, creating it on first use."""
        if label not in self._generators:
            self._generators[label] = np.random.default_rng(derive_seed(self.master_seed, label))
        return self._generators[label]

    def python_random(self, label: str) -> random.Random:
        """Return a stdlib `random.Random` for components that need it."""
        return random.Random(derive_seed(self.master_seed, label))

    def seed_for(self, label: str) -> int:
        return derive_seed(self.master_seed, label)

    def fingerprint(self) -> str:
        """Short digest of the master seed, recorded in the run manifest."""
        return hashlib.blake2b(str(self.master_seed).encode(), digest_size=8).hexdigest()

    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self._generators))


def seed_global_libraries(master_seed: int) -> None:
    """Seed process-global RNGs.

    Component code must use `SeedBundle` instead. This exists only for
    third-party libraries that read global state and offer no explicit
    generator argument; it is called once per run and logged.
    """
    random.seed(derive_seed(master_seed, "global::python"))
    np.random.seed(derive_seed(master_seed, "global::numpy") % (2**32))
