"""Simulated power of the procedure the plan actually uses (decision 5).

A closed-form power formula describes a fixed-n z-test, not a predictable
aGRAPA e-process stopped at its first crossing of 2k/α. So the power is
simulated: paths of paired differences are generated with the dispersion the
pilot measured, shifted by the declared effect, and run through
`analysis.agrapa.run_e_process` itself. The power at n is the fraction of paths
that cross the threshold by replicate n, and n is the smallest replicate count
at which that fraction reaches the target.

Two generators are used, and the larger requirement wins:

* **bootstrap**: resample the pilot's differences, centred on zero, then add
  the effect. Keeps the pilot's shape, skew and tails;
* **normal**: a normal with the pilot's standard deviation, plus the effect.

A two-sided hypothesis is simulated with the effect in both directions, and
its power is the smaller of the two. The same simulation with no effect gives
the rejection rate under the null, which must stay below 1/threshold; it is
reported as a check on the whole chain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from xai_gov.analysis.agrapa import familywise_threshold, run_e_process

GENERATORS = ("bootstrap", "normal")


@dataclass(frozen=True, slots=True)
class PowerSpec:
    """What one hypothesis must detect."""

    hypothesis_id: str
    differences: tuple[float, ...]
    two_sided: bool
    effect: float = 0.05
    scale: float = 1.0

    def __post_init__(self) -> None:
        if len(self.differences) < 2:
            raise ValueError(f"{self.hypothesis_id}: the pilot needs at least two pairs")
        if self.effect <= 0.0:
            raise ValueError("the declared effect must be positive")


def _paths(
    spec: PowerSpec, *, shift: float, generator: str, sims: int, n_max: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pilot = np.asarray(spec.differences, dtype=float)
    if generator == "bootstrap":
        centred = pilot - pilot.mean()
        draws = rng.choice(centred, size=(sims, n_max), replace=True)
    elif generator == "normal":
        sd = float(pilot.std(ddof=1))
        draws = rng.normal(0.0, sd, size=(sims, n_max))
    else:
        raise ValueError(f"unknown generator {generator!r}; available: {GENERATORS}")
    # The declared range bounds every difference; the test refuses anything
    # beyond it, so the simulated world respects it too.
    return np.clip(draws + shift, -spec.scale, spec.scale)


def crossing_times(
    spec: PowerSpec, *, shift: float, generator: str, sims: int, n_max: int,
    threshold: float, seed: int,
) -> list[int | None]:
    """First crossing of each simulated path, or None if it never crossed."""
    rng = np.random.default_rng(seed)
    paths = _paths(spec, shift=shift, generator=generator, sims=sims, n_max=n_max, rng=rng)
    return [
        run_e_process(
            [float(v) for v in path], two_sided=spec.two_sided, scale=spec.scale,
            threshold=threshold,
        ).first_crossing
        for path in paths
    ]


def power_curve(crossings: list[int | None], n_values: list[int]) -> dict[int, float]:
    total = len(crossings)
    return {
        n: sum(1 for c in crossings if c is not None and c <= n) / total for n in n_values
    }


def required_n(crossings: list[int | None], *, target: float, n_max: int) -> int | None:
    """Smallest n whose power reaches ``target``; None if not within n_max."""
    times = sorted(c for c in crossings if c is not None)
    needed = math.ceil(target * len(crossings))
    if len(times) < needed or needed == 0:
        return None
    return int(times[needed - 1]) if times[needed - 1] <= n_max else None


def _max_or_none(values: list[int | None]) -> int | None:
    """The largest requirement, or None if any case was not met within n_max."""
    met = [v for v in values if v is not None]
    return None if len(met) < len(values) or not met else max(met)


def simulate_hypothesis(
    spec: PowerSpec, *, alpha: float = 0.05, hypotheses: int = 6, target: float = 0.80,
    sims: int = 2000, n_max: int = 1000, seed: int = 0,
) -> dict[str, Any]:
    """Power and required n for one hypothesis, under both generators."""
    threshold = familywise_threshold(hypotheses=hypotheses, alpha=alpha)
    shifts = (spec.effect, -spec.effect) if spec.two_sided else (spec.effect,)
    grid = sorted({*range(25, n_max + 1, 25), n_max})
    by_generator: dict[str, Any] = {}
    for g_index, generator in enumerate(GENERATORS):
        per_shift: list[dict[str, Any]] = []
        requirements: list[int | None] = []
        for s_index, shift in enumerate(shifts):
            crossings = crossing_times(
                spec, shift=shift, generator=generator, sims=sims, n_max=n_max,
                threshold=threshold, seed=seed + 100 * g_index + 10 * s_index + 1,
            )
            needed = required_n(crossings, target=target, n_max=n_max)
            requirements.append(needed)
            per_shift.append({
                "shift": shift,
                "required_n": needed,
                "power_at_n_max": power_curve(crossings, [n_max])[n_max],
                "curve": {str(n): round(p, 4) for n, p in power_curve(crossings, grid).items()},
            })
        null = crossing_times(
            spec, shift=0.0, generator=generator, sims=sims, n_max=n_max,
            threshold=threshold, seed=seed + 100 * g_index + 99,
        )
        by_generator[generator] = {
            "required_n": _max_or_none(requirements),
            "by_shift": per_shift,
            "null_rejection_rate": round(
                sum(1 for c in null if c is not None) / len(null), 6
            ),
        }
    overall = _max_or_none([entry["required_n"] for entry in by_generator.values()])
    return {
        "hypothesis": spec.hypothesis_id,
        "two_sided": spec.two_sided,
        "effect": spec.effect,
        "scale": spec.scale,
        "pilot_pairs": len(spec.differences),
        "pilot_sd": round(float(np.std(spec.differences, ddof=1)), 6),
        "threshold": threshold,
        "target_power": target,
        "sims": sims,
        "n_max": n_max,
        "required_n": overall,
        "generators": by_generator,
        "null_bound": round(1.0 / threshold, 6),
    }


def plan_replicates(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Arms share replicates, so the campaign needs the largest requirement."""
    requirements = {r["hypothesis"]: r["required_n"] for r in results}
    unmet = sorted(h for h, n in requirements.items() if n is None)
    return {
        "required_n_by_hypothesis": requirements,
        "binding_hypothesis": (
            None if unmet else max(requirements, key=lambda h: requirements[h] or 0)
        ),
        "replicates": None if unmet else max(n for n in requirements.values() if n),
        "unmet_within_n_max": unmet,
    }
