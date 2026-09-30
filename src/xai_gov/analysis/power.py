"""Statistical power, computed rather than asserted.

The protocol commits to n >= 30 replicates per cell for power 0.80 at a medium
effect size. That number is a convention until someone checks it against the
design actually being run, and this module does the checking.

Two things make the calculation here different from the textbook one.

The comparisons are **paired**: arms share seeds, so the relevant variance is
the variance of the *differences*, not of the arms. Paired designs need far
fewer replicates for the same power, and using the unpaired formula would
overstate the requirement — leading to a campaign that burns compute proving
something the design had already secured.

The test is **anytime-valid**, which costs power relative to a fixed-n test at
the same alpha. That cost is real and is not hidden: `sequential_penalty`
inflates the required sample accordingly. A power calculation that quoted the
t-test number while the analysis used e-values would promise a sensitivity the
campaign does not have.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

#: Normal quantiles, so the module needs no SciPy at import time.
_Z = {0.80: 0.8416, 0.90: 1.2816, 0.95: 1.6449, 0.975: 1.9600, 0.99: 2.3263}


def _z(p: float) -> float:
    """Standard normal quantile, interpolated over the tabulated points."""
    keys = sorted(_Z)
    if p <= keys[0]:
        return _Z[keys[0]]
    if p >= keys[-1]:
        return _Z[keys[-1]]
    for low, high in pairwise(keys):
        if low <= p <= high:
            weight = (p - low) / (high - low)
            return _Z[low] + weight * (_Z[high] - _Z[low])
    return _Z[keys[-1]]


#: How much larger a sample an anytime-valid test needs for the same power.
#: The sequential test pays for the right to stop at any time; roughly a
#: 25 percent inflation at conventional alpha, which is the accepted figure for
#: mixture-martingale tests against a fixed-n z-test.
SEQUENTIAL_PENALTY = 1.25


@dataclass(frozen=True, slots=True)
class PowerAnalysis:
    """What a campaign can and cannot detect."""

    effect_size: float
    alpha: float
    power: float
    required_paired: int
    required_sequential: int
    planned: int
    adequate: bool
    detectable_effect: float

    def to_payload(self) -> dict[str, Any]:
        return {
            "effect_size": round(self.effect_size, 4),
            "alpha": self.alpha,
            "target_power": self.power,
            "required_paired_fixed_n": self.required_paired,
            "required_anytime_valid": self.required_sequential,
            "planned_replicates": self.planned,
            "adequate": self.adequate,
            "minimum_detectable_effect": round(self.detectable_effect, 4),
            "note": (
                "paired design: the variance is that of the differences, not of "
                "the arms; the sequential requirement includes the price of "
                "being allowed to stop at any time"
            ),
        }


def required_replicates(
    *, effect_size: float = 0.5, alpha: float = 0.05, power: float = 0.80
) -> int:
    """Replicates needed for a paired comparison at a fixed sample size."""
    if effect_size <= 0.0:
        raise ValueError("effect_size must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must lie in (0, 1)")
    z_alpha = _z(1.0 - alpha / 2.0)
    z_beta = _z(power)
    return max(math.ceil(((z_alpha + z_beta) / effect_size) ** 2), 2)


def analyse_power(
    *,
    planned: int,
    effect_size: float = 0.5,
    alpha: float = 0.05,
    power: float = 0.80,
) -> PowerAnalysis:
    """Check a planned replicate count against what the design needs."""
    fixed = required_replicates(effect_size=effect_size, alpha=alpha, power=power)
    sequential = math.ceil(fixed * SEQUENTIAL_PENALTY)

    # What the planned sample can actually detect, which is the number a
    # reader needs when a hypothesis comes back unsupported.
    z_alpha = _z(1.0 - alpha / 2.0)
    z_beta = _z(power)
    detectable = (z_alpha + z_beta) / math.sqrt(max(planned, 1) / SEQUENTIAL_PENALTY)

    return PowerAnalysis(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        required_paired=fixed,
        required_sequential=sequential,
        planned=planned,
        adequate=planned >= sequential,
        detectable_effect=detectable,
    )
