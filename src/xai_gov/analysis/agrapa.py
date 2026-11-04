"""The predictable aGRAPA e-process (Waudby-Smith and Ramdas, 2024).

A test of H0: E[d] ≤ 0 for paired differences d bounded by a declared range R,
as a betting game. The wealth after t replicates is

    K_t = Π_{i ≤ t} (1 + λ_i x_i),    x_i = d_i / R ∈ [−1, 1],

and under H0 it is a nonnegative supermartingale for any bet λ_i ≥ 0 that is
fixed *before* x_i is seen. Ville's inequality then gives P(sup_t K_t ≥ c) ≤ 1/c,
so rejecting at the first t with K_t ≥ c is valid however long one watches.

The bet is aGRAPA, the approximate growth-rate-adaptive plug-in:

    λ_t = clip( μ̂_{t−1} / (σ̂²_{t−1} + μ̂²_{t−1}), 0, 1/2 ),

where μ̂ and σ̂² are computed from x_1 … x_{t−1} only, with a conservative prior
(mean 0, the null boundary, and variance 1, the largest a variable on [−1, 1]
can have), each worth one observation:

    μ̂_t = (0 + Σ_{i ≤ t} x_i) / (t + 1),
    σ̂²_t = (1 + Σ_{i ≤ t} (x_i − μ̂_i)²) / (t + 1).

Three commitments make the test what the plan says it is:

* **predictable**: λ_t uses past differences only, so no observation sets its
  own bet; this is checked by a test;
* **bounded bets**: λ ∈ [0, 1/2], so each factor is at least 1/2 and one bad
  replicate cannot ruin the wealth;
* **fixed scale**: R is the declared range of the indicator (1 for a fill rate,
  whose differences lie in [−1, 1]), never estimated from the data. A
  difference outside it means the declaration was wrong, and is refused rather
  than clipped.

A two-sided test of H0: E[d] = 0 averages the wealth of a bet on d and a bet
on −d: (K⁺_t + K⁻_t)/2 is again a nonnegative supermartingale under H0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

#: The bet is confined to [0, BET_CAP].
BET_CAP = 0.5
_LOG_CAP = 700.0
_TOLERANCE = 1e-9


@dataclass(slots=True)
class AgrapaEProcess:
    """One-sided aGRAPA wealth process for H0: E[d] ≤ 0."""

    scale: float = 1.0
    cap: float = BET_CAP
    prior_mean: float = 0.0
    prior_variance: float = 1.0
    observations: int = 0
    log_wealth: float = 0.0
    peak_log_wealth: float = 0.0
    first_crossing: int | None = None
    _sum: float = 0.0
    _sum_sq_dev: float = 0.0
    bets: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("scale must be positive: it is the declared range of |d|")
        if not 0.0 < self.cap < 1.0:
            raise ValueError("the bet cap must lie in (0, 1) so every factor stays positive")
        if self.prior_variance <= 0.0:
            raise ValueError("prior_variance must be positive")

    # -- the predictable plug-ins ---------------------------------------------------
    @property
    def mean_estimate(self) -> float:
        return (self.prior_mean + self._sum) / (self.observations + 1)

    @property
    def variance_estimate(self) -> float:
        return (self.prior_variance + self._sum_sq_dev) / (self.observations + 1)

    def next_bet(self) -> float:
        """λ for the next replicate, from the past only."""
        mean = self.mean_estimate
        raw = mean / (self.variance_estimate + mean * mean)
        return min(max(raw, 0.0), self.cap)

    # -- one replicate --------------------------------------------------------------
    def normalize(self, difference: float) -> float:
        x = difference / self.scale
        if abs(x) > 1.0 + _TOLERANCE:
            raise ValueError(
                f"difference {difference} lies outside the declared range ±{self.scale}; "
                "the scale is part of the plan and is not adjusted to the data"
            )
        return max(-1.0, min(1.0, x))

    def update(self, difference: float, *, threshold: float | None = None) -> float:
        """Bet on one difference; returns the log wealth."""
        x = self.normalize(difference)
        bet = self.next_bet()
        self.bets.append(bet)
        self.log_wealth += math.log1p(bet * x)
        self.observations += 1
        self.peak_log_wealth = max(self.peak_log_wealth, self.log_wealth)
        if (
            threshold is not None
            and self.first_crossing is None
            and self.log_wealth >= math.log(threshold)
        ):
            self.first_crossing = self.observations
        # The plug-ins absorb x only after the bet on it was placed.
        self._sum += x
        self._sum_sq_dev += (x - self.mean_estimate) ** 2
        return self.log_wealth

    @property
    def wealth(self) -> float:
        return math.exp(min(self.log_wealth, _LOG_CAP))

    @property
    def peak_wealth(self) -> float:
        return math.exp(min(self.peak_log_wealth, _LOG_CAP))


@dataclass(frozen=True, slots=True)
class EProcessPath:
    """An e-process run over a sequence: its value at the end and its peak."""

    final: float
    peak: float
    first_crossing: int | None
    observations: int


def _log_mean_exp(a: float, b: float) -> float:
    top = max(a, b)
    return top + math.log((math.exp(a - top) + math.exp(b - top)) / 2.0)


def run_e_process(
    differences: Sequence[float],
    *,
    two_sided: bool = False,
    scale: float = 1.0,
    threshold: float | None = None,
) -> EProcessPath:
    """The one-sided wealth on ``differences``, or the two-sided average.

    ``first_crossing`` is the first replicate at which the process reached
    ``threshold``: the stopping rule the plan uses.
    """
    up = AgrapaEProcess(scale=scale)
    down = AgrapaEProcess(scale=scale) if two_sided else None
    log_threshold = math.log(threshold) if threshold is not None else None
    peak = 0.0
    crossing: int | None = None
    current = 0.0
    for index, difference in enumerate(differences, start=1):
        up.update(difference)
        if down is not None:
            down.update(-difference)
            current = _log_mean_exp(up.log_wealth, down.log_wealth)
        else:
            current = up.log_wealth
        peak = max(peak, current)
        if log_threshold is not None and crossing is None and current >= log_threshold:
            crossing = index
    return EProcessPath(
        final=math.exp(min(current, _LOG_CAP)),
        peak=math.exp(min(peak, _LOG_CAP)),
        first_crossing=crossing,
        observations=len(differences),
    )


def familywise_threshold(*, hypotheses: int, alpha: float) -> float:
    """2k/α: each of k hypotheses may be claimed in either direction."""
    if hypotheses < 1:
        raise ValueError("hypotheses must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return 2.0 * hypotheses / alpha


def describe() -> dict[str, Any]:
    """The procedure, for the sealed plan and the report."""
    return {
        "procedure": "predictable aGRAPA e-process (Waudby-Smith and Ramdas, 2024)",
        "bet": "lambda_t = clip(mu_{t-1} / (sigma2_{t-1} + mu_{t-1}^2), 0, 1/2)",
        "plug_ins": "past differences only; prior mean 0 and variance 1, weight 1",
        "scale": "declared range of the indicator difference, fixed in advance",
        "two_sided": "average of the bets on d and on -d",
        "decision": "reject at the first replicate whose e-value reaches 2k/alpha",
    }
