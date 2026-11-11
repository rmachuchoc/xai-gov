"""Anytime-valid inference.

Classical hypothesis testing fixes the sample size in advance. Look at the data
early and stop when the result pleases you, and the error rate you reported is
no longer the error rate you have — the practice has a name, optional stopping,
and it is one of the main sources of irreproducible findings.

That constraint is impossible to honour in simulation research. What limits a
campaign is compute time, which is not known in advance and does not respect a
sample size chosen months earlier. So the protocol uses e-values instead of
p-values, and the difference is precise: an e-value is a bet. Its expectation
under the null is at most 1, so by Ville's inequality

    P( sup_t E_t >= 1/alpha ) <= alpha

and the analyst may look at any time, stop at any time, and extend a campaign
that ran out of budget — without any correction and without invalidating
anything already reported.

Two things follow that are worth stating plainly. Evidence *accumulates*: two
independent campaigns multiply their e-values, so a follow-up study strengthens
the first rather than requiring a fresh one. And an e-value below the threshold
is not "no effect" but "not enough evidence yet", which is the honest reading
and the one a p-value near 0.06 never gets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from xai_gov.analysis.agrapa import BET_CAP

if TYPE_CHECKING:
    from collections.abc import Sequence

_LOG_CAP = 700.0
CS_PROCEDURE = "betting confidence sequence: inverted two-sided aGRAPA e-process"


@dataclass(frozen=True, slots=True)
class EvidenceVerdict:
    """The state of evidence at one moment in a campaign."""

    e_value: float
    log_e_value: float
    threshold: float
    alpha: float
    decided: bool
    observations: int
    interpretation: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "e_value": round(self.e_value, 6),
            "log_e_value": round(self.log_e_value, 6),
            "threshold": round(self.threshold, 6),
            "alpha": self.alpha,
            "decided": self.decided,
            "observations": self.observations,
            "interpretation": self.interpretation,
        }


@dataclass(slots=True)
class MeanShiftMartingale:
    """Sequential test that one arm's mean exceeds another's.

    Implemented as a betting martingale on paired differences. Pairing is not
    an optimization: the arms share a seed, a network and a demand path by
    construction, so an unpaired test would spend its power on variance the
    design already removed.

    The bet fraction is chosen by a mixture over a grid rather than by a
    plug-in estimate, because a plug-in fraction estimated from the same data
    it bets on is a subtle form of the leakage this whole module exists to
    avoid.
    """

    alpha: float = 0.05
    fractions: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.8)
    scale: float = 1.0
    observations: int = 0
    _log_components: list[float] = field(default_factory=list, repr=False)
    _peak_log: float = 0.0
    _decided_at: int | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not self.fractions:
            raise ValueError("at least one betting fraction is required")
        for fraction in self.fractions:
            if not 0.0 < fraction < 1.0:
                raise ValueError(f"betting fraction {fraction} must lie in (0, 1)")
        if self.scale <= 0.0:
            raise ValueError("scale must be positive; it normalizes the differences")
        self._log_components = [0.0] * len(self.fractions)

    @property
    def threshold(self) -> float:
        return 1.0 / self.alpha

    @property
    def log_value(self) -> float:
        if not self._log_components:
            return 0.0
        largest = max(self._log_components)
        total = sum(math.exp(c - largest) for c in self._log_components)
        return largest + math.log(total / len(self._log_components))

    @property
    def value(self) -> float:
        return math.exp(min(self.log_value, _LOG_CAP))

    @property
    def decided(self) -> bool:
        return self._decided_at is not None

    def update(self, difference: float) -> bool:
        """Place one round of bets on a paired difference.

        Differences are normalized and clipped to [-1, 1]. The clip is what
        keeps the payoff bounded and therefore keeps the martingale property
        exact; an outlier that would otherwise dominate is capped rather than
        allowed to manufacture evidence.
        """
        normalized = max(-1.0, min(1.0, difference / self.scale))
        for index, fraction in enumerate(self.fractions):
            payoff = 1.0 + fraction * normalized
            # A payoff of zero would make the log undefined and the bet ruinous;
            # the floor keeps a single extreme observation from ending the test.
            self._log_components[index] += math.log(max(payoff, 1e-12))
        self.observations += 1
        self._peak_log = max(self._peak_log, self.log_value)

        if self._decided_at is None and self.log_value >= math.log(self.threshold):
            self._decided_at = self.observations
            return True
        return False

    def verdict(self) -> EvidenceVerdict:
        decided = self.decided
        if decided:
            interpretation = (
                f"evidence exceeded 1/alpha at observation {self._decided_at}; "
                "the effect is supported at this level"
            )
        else:
            # Not "no effect". An e-value below threshold means the evidence has
            # not accumulated yet, and the campaign may legitimately continue.
            interpretation = (
                "evidence has not reached 1/alpha; this is insufficient evidence, "
                "not evidence of no effect, and the campaign may be extended"
            )
        return EvidenceVerdict(
            e_value=self.value,
            log_e_value=self.log_value,
            threshold=self.threshold,
            alpha=self.alpha,
            decided=decided,
            observations=self.observations,
            interpretation=interpretation,
        )

    def to_payload(self) -> dict[str, Any]:
        payload = self.verdict().to_payload()
        payload["peak_log_e_value"] = round(self._peak_log, 6)
        payload["decided_at"] = self._decided_at
        payload["fractions"] = list(self.fractions)
        return payload


@dataclass(frozen=True, slots=True)
class DirectionalEvidence:
    """Evidence for a directional claim and for its opposite.

    Two e-values, not one, because a single e-value cannot answer two
    questions. A large e-value is evidence against its own null; a *small* one
    means only that the bet lost. It does not license a claim in the opposite
    direction, and reading it that way was an error this project shipped.

    To assert that an effect runs opposite to a prediction, the opposite claim
    needs its own e-value — the same betting martingale run on negated
    differences, testing its own null. Both are reported with their nulls
    named, so a reader can see which hypothesis each number is about.
    """

    forward: float
    reverse: float | None
    threshold: float
    familywise_threshold: float
    observations: int
    forward_null: str
    reverse_null: str | None
    #: The e-values at the last replicate. ``forward``/``reverse`` are the
    #: running maxima, which the first-crossing rule decides on.
    forward_final: float | None = None
    reverse_final: float | None = None
    procedure: str = "aGRAPA"

    @property
    def supported(self) -> bool:
        """Evidence for the predicted direction at the familywise threshold."""
        return self.forward >= self.familywise_threshold

    @property
    def refuted(self) -> bool:
        """Evidence for the opposite direction, at the same threshold.

        This is the claim a small forward e-value cannot make on its own. Only
        a reverse e-value past the threshold establishes that the effect runs
        the other way.
        """
        return self.reverse is not None and self.reverse >= self.familywise_threshold

    @property
    def verdict(self) -> str:
        if self.supported:
            return "supported"
        if self.refuted:
            return "refuted: the effect runs opposite to the prediction"
        if self.forward >= self.threshold:
            return "supported per-comparison only; below the familywise threshold"
        if self.reverse is not None and self.reverse >= self.threshold:
            return "opposite direction per-comparison only; below familywise"
        return "inconclusive at this sample size"

    def to_payload(self) -> dict[str, Any]:
        return {
            "procedure": self.procedure,
            "forward_e_value": round(self.forward, 6),
            "reverse_e_value": None if self.reverse is None else round(self.reverse, 6),
            "forward_final_e_value": (
                None if self.forward_final is None else round(self.forward_final, 6)
            ),
            "reverse_final_e_value": (
                None if self.reverse_final is None else round(self.reverse_final, 6)
            ),
            "per_comparison_threshold": round(self.threshold, 4),
            "familywise_threshold": round(self.familywise_threshold, 4),
            "observations": self.observations,
            "forward_null": self.forward_null,
            "reverse_null": self.reverse_null,
            "supported": self.supported,
            "refuted": self.refuted,
            "verdict": self.verdict,
            "note": (
                "a forward e-value below 1 means the bet lost, not that the "
                "effect runs the other way; only the reverse e-value can "
                "establish that"
            ),
        }


def directional_evidence(
    differences: list[float],
    *,
    alpha: float = 0.05,
    hypotheses: int = 1,
    two_sided: bool = False,
    indicator: str = "the indicator",
    scale: float = 1.0,
) -> DirectionalEvidence:
    """Run the aGRAPA e-process forward and, for a directional claim, backward.

    The e-process is the predictable aGRAPA bet of `analysis.agrapa`, on
    differences scaled by the declared range ``scale``. ``forward`` and
    ``reverse`` are the running maxima: the plan rejects at the first replicate
    whose e-value reaches the familywise threshold, which Ville's inequality
    makes valid at any stopping time.

    ``differences`` are already oriented so that positive values support the
    prediction. The reverse test is the same construction on their negation,
    which is a valid e-value for the opposite null by the same argument.

    ``hypotheses`` sets the familywise threshold. Anytime validity handles
    optional stopping and says nothing about multiplicity: with k preregistered
    hypotheses, a union bound gives familywise control at alpha by requiring
    e >= k/alpha, which is the e-value analogue of a Bonferroni correction and
    is valid because e-values admit the union bound directly.

    A two-sided claim is tested against the point null "mean difference is
    zero" by averaging a bet on each direction: e = (e_up + e_down) / 2. Each
    component is a valid e-value under that null, and so is any convex
    combination of them, so the average is valid without correction.

    An earlier version bet on the magnitudes |d| instead. That is not a valid
    test: under the null E|d| > 0 whenever d has any spread, so the bet
    accumulates evidence where there is no effect. On the campaign's data the
    verdict did not change, but the construction was wrong in principle and the
    reported value was inflated by a factor of about two.

    ``differences`` must therefore be the signed values for a two-sided claim,
    not their absolute values. There is still no separate reverse e-value: the
    two directions are already pooled into the single two-sided one.
    """
    if hypotheses < 1:
        raise ValueError("hypotheses must be at least 1")
    if not differences:
        raise ValueError("cannot test an empty set of differences")

    from xai_gov.analysis.agrapa import familywise_threshold, run_e_process

    threshold = familywise_threshold(hypotheses=hypotheses, alpha=alpha)
    forward_path = run_e_process(
        differences, two_sided=two_sided, scale=scale, threshold=threshold
    )
    forward, forward_final = forward_path.peak, forward_path.final
    reverse: float | None = None
    reverse_final: float | None = None
    if not two_sided:
        reverse_path = run_e_process(
            [-value for value in differences], scale=scale, threshold=threshold
        )
        reverse, reverse_final = reverse_path.peak, reverse_path.final

    return DirectionalEvidence(
        forward=forward,
        reverse=reverse,
        threshold=1.0 / alpha,
        familywise_threshold=2 * hypotheses / alpha,
        observations=len(differences),
        forward_null=(
            f"no difference or an effect opposite to the prediction on {indicator}"
            if not two_sided
            else f"zero mean difference on {indicator}"
        ),
        reverse_null=(
            None if two_sided else f"no difference or the predicted effect on {indicator}"
        ),
        forward_final=forward_final,
        reverse_final=reverse_final,
    )


def combine_evidence(e_values: list[float]) -> float:
    """Combine independent e-values by multiplication.

    This is the property that makes evidence cumulative: a replication does not
    start from zero, it multiplies into what the first study established. The
    same operation on p-values is invalid, which is why meta-analysis normally
    needs machinery that this does not.
    """
    if not e_values:
        return 1.0
    if any(value < 0.0 for value in e_values):
        raise ValueError("e-values are non-negative by construction")
    log_total = sum(math.log(max(value, 1e-300)) for value in e_values)
    return math.exp(min(log_total, _LOG_CAP))


@dataclass(frozen=True, slots=True)
class ConfidenceSequence:
    """An interval valid at every sample size simultaneously.

    A confidence interval is valid at the one sample size it was computed for.
    Watching it shrink and stopping when it excludes zero destroys its coverage.
    A confidence sequence is valid at all sample sizes at once, so it can be
    monitored — which is what a running campaign needs.

    Built by inverting the same aGRAPA betting e-process that decides the
    hypotheses (see :func:`confidence_sequence`), so the interval and the test
    cannot disagree about zero in the direction that matters: a supported
    hypothesis always has an interval that excludes it.
    """

    mean: float
    lower: float
    upper: float
    n: int
    alpha: float
    procedure: str = CS_PROCEDURE

    @property
    def excludes_zero(self) -> bool:
        return self.lower > 0.0 or self.upper < 0.0

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def to_payload(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 6),
            "lower": round(self.lower, 6),
            "upper": round(self.upper, 6),
            "width": round(self.width, 6),
            "n": self.n,
            "alpha": self.alpha,
            "excludes_zero": self.excludes_zero,
            "procedure": self.procedure,
        }


def two_sided_peak_log_wealth(
    values: Sequence[float], nulls: Any, *, scale: float = 1.0
) -> Any:
    """Running maximum of the two-sided aGRAPA log-wealth at each null mean.

    For a candidate mean ``m`` the bets are placed on x_i = (d_i - m) / scale,
    with exactly the plug-ins of :class:`xai_gov.analysis.agrapa.AgrapaEProcess`
    (prior mean 0 and variance 1, each worth one observation; bet clipped to
    [0, 1/2]), once on x and once on -x, and the two wealths are averaged. At
    ``m = 0`` this is the two-sided e-process of the tests themselves.

    With |d| <= scale and |m| <= scale, |x| <= 2, so a bet of at most 1/2 keeps
    every factor 1 + lambda * x nonnegative: under E[d] = m each wealth is a
    nonnegative martingale, and so is their average. Vectorized over ``nulls``.
    """
    import numpy as np

    grid = np.atleast_1d(np.asarray(nulls, dtype=float))
    peak = np.zeros_like(grid)
    # Row 0 bets on x, row 1 on -x; each row is one aGRAPA wealth per null.
    signs = np.array([[1.0], [-1.0]])
    sums = np.zeros((2, grid.size))
    squares = np.zeros((2, grid.size))
    logs = np.zeros((2, grid.size))
    with np.errstate(divide="ignore", invalid="ignore"):
        for t, value in enumerate(values):
            x = signs * (float(value) - grid) / scale
            mean = sums / (t + 1)
            variance = (1.0 + squares) / (t + 1)
            bet = np.clip(mean / (variance + mean * mean), 0.0, BET_CAP)
            logs = logs + np.log1p(bet * x)
            sums = sums + x
            squares = squares + (x - sums / (t + 2)) ** 2
            current = np.logaddexp(logs[0], logs[1]) - math.log(2.0)
            peak = np.maximum(peak, current)
    return peak


def confidence_sequence(
    values: list[float], *, alpha: float = 0.05, bound: float = 1.0
) -> ConfidenceSequence:
    """A betting confidence sequence for the mean of ``values``.

    The interval is the set of means ``m`` that the two-sided aGRAPA e-process
    has never rejected at level ``alpha``:

        C_n = { m : sup_{t <= n} (K+_t(m) + K-_t(m)) / 2 < 1/alpha }.

    For the true mean, Ville's inequality bounds the probability that the
    e-process ever reaches 1/alpha by alpha, so C_n contains it at every n at
    once (Waudby-Smith and Ramdas, 2024). Because C_n is built from running
    maxima, it is the running intersection of the intervals at each n.

    ``bound`` is the declared range of the values, as in the test; a value
    outside it is refused rather than clipped. The boundary is located on a
    grid of step ``bound``/2000 and refined by bisection to about 1e-7.

    Coherence with the test: a one-sided e-value at or above 2k/alpha implies a
    two-sided e-value of at least k/alpha >= 1/alpha at zero, so a supported
    hypothesis always has an interval that excludes zero.
    """
    import numpy as np

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if bound <= 0.0:
        raise ValueError("bound must be positive; it is the range of the values")
    n = len(values)
    if n == 0:
        return ConfidenceSequence(mean=0.0, lower=-bound, upper=bound, n=0, alpha=alpha)
    if any(abs(v) > bound * (1.0 + 1e-9) for v in values):
        raise ValueError(
            f"a value lies outside the declared range ±{bound}; the range is part "
            "of the plan and is not adjusted to the data"
        )
    mean = float(sum(values)) / n
    log_threshold = math.log(1.0 / alpha)

    def inside(points: Any) -> Any:
        return two_sided_peak_log_wealth(values, points, scale=bound) < log_threshold

    grid = np.linspace(-bound, bound, 4001)
    kept = inside(grid)
    if not kept.any():
        # Cannot happen for alpha < 1 at the sample mean unless the data are
        # degenerate at a bound; report the point rather than an empty set.
        return ConfidenceSequence(mean=mean, lower=mean, upper=mean, n=n, alpha=alpha)
    first, last = int(np.argmax(kept)), int(len(kept) - 1 - np.argmax(kept[::-1]))

    def refine(outside: float, inner: float) -> float:
        for _ in range(40):
            middle = 0.5 * (outside + inner)
            if bool(inside(np.array([middle]))[0]):
                inner = middle
            else:
                outside = middle
            if abs(inner - outside) < 1e-7:
                break
        return inner

    lower = float(grid[first]) if first == 0 else refine(float(grid[first - 1]), float(grid[first]))
    upper = (
        float(grid[last])
        if last == len(grid) - 1
        else refine(float(grid[last + 1]), float(grid[last]))
    )
    return ConfidenceSequence(mean=mean, lower=lower, upper=upper, n=n, alpha=alpha)
