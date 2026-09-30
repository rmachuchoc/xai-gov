"""Belief over the latent risk regime.

The governance agent does not observe whether the system is in a nominal,
drifting or disrupted regime; it observes a conformal signal that is
informative about it. The belief is therefore a filtering problem, and this
module is the filter.

Two commitments are worth stating.

The observation model is declared, not learned. `emission` gives the
probability of a flag under each regime, and those numbers are experimental
parameters subject to the sensitivity analysis of the protocol — not
constants tuned until the results looked right. Fitting them to the same
runs used to evaluate governance would be circular.

The martingale enters as evidence, not as a switch. A declared change makes
the disruption regime far more likely, but it does not set the belief to
certainty: a detector with a false-alarm rate of δ is wrong δ of the time,
and encoding it as certainty would make the agent unable to recover from its
own false alarm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xai_gov.io.decision_record import ConformalSignal, RiskRegime

_REGIMES: tuple[RiskRegime, ...] = (
    RiskRegime.NOMINAL,
    RiskRegime.DRIFT,
    RiskRegime.DISRUPTION,
)
_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class RegimeModel:
    """Transition and emission parameters of the latent regime.

    ``persistence`` is the probability a regime holds from one period to the
    next. High persistence is what makes the belief stable enough to act on;
    without it the filter tracks noise and the agent oscillates between
    approving and vetoing the same decision.
    """

    persistence: float = 0.9
    emission: dict[str, float] = field(
        default_factory=lambda: {
            RiskRegime.NOMINAL.value: 0.05,
            RiskRegime.DRIFT.value: 0.35,
            RiskRegime.DISRUPTION.value: 0.80,
        }
    )
    change_likelihood_ratio: float = 8.0

    def __post_init__(self) -> None:
        if not 0.0 < self.persistence < 1.0:
            raise ValueError("persistence must lie in (0, 1)")
        missing = {regime.value for regime in _REGIMES} - set(self.emission)
        if missing:
            raise ValueError(f"emission probabilities missing for {sorted(missing)}")
        for regime, probability in self.emission.items():
            if not 0.0 < probability < 1.0:
                raise ValueError(f"emission probability for {regime} must lie in (0, 1)")
        if self.change_likelihood_ratio < 1.0:
            raise ValueError("change_likelihood_ratio must be at least 1")
        # A flag must be more likely under a worse regime, or the filter would
        # read evidence backwards while still appearing to work.
        ordered = [self.emission[regime.value] for regime in _REGIMES]
        if not ordered[0] < ordered[1] < ordered[2]:
            raise ValueError(
                "emission probabilities must increase from nominal to disruption; "
                f"got {ordered}"
            )

    def transition(self, from_regime: RiskRegime, to_regime: RiskRegime) -> float:
        """Probability of moving between regimes in one period."""
        if from_regime is to_regime:
            return self.persistence
        return (1.0 - self.persistence) / (len(_REGIMES) - 1)

    def flag_probability(self, regime: RiskRegime) -> float:
        return self.emission[regime.value]

    def to_payload(self) -> dict[str, Any]:
        return {
            "persistence": self.persistence,
            "emission": dict(sorted(self.emission.items())),
            "change_likelihood_ratio": self.change_likelihood_ratio,
        }


@dataclass(slots=True)
class BeliefFilter:
    """Forward filter over the three latent regimes."""

    model: RegimeModel = field(default_factory=RegimeModel)
    belief: dict[str, float] = field(init=False)
    updates: int = 0

    def __post_init__(self) -> None:
        # The prior places most mass on nominal: a system is presumed to be
        # operating normally until evidence says otherwise, which is also the
        # assumption that makes a false alarm costly and therefore measurable.
        self.belief = {
            RiskRegime.NOMINAL.value: 0.9,
            RiskRegime.DRIFT.value: 0.08,
            RiskRegime.DISRUPTION.value: 0.02,
        }

    # -- accessors --------------------------------------------------------
    def probability(self, regime: RiskRegime) -> float:
        return self.belief[regime.value]

    @property
    def disruption(self) -> float:
        return self.probability(RiskRegime.DISRUPTION)

    @property
    def adverse(self) -> float:
        """Mass on anything other than nominal."""
        return 1.0 - self.probability(RiskRegime.NOMINAL)

    @property
    def most_likely(self) -> RiskRegime:
        return max(_REGIMES, key=self.probability)

    def snapshot(self) -> dict[str, float]:
        """A copy safe to seal into a decision record."""
        return {key: round(value, 10) for key, value in sorted(self.belief.items())}

    # -- filtering --------------------------------------------------------
    def update(self, signal: ConformalSignal) -> dict[str, float]:
        """Advance the belief by one period given the conformal signal.

        An inactive signal advances the prediction step only. That is the
        correct treatment of a detector that is still warming up: time has
        passed, so the belief should drift toward the stationary distribution,
        but no evidence arrived, so nothing should sharpen. Treating an absent
        observation as a negative one would let a warming detector actively
        argue that everything is fine.
        """
        predicted = {
            regime.value: sum(
                self.model.transition(source, regime) * self.belief[source.value]
                for source in _REGIMES
            )
            for regime in _REGIMES
        }

        if not signal.active:
            self.belief = _normalize(predicted)
            self.updates += 1
            return self.snapshot()

        posterior: dict[str, float] = {}
        for regime in _REGIMES:
            flag_probability = self.model.flag_probability(regime)
            likelihood = flag_probability if signal.flagged_ood else 1.0 - flag_probability
            if signal.change_declared and regime is RiskRegime.DISRUPTION:
                likelihood *= self.model.change_likelihood_ratio
            posterior[regime.value] = predicted[regime.value] * likelihood

        self.belief = _normalize(posterior)
        self.updates += 1
        return self.snapshot()

    def to_payload(self) -> dict[str, Any]:
        return {
            "model": self.model.to_payload(),
            "updates": self.updates,
            "belief": self.snapshot(),
            "most_likely": self.most_likely.value,
        }


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    """Normalize a weight vector, falling back to the prior on collapse.

    Total mass can underflow when several very small likelihoods multiply.
    Returning a uniform belief in that case is the honest response: the filter
    has lost track, and a uniform belief says so, whereas renormalizing
    denormals would produce a confident-looking number backed by nothing.
    """
    total = sum(weights.values())
    if total <= _EPS:
        share = 1.0 / len(weights)
        return dict.fromkeys(weights, share)
    return {key: value / total for key, value in weights.items()}


def build_regime_model(config: dict[str, Any]) -> RegimeModel:
    """Construct the regime model from configuration."""
    if not config:
        return RegimeModel()
    params = dict(config)
    emission = params.get("emission")
    if emission is not None:
        if not isinstance(emission, dict):
            raise ValueError("regime 'emission' must be a mapping")
        params["emission"] = {str(key): float(value) for key, value in emission.items()}
    try:
        return RegimeModel(**params)
    except TypeError as error:
        raise ValueError(f"regime model rejected its parameters: {error}") from error
