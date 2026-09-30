"""Learned delegation, and the complementarity it must earn.

Escalation is not a failure of the system. It is a decision with a cost and a
benefit, and the protocol models it as one: a rejection function decides, per
decision, whether the system acts or transfers control to a human whose
competence is uneven and whose attention is finite.

The objective is the *team's* error, not the model's. That is the whole shift.
A system minimizing its own error will never defer, because deferring cannot
improve a metric that only counts its own decisions; a system minimizing team
error defers exactly where the human is better and the attention is worth
spending.

`ThresholdDeferral` is the control arm: escalate above a fixed risk level, which
is what the field usually calls human oversight. `LearnedDeferral` estimates
who is better in each regime from the outcomes it has seen, and defers when the
expected gain exceeds the attention price.

The claim being tested is **strict complementarity**: that the team beats both
the human alone and the system alone in at least one regime. It is not assumed
anywhere in this design. `ComplementarityLedger` records the counterfactual
outcomes needed to test it, and the answer may well be no — which would be a
finding, and one the protocol is built to be able to report.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from xai_gov.core.variants import build_variant
from xai_gov.io.decision_record import RiskRegime

_REGIMES: tuple[RiskRegime, ...] = (
    RiskRegime.NOMINAL,
    RiskRegime.DRIFT,
    RiskRegime.DISRUPTION,
)


@dataclass(frozen=True, slots=True)
class DeferralDecision:
    """Whether to defer, and the reasoning that produced it."""

    defer: bool
    expected_gain: float
    attention_price: float
    reason: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "defer": self.defer,
            "expected_gain": round(self.expected_gain, 6),
            "attention_price": round(self.attention_price, 6),
            "reason": self.reason,
        }


class DeferralPolicy(ABC):
    """Decides whether a decision goes to the human."""

    name: str = "abstract"

    @abstractmethod
    def should_defer(
        self,
        *,
        regime: RiskRegime,
        disruption_belief: float,
        attention_remaining: float,
    ) -> DeferralDecision:
        """Whether to transfer this decision to the supervisor."""

    def observe_outcome(
        self, *, regime: RiskRegime, deferred: bool, correct: bool
    ) -> None:
        """Record how a decision turned out.

        Concrete rather than abstract: a policy that does not learn has nothing
        to record, and forcing every such policy to write an empty override
        would be ceremony. The default is a deliberate no-op, not an omission.
        """
        del regime, deferred, correct

    def to_payload(self) -> dict[str, Any]:
        return {"policy": self.name}


@dataclass(slots=True)
class NoDeferral(DeferralPolicy):
    """Full automation: nothing is ever escalated."""

    name: str = "none"

    def should_defer(
        self, *, regime: RiskRegime, disruption_belief: float, attention_remaining: float
    ) -> DeferralDecision:
        del regime, disruption_belief, attention_remaining
        return DeferralDecision(
            defer=False,
            expected_gain=0.0,
            attention_price=0.0,
            reason="deferral disabled; the system decides alone",
        )


@dataclass(slots=True)
class ThresholdDeferral(DeferralPolicy):
    """The control arm: escalate above a fixed belief in disruption.

    This is what a fixed escalation threshold in a config file amounts to, and
    it is the practice the protocol argues is insufficient. Its defect is not
    that the threshold is wrong but that it is blind: it cannot know whether
    the person it is escalating to is better than the system in this regime, so
    it spends attention identically whether or not doing so helps.
    """

    name: str = "threshold"
    threshold: float = 0.6

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold <= 1.0:
            raise ValueError("threshold must lie in (0, 1]")

    def should_defer(
        self, *, regime: RiskRegime, disruption_belief: float, attention_remaining: float
    ) -> DeferralDecision:
        del regime
        defer = disruption_belief >= self.threshold and attention_remaining > 0.0
        return DeferralDecision(
            defer=defer,
            expected_gain=0.0,
            attention_price=0.0,
            reason=(
                f"belief {disruption_belief:.3f} "
                f"{'at or above' if defer else 'below'} fixed threshold {self.threshold:.2f}"
            ),
        )


@dataclass(slots=True)
class LearnedDeferral(DeferralPolicy):
    """Defers when the human is expected to do better, at a price.

    Both accuracies are estimated online from realized outcomes, per regime,
    with a Beta-style prior so that a single early outcome does not settle the
    question. The prior starts the two actors *equal*: assuming the human is
    better would build the conclusion into the design, and assuming the system
    is better would prevent the policy from ever gathering evidence.

    The attention price rises as the budget depletes, which is what makes this
    a resource-allocation policy rather than a comparison. Late in a run,
    deferral must clear a higher bar because the attention it spends is no
    longer replaceable.
    """

    name: str = "learned"
    prior_strength: float = 4.0
    attention_weight: float = 0.3
    min_gain: float = 0.05
    #: Fraction of eligible decisions deferred regardless of the estimate, so
    #: the policy can observe outcomes it would not otherwise have seen.
    #: Without it the policy cannot start: it learns only from decisions it
    #: delegates and delegates only once it has learned, and with equal priors
    #: those two conditions are mutually exclusive. The deadlock is silent — the
    #: run completes, the arm is simply identical to its control.
    exploration: float = 0.1
    _rng: random.Random = field(init=False, repr=False)
    _human_correct: dict[str, float] = field(default_factory=dict, repr=False)
    _human_total: dict[str, float] = field(default_factory=dict, repr=False)
    _system_correct: dict[str, float] = field(default_factory=dict, repr=False)
    _system_total: dict[str, float] = field(default_factory=dict, repr=False)
    deferrals: int = 0
    retentions: int = 0

    def __post_init__(self) -> None:
        if self.prior_strength <= 0.0:
            raise ValueError("prior_strength must be positive")
        if self.attention_weight < 0.0:
            raise ValueError("attention_weight must be non-negative")
        if not 0.0 <= self.exploration < 1.0:
            raise ValueError(
                "exploration must lie in [0, 1); a policy that always explores is "
                "not learning, and one that never explores cannot start"
            )
        self._rng = random.Random(20260703)
        for regime in _REGIMES:
            # Equal priors: the policy must earn its belief that either actor
            # is better in a given regime.
            self._human_correct[regime.value] = self.prior_strength / 2.0
            self._human_total[regime.value] = self.prior_strength
            self._system_correct[regime.value] = self.prior_strength / 2.0
            self._system_total[regime.value] = self.prior_strength

    def human_accuracy(self, regime: RiskRegime) -> float:
        return self._human_correct[regime.value] / self._human_total[regime.value]

    def system_accuracy(self, regime: RiskRegime) -> float:
        return self._system_correct[regime.value] / self._system_total[regime.value]

    def should_defer(
        self, *, regime: RiskRegime, disruption_belief: float, attention_remaining: float
    ) -> DeferralDecision:
        if attention_remaining <= 0.0:
            return DeferralDecision(
                defer=False,
                expected_gain=0.0,
                attention_price=1.0,
                reason="no attention remains; the decision stays with the system",
            )

        gain = self.human_accuracy(regime) - self.system_accuracy(regime)
        # Scarcity raises the price: the same gain justifies deferral early in
        # a run and not late in one.
        scarcity = 1.0 / (1.0 + attention_remaining)
        price = self.attention_weight * scarcity
        defer = gain - price >= self.min_gain

        # Forced exploration. The estimate can only move on a decision the
        # policy actually delegated, so a policy that defers strictly on the
        # estimate never gathers the evidence it needs to defer.
        explored = False
        if not defer and self.exploration > 0.0:
            explored = self._rng.random() < self.exploration
            defer = explored

        if defer:
            self.deferrals += 1
        else:
            self.retentions += 1

        return DeferralDecision(
            defer=defer,
            expected_gain=gain,
            attention_price=price,
            reason=(
                f"exploring in {regime.value} at rate {self.exploration:.2f}; the "
                "estimate cannot move without a delegated outcome to learn from"
                if explored
                else f"in {regime.value} the supervisor is estimated "
                f"{'better' if gain > 0 else 'no better'} by {gain:+.3f}; "
                f"attention price {price:.3f}"
            ),
        )

    def observe_outcome(
        self, *, regime: RiskRegime, deferred: bool, correct: bool
    ) -> None:
        """Attribute an outcome to whoever actually made the decision."""
        correct_counts = self._human_correct if deferred else self._system_correct
        total_counts = self._human_total if deferred else self._system_total
        correct_counts[regime.value] += float(correct)
        total_counts[regime.value] += 1.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "policy": self.name,
            "deferrals": self.deferrals,
            "retentions": self.retentions,
            "estimated_human_accuracy": {
                regime.value: round(self.human_accuracy(regime), 4) for regime in _REGIMES
            },
            "estimated_system_accuracy": {
                regime.value: round(self.system_accuracy(regime), 4) for regime in _REGIMES
            },
            "attention_weight": self.attention_weight,
            "exploration": self.exploration,
            "min_gain": self.min_gain,
        }


@dataclass(slots=True)
class ComplementarityLedger:
    """Records what the team, the human and the system would each have done.

    Complementarity cannot be read off team performance alone: a team that
    outperforms the system might simply be the human doing all the work. The
    ledger keeps all three counterfactual outcomes per regime so the strict
    test — team beats *both*, in at least one regime — can be evaluated.
    """

    team_correct: dict[str, int] = field(default_factory=dict)
    human_correct: dict[str, int] = field(default_factory=dict)
    system_correct: dict[str, int] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        *,
        regime: RiskRegime,
        team: bool,
        human_alone: bool,
        system_alone: bool,
    ) -> None:
        key = regime.value
        self.counts[key] = self.counts.get(key, 0) + 1
        self.team_correct[key] = self.team_correct.get(key, 0) + int(team)
        self.human_correct[key] = self.human_correct.get(key, 0) + int(human_alone)
        self.system_correct[key] = self.system_correct.get(key, 0) + int(system_alone)

    def rates(self, source: dict[str, int]) -> dict[str, float]:
        return {
            regime: round(source.get(regime, 0) / total, 6)
            for regime, total in sorted(self.counts.items())
            if total
        }

    def strict_complementarity(self, *, margin: float = 0.0) -> tuple[str, ...]:
        """Regimes where the team strictly beats both actors alone."""
        team = self.rates(self.team_correct)
        human = self.rates(self.human_correct)
        system = self.rates(self.system_correct)
        return tuple(
            regime
            for regime in sorted(team)
            if team[regime] > human.get(regime, 0.0) + margin
            and team[regime] > system.get(regime, 0.0) + margin
        )

    def to_payload(self) -> dict[str, Any]:
        complementary = self.strict_complementarity()
        return {
            "decisions": dict(sorted(self.counts.items())),
            "team_accuracy": self.rates(self.team_correct),
            "human_alone_accuracy": self.rates(self.human_correct),
            "system_alone_accuracy": self.rates(self.system_correct),
            "strict_complementarity_regimes": list(complementary),
            # The RQ7 verdict, stated as a fact about this run rather than left
            # for a reader to infer from three tables.
            "strict_complementarity": bool(complementary),
        }


def available_deferral_policies() -> tuple[str, ...]:
    return ("none", "threshold", "learned")


def build_deferral_policy(config: dict[str, Any] | None) -> DeferralPolicy:
    """Construct the deferral policy named in configuration."""
    settings = config or {}
    name = str(settings.get("policy", "none"))
    if name == "none":
        return NoDeferral()
    variants: dict[str, type[DeferralPolicy]] = {
        "threshold": ThresholdDeferral,
        "learned": LearnedDeferral,
    }
    return build_variant(
        kind="deferral policy",
        name=name,
        params=settings.get("params", {}),
        variants=variants,
    )
