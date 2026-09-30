"""The human supervisor, modelled rather than assumed.

Escalation only means something if there is someone at the other end, and the
protocol's claim about effective human oversight is empty unless that person is
represented with the two properties that make oversight hard: their attention
is finite, and their competence is uneven.

This module is a simulated supervisor for exactly that purpose. It is not a
model of human cognition and does not claim to be. It is a decision-theoretic
stand-in with three declared parameters — accuracy by regime, attention budget,
and a bias — chosen so the delegation policy has something non-trivial to learn
and so RQ7 can fail.

The uneven competence is the point. A supervisor who is uniformly better than
the system makes delegation trivial (defer everything) and a supervisor who is
uniformly worse makes it pointless (defer nothing). Real oversight is valuable
precisely because the human is better in some regimes and worse in others, and
the delegation policy has to find which.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from xai_gov.core.seeds import derive_seed
from xai_gov.io.decision_record import RiskRegime


@dataclass(frozen=True, slots=True)
class SupervisorJudgement:
    """One human decision, with what it cost."""

    approved: bool
    correct: bool
    attention_spent: float
    available: bool
    reason: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "approved": self.approved,
            "correct": self.correct,
            "attention_spent": round(self.attention_spent, 6),
            "available": self.available,
            "reason": self.reason,
        }


@dataclass(slots=True)
class SimulatedSupervisor:
    """A human overseer with finite attention and regime-dependent accuracy.

    ``accuracy`` is the probability of judging correctly in each latent regime.
    The default encodes the case that makes oversight interesting: the human is
    *worse* than a calibrated detector under nominal conditions, where nothing
    is obviously wrong and vigilance decays, and *better* under disruption,
    where context and judgement outperform a statistic fitted to a world that
    no longer applies.

    ``fatigue`` degrades accuracy as attention is consumed. Without it, a
    supervisor with a large budget is simply a better system, and the attention
    constraint would only ever bind arithmetically.
    """

    seed: int = 0
    accuracy: dict[str, float] = field(
        default_factory=lambda: {
            RiskRegime.NOMINAL.value: 0.55,
            RiskRegime.DRIFT.value: 0.70,
            RiskRegime.DISRUPTION.value: 0.88,
        }
    )
    attention_budget: float = 12.0
    attention_per_review: float = 1.0
    fatigue: float = 0.15
    approval_bias: float = 0.5
    attention_spent: float = 0.0
    reviews: int = 0
    correct_reviews: int = 0
    declined: int = 0
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        for regime, value in self.accuracy.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"accuracy for {regime} must be a probability")
        if self.attention_budget < 0.0:
            raise ValueError("attention_budget must be non-negative")
        if self.attention_per_review <= 0.0:
            raise ValueError("attention_per_review must be positive")
        if not 0.0 <= self.fatigue <= 1.0:
            raise ValueError("fatigue must lie in [0, 1]")
        if not 0.0 <= self.approval_bias <= 1.0:
            raise ValueError("approval_bias must be a probability")
        self._rng = random.Random(derive_seed(self.seed, "supervisor"))

    @property
    def attention_remaining(self) -> float:
        return max(self.attention_budget - self.attention_spent, 0.0)

    @property
    def exhausted(self) -> bool:
        return self.attention_remaining < self.attention_per_review

    @property
    def effective_accuracy_multiplier(self) -> float:
        """Accuracy decay from consumed attention.

        Linear in the fraction of budget spent, floored so a tired supervisor
        is degraded rather than random: an overseer who has been working all
        day is worse, not useless, and modelling them as a coin flip would
        overstate the case for automation.
        """
        if self.attention_budget <= 0.0:
            return 1.0
        used = min(self.attention_spent / self.attention_budget, 1.0)
        return 1.0 - self.fatigue * used

    def review(self, *, regime: RiskRegime, truly_unsafe: bool) -> SupervisorJudgement:
        """Review one escalated decision.

        ``truly_unsafe`` is ground truth available only to the simulator. It
        never reaches the agent; it exists so the study can measure whether the
        team was right, which is the only way complementarity can be tested.
        """
        if self.exhausted:
            self.declined += 1
            # An unavailable supervisor is not a neutral event. The decision
            # falls back to the system, and the run must show that oversight
            # was requested and not delivered.
            return SupervisorJudgement(
                approved=True,
                correct=not truly_unsafe,
                attention_spent=0.0,
                available=False,
                reason="attention budget exhausted; decision returned to the system",
            )

        base = self.accuracy.get(regime.value, 0.5)
        accuracy = base * self.effective_accuracy_multiplier
        self.attention_spent += self.attention_per_review
        self.reviews += 1

        correct = bool(self._rng.random() < accuracy)
        # A wrong judgement is not a coin flip either: the bias decides which
        # way the error falls, so a lenient supervisor waves through unsafe
        # actions and a strict one blocks safe ones.
        approved = (
            (not truly_unsafe)
            if correct
            else bool(self._rng.random() < self.approval_bias)
        )
        self.correct_reviews += int(correct)

        return SupervisorJudgement(
            approved=approved,
            correct=correct,
            attention_spent=self.attention_per_review,
            available=True,
            reason=f"reviewed under {regime.value} at accuracy {accuracy:.2f}",
        )

    @property
    def realized_accuracy(self) -> float:
        return self.correct_reviews / self.reviews if self.reviews else 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "accuracy": dict(sorted(self.accuracy.items())),
            "attention_budget": self.attention_budget,
            "attention_spent": round(self.attention_spent, 6),
            "attention_remaining": round(self.attention_remaining, 6),
            "fatigue": self.fatigue,
            "reviews": self.reviews,
            "declined": self.declined,
            "realized_accuracy": round(self.realized_accuracy, 6),
        }
