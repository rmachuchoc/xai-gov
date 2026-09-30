"""Learned delegation and the complementarity it must earn."""

from __future__ import annotations

import pytest

from xai_gov.io.decision_record import RiskRegime
from xai_gov.oversight.delegation import (
    ComplementarityLedger,
    LearnedDeferral,
    NoDeferral,
    ThresholdDeferral,
    available_deferral_policies,
    build_deferral_policy,
)
from xai_gov.oversight.supervisor import SimulatedSupervisor


def test_full_automation_never_defers() -> None:
    decision = NoDeferral().should_defer(
        regime=RiskRegime.DISRUPTION, disruption_belief=0.99, attention_remaining=100.0
    )
    assert decision.defer is False


def test_the_threshold_arm_ignores_who_it_is_escalating_to() -> None:
    """Its defect is not that the threshold is wrong but that it is blind."""
    policy = ThresholdDeferral(threshold=0.6)
    nominal = policy.should_defer(
        regime=RiskRegime.NOMINAL, disruption_belief=0.7, attention_remaining=10.0
    )
    disruption = policy.should_defer(
        regime=RiskRegime.DISRUPTION, disruption_belief=0.7, attention_remaining=10.0
    )
    assert nominal.defer == disruption.defer is True


def test_learned_deferral_starts_from_equal_priors() -> None:
    """Assuming the human is better would build the conclusion into the design;
    assuming the system is would prevent gathering evidence."""
    policy = LearnedDeferral()
    for regime in (RiskRegime.NOMINAL, RiskRegime.DRIFT, RiskRegime.DISRUPTION):
        assert policy.human_accuracy(regime) == pytest.approx(
            policy.system_accuracy(regime)
        )


def test_learned_deferral_finds_where_the_human_is_better() -> None:
    policy = LearnedDeferral(min_gain=0.05, attention_weight=0.0)
    for _ in range(40):
        policy.observe_outcome(regime=RiskRegime.DISRUPTION, deferred=True, correct=True)
        policy.observe_outcome(regime=RiskRegime.DISRUPTION, deferred=False, correct=False)
        policy.observe_outcome(regime=RiskRegime.NOMINAL, deferred=True, correct=False)
        policy.observe_outcome(regime=RiskRegime.NOMINAL, deferred=False, correct=True)

    assert policy.should_defer(
        regime=RiskRegime.DISRUPTION, disruption_belief=0.7, attention_remaining=10.0
    ).defer is True
    assert policy.should_defer(
        regime=RiskRegime.NOMINAL, disruption_belief=0.7, attention_remaining=10.0
    ).defer is False


def test_scarce_attention_raises_the_bar() -> None:
    """The same gain justifies deferral early in a run and not late in one."""
    policy = LearnedDeferral(attention_weight=2.0, min_gain=0.0)
    for _ in range(30):
        policy.observe_outcome(regime=RiskRegime.DRIFT, deferred=True, correct=True)
        policy.observe_outcome(regime=RiskRegime.DRIFT, deferred=False, correct=False)
    plentiful = policy.should_defer(
        regime=RiskRegime.DRIFT, disruption_belief=0.7, attention_remaining=50.0
    )
    scarce = policy.should_defer(
        regime=RiskRegime.DRIFT, disruption_belief=0.7, attention_remaining=0.05
    )
    assert scarce.attention_price > plentiful.attention_price


def test_no_attention_means_no_deferral() -> None:
    decision = LearnedDeferral().should_defer(
        regime=RiskRegime.DISRUPTION, disruption_belief=0.99, attention_remaining=0.0
    )
    assert decision.defer is False
    assert "no attention remains" in decision.reason


def test_outcomes_are_attributed_to_whoever_decided() -> None:
    policy = LearnedDeferral()
    before = policy.system_accuracy(RiskRegime.NOMINAL)
    policy.observe_outcome(regime=RiskRegime.NOMINAL, deferred=True, correct=True)
    assert policy.system_accuracy(RiskRegime.NOMINAL) == before
    assert policy.human_accuracy(RiskRegime.NOMINAL) > before


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="threshold"):
        ThresholdDeferral(threshold=0.0)
    with pytest.raises(ValueError, match="prior_strength"):
        LearnedDeferral(prior_strength=0.0)
    with pytest.raises(ValueError, match="attention_weight"):
        LearnedDeferral(attention_weight=-1.0)


def test_the_registry_builds_every_arm() -> None:
    assert available_deferral_policies() == ("none", "threshold", "learned")
    for name in available_deferral_policies():
        assert build_deferral_policy({"policy": name}).name == name
    with pytest.raises(ValueError, match="unknown deferral policy"):
        build_deferral_policy({"policy": "telepathy"})


# -- the supervisor -------------------------------------------------------
def test_the_supervisor_is_better_in_some_regimes_and_worse_in_others() -> None:
    """Uniform competence makes delegation either trivial or pointless."""
    accuracy = SimulatedSupervisor().accuracy
    assert accuracy[RiskRegime.NOMINAL.value] < accuracy[RiskRegime.DISRUPTION.value]


def test_attention_is_finite_and_running_out_is_recorded() -> None:
    """An unavailable supervisor is not a neutral event."""
    supervisor = SimulatedSupervisor(attention_budget=2.0)
    for _ in range(5):
        supervisor.review(regime=RiskRegime.DRIFT, truly_unsafe=True)
    assert supervisor.exhausted is True
    assert supervisor.declined > 0
    judgement = supervisor.review(regime=RiskRegime.DRIFT, truly_unsafe=True)
    assert judgement.available is False
    assert "returned to the system" in judgement.reason


def test_fatigue_degrades_but_does_not_randomize() -> None:
    """A tired overseer is worse, not useless; modelling them as a coin flip
    would overstate the case for automation."""
    supervisor = SimulatedSupervisor(attention_budget=10.0, fatigue=0.5)
    fresh = supervisor.effective_accuracy_multiplier
    for _ in range(10):
        supervisor.review(regime=RiskRegime.NOMINAL, truly_unsafe=False)
    assert supervisor.effective_accuracy_multiplier < fresh
    assert supervisor.effective_accuracy_multiplier >= 0.5


def test_the_supervisor_is_reproducible_from_its_seed() -> None:
    def run() -> list[bool]:
        supervisor = SimulatedSupervisor(seed=7, attention_budget=50.0)
        return [
            supervisor.review(regime=RiskRegime.DRIFT, truly_unsafe=True).approved
            for _ in range(20)
        ]

    assert run() == run()


def test_a_wrong_judgement_falls_the_way_the_bias_points() -> None:
    lenient = SimulatedSupervisor(
        seed=1,
        attention_budget=100.0,
        approval_bias=1.0,
        accuracy={"nominal": 0.0, "drift": 0.0, "disruption": 0.0},
    )
    approvals = [
        lenient.review(regime=RiskRegime.DRIFT, truly_unsafe=True).approved
        for _ in range(20)
    ]
    assert all(approvals)


def test_invalid_supervisor_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="accuracy"):
        SimulatedSupervisor(accuracy={"nominal": 1.5})
    with pytest.raises(ValueError, match="fatigue"):
        SimulatedSupervisor(fatigue=2.0)


# -- complementarity ------------------------------------------------------
def test_strict_complementarity_requires_beating_both() -> None:
    """A team that outperforms the system might simply be the human doing all
    the work."""
    ledger = ComplementarityLedger()
    for _ in range(10):
        ledger.record(
            regime=RiskRegime.DRIFT, team=True, human_alone=True, system_alone=False
        )
    # The team matches the human exactly, so this is not complementarity.
    assert ledger.strict_complementarity() == ()


def test_complementarity_is_detected_when_it_exists() -> None:
    ledger = ComplementarityLedger()
    for index in range(10):
        ledger.record(
            regime=RiskRegime.DRIFT,
            team=True,
            human_alone=index < 6,
            system_alone=index < 7,
        )
    assert "drift" in ledger.strict_complementarity()
    assert ledger.to_payload()["strict_complementarity"] is True


def test_the_verdict_is_stated_not_left_to_be_inferred() -> None:
    payload = ComplementarityLedger().to_payload()
    assert payload["strict_complementarity"] is False
    assert payload["strict_complementarity_regimes"] == []
