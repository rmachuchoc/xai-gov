"""The belief filter: evidence, not switches."""

from __future__ import annotations

import pytest

from xai_gov.governance.belief import BeliefFilter, RegimeModel, build_regime_model
from xai_gov.io.decision_record import ConformalSignal, RiskRegime


def signal(*, flagged: bool, change: bool = False, active: bool = True) -> ConformalSignal:
    if not active:
        return ConformalSignal(
            score=0.0, threshold=None, level=0.1, flagged_ood=False, scheme="none"
        )
    return ConformalSignal(
        score=2.0,
        threshold=1.0,
        level=0.1,
        flagged_ood=flagged,
        scheme="aci",
        change_declared=change,
    )


def test_the_prior_presumes_normal_operation() -> None:
    filt = BeliefFilter()
    assert filt.probability(RiskRegime.NOMINAL) > 0.8
    assert filt.most_likely is RiskRegime.NOMINAL


def test_the_belief_is_always_a_distribution() -> None:
    filt = BeliefFilter()
    for flagged in (True, False, True, True, False):
        belief = filt.update(signal(flagged=flagged))
        assert sum(belief.values()) == pytest.approx(1.0, abs=1e-9)


def test_repeated_flags_shift_mass_toward_disruption() -> None:
    filt = BeliefFilter()
    start = filt.disruption
    for _ in range(10):
        filt.update(signal(flagged=True))
    assert filt.disruption > start
    assert filt.most_likely is RiskRegime.DISRUPTION


def test_the_belief_recovers_when_flags_stop() -> None:
    """A detector's false alarm must not be permanent: an agent that cannot
    come back from its own mistake is unusable."""
    filt = BeliefFilter()
    for _ in range(10):
        filt.update(signal(flagged=True))
    peak = filt.disruption
    for _ in range(30):
        filt.update(signal(flagged=False))
    assert filt.disruption < peak / 2


def test_a_declared_change_is_evidence_not_certainty() -> None:
    """Encoding a detector with false-alarm rate delta as certainty would make
    recovery impossible."""
    filt = BeliefFilter()
    filt.update(signal(flagged=True, change=True))
    assert filt.disruption < 1.0
    assert filt.disruption > BeliefFilter().disruption


def test_a_change_declaration_moves_more_mass_than_a_bare_flag() -> None:
    with_change = BeliefFilter()
    without = BeliefFilter()
    with_change.update(signal(flagged=True, change=True))
    without.update(signal(flagged=True))
    assert with_change.disruption > without.disruption


def test_an_inactive_signal_only_advances_time() -> None:
    """A warming detector must not be able to argue that all is well."""
    filt = BeliefFilter()
    before = filt.disruption
    for _ in range(5):
        filt.update(signal(flagged=False, active=False))
    # Drift toward the stationary distribution raises, never lowers, the mass
    # on adverse regimes from a confident nominal prior.
    assert filt.disruption >= before


def test_emission_probabilities_must_be_ordered() -> None:
    """A model that reads evidence backwards would still appear to work."""
    with pytest.raises(ValueError, match="must increase"):
        RegimeModel(emission={"nominal": 0.8, "drift": 0.35, "disruption": 0.05})


def test_missing_emission_probability_is_refused() -> None:
    with pytest.raises(ValueError, match="missing"):
        RegimeModel(emission={"nominal": 0.05, "drift": 0.35})


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="persistence"):
        RegimeModel(persistence=1.0)
    with pytest.raises(ValueError, match="change_likelihood_ratio"):
        RegimeModel(change_likelihood_ratio=0.5)


def test_low_persistence_pulls_the_belief_back_toward_mixing() -> None:
    """Persistence is not inertia against evidence, it is resistance to
    mixing.

    Each prediction step moves mass toward the mixing distribution before the
    observation is folded in. A fluid regime therefore *discards* accumulated
    evidence every period, so after a run of flags it holds less disruption
    mass than a sticky one — the opposite of the intuition that low persistence
    means "moves faster".
    """
    sticky = BeliefFilter(model=RegimeModel(persistence=0.98))
    fluid = BeliefFilter(model=RegimeModel(persistence=0.6))
    for _ in range(3):
        sticky.update(signal(flagged=True))
        fluid.update(signal(flagged=True))
    assert sticky.disruption > fluid.disruption
    # Both still moved off the prior: evidence is being read.
    assert fluid.disruption > BeliefFilter().disruption


def test_registry_builds_defaults_and_rejects_typos() -> None:
    assert build_regime_model({}).persistence == 0.9
    with pytest.raises(ValueError, match="rejected its parameters"):
        build_regime_model({"persistance": 0.9})
