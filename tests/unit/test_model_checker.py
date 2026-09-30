"""Model checking: exact, conservative, and honest about its abstraction."""

from __future__ import annotations

import pytest

from xai_gov.verification.model_checker import (
    Abstraction,
    AbstractState,
    ModelChecker,
    estimate_transitions,
)
from xai_gov.verification.specification import (
    Operator,
    SafetyProperty,
    Specification,
)


def prop(**kwargs: object) -> SafetyProperty:
    defaults: dict[str, object] = {
        "property_id": "P",
        "operator": Operator.BOUNDED_EVENTUALLY,
        "predicate": "stockout",
        "bound": 0.1,
        "horizon": 3,
    }
    return SafetyProperty(**(defaults | kwargs))  # type: ignore[arg-type]


def test_bands_must_ascend() -> None:
    with pytest.raises(ValueError, match="ascending"):
        Abstraction(inventory_bands=(10.0, 5.0, 20.0))


def test_an_abstraction_needs_more_than_one_edge() -> None:
    with pytest.raises(ValueError, match="at least two edges"):
        Abstraction(backlog_bands=(1.0,))


def test_states_map_into_bands() -> None:
    abstraction = Abstraction()
    assert abstraction.abstract(0.0, 0.0) == AbstractState(0, 0)
    assert abstraction.abstract(1000.0, 1000.0).inventory_band == len(
        abstraction.inventory_bands
    )


def test_a_safe_model_satisfies_its_property() -> None:
    """Two healthy bands that never reach the bad one."""
    checker = ModelChecker()
    transitions = {(3, 0): {(3, 0): 0.9, (4, 0): 0.1}, (4, 0): {(3, 0): 1.0}}
    results = checker.check(
        Specification((prop(),)), transitions, initial=AbstractState(3, 0)
    )
    assert results[0].satisfied is True
    assert results[0].discharged is True


def test_an_unsafe_model_fails_and_produces_a_counterexample() -> None:
    """A failed check without a path tells an engineer something is wrong but
    not what."""
    checker = ModelChecker()
    transitions = {(3, 0): {(0, 0): 1.0}, (0, 0): {(0, 0): 1.0}}
    results = checker.check(
        Specification((prop(bound=0.01),)), transitions, initial=AbstractState(3, 0)
    )
    assert results[0].satisfied is False
    assert results[0].counterexample
    assert results[0].counterexample[-1] == (0, 0)


def test_the_probability_is_the_maximum_over_reachable_states() -> None:
    """A property that holds from the observed start and fails from a state the
    system can reach is not a safety property."""
    checker = ModelChecker()
    transitions = {
        (4, 0): {(4, 0): 0.99, (3, 0): 0.01},
        (3, 0): {(0, 0): 1.0},
        (0, 0): {(0, 0): 1.0},
    }
    results = checker.check(
        Specification((prop(bound=0.5),)), transitions, initial=AbstractState(4, 0)
    )
    # From (3,0) the bad state is certain, so the maximum is 1 even though the
    # declared start is safe for a long time.
    assert results[0].probability == pytest.approx(1.0)


def test_a_predicate_outside_the_abstraction_is_not_discharged() -> None:
    """Silently passing it would be the one failure mode a verification layer
    must not have."""
    checker = ModelChecker()
    results = checker.check(
        Specification((prop(predicate="capacity_exhausted", operator=Operator.EVENTUALLY),)),
        {(3, 0): {(3, 0): 1.0}},
        initial=AbstractState(3, 0),
    )
    assert results[0].discharged is False
    assert results[0].satisfied is False
    assert "runtime monitor" in results[0].reason


def test_the_probability_excludes_states_that_are_already_bad() -> None:
    """A bad state reaches itself with probability 1. Including it would make
    every property whose bad region is reachable report exactly 1, collapsing
    the bound into 'is the bad state reachable' and discarding the
    quantitative content the bound exists for."""
    checker = ModelChecker()
    transitions = {
        (3, 0): {(3, 0): 0.98, (0, 0): 0.02},
        (0, 0): {(0, 0): 1.0},
    }
    result = checker.check(
        Specification((prop(bound=0.5, horizon=1),)), transitions, initial=AbstractState(3, 0)
    )[0]
    assert result.probability == pytest.approx(0.02)


def test_bounded_and_unbounded_operators_differ() -> None:
    """A rare transition reaches the bad state eventually but not in one step."""
    checker = ModelChecker()
    transitions = {
        (3, 0): {(3, 0): 0.98, (0, 0): 0.02},
        (0, 0): {(0, 0): 1.0},
    }
    bounded = checker.check(
        Specification((prop(bound=0.05, horizon=1),)), transitions, initial=AbstractState(3, 0)
    )[0]
    unbounded = checker.check(
        Specification((prop(operator=Operator.EVENTUALLY, bound=0.05),)),
        transitions,
        initial=AbstractState(3, 0),
    )[0]
    assert bounded.probability < unbounded.probability


def test_transition_estimation_smooths_single_observations() -> None:
    """Smoothing over the observed successors alone smooths nothing: a source
    seen going to exactly one place would still get probability 1, which is the
    fabricated certainty the smoothing exists to prevent."""
    model = estimate_transitions([((0, 0), (1, 0))])
    assert model[(0, 0)][(1, 0)] < 1.0
    # Mass is spread over states the run actually visited, not invented ones.
    assert set(model[(0, 0)]) == {(0, 0), (1, 0)}


def test_more_evidence_sharpens_the_estimate() -> None:
    """Smoothing must yield to data rather than permanently flattening it."""
    few = estimate_transitions([((0, 0), (1, 0))] * 2)
    many = estimate_transitions([((0, 0), (1, 0))] * 50)
    assert many[(0, 0)][(1, 0)] > few[(0, 0)][(1, 0)]


def test_estimated_rows_are_distributions() -> None:
    observations = [((0, 0), (1, 0)), ((0, 0), (1, 0)), ((0, 0), (2, 0))]
    model = estimate_transitions(observations)
    assert sum(model[(0, 0)].values()) == pytest.approx(1.0)


def test_zero_smoothing_recovers_the_raw_estimate() -> None:
    """The smoothing is a declared parameter, so switching it off must give
    back exactly the empirical frequencies."""
    model = estimate_transitions([((0, 0), (1, 0))], laplace=0.0)
    assert model[(0, 0)] == {(1, 0): 1.0}


def test_negative_smoothing_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        estimate_transitions([((0, 0), (1, 0))], laplace=-1.0)
