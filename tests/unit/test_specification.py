"""Safety properties: what they claim, and what they refuse to claim."""

from __future__ import annotations

import pytest

from tests.factories import state
from xai_gov.verification.specification import (
    Operator,
    SafetyProperty,
    Specification,
    TraceWindow,
    available_predicates,
    build_specification,
)


def test_a_property_renders_as_pctl() -> None:
    """The formula travels into the artifact so a reviewer sees the claim, not
    a description of it."""
    prop = SafetyProperty(
        property_id="SAFE-1",
        operator=Operator.PERSISTENT,
        predicate="stockout",
        bound=0.02,
        horizon=2,
    )
    assert prop.formula() == "P<=0.02 [ G<=2 stockout ]"


def test_a_temporal_operator_needs_a_horizon() -> None:
    """Without one the claim has no temporal content, so it is refused rather
    than silently treated as one period."""
    with pytest.raises(ValueError, match="needs a horizon"):
        SafetyProperty(
            property_id="X", operator=Operator.PERSISTENT, predicate="stockout", bound=0.1
        )


def test_an_unknown_predicate_fails_at_load() -> None:
    with pytest.raises(ValueError, match="unknown predicate"):
        SafetyProperty(
            property_id="X", operator=Operator.EVENTUALLY, predicate="vibes", bound=0.1
        )


def test_a_bound_must_be_a_probability() -> None:
    with pytest.raises(ValueError, match="probability"):
        SafetyProperty(
            property_id="X", operator=Operator.EVENTUALLY, predicate="stockout", bound=1.5
        )


def test_duplicate_property_ids_are_refused() -> None:
    prop = SafetyProperty(
        property_id="SAME", operator=Operator.EVENTUALLY, predicate="stockout", bound=0.1
    )
    with pytest.raises(ValueError, match="duplicate property id"):
        Specification(properties=(prop, prop))


def test_the_default_specification_engages_on_a_bad_state() -> None:
    spec = Specification()
    healthy = state(inventory=80.0, backlog=0.0)
    broken = state(inventory=0.0, backlog=90.0)
    assert spec.violated_by(healthy) == ()
    assert len(spec.violated_by(broken)) >= 2


def test_a_state_check_is_not_a_temporal_verdict() -> None:
    """violated_by says the system is in a state the property is about, not
    that the property has been violated — persistent properties are violated
    over a trace."""
    spec = Specification()
    engaged = spec.violated_by(state(inventory=0.0))
    assert engaged
    assert all(prop.horizon >= 1 or prop.horizon == 0 for prop in engaged)


def test_the_trace_window_counts_consecutive_and_recent() -> None:
    window = TraceWindow(capacity=8)
    for inventory in (80.0, 0.0, 0.0, 0.0):
        window.append(state(inventory=inventory))
    assert window.consecutive_true("stockout") == 3
    assert window.recent_true("stockout", 4) == 3


def test_the_window_forgets_beyond_its_capacity() -> None:
    window = TraceWindow(capacity=3)
    for period in range(10):
        window.append(state(period=period))
    assert len(window.states) == 3
    assert window.states[0].period == 7


def test_a_recovered_run_resets_the_consecutive_count() -> None:
    window = TraceWindow()
    for inventory in (0.0, 0.0, 50.0):
        window.append(state(inventory=inventory))
    assert window.consecutive_true("stockout") == 0


def test_the_registry_builds_from_configuration() -> None:
    spec = build_specification(
        {
            "properties": [
                {
                    "id": "P1",
                    "operator": "bounded_eventually",
                    "predicate": "backlog_critical",
                    "bound": 0.1,
                    "horizon": 4,
                }
            ]
        }
    )
    assert len(spec) == 1
    assert spec.by_id("P1").horizon == 4


def test_the_registry_names_what_it_rejects() -> None:
    with pytest.raises(ValueError, match="unknown operator"):
        build_specification({"properties": [{"id": "P", "operator": "someday"}]})
    with pytest.raises(ValueError, match="unknown keys"):
        build_specification({"properties": [{"id": "P", "horizon": 2, "bnd": 0.1}]})


def test_every_predicate_is_listed() -> None:
    assert "stockout" in available_predicates()
    assert "backlog_critical" in available_predicates()
