"""Specification conflicts: a formal-methods finding, not a log line."""

from __future__ import annotations

from tests.factories import state
from xai_gov.io.decision_record import ActionKind
from xai_gov.verification.shield import NoShield, Shield
from xai_gov.verification.specification import (
    Operator,
    SafetyProperty,
    Specification,
)


def conflicting() -> Specification:
    """A specification whose properties cannot both hold at zero capacity.

    One demands ordering enough to clear the stockout; the other forbids
    ordering at all. Both are individually reasonable and jointly unsatisfiable
    in a reachable state, which is exactly the situation a specification review
    needs to be told about.
    """
    return Specification(
        (
            SafetyProperty(
                property_id="STOCK",
                operator=Operator.PERSISTENT,
                predicate="stockout",
                bound=0.02,
                horizon=2,
            ),
            SafetyProperty(
                property_id="CAP",
                operator=Operator.EVENTUALLY,
                predicate="capacity_exhausted",
                bound=0.0,
            ),
        )
    )


def test_a_satisfiable_specification_reports_no_conflicts() -> None:
    shield = Shield()
    for _ in range(10):
        shield.filter(
            state=state(inventory=80.0), action=ActionKind.HOLD, quantity=0.0
        )
    assert shield.conflicts == 0
    assert shield.conflict_rate == 0.0


def test_a_conflict_is_counted_and_the_pair_named() -> None:
    """The rate and the pair are what a review needs; a warning per occurrence
    gives neither and buries the phase summaries."""
    shield = Shield(specification=conflicting())
    shield.filter(
        state=state(inventory=0.0, in_transit=0.0, demand_observed=20.0, capacity=0.0),
        action=ActionKind.HOLD,
        quantity=0.0,
    )
    assert shield.conflicts == 1
    assert shield.conflicting_pairs
    assert any("STOCK" in pair for pair in shield.conflicting_pairs)


def test_the_conflict_rate_is_reported_in_the_payload() -> None:
    shield = Shield(specification=conflicting())
    for _ in range(4):
        shield.filter(
            state=state(inventory=0.0, in_transit=0.0, demand_observed=20.0, capacity=0.0),
            action=ActionKind.HOLD,
            quantity=0.0,
        )
    for _ in range(6):
        shield.filter(
            state=state(inventory=80.0, capacity=60.0),
            action=ActionKind.HOLD,
            quantity=0.0,
        )
    payload = shield.to_payload()
    assert payload["conflicts"] == 4
    assert payload["conflict_rate"] == 0.4
    assert payload["conflicting_pairs"]


def test_a_conflict_resolves_in_favour_of_capacity() -> None:
    """Ordering beyond capacity is not an action the world can execute, so the
    tie-break is physical rather than arbitrary."""
    shield = Shield(specification=conflicting())
    _, quantity, _ = shield.filter(
        state=state(inventory=0.0, in_transit=0.0, demand_observed=20.0, capacity=0.0),
        action=ActionKind.REORDER,
        quantity=25.0,
    )
    assert quantity == 0.0


def test_the_control_arm_reports_zero_conflicts() -> None:
    assert NoShield().to_payload()["conflict_rate"] == 0.0
