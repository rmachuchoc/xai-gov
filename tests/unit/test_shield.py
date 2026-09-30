"""The shield: minimal, derived, and able to repair rather than only refuse."""

from __future__ import annotations

import pytest

from tests.factories import state
from xai_gov.io.decision_record import ActionKind
from xai_gov.verification.shield import ActionEnvelope, NoShield, Shield
from xai_gov.verification.specification import (
    Operator,
    SafetyProperty,
    Specification,
)


def test_an_unengaged_state_is_not_restricted() -> None:
    """The shield is inactive by default and earns each restriction."""
    shield = Shield()
    envelope = shield.envelope(state(inventory=80.0, backlog=0.0, demand_observed=10.0))
    assert envelope.minimum == 0.0
    assert envelope.binding_property is None


def test_a_healthy_action_passes_untouched() -> None:
    shield = Shield()
    action, quantity, outcome = shield.filter(
        state=state(inventory=80.0, backlog=0.0), action=ActionKind.REORDER, quantity=10.0
    )
    assert outcome.activated is False
    assert (action, quantity) == (ActionKind.REORDER, 10.0)


def test_the_shield_repairs_rather_than_only_refusing() -> None:
    """Cancelling a replenishment during a stockout is itself a way to violate
    a safety property; an agent whose only tool is refusal will do it."""
    shield = Shield()
    action, quantity, outcome = shield.filter(
        state=state(inventory=0.0, in_transit=0.0, demand_observed=20.0),
        action=ActionKind.HOLD,
        quantity=0.0,
    )
    assert outcome.activated is True
    assert quantity > 0.0
    assert action is ActionKind.REORDER
    assert outcome.property_id is not None


def test_substitution_is_minimal() -> None:
    """Every unit of substitution is performance taken from a policy that may
    have been right."""
    shield = Shield()
    engaged = state(inventory=0.0, in_transit=0.0, demand_observed=20.0)
    envelope = shield.envelope(engaged)
    _, quantity, _ = shield.filter(
        state=engaged, action=ActionKind.HOLD, quantity=0.0
    )
    assert quantity == pytest.approx(envelope.minimum)


def test_capacity_caps_what_the_shield_can_ask_for() -> None:
    shield = Shield()
    _, quantity, _ = shield.filter(
        state=state(inventory=0.0, in_transit=0.0, demand_observed=200.0, capacity=15.0),
        action=ActionKind.HOLD,
        quantity=0.0,
    )
    assert quantity <= 15.0


def test_the_envelope_is_derived_from_the_specification() -> None:
    """Adding a property changes the shield automatically; a hand-written safe
    mode drifts from the specification the first time either is edited alone."""
    empty = Shield(specification=Specification(properties=()))
    engaged = state(inventory=0.0, in_transit=0.0, demand_observed=20.0)
    assert empty.envelope(engaged).minimum == 0.0
    assert Shield().envelope(engaged).minimum > 0.0


def test_an_empty_envelope_is_refused() -> None:
    """Properties that forbid every action here are jointly unsatisfiable and
    must be revised, not silently resolved."""
    with pytest.raises(ValueError, match="empty envelope"):
        ActionEnvelope(minimum=10.0, maximum=5.0, binding_property="X", reason="")


def test_an_escalation_survives_the_shield() -> None:
    """Safety machinery that silently resolves a question posed to a human has
    removed the oversight it was meant to support."""
    shield = Shield()
    shield.filter(
        state=state(inventory=0.0, in_transit=0.0, demand_observed=20.0),
        action=ActionKind.HOLD,
        quantity=0.0,
        escalated=True,
    )
    assert shield.escalations_preserved == 1


def test_the_price_of_safety_is_reported_per_activation() -> None:
    """Averaging over untouched decisions would hide the size of the
    correction, which is what Proposition 2 bounds."""
    shield = Shield()
    for _ in range(8):
        shield.filter(state=state(inventory=80.0), action=ActionKind.HOLD, quantity=0.0)
    shield.filter(
        state=state(inventory=0.0, in_transit=0.0, demand_observed=20.0),
        action=ActionKind.HOLD,
        quantity=0.0,
    )
    assert shield.activation_rate < 0.2
    assert shield.mean_substitution > 0.0


def test_a_capacity_property_forces_a_hold() -> None:
    spec = Specification(
        (
            SafetyProperty(
                property_id="CAP",
                operator=Operator.EVENTUALLY,
                predicate="capacity_exhausted",
                bound=0.0,
            ),
        )
    )
    shield = Shield(specification=spec)
    action, quantity, outcome = shield.filter(
        state=state(capacity=0.0), action=ActionKind.REORDER, quantity=25.0
    )
    assert outcome.activated is True
    assert (action, quantity) == (ActionKind.HOLD, 0.0)


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="horizon_lookahead"):
        Shield(horizon_lookahead=0)
    with pytest.raises(ValueError, match="max_order"):
        Shield(max_order=0.0)


def test_the_control_arm_filters_nothing() -> None:
    """Required for RQ8: the price of safety is a difference against this."""
    shield = NoShield()
    action, quantity, outcome = shield.filter(
        state=state(inventory=0.0), action=ActionKind.HOLD, quantity=0.0
    )
    assert outcome.activated is False
    assert (action, quantity) == (ActionKind.HOLD, 0.0)
    assert shield.activation_rate == 0.0
