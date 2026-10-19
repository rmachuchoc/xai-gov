"""Fallback versus projection: the same envelope, a different operator."""

from __future__ import annotations

import pytest

from tests.factories import state
from xai_gov.io.decision_record import ActionKind
from xai_gov.verification.shield import Shield

STARVED = {"inventory": 0.0, "in_transit": 0.0, "demand_observed": 20.0}


def test_projection_is_the_default() -> None:
    """Existing arms must behave exactly as before the operator was added."""
    assert Shield().mode == "projection"


def test_projection_repairs_a_starving_hold() -> None:
    _, quantity, outcome = Shield().filter(
        state=state(**STARVED), action=ActionKind.HOLD, quantity=0.0
    )
    assert outcome.activated and quantity > 0.0


def test_fallback_blocks_instead_of_repairing() -> None:
    """The unsafe hold is replaced by the null action, which is the same hold:
    the shield fires and cannot leave the inadmissible region."""
    shield = Shield(mode="fallback")
    action, quantity, outcome = shield.filter(
        state=state(**STARVED), action=ActionKind.HOLD, quantity=0.0
    )
    assert outcome.activated
    assert (action, quantity) == (ActionKind.HOLD, 0.0)
    assert shield.fallback_inadmissible == 1


def test_fallback_also_blocks_an_oversized_order() -> None:
    shield = Shield(mode="fallback")
    _, quantity, outcome = shield.filter(
        state=state(inventory=80.0), action=ActionKind.REORDER, quantity=500.0
    )
    assert outcome.activated and quantity == 0.0


def test_both_operators_pass_an_admissible_action_untouched() -> None:
    for mode in ("projection", "fallback"):
        _, quantity, outcome = Shield(mode=mode).filter(
            state=state(inventory=80.0), action=ActionKind.REORDER, quantity=10.0
        )
        assert not outcome.activated and quantity == 10.0


def test_an_unknown_operator_is_refused() -> None:
    with pytest.raises(ValueError, match="mode"):
        Shield(mode="veto")


def test_the_payload_records_the_operator() -> None:
    payload = Shield(mode="fallback").to_payload()
    assert payload["mode"] == "fallback"
    assert payload["fallback_inadmissible"] == 0


def test_the_registry_builds_a_fallback_shield() -> None:
    from xai_gov.governance.registry import _build_shield

    shield = _build_shield({"shield": True, "shield_mode": "fallback"})
    assert shield.mode == "fallback"
