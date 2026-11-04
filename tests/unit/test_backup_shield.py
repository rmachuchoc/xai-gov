"""F-backup: an inadmissible action is replaced by a base-stock order in the envelope."""

from __future__ import annotations

import pytest

from tests.factories import state
from xai_gov.io.decision_record import ActionKind
from xai_gov.simulation.network import EXTERNAL_SOURCE, NodeSpec
from xai_gov.verification.shield import Shield, base_stock_order

STARVED = {"inventory": 0.0, "in_transit": 0.0, "demand_observed": 20.0}

# S = s + Q = 25 + 35 = 60, the retailer of the three-echelon network.
RETAIL = NodeSpec(
    node_id="retail", echelon=0, supplier_id=EXTERNAL_SOURCE, initial_inventory=45.0,
    capacity=120.0, reorder_point=25.0, order_quantity=35.0, lead_time_mean=2.0,
)


def test_base_stock_orders_up_to_s_plus_q_from_the_observed_position() -> None:
    observed = state(inventory=10.0, in_transit=15.0, backlog=5.0)
    # position = 10 + 15 - 5 = 20, so the order is 60 - 20.
    assert base_stock_order(RETAIL, observed) == pytest.approx(40.0)


def test_base_stock_never_orders_a_negative_quantity() -> None:
    assert base_stock_order(RETAIL, state(inventory=100.0, in_transit=0.0)) == 0.0


def test_backup_repairs_a_starving_hold_where_fallback_cannot() -> None:
    backup = Shield(mode="backup")
    action, quantity, outcome = backup.filter(
        state=state(**STARVED), action=ActionKind.HOLD, quantity=0.0, spec=RETAIL
    )
    envelope = Shield().envelope(state(**STARVED))
    assert outcome.activated
    assert action is ActionKind.REORDER and quantity > 0.0
    assert envelope.contains(quantity)
    assert backup.fallback_inadmissible == 0


def test_backup_is_the_base_stock_order_clamped_to_the_envelope() -> None:
    observed = state(inventory=80.0, in_transit=0.0, capacity=60.0)
    _, quantity, outcome = Shield(mode="backup").filter(
        state=observed, action=ActionKind.REORDER, quantity=500.0, spec=RETAIL
    )
    envelope = Shield().envelope(observed)
    assert outcome.activated
    assert quantity == pytest.approx(envelope.clamp(base_stock_order(RETAIL, observed)))


@pytest.mark.parametrize("inventory", [0.0, 3.0, 10.0, 40.0, 80.0, 150.0])
@pytest.mark.parametrize("backlog", [0.0, 12.0, 60.0])
@pytest.mark.parametrize("proposed", [0.0, 5.0, 60.0, 500.0])
def test_backup_always_executes_an_admissible_quantity(
    inventory: float, backlog: float, proposed: float
) -> None:
    """Admissible by construction: whatever it replaces lies in the envelope."""
    observed = state(inventory=inventory, backlog=backlog, in_transit=0.0, capacity=120.0)
    envelope = Shield().envelope(observed)
    _, quantity, _ = Shield(mode="backup").filter(
        state=observed, action=ActionKind.REORDER, quantity=proposed, spec=RETAIL
    )
    assert envelope.contains(quantity)


def test_backup_passes_an_admissible_action_untouched() -> None:
    _, quantity, outcome = Shield(mode="backup").filter(
        state=state(inventory=80.0), action=ActionKind.REORDER, quantity=10.0, spec=RETAIL
    )
    assert not outcome.activated and quantity == 10.0


def test_backup_without_a_spec_is_refused() -> None:
    with pytest.raises(ValueError, match="node spec"):
        Shield(mode="backup").filter(state=state(), action=ActionKind.HOLD, quantity=0.0)


def test_the_registry_builds_a_backup_shield() -> None:
    from xai_gov.governance.registry import _build_shield

    assert _build_shield({"shield": True, "shield_mode": "backup"}).mode == "backup"
