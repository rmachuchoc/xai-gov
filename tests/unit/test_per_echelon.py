"""Per-echelon indicators: where aggregation destroys the quantity."""

from __future__ import annotations

from tests.factories import state
from xai_gov.io.decision_record import (
    ActionKind,
    CausalDescriptor,
    ConformalSignal,
    DecisionRecord,
    GovernanceAction,
    ShieldOutcome,
)
from xai_gov.kpis.layers import per_echelon_layer

NEUTRAL_SIGNAL = ConformalSignal(
    score=0.0, threshold=None, level=0.0, flagged_ood=False, scheme="none"
)
NEUTRAL_DESCRIPTOR = CausalDescriptor(
    actionability_score=0.0,
    feature_effects={},
    explanatory_fidelity=0.0,
    identifiable=False,
    informative=False,
    method="none",
)


def record(
    node: str,
    period: int,
    *,
    inventory: float = 40.0,
    quantity: float = 10.0,
    demand: float = 20.0,
    action: GovernanceAction = GovernanceAction.APPROVE,
) -> DecisionRecord:
    return DecisionRecord(
        period=period,
        node_id=node,
        state=state(period=period, inventory=inventory, demand_observed=demand),
        policy_id="test",
        policy_version="1",
        proposed_action=ActionKind.REORDER,
        proposed_quantity=quantity,
        descriptor=NEUTRAL_DESCRIPTOR,
        conformal=NEUTRAL_SIGNAL,
        belief={},
        governance_action=action,
        governance_rationale="",
        autonomy_regime="H2",
        escalated=False,
        safe_mode=False,
        intervention_cost=0.0,
        shield=ShieldOutcome(),
        final_action=ActionKind.REORDER,
        final_quantity=quantity,
    )


def test_no_records_leaves_the_layer_unavailable() -> None:
    layer = per_echelon_layer([])
    assert layer["status"] == "not_available"


def test_each_node_gets_its_own_indicators() -> None:
    records = [record("retail", p) for p in range(4)] + [
        record("plant", p, inventory=90.0) for p in range(4)
    ]
    layer = per_echelon_layer(records)
    assert layer["echelons"] == 2
    assert set(layer["nodes"]) == {"plant", "retail"}
    assert layer["nodes"]["plant"]["mean_inventory"] == 90.0


def test_the_first_node_to_starve_is_named() -> None:
    """Which echelon fails first is the mechanism behind an aggregate collapse,
    and an averaged indicator cannot show it."""
    records = [record("retail", p, inventory=0.0 if p >= 1 else 40.0) for p in range(5)]
    records += [record("plant", p, inventory=0.0 if p >= 3 else 80.0) for p in range(5)]
    layer = per_echelon_layer(records)
    assert layer["first_stockout_period"]["retail"] == 1
    assert layer["first_stockout_period"]["plant"] == 3
    assert layer["first_node_to_starve"] == "retail"


def test_a_node_that_never_starves_reports_none_not_zero() -> None:
    layer = per_echelon_layer([record("retail", p) for p in range(4)])
    assert layer["first_stockout_period"]["retail"] is None
    assert layer["first_node_to_starve"] is None


def test_local_bullwhip_is_scale_free() -> None:
    """A node with larger throughput must not read as more amplifying merely
    for being larger."""
    small = [record("a", p, quantity=10.0 + p, demand=20.0 + p) for p in range(6)]
    large = [record("b", p, quantity=100.0 + 10 * p, demand=200.0 + 10 * p) for p in range(6)]
    layer = per_echelon_layer(small + large)
    assert layer["nodes"]["a"]["local_bullwhip"] > 0.0
    assert abs(
        layer["nodes"]["a"]["local_bullwhip"] - layer["nodes"]["b"]["local_bullwhip"]
    ) < 0.5


def test_intervention_rate_is_per_node() -> None:
    records = [record("retail", p, action=GovernanceAction.VETO) for p in range(4)]
    records += [record("plant", p) for p in range(4)]
    layer = per_echelon_layer(records)
    assert layer["nodes"]["retail"]["intervention_rate"] == 1.0
    assert layer["nodes"]["plant"]["intervention_rate"] == 0.0
