"""KPI layers: an absent layer must say so, never report a silent zero."""

from __future__ import annotations

from tests.unit.test_decision_record import make_record
from xai_gov.kpis.layers import compute_kpis, governance_layer, operational_layer
from xai_gov.simulation.engine import PeriodTrace, SimulationResult


def trace(period: int, demand: float, served: float, **kwargs: float) -> PeriodTrace:
    defaults: dict[str, float] = {
        "orders_total": 0.0, "inventory_total": 30.0, "backlog_total": 0.0,
        "holding_cost": 10.0, "backlog_cost": 0.0, "order_cost": 0.0,
        "intervention_cost": 0.0,
    }
    merged = defaults | kwargs
    return PeriodTrace(
        period=period,
        demand_total=demand,
        served_total=served,
        orders_total=merged["orders_total"],
        inventory_total=merged["inventory_total"],
        backlog_total=merged["backlog_total"],
        stocked_out_nodes=int(merged.get("stocked_out_nodes", 0)),
        holding_cost=merged["holding_cost"],
        backlog_cost=merged["backlog_cost"],
        order_cost=merged["order_cost"],
        intervention_cost=merged["intervention_cost"],
    )


def test_perfect_service_is_reported_as_one() -> None:
    result = SimulationResult(periods=3, traces=[trace(p, 10.0, 10.0) for p in range(3)])
    layer = operational_layer(result)
    assert layer["service_level"] == 1.0
    assert layer["fill_rate"] == 1.0
    assert layer["unmet_demand"] == 0.0


def test_service_level_and_fill_rate_differ_and_both_are_reported() -> None:
    """A node slightly short every period must not look like a node that
    failed outright twice.

    Chronic loses one unit in each of four periods (fill 0.975, service 0.0);
    acute loses five units in one period (fill 0.875, service 0.75). Ranking
    them by service level and by fill rate gives opposite answers, which is
    exactly why the protocol reports both.
    """
    chronic = SimulationResult(periods=4, traces=[trace(p, 10.0, 9.75) for p in range(4)])
    acute = SimulationResult(
        periods=4,
        traces=[
            trace(0, 10.0, 10.0), trace(1, 10.0, 10.0),
            trace(2, 10.0, 10.0), trace(3, 10.0, 5.0),
        ],
    )
    chronic_layer = operational_layer(chronic)
    acute_layer = operational_layer(acute)

    # Service level says acute is healthier; fill rate says chronic is.
    assert chronic_layer["service_level"] == 0.0
    assert acute_layer["service_level"] == 0.75
    assert chronic_layer["fill_rate"] > acute_layer["fill_rate"]


def test_ungoverned_arm_reports_zero_rates_not_absence() -> None:
    from xai_gov.io.decision_record import ActionKind, GovernanceAction

    records = [
        make_record(
            period=p,
            governance_action=GovernanceAction.APPROVE,
            final_action=ActionKind.HOLD,
            final_quantity=0.0,
            proposed_action=ActionKind.HOLD,
            proposed_quantity=0.0,
        )
        for p in range(3)
    ]
    layer = governance_layer(records)
    assert layer["status"] == "available"
    assert layer["intervention_rate"] == 0.0
    assert layer["traceability"] == 1.0


def test_layers_that_need_later_stages_declare_what_they_need() -> None:
    """Under the ungoverned arm neither conformal nor causal signals exist,
    and those layers must say so rather than report a fabricated zero."""
    from xai_gov.governance.agent import NEUTRAL_CONFORMAL, NEUTRAL_DESCRIPTOR

    result = SimulationResult(
        periods=2,
        traces=[trace(p, 10.0, 10.0) for p in range(2)],
        records=[
            make_record(period=p, conformal=NEUTRAL_CONFORMAL, descriptor=NEUTRAL_DESCRIPTOR)
            for p in range(2)
        ],
    )
    kpis = compute_kpis(result)

    assert kpis["guarantees"]["status"] == "not_available"
    assert "conformal" in kpis["guarantees"]["requires"]
    assert kpis["xai"]["status"] == "not_available"
    assert "descriptor" in kpis["xai"]["requires"]
    assert kpis["organizational_value"]["status"] == "partial"

    # The layers that ARE computable stay computable.
    assert kpis["operational"]["status"] == "available"
    assert kpis["governance"]["status"] == "available"


def test_an_active_conformal_scheme_makes_the_guarantees_layer_available() -> None:
    result = SimulationResult(
        periods=2,
        traces=[trace(p, 10.0, 10.0) for p in range(2)],
        records=[make_record(period=p) for p in range(2)],
    )
    layer = compute_kpis(result)["guarantees"]
    assert layer["status"] == "available"
    assert layer["ood_rate"] == 1.0


def test_bullwhip_is_zero_when_orders_track_demand_exactly() -> None:
    traces = [trace(p, 10.0 + p, 10.0 + p, orders_total=10.0 + p) for p in range(6)]
    result = SimulationResult(periods=6, traces=traces)
    kpis = compute_kpis(result)
    assert kpis["systemic_risk"]["bullwhip_ratio"] < 1.5


def test_shock_analysis_reports_degradation_and_recovery() -> None:
    traces = [trace(p, 10.0, 10.0) for p in range(5)]
    traces += [trace(5, 10.0, 4.0), trace(6, 10.0, 6.0), trace(7, 10.0, 10.0)]
    result = SimulationResult(periods=8, traces=traces)
    layer = compute_kpis(result, shock_periods=(5,))["systemic_risk"]
    assert layer["degradation"] > 0.0
    assert layer["recovery_periods"] == 2


def test_unrecovered_shock_reports_none_rather_than_the_horizon() -> None:
    traces = [trace(p, 10.0, 10.0) for p in range(4)]
    traces += [trace(p, 10.0, 3.0) for p in range(4, 8)]
    result = SimulationResult(periods=8, traces=traces)
    layer = compute_kpis(result, shock_periods=(4,))["systemic_risk"]
    assert layer["recovery_periods"] is None
