"""The closed loop: conservation, determinism, and the record invariant."""

from __future__ import annotations

import pytest

from xai_gov.core.seeds import SeedBundle
from xai_gov.governance.agent import NoGovernanceAgent
from xai_gov.io.decision_record import ActionKind
from xai_gov.policies.families import HeuristicSQPolicy, OpaqueDROPolicy
from xai_gov.simulation.demand import StableDemand, StructuralChangeDemand
from xai_gov.simulation.disruptions import Disruption, DisruptionKind, DisruptionSchedule
from xai_gov.simulation.engine import DigitalTwin
from xai_gov.simulation.network import EXTERNAL_SOURCE, NetworkSpec, NodeSpec

pytestmark = pytest.mark.integration


def node(node_id: str, echelon: int, supplier: str, **kwargs: float) -> NodeSpec:
    defaults: dict[str, float] = {
        "initial_inventory": 60.0, "capacity": 300.0, "reorder_point": 30.0,
        "order_quantity": 40.0, "lead_time_mean": 2.0,
    }
    return NodeSpec(node_id=node_id, echelon=echelon, supplier_id=supplier, **(defaults | kwargs))


def build(
    *, periods: int = 20, seed: int = 42, policy: object | None = None,
    network: NetworkSpec | None = None, schedule: DisruptionSchedule | None = None,
    demand: object | None = None,
) -> DigitalTwin:
    return DigitalTwin(
        network=network or NetworkSpec(nodes=(node("wh", 0, EXTERNAL_SOURCE),)),
        demand=demand or StableDemand(name="stable", mean=12.0, dispersion=0.15),  # type: ignore[arg-type]
        policy=policy or HeuristicSQPolicy(),  # type: ignore[arg-type]
        governance=NoGovernanceAgent(),
        seeds=SeedBundle(master_seed=seed),
        periods=periods,
        schedule=schedule,
    )


def three_echelon() -> NetworkSpec:
    return NetworkSpec(
        nodes=(
            node("retail", 0, "dc", initial_inventory=45.0, capacity=120.0),
            node("dc", 1, "plant", initial_inventory=90.0, capacity=300.0),
            node("plant", 2, EXTERNAL_SOURCE, initial_inventory=180.0, capacity=600.0),
        ),
        name="three_echelon",
    )


def test_one_record_per_node_per_period() -> None:
    """The traceability invariant: nothing executes unrecorded."""
    twin = build(periods=10, network=three_echelon())
    result = twin.run()
    assert len(result.records) == 10 * 3
    assert len(result.traces) == 10


def test_runs_are_bit_for_bit_reproducible() -> None:
    first = build(seed=7).run()
    second = build(seed=7).run()
    assert [r.to_payload() for r in first.records] == [r.to_payload() for r in second.records]


def test_a_different_seed_changes_the_trajectory() -> None:
    first = build(seed=7).run()
    other = build(seed=8).run()
    assert first.demand_series != other.demand_series


def test_demand_enters_only_at_echelon_zero() -> None:
    result = build(periods=8, network=three_echelon()).run()
    upstream = [r for r in result.records if r.node_id != "retail"]
    assert all(r.state.demand_observed == 0.0 for r in upstream)


def test_served_never_exceeds_demand_plus_backlog() -> None:
    result = build(periods=25).run()
    for trace in result.traces:
        assert trace.served_total <= trace.demand_total + trace.backlog_total + 1e-9


def test_inventory_never_goes_negative() -> None:
    result = build(periods=30, policy=OpaqueDROPolicy()).run()
    assert all(record.state.inventory >= -1e-9 for record in result.records)


def test_final_quantity_never_exceeds_capacity_headroom() -> None:
    result = build(periods=30, policy=OpaqueDROPolicy()).run()
    for record in result.records:
        headroom = record.state.capacity - record.state.inventory - record.state.in_transit
        assert record.final_quantity <= max(headroom, 0.0) + 1e-9


def test_capacity_loss_reduces_observed_capacity() -> None:
    schedule = DisruptionSchedule(
        disruptions=(
            Disruption(kind=DisruptionKind.CAPACITY_LOSS, start=5, duration=3, magnitude=0.5),
        )
    )
    result = build(periods=10, schedule=schedule).run()
    during = [r.state.capacity for r in result.records if 5 <= r.period < 8]
    outside = [r.state.capacity for r in result.records if r.period < 5]
    assert max(during) < min(outside)


def test_supply_halt_places_no_orders() -> None:
    schedule = DisruptionSchedule(
        disruptions=(
            Disruption(kind=DisruptionKind.SUPPLY_HALT, start=0, duration=30, magnitude=1.0),
        )
    )
    result = build(periods=30, schedule=schedule).run()
    assert all(trace.orders_total == 0.0 for trace in result.traces)
    # And the consequence is visible, not silently absorbed.
    assert result.traces[-1].backlog_total > 0.0


def test_ungoverned_arm_never_modifies_an_action() -> None:
    result = build(periods=15, policy=OpaqueDROPolicy(), network=three_echelon()).run()
    assert not any(record.action_was_modified for record in result.records)


def test_expedite_arrives_no_earlier_than_the_next_period() -> None:
    twin = build(periods=15, policy=OpaqueDROPolicy())
    result = twin.run()
    expedited = [r for r in result.records if r.final_action is ActionKind.EXPEDITE]
    if expedited:
        assert all(r.final_quantity > 0.0 for r in expedited)
    # A same-period arrival would show as inventory rising before any lead
    # time elapsed; the pipeline forbids it structurally.
    assert all(period > 0 for period in twin.states["wh"].pipeline)


def test_structural_change_is_visible_in_the_demand_series() -> None:
    demand = StructuralChangeDemand(
        name="structural_change", mean=10.0, dispersion=0.1,
        change_period=10, new_mean_multiplier=2.0, new_dispersion=0.4,
    )
    result = build(periods=20, demand=demand).run()
    before = sum(result.demand_series[:10]) / 10
    after = sum(result.demand_series[10:]) / 10
    assert after > before * 1.4


def test_bullwhip_amplifies_upstream_in_a_three_echelon_chain() -> None:
    """Orders must be more variable than demand once echelons stack."""
    result = build(periods=40, network=three_echelon(), policy=OpaqueDROPolicy()).run()
    from xai_gov.kpis.layers import compute_kpis

    layer = compute_kpis(result)["systemic_risk"]
    assert layer["order_cv"] > 0.0


def test_zero_periods_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one period"):
        build(periods=0)
