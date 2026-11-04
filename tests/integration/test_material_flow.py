"""Material flow contracts of the serial chain (review make-4658289, R1.1-R1.2).

Three properties any correct serial model with backorders satisfies:

1. material is conserved in every period: initial inventory + cumulative
   production = inventory at nodes + material in transit + cumulative sales
   to the end customer;
2. the fill rate β at the retailer lies in [0, 1];
3. a backorder raised at a supplier eventually reaches the node that
   ordered it.

The scripted scenario never orders from the external source, so production
is zero by construction and the conservation check does not depend on how
the plant is modelled. It also avoids fixing the exact period a shipment
arrives, which depends on the timing of the period step; the contracts hold
under any timing that conserves material.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from xai_gov.core.seeds import SeedBundle
from xai_gov.governance.agent import NoGovernanceAgent
from xai_gov.io.decision_record import ActionKind
from xai_gov.kpis.layers import compute_kpis
from xai_gov.policies.base import Policy, PolicyProposal
from xai_gov.policies.families import HeuristicSQPolicy, OpaqueDROPolicy
from xai_gov.simulation.demand import StableDemand
from xai_gov.simulation.disruptions import Disruption, DisruptionKind, DisruptionSchedule
from xai_gov.simulation.engine import DigitalTwin
from xai_gov.simulation.network import EXTERNAL_SOURCE, NetworkSpec, NodeSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from xai_gov.io.decision_record import OperatingState

pytestmark = pytest.mark.integration

TOL = 1e-9


# The scripted scenario: the retailer places one order of 20 on a DC that
# holds 12, so the DC is short by exactly 8 and must replenish from the plant.
RETAIL_ORDER = 20.0
DC_STOCK = 12.0
SHORTFALL = RETAIL_ORDER - DC_STOCK  # 8 units on backorder at the DC
RETAIL_STOCK = 50.0
PLANT_STOCK = 100.0
CUSTOMER_DEMAND = 2.0
HORIZON = 10


class ScriptedPolicy(Policy):
    """Deterministic orders: one retail order, then cover backorders only.

    The retailer orders ``RETAIL_ORDER`` in period 0 and nothing after.
    Every other node orders exactly what it needs to cover its uncovered
    backorder, ``max(backlog - inventory - in_transit, 0)``, so the plant —
    which never falls short — never orders from the external source.
    """

    policy_id = "scripted_material_flow"
    version = "1.0.0"

    def propose(
        self, *, spec: NodeSpec, state: OperatingState, rng: np.random.Generator
    ) -> PolicyProposal:
        if spec.echelon == 0:
            quantity = RETAIL_ORDER if state.period == 0 else 0.0
        else:
            quantity = max(state.backlog - state.inventory - state.in_transit, 0.0)
        if quantity <= 0.0:
            return PolicyProposal(ActionKind.HOLD, 0.0, {}, "nothing to cover")
        return PolicyProposal(ActionKind.REORDER, quantity, {}, "scripted order")


def scripted_node(node_id: str, echelon: int, supplier: str, stock: float) -> NodeSpec:
    return NodeSpec(
        node_id=node_id, echelon=echelon, supplier_id=supplier,
        initial_inventory=stock, capacity=1_000.0, reorder_point=0.0,
        order_quantity=0.0, lead_time_mean=1.0,
    )


SCRIPTED_NETWORK = NetworkSpec(
    nodes=(
        scripted_node("retail", 0, "dc", RETAIL_STOCK),
        scripted_node("dc", 1, "plant", DC_STOCK),
        scripted_node("plant", 2, EXTERNAL_SOURCE, PLANT_STOCK),
    ),
    name="scripted_three_echelon",
)


def scripted_twin(periods: int) -> DigitalTwin:
    return DigitalTwin(
        network=SCRIPTED_NETWORK,
        demand=StableDemand(name="stable", mean=CUSTOMER_DEMAND, dispersion=0.0),
        policy=ScriptedPolicy(),
        governance=NoGovernanceAgent(),
        seeds=SeedBundle(master_seed=0),
        periods=periods,
    )


def stochastic_twin(
    policy: Policy, seed: int, *, periods: int = 30, schedule: DisruptionSchedule | None = None
) -> DigitalTwin:
    def node(node_id: str, echelon: int, supplier: str, **kwargs: float) -> NodeSpec:
        defaults: dict[str, float] = {
            "initial_inventory": 60.0, "capacity": 300.0, "reorder_point": 30.0,
            "order_quantity": 40.0, "lead_time_mean": 2.0, "lead_time_dispersion": 0.25,
        }
        return NodeSpec(
            node_id=node_id, echelon=echelon, supplier_id=supplier, **(defaults | kwargs)
        )

    network = NetworkSpec(
        nodes=(
            node("retail", 0, "dc", initial_inventory=45.0, capacity=120.0),
            node("dc", 1, "plant", initial_inventory=90.0),
            node("plant", 2, EXTERNAL_SOURCE, initial_inventory=180.0, capacity=600.0),
        ),
        name="three_echelon",
    )
    return DigitalTwin(
        network=network,
        demand=StableDemand(name="stable", mean=12.0, dispersion=0.15),
        policy=policy,
        governance=NoGovernanceAgent(),
        seeds=SeedBundle(master_seed=seed),
        periods=periods,
        schedule=schedule,
    )


def customer_sales(twin: DigitalTwin) -> float:
    """Cumulative units delivered to the end customer, on time or late."""
    return sum(
        twin.states[spec.node_id].cumulative_served for spec in twin.network.demand_facing
    )


# -- 1. conservation ------------------------------------------------------------
@pytest.mark.parametrize("periods", range(1, HORIZON + 1))
def test_material_is_conserved_at_the_end_of_every_period(periods: int) -> None:
    """Initial stock = stock at nodes + in transit + sales (no production here).

    Each case runs the same seeded scenario for one more period, so the
    balance is checked at the close of every period, not only at the end.
    """
    twin = scripted_twin(periods)
    twin.run()

    initial = sum(spec.initial_inventory for spec in twin.network.nodes)
    on_hand = sum(state.inventory for state in twin.states.values())
    in_transit = sum(state.in_transit for state in twin.states.values())
    sales = customer_sales(twin)

    assert on_hand + in_transit + sales == pytest.approx(initial, abs=TOL), (
        f"period {periods - 1}: {initial} units at start, but {on_hand} on hand + "
        f"{in_transit} in transit + {sales} sold = {on_hand + in_transit + sales}"
    )


DISRUPTED = DisruptionSchedule(
    disruptions=(
        Disruption(kind=DisruptionKind.CAPACITY_LOSS, start=4, duration=4, magnitude=0.5),
        Disruption(kind=DisruptionKind.LEAD_TIME_SPIKE, start=8, duration=4, magnitude=2.0),
        Disruption(kind=DisruptionKind.SUPPLY_HALT, start=12, duration=3, magnitude=1.0),
        Disruption(kind=DisruptionKind.DEMAND_SURGE, start=16, duration=4, magnitude=1.8),
    )
)


@pytest.mark.parametrize("schedule", [None, DISRUPTED], ids=["calm", "disrupted"])
@pytest.mark.parametrize("policy", [HeuristicSQPolicy, OpaqueDROPolicy])
@pytest.mark.parametrize("seed", [7, 42])
def test_material_is_conserved_with_production_in_a_stochastic_chain(
    policy: type[Policy], seed: int, schedule: DisruptionSchedule | None
) -> None:
    """The full balance, production included, at the close of every period."""
    for periods in range(1, 26):
        twin = stochastic_twin(policy(), seed, periods=periods, schedule=schedule)
        result = twin.run()
        initial = sum(spec.initial_inventory for spec in twin.network.nodes)
        production = sum(trace.production_total for trace in result.traces)
        on_hand = sum(state.inventory for state in twin.states.values())
        in_transit = sum(state.in_transit for state in twin.states.values())
        sales = customer_sales(twin)
        assert on_hand + in_transit + sales == pytest.approx(initial + production, abs=1e-6), (
            f"period {periods - 1}: balance off by "
            f"{on_hand + in_transit + sales - initial - production}"
        )
        assert sales == pytest.approx(sum(t.customer_shipped for t in result.traces))


# -- 2. β in [0, 1] -------------------------------------------------------------
FILL_RATE_CASES: dict[str, Callable[[], DigitalTwin]] = {
    "scripted": lambda: scripted_twin(HORIZON),
    "heuristic_sq-seed42": lambda: stochastic_twin(HeuristicSQPolicy(), seed=42),
    "opaque_dro-seed42": lambda: stochastic_twin(OpaqueDROPolicy(), seed=42),
}


@pytest.mark.parametrize("case", sorted(FILL_RATE_CASES))
def test_fill_rate_is_a_proportion(case: str) -> None:
    """β = customer units served from stock in the same period / customer demand."""
    result = FILL_RATE_CASES[case]().run()

    beta = compute_kpis(result)["operational"]["fill_rate"]
    assert 0.0 <= beta <= 1.0 + TOL, f"fill rate {beta} outside [0, 1]"

    for trace in result.traces:
        assert trace.served_total <= trace.demand_total + TOL, (
            f"period {trace.period}: served {trace.served_total} "
            f"exceeds customer demand {trace.demand_total}"
        )


# -- 3. a backorder reaches the node that ordered it ----------------------------
def test_a_backorder_of_eight_units_reaches_the_downstream_node() -> None:
    """The DC ships 12 at once and owes 8; all 20 must reach the retailer."""
    twin = scripted_twin(HORIZON)
    twin.run()
    retail = twin.states["retail"]
    dc = twin.states["dc"]

    received = retail.inventory + retail.cumulative_served - RETAIL_STOCK
    assert retail.in_transit == pytest.approx(0.0, abs=TOL)
    assert dc.backlog == pytest.approx(0.0, abs=TOL)
    assert received == pytest.approx(RETAIL_ORDER, abs=TOL), (
        f"retailer ordered {RETAIL_ORDER} but received {received}: "
        f"{RETAIL_ORDER - received} units of the {SHORTFALL}-unit backorder never arrived"
    )
