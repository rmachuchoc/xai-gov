"""The digital twin: a serial supply chain with backorders.

The period step is the standard serial model with backorders (the beer
game; Lee et al.). Within period t, nodes are visited from the retailer
upstream, and each node j:

1. receives the shipments due at t;
2. sees its demand: customer demand at the retailer, and elsewhere the
   order its downstream node placed at t − 1;
3. ships ``min(inventory, backlog + demand)`` into the downstream node's
   pipeline with that node's lead time (to the customer at the retailer);
   the rest stays on backorder;
4. decides its order (policy → governance → shield), seals the decision
   record, and books the order as the supplier's demand for t + 1.
   Placing an order moves no material: the supplier ships it when it
   sees it, from what it has on hand;
5. at the plant (the node supplied externally) the order is a production
   release, capped by capacity, that lands after the production time.

Every period the balance holds: initial inventory + cumulative production =
inventory at nodes + material in transit + cumulative sales to the customer.

The fill rate β counts only what the retailer ships to the customer from
stock in the same period the demand arrives (``PeriodTrace.served_total``
over ``demand_total``), so it lies in [0, 1] by construction. A backorder
filled late counts in the customer backlog, not in β. Upstream shipments
are flows inside the chain and never enter it.

The decision record is written *before* the order is booked, so a crash
can leave an unexecuted record but never an unrecorded execution. The
traceability contract is one-directional on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from xai_gov.core.logging import get_logger
from xai_gov.io.decision_record import ActionKind, DecisionRecord
from xai_gov.simulation.disruptions import DisruptionSchedule
from xai_gov.simulation.network import EXTERNAL_SOURCE
from xai_gov.simulation.state import NodeState

if TYPE_CHECKING:  # The twin is handed its policy and governance agent; it
    # does not import them. Depending on those layers at runtime would make
    # the package cyclic (engine -> governance -> policies -> simulation) and,
    # more importantly, would let the simulation reach into the authorities
    # that are supposed to constrain it.
    import numpy as np

    from xai_gov.core.seeds import SeedBundle
    from xai_gov.governance.agent import GovernanceAgent
    from xai_gov.policies.base import Policy
    from xai_gov.simulation.demand import DemandProcess
    from xai_gov.simulation.network import NetworkSpec, NodeSpec

_LOG = get_logger("simulation.engine")


@dataclass(frozen=True, slots=True)
class PeriodTrace:
    """Aggregate outcome of one period, for the KPI layers."""

    period: int
    demand_total: float
    served_total: float
    orders_total: float
    inventory_total: float
    backlog_total: float
    stocked_out_nodes: int
    holding_cost: float
    backlog_cost: float
    order_cost: float
    intervention_cost: float
    # Retailer only: everything shipped to the customer (on time or late),
    # and the customer backorders outstanding at the close of the period.
    customer_shipped: float = 0.0
    customer_backlog: float = 0.0
    production_total: float = 0.0
    in_transit_total: float = 0.0

    @property
    def logistics_cost(self) -> float:
        return self.holding_cost + self.backlog_cost + self.order_cost

    def to_payload(self) -> dict[str, Any]:
        return {
            "period": self.period,
            "demand_total": self.demand_total,
            "served_total": self.served_total,
            "orders_total": self.orders_total,
            "inventory_total": self.inventory_total,
            "backlog_total": self.backlog_total,
            "stocked_out_nodes": self.stocked_out_nodes,
            "logistics_cost": self.logistics_cost,
            "intervention_cost": self.intervention_cost,
            "customer_shipped": self.customer_shipped,
            "customer_backlog": self.customer_backlog,
            "production_total": self.production_total,
            "in_transit_total": self.in_transit_total,
        }


@dataclass
class SimulationResult:
    """Everything one run produces, before KPI aggregation."""

    periods: int
    traces: list[PeriodTrace] = field(default_factory=list)
    records: list[DecisionRecord] = field(default_factory=list)
    final_states: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def demand_series(self) -> list[float]:
        return [trace.demand_total for trace in self.traces]

    @property
    def order_series(self) -> list[float]:
        return [trace.orders_total for trace in self.traces]


class DigitalTwin:
    """The simulator: a network, a demand process, policies and governance."""

    def __init__(
        self,
        *,
        network: NetworkSpec,
        demand: DemandProcess,
        policy: Policy,
        governance: GovernanceAgent,
        seeds: SeedBundle,
        periods: int = 30,
        schedule: DisruptionSchedule | None = None,
        data_delay: int = 0,
    ) -> None:
        if periods < 1:
            raise ValueError("a run must cover at least one period")
        self.network = network
        self.demand = demand
        self.policy = policy
        self.governance = governance
        self.seeds = seeds
        self.periods = periods
        self.schedule = schedule or DisruptionSchedule()
        self.data_delay = max(int(data_delay), 0)
        self.states: dict[str, NodeState] = {
            spec.node_id: NodeState(
                node_id=spec.node_id,
                inventory=spec.initial_inventory,
                capacity=spec.capacity,
            )
            for spec in network.nodes
        }
        self._customer: dict[str, NodeSpec] = {}
        for spec in network.nodes:
            customers = network.customers_of(spec.node_id)
            if len(customers) > 1:
                raise ValueError(
                    f"{spec.node_id} supplies {len(customers)} nodes; the twin models a "
                    "serial chain, where every supplier has exactly one customer"
                )
            if customers:
                self._customer[spec.node_id] = customers[0]

    # -- random streams ---------------------------------------------------
    def _rng(self, label: str) -> np.random.Generator:
        return self.seeds.generator(label)

    def _lead_time(self, spec: NodeSpec, period: int) -> int:
        """Realized lead time, in whole periods, never below one."""
        multiplier = self.schedule.lead_time_multiplier(period, spec.node_id)
        mean = spec.lead_time_mean * multiplier
        if spec.lead_time_dispersion <= 0.0:
            return max(round(mean), 1)
        rng = self._rng(f"lead_time::{spec.node_id}")
        shape = 1.0 / (spec.lead_time_dispersion**2)
        drawn = float(rng.gamma(shape=shape, scale=mean / shape))
        return max(round(drawn), 1)

    # -- the loop ---------------------------------------------------------
    def run(self) -> SimulationResult:
        result = SimulationResult(periods=self.periods)
        for period in range(self.periods):
            result.traces.append(self._step(period, result.records))
        result.final_states = {
            node_id: state.to_payload() for node_id, state in self.states.items()
        }
        _LOG.info(
            "simulation finished",
            extra={
                "periods": self.periods,
                "decisions": len(result.records),
                "policy": self.policy.policy_id,
                "governance": self.governance.agent_id,
            },
        )
        return result

    def _step(self, period: int, records: list[DecisionRecord]) -> PeriodTrace:
        demand_total = served_total = orders_total = 0.0
        customer_shipped = production_total = 0.0
        holding = backlog_cost = order_cost = intervention_cost = 0.0

        for spec in self.network.downstream_first():
            state = self.states[spec.node_id]
            state.receive(period)
            state.capacity = spec.capacity * self.schedule.capacity_multiplier(
                period, spec.node_id
            )

            demand, expedited = self._demand_for(spec, state, period)
            shipped, on_time = state.serve(demand)
            if spec.echelon == 0:
                demand_total += demand
                served_total += on_time
                customer_shipped += shipped
            else:
                self._ship_downstream(spec, shipped, period, expedited=expedited)

            observed = state.operating_state(period, data_delay=self.data_delay)
            proposal = self.policy.propose(
                spec=spec, state=observed, rng=self._rng(f"policy::{spec.node_id}")
            )
            verdict = self.governance.mediate(spec=spec, state=observed, proposal=proposal)

            period_holding = spec.holding_cost * max(state.inventory, 0.0)
            period_backlog = spec.backlog_cost * state.backlog
            period_order = spec.order_cost * verdict.final_quantity
            holding += period_holding
            backlog_cost += period_backlog
            order_cost += period_order
            intervention_cost += verdict.intervention_cost

            records.append(
                DecisionRecord(
                    period=period,
                    node_id=spec.node_id,
                    state=observed,
                    policy_id=self.policy.policy_id,
                    policy_version=self.policy.version,
                    proposed_action=proposal.action,
                    proposed_quantity=proposal.quantity,
                    descriptor=verdict.descriptor,
                    conformal=verdict.conformal,
                    belief=verdict.belief,
                    governance_action=verdict.action,
                    governance_rationale=verdict.rationale,
                    autonomy_regime=self.governance.autonomy_regime,
                    escalated=verdict.escalated,
                    safe_mode=verdict.safe_mode,
                    intervention_cost=verdict.intervention_cost,
                    shield=verdict.shield,
                    final_action=verdict.final_action,
                    final_quantity=verdict.final_quantity,
                    operating_cost=round(period_holding + period_backlog + period_order, 10),
                )
            )
            placed, produced = self._place_order(
                spec, state, period, verdict.final_action, verdict.final_quantity
            )
            orders_total += placed
            production_total += produced

        retailers = self.network.demand_facing
        return PeriodTrace(
            period=period,
            demand_total=round(demand_total, 10),
            served_total=round(served_total, 10),
            orders_total=round(orders_total, 10),
            inventory_total=round(sum(s.inventory for s in self.states.values()), 10),
            backlog_total=round(sum(s.backlog for s in self.states.values()), 10),
            stocked_out_nodes=sum(1 for s in self.states.values() if s.stocked_out),
            holding_cost=round(holding, 10),
            backlog_cost=round(backlog_cost, 10),
            order_cost=round(order_cost, 10),
            intervention_cost=round(intervention_cost, 10),
            customer_shipped=round(customer_shipped, 10),
            customer_backlog=round(sum(self.states[r.node_id].backlog for r in retailers), 10),
            production_total=round(production_total, 10),
            in_transit_total=round(sum(s.in_transit for s in self.states.values()), 10),
        )

    def _demand_for(
        self, spec: NodeSpec, state: NodeState, period: int
    ) -> tuple[float, bool]:
        """Customer demand at echelon 0; last period's order elsewhere.

        Upstream nodes see demand only through what their customer ordered,
        never through the true customer process. That informational
        separation is what makes the network a decentralized POMDP. The
        flag says whether the order being filled was expedited.
        """
        if spec.echelon != 0:
            return state.take_orders(period)
        rng = self._rng(f"demand::{spec.node_id}")
        drawn = self.demand.draw(period, rng)
        return drawn * self.schedule.demand_multiplier(period, spec.node_id), False

    def _ship_downstream(
        self, spec: NodeSpec, quantity: float, period: int, *, expedited: bool
    ) -> None:
        """Put a shipment into the customer node's pipeline with its lead time."""
        if quantity <= 0.0:
            return
        customer = self._customer[spec.node_id]
        lead_time = self._lead_time(customer, period)
        # Expediting buys one period on the shipment that answers the order;
        # a remainder left on backorder ships later at the normal lead time.
        if expedited:
            lead_time = max(lead_time - 1, 1)
        customer_state = self.states[customer.node_id]
        customer_state.awaiting -= quantity
        customer_state.schedule_arrival(period + lead_time, quantity)

    def _place_order(
        self,
        spec: NodeSpec,
        state: NodeState,
        period: int,
        action: ActionKind,
        quantity: float,
    ) -> tuple[float, float]:
        """Book the executed order; returns (quantity ordered, quantity produced).

        An internal order only becomes the supplier's demand for the next
        period: no material moves until the supplier ships it. At the plant
        the order is a production release, capped by this period's capacity,
        that arrives after the production time.
        """
        if action in (ActionKind.HOLD, ActionKind.CANCEL) or quantity <= 0.0:
            return 0.0, 0.0
        if self.schedule.supply_halted(period, spec.node_id):
            _LOG.debug("supply halted", extra={"node": spec.node_id, "period": period})
            return 0.0, 0.0

        expedited = action is ActionKind.EXPEDITE
        state.orders_placed += 1

        if spec.supplier_id == EXTERNAL_SOURCE:
            produced = min(quantity, max(state.capacity, 0.0))
            lead_time = self._lead_time(spec, period)
            if expedited:
                lead_time = max(lead_time - 1, 1)
            state.schedule_arrival(period + lead_time, produced)
            state.cumulative_production += produced
            return float(quantity), float(produced)

        self.states[spec.supplier_id].book_order(period + 1, quantity, expedited=expedited)
        state.awaiting += quantity
        return float(quantity), 0.0
