"""Mutable per-node state and the in-transit pipeline.

`NodeState` is the only mutable object in the simulation. Everything else
(specs, decisions, records) is frozen, so when a value changes it is
unambiguous which object changed it.

The pipeline is a dict keyed by arrival period rather than a list of
shipments. Two consequences: a shipment sent at period t with lead time L
is settled by a single lookup at t+L, and out-of-order arrivals — which a
stochastic lead time produces — need no sorting pass.

Two quantities are kept apart on purpose. ``in_transit`` is physical
material on its way to the node, and enters the conservation balance.
``on_order`` adds what the node ordered and its supplier has not shipped
yet; it is not material, but it is what the node has asked for, so it is
what the node observes. Observing only ``in_transit`` would make every
policy re-order its own pending orders each period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xai_gov.io.decision_record import OperatingState


@dataclass(slots=True)
class NodeState:
    """The evolving state of one node."""

    node_id: str
    inventory: float
    capacity: float
    backlog: float = 0.0
    pipeline: dict[int, float] = field(default_factory=dict)
    # Ordered from the supplier and not shipped yet (pending or backordered).
    awaiting: float = 0.0
    # Orders placed on this node by its customer, keyed by the period in
    # which this node first sees them: (quantity, expedited).
    orders_due: dict[int, tuple[float, bool]] = field(default_factory=dict)
    demand_observed: float = 0.0
    demand_served: float = 0.0
    served_on_time: float = 0.0
    orders_placed: int = 0
    stockout_periods: int = 0
    cumulative_demand: float = 0.0
    cumulative_served: float = 0.0
    cumulative_served_on_time: float = 0.0
    cumulative_production: float = 0.0

    # -- pipeline ---------------------------------------------------------
    @property
    def in_transit(self) -> float:
        """Material physically on its way to this node."""
        return float(sum(self.pipeline.values()))

    @property
    def on_order(self) -> float:
        """Everything ordered and not yet received: in transit plus awaiting."""
        return float(self.in_transit + self.awaiting)

    def schedule_arrival(self, period: int, quantity: float) -> None:
        if quantity <= 0.0:
            return
        self.pipeline[period] = self.pipeline.get(period, 0.0) + quantity

    def receive(self, period: int) -> float:
        """Move everything due at ``period`` into inventory."""
        arriving = self.pipeline.pop(period, 0.0)
        self.inventory += arriving
        return float(arriving)

    # -- orders from the customer -------------------------------------------
    def book_order(self, period: int, quantity: float, *, expedited: bool) -> None:
        """Register a customer order that this node will see at ``period``."""
        if quantity <= 0.0:
            return
        previous, was_expedited = self.orders_due.get(period, (0.0, False))
        self.orders_due[period] = (previous + quantity, was_expedited or expedited)

    def take_orders(self, period: int) -> tuple[float, bool]:
        """The customer orders due at ``period``, removed from the book."""
        return self.orders_due.pop(period, (0.0, False))

    # -- demand -----------------------------------------------------------
    def serve(self, demand: float) -> tuple[float, float]:
        """Ship ``min(inventory, backlog + demand)``; the rest stays on backorder.

        Backlog is served before new demand (first come, first served): an
        order already late is more costly than one about to be placed, and
        serving new demand first would let a node hide a growing backlog
        behind a healthy fill rate.

        Returns the quantity shipped (late and on-time together) and the
        part of this period's ``demand`` shipped in this same period, which
        is what the fill rate β counts. A backorder shipped late counts in
        the backlog, never in β.
        """
        previous_backlog = self.backlog
        required = previous_backlog + demand
        shipped = min(self.inventory, required)
        on_time = min(max(shipped - previous_backlog, 0.0), demand)
        self.inventory -= shipped
        self.backlog = required - shipped

        self.demand_observed = demand
        self.demand_served = shipped
        self.served_on_time = on_time
        self.cumulative_demand += demand
        self.cumulative_served += shipped
        self.cumulative_served_on_time += on_time
        if self.inventory <= 0.0 and self.backlog > 0.0:
            self.stockout_periods += 1
        return float(shipped), float(on_time)

    @property
    def stocked_out(self) -> bool:
        return self.inventory <= 0.0 and self.backlog > 0.0

    # -- projections ------------------------------------------------------
    def operating_state(self, period: int, data_delay: int = 0) -> OperatingState:
        """The observable state x_t handed to policies and to the log.

        Its ``in_transit`` field is the node's ``on_order``: what it has
        asked for and not received, whether shipped or not.
        """
        return OperatingState(
            period=period,
            inventory=round(self.inventory, 10),
            backlog=round(self.backlog, 10),
            in_transit=round(self.on_order, 10),
            capacity=self.capacity,
            demand_observed=round(self.demand_observed, 10),
            data_delay=data_delay,
        )

    def inventory_position(self) -> float:
        """Inventory position: on hand plus on order minus backlog."""
        return float(self.inventory + self.on_order - self.backlog)

    def to_payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "inventory": self.inventory,
            "backlog": self.backlog,
            "in_transit": self.in_transit,
            "awaiting": self.awaiting,
            "orders_placed": self.orders_placed,
            "stockout_periods": self.stockout_periods,
            "cumulative_demand": self.cumulative_demand,
            "cumulative_served": self.cumulative_served,
            "cumulative_served_on_time": self.cumulative_served_on_time,
            "cumulative_production": self.cumulative_production,
        }
