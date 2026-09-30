"""Mutable per-node state and the in-transit pipeline.

`NodeState` is the only mutable object in the simulation. Everything else
(specs, decisions, records) is frozen, so when a value changes it is
unambiguous which object changed it.

The pipeline is a dict keyed by arrival period rather than a list of
shipments. Two consequences: an order placed at period t with lead time L
is settled by a single lookup at t+L, and out-of-order arrivals — which a
stochastic lead time produces — need no sorting pass.
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
    demand_observed: float = 0.0
    demand_served: float = 0.0
    orders_placed: int = 0
    stockout_periods: int = 0
    cumulative_demand: float = 0.0
    cumulative_served: float = 0.0

    # -- pipeline ---------------------------------------------------------
    @property
    def in_transit(self) -> float:
        return float(sum(self.pipeline.values()))

    def schedule_arrival(self, period: int, quantity: float) -> None:
        if quantity <= 0.0:
            return
        self.pipeline[period] = self.pipeline.get(period, 0.0) + quantity

    def receive(self, period: int) -> float:
        """Move everything due at ``period`` into inventory."""
        arriving = self.pipeline.pop(period, 0.0)
        self.inventory += arriving
        return float(arriving)

    # -- demand -----------------------------------------------------------
    def serve(self, demand: float) -> tuple[float, float]:
        """Serve ``demand`` plus outstanding backlog from inventory.

        Backlog is served before new demand: an order already late is more
        costly than one about to be placed, and serving new demand first
        would let a node hide a growing backlog behind a healthy fill rate.

        Returns the quantity served and the quantity added to backlog.
        """
        required = self.backlog + demand
        served = min(self.inventory, required)
        self.inventory -= served
        self.backlog = required - served

        self.demand_observed = demand
        self.demand_served = served
        self.cumulative_demand += demand
        self.cumulative_served += served
        if self.inventory <= 0.0 and self.backlog > 0.0:
            self.stockout_periods += 1
        return float(served), float(self.backlog)

    @property
    def stocked_out(self) -> bool:
        return self.inventory <= 0.0 and self.backlog > 0.0

    # -- projections ------------------------------------------------------
    def operating_state(self, period: int, data_delay: int = 0) -> OperatingState:
        """The observable state x_t handed to policies and to the log."""
        return OperatingState(
            period=period,
            inventory=round(self.inventory, 10),
            backlog=round(self.backlog, 10),
            in_transit=round(self.in_transit, 10),
            capacity=self.capacity,
            demand_observed=round(self.demand_observed, 10),
            data_delay=data_delay,
        )

    def inventory_position(self) -> float:
        """Inventory position: on hand plus in transit minus backlog."""
        return float(self.inventory + self.in_transit - self.backlog)

    def to_payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "inventory": self.inventory,
            "backlog": self.backlog,
            "in_transit": self.in_transit,
            "orders_placed": self.orders_placed,
            "stockout_periods": self.stockout_periods,
            "cumulative_demand": self.cumulative_demand,
            "cumulative_served": self.cumulative_served,
        }
