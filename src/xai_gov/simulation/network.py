"""Network topology.

The twin is a multi-echelon network, not a single warehouse: the single
warehouse is the degenerate case used for validation. Each node observes
only its own state and the orders its customers place, which is what makes
the system a decentralized POMDP rather than a centralized one.

Echelon 0 is the most downstream (it faces exogenous customer demand);
higher indices are upstream. A node's supplier is a node one echelon up,
or the external source for the topmost echelon, which is modeled as
uncapacitated with a fixed lead time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

EXTERNAL_SOURCE = "__external__"


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Static parameters of one node."""

    node_id: str
    echelon: int
    supplier_id: str
    initial_inventory: float
    capacity: float
    reorder_point: float
    order_quantity: float
    lead_time_mean: float
    lead_time_dispersion: float = 0.0
    holding_cost: float = 1.0
    backlog_cost: float = 5.0
    order_cost: float = 0.5

    def __post_init__(self) -> None:
        if self.echelon < 0:
            raise ValueError(f"{self.node_id}: echelon must be non-negative")
        if self.capacity <= 0.0:
            raise ValueError(f"{self.node_id}: capacity must be positive")
        if self.lead_time_mean < 1.0:
            raise ValueError(f"{self.node_id}: lead_time_mean must be at least one period")

    @property
    def supplies_externally(self) -> bool:
        return self.supplier_id == EXTERNAL_SOURCE

    def to_payload(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "echelon": self.echelon,
            "supplier_id": self.supplier_id,
            "capacity": self.capacity,
            "reorder_point": self.reorder_point,
            "order_quantity": self.order_quantity,
            "lead_time_mean": self.lead_time_mean,
        }


@dataclass(frozen=True)
class NetworkSpec:
    """A validated network: nodes, their links, and the demand entry point."""

    nodes: tuple[NodeSpec, ...]
    name: str = "unnamed"
    _by_id: dict[str, NodeSpec] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.nodes:
            raise ValueError("a network must declare at least one node")
        by_id: dict[str, NodeSpec] = {}
        for node in self.nodes:
            if node.node_id in by_id:
                raise ValueError(f"duplicate node id {node.node_id!r}")
            by_id[node.node_id] = node
        object.__setattr__(self, "_by_id", by_id)

        for node in self.nodes:
            if node.supplies_externally:
                continue
            supplier = by_id.get(node.supplier_id)
            if supplier is None:
                raise ValueError(f"{node.node_id}: unknown supplier {node.supplier_id!r}")
            if supplier.echelon != node.echelon + 1:
                raise ValueError(
                    f"{node.node_id} (echelon {node.echelon}) is supplied by "
                    f"{supplier.node_id} (echelon {supplier.echelon}); a supplier must sit "
                    "exactly one echelon upstream"
                )
        if not self.demand_facing:
            raise ValueError("no node at echelon 0; nothing faces customer demand")

    def node(self, node_id: str) -> NodeSpec:
        try:
            return self._by_id[node_id]
        except KeyError as error:
            raise KeyError(f"unknown node {node_id!r}") from error

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(node.node_id for node in self.nodes)

    @property
    def demand_facing(self) -> tuple[NodeSpec, ...]:
        """Nodes at echelon 0, which receive exogenous demand."""
        return tuple(node for node in self.nodes if node.echelon == 0)

    @property
    def echelons(self) -> int:
        return max(node.echelon for node in self.nodes) + 1

    def downstream_first(self) -> tuple[NodeSpec, ...]:
        """Nodes ordered echelon 0 upward.

        Demand and its propagation are resolved in this order within a
        period, so an order placed downstream is visible upstream in the
        same period: that is what produces the bullwhip effect the
        systemic-risk KPIs measure.
        """
        return tuple(sorted(self.nodes, key=lambda node: (node.echelon, node.node_id)))

    def customers_of(self, node_id: str) -> tuple[NodeSpec, ...]:
        return tuple(node for node in self.nodes if node.supplier_id == node_id)

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "echelons": self.echelons,
            "nodes": [node.to_payload() for node in self.nodes],
        }


def build_network(config: dict[str, Any]) -> NetworkSpec:
    """Construct a network from its configuration block.

    Defaults declared under ``node_defaults`` apply to every node, so a
    three-echelon chain that differs only in reorder point stays legible.
    """
    raw_nodes = config.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("network configuration must declare a non-empty 'nodes' list")
    defaults = config.get("node_defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("network 'node_defaults' must be a mapping")

    specs: list[NodeSpec] = []
    accepted = set(NodeSpec.__dataclass_fields__)
    for index, entry in enumerate(raw_nodes):
        if not isinstance(entry, dict):
            raise ValueError(f"node at position {index} is not a mapping")
        merged: dict[str, Any] = {**defaults, **entry}
        unknown = sorted(set(merged) - accepted)
        if unknown:
            raise ValueError(f"node {merged.get('node_id', index)!r}: unknown keys {unknown}")
        specs.append(NodeSpec(**merged))
    return NetworkSpec(nodes=tuple(specs), name=str(config.get("name", "unnamed")))
