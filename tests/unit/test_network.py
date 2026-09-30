"""Topology validation: a malformed network must fail at construction."""

from __future__ import annotations

import pytest

from xai_gov.simulation.network import EXTERNAL_SOURCE, NetworkSpec, NodeSpec, build_network


def node(node_id: str, echelon: int, supplier: str, **kwargs: float) -> NodeSpec:
    defaults: dict[str, float] = {
        "initial_inventory": 50.0,
        "capacity": 200.0,
        "reorder_point": 25.0,
        "order_quantity": 40.0,
        "lead_time_mean": 2.0,
    }
    return NodeSpec(node_id=node_id, echelon=echelon, supplier_id=supplier, **(defaults | kwargs))


def test_single_node_network_is_valid() -> None:
    spec = NetworkSpec(nodes=(node("wh", 0, EXTERNAL_SOURCE),))
    assert spec.echelons == 1
    assert spec.demand_facing[0].node_id == "wh"


def test_three_echelon_chain_orders_downstream_first() -> None:
    spec = NetworkSpec(
        nodes=(
            node("plant", 2, EXTERNAL_SOURCE),
            node("retail", 0, "dc"),
            node("dc", 1, "plant"),
        )
    )
    assert [n.node_id for n in spec.downstream_first()] == ["retail", "dc", "plant"]
    assert spec.echelons == 3


def test_supplier_must_sit_one_echelon_upstream() -> None:
    with pytest.raises(ValueError, match="exactly one echelon upstream"):
        NetworkSpec(nodes=(node("retail", 0, "plant"), node("plant", 2, EXTERNAL_SOURCE)))


def test_unknown_supplier_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown supplier"):
        NetworkSpec(nodes=(node("retail", 0, "ghost"),))


def test_duplicate_node_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate node id"):
        NetworkSpec(nodes=(node("a", 0, EXTERNAL_SOURCE), node("a", 0, EXTERNAL_SOURCE)))


def test_network_without_echelon_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="echelon 0"):
        NetworkSpec(nodes=(node("plant", 1, EXTERNAL_SOURCE),))


def test_customers_are_resolvable() -> None:
    spec = NetworkSpec(nodes=(node("retail", 0, "dc"), node("dc", 1, EXTERNAL_SOURCE)))
    assert [n.node_id for n in spec.customers_of("dc")] == ["retail"]


def test_zero_capacity_is_rejected() -> None:
    with pytest.raises(ValueError, match="capacity must be positive"):
        node("wh", 0, EXTERNAL_SOURCE, capacity=0.0)


def test_sub_period_lead_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="lead_time_mean"):
        node("wh", 0, EXTERNAL_SOURCE, lead_time_mean=0.5)


def test_build_network_applies_defaults_and_rejects_typos() -> None:
    config = {
        "name": "two",
        "node_defaults": {
            "initial_inventory": 10.0,
            "capacity": 100.0,
            "reorder_point": 5.0,
            "order_quantity": 10.0,
            "lead_time_mean": 1.0,
        },
        "nodes": [
            {"node_id": "retail", "echelon": 0, "supplier_id": "dc"},
            {"node_id": "dc", "echelon": 1, "supplier_id": EXTERNAL_SOURCE},
        ],
    }
    spec = build_network(config)
    assert spec.node("retail").capacity == 100.0

    typo = {**config, "nodes": [{**config["nodes"][0], "reorderpoint": 3.0}]}  # type: ignore[index]
    with pytest.raises(ValueError, match="unknown keys"):
        build_network(typo)
