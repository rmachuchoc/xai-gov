"""The network diagram: documentation generated from the configuration."""

from __future__ import annotations

import re
from pathlib import Path

from xai_gov.analysis.figures import load_network, network_diagram

THREE = [
    {"node_id": "plant", "echelon": 2, "initial_inventory": 120.0, "capacity": 150.0,
     "reorder_point": 60.0, "lead_time_mean": 4.0, "lead_time_dispersion": 1.0},
    {"node_id": "distribution", "echelon": 1, "initial_inventory": 80.0,
     "capacity": 100.0, "reorder_point": 45.0, "lead_time_mean": 3.0,
     "lead_time_dispersion": 0.8},
    {"node_id": "retail", "echelon": 0, "initial_inventory": 40.0, "capacity": 60.0,
     "reorder_point": 30.0, "lead_time_mean": 2.0, "lead_time_dispersion": 0.5},
]


def test_every_node_appears() -> None:
    svg = network_diagram(THREE)
    for node in THREE:
        assert str(node["node_id"]) in svg


def test_nodes_are_ordered_upstream_to_downstream() -> None:
    """Read left to right the chain must run plant to retail, whatever order
    the configuration listed them in."""
    svg = network_diagram(THREE)
    assert svg.index("plant") < svg.index("distribution") < svg.index("retail")


def test_both_flow_directions_are_drawn() -> None:
    """The paper's mechanism is upstream starvation, so a single-arrow diagram
    would make the propagation path invisible."""
    svg = network_diagram(THREE)
    assert "arrow-fwd" in svg
    assert "arrow-back" in svg
    assert "material" in svg and "orders" in svg


def test_the_parameters_are_annotated() -> None:
    """The figure documents the parameterization rather than illustrating it."""
    svg = network_diagram(THREE)
    assert "inv 40" in svg
    assert "cap 60" in svg
    assert "s 30" in svg
    assert "L 2" in svg


def test_the_system_boundary_is_shown() -> None:
    """Demand enters, disruptions enter: without both the diagram omits the
    inputs the reviewer asked to see."""
    svg = network_diagram(THREE)
    assert "demand" in svg
    assert "disruptions" in svg


def test_an_empty_network_yields_nothing() -> None:
    assert network_diagram([]) == ""


def test_a_single_node_network_still_renders() -> None:
    """The degenerate topology is a declared arm, so it must not crash on the
    flow arrows it has no room for."""
    svg = network_diagram([THREE[2]])
    assert "retail" in svg
    assert svg.startswith("<svg")


def test_a_node_missing_parameters_omits_them_rather_than_defaulting() -> None:
    """Checked against the annotation pattern, not the substring: the legend
    spells out 'inv initial inventory', so a bare containment test passes on the
    legend and says nothing about whether a value was invented."""
    annotation = re.compile(r">(inv|cap|s) -?\d")
    assert annotation.search(network_diagram([{"node_id": "bare", "echelon": 0}])) is None
    assert annotation.search(network_diagram(THREE)) is not None


def test_arm_names_are_escaped() -> None:
    svg = network_diagram([{"node_id": "a<b>", "echelon": 0}])
    assert "&lt;b&gt;" in svg
    assert "<b>" not in svg


def test_the_shipped_configurations_load(project_root: Path) -> None:
    """The diagram and the simulator read the same file through the same loader;
    a second parser is a second thing that can disagree."""
    for name, expected in (("three_echelon", 3), ("five_echelon", 5), ("single_echelon", 1)):
        path = project_root / "configs" / "network" / f"{name}.yaml"
        if not path.exists():
            continue
        nodes = load_network(path)
        assert len(nodes) == expected, name
        assert network_diagram(nodes).startswith("<svg")


def test_shared_defaults_reach_the_diagram(project_root: Path) -> None:
    """A parameter declared once for every node is still that node's parameter.
    A diagram reading only the per-node block would omit exactly the values the
    configuration factored out."""
    path = project_root / "configs" / "network" / "three_echelon.yaml"
    if not path.exists():
        return
    nodes = load_network(path)
    # lead_time_dispersion lives in node_defaults, not on any node.
    assert all("lead_time_dispersion" in node for node in nodes)
    assert "±" in network_diagram(nodes)
