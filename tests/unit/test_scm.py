"""The structural causal model: the graph, and what it refuses to claim."""

from __future__ import annotations

import pytest

from tests.factories import state
from xai_gov.causal.scm import (
    OBSERVED,
    StructuralCausalModel,
    Variable,
    reorder_rule,
    world_from_state,
)


def test_the_graph_is_acyclic() -> None:
    """A cyclic graph has no counterfactuals to compute."""
    StructuralCausalModel()  # constructing asserts acyclicity


def test_in_transit_is_a_root_not_a_child_of_the_order() -> None:
    """Within one period the order placed now has not arrived.

    Making in_transit a child of order_quantity collapses the lead time and
    creates the cycle order → in transit → position → order. The acyclicity
    check refuses that, correctly: a model with the cycle has no
    counterfactuals to compute. A cross-period twin would carry the lagged
    edge in_transit[t] ← order[t-1]; this is a single-decision model.
    """
    model = StructuralCausalModel()
    assert model.parents[Variable.IN_TRANSIT] == ()
    assert Variable.IN_TRANSIT in model.ancestors(Variable.ORDER_QUANTITY)
    assert Variable.ORDER_QUANTITY not in model.ancestors(Variable.IN_TRANSIT)


def test_a_cycle_is_refused() -> None:
    parents = {
        Variable.INVENTORY: (Variable.BACKLOG,),
        Variable.BACKLOG: (Variable.INVENTORY,),
    }
    with pytest.raises(ValueError, match="cycle"):
        StructuralCausalModel(parents=parents)


def test_intervening_recomputes_descendants() -> None:
    """Setting inventory while leaving inventory position untouched would
    describe a world where the accounting identity is broken."""
    world = world_from_state(state(inventory=40.0, in_transit=10.0, backlog=0.0))
    assert world.get(Variable.INVENTORY_POSITION) == 50.0
    intervened = world.with_intervention({Variable.INVENTORY: 10.0})
    assert intervened.get(Variable.INVENTORY_POSITION) == 20.0


def test_intervention_does_not_mutate_the_original_world() -> None:
    world = world_from_state(state(inventory=40.0))
    world.with_intervention({Variable.INVENTORY: 0.0})
    assert world.get(Variable.INVENTORY) == 40.0


def test_demand_and_backlog_are_not_levers() -> None:
    """An explanation whose recourse is 'have had less demand' is
    correlationally true and operationally useless."""
    model = StructuralCausalModel()
    assert model.is_admissible(Variable.DEMAND, 5.0) is False
    assert model.is_admissible(Variable.BACKLOG, 0.0) is False
    assert model.is_admissible(Variable.INVENTORY, 5.0) is True


def test_worlds_that_cannot_exist_are_refused() -> None:
    model = StructuralCausalModel()
    assert model.is_admissible(Variable.INVENTORY, -1.0) is False
    assert model.is_admissible(Variable.DATA_DELAY, 99.0) is False


def test_only_causally_relevant_levers_are_admissible() -> None:
    """A variable with no path to the decision cannot explain it."""
    model = StructuralCausalModel()
    for lever in model.admissible_interventions():
        assert model.affects_decision(lever), lever


def test_identifiability_is_declared_not_assumed() -> None:
    model = StructuralCausalModel()
    for variable in OBSERVED:
        assert model.is_identifiable(variable) is True
    assert model.is_identifiable(Variable.ORDER_QUANTITY) is False


def test_ancestors_and_descendants_agree() -> None:
    model = StructuralCausalModel()
    assert Variable.INVENTORY in model.ancestors(Variable.ORDER_QUANTITY)
    assert Variable.ORDER_QUANTITY in model.descendants(Variable.INVENTORY)


def test_the_model_matches_the_engine_state() -> None:
    """A silent divergence here is this project's most dangerous failure: every
    downstream number stays well formed while describing a mechanism the twin
    does not run."""
    engine_fields = frozenset(state().to_payload())
    StructuralCausalModel().assert_matches_engine(engine_fields)


def test_drift_from_the_engine_is_caught() -> None:
    model = StructuralCausalModel()
    with pytest.raises(ValueError, match="the engine does not expose"):
        model.assert_matches_engine(frozenset({"inventory"}))


def test_the_reorder_rule_fires_below_the_reorder_point() -> None:
    rule = reorder_rule(reorder_point=30.0, order_quantity=25.0)
    assert rule(world_from_state(state(inventory=40.0, in_transit=10.0))) == 0.0
    assert rule(world_from_state(state(inventory=5.0, in_transit=0.0))) == 25.0


def test_the_order_is_capped_by_capacity() -> None:
    rule = reorder_rule(reorder_point=30.0, order_quantity=100.0)
    assert rule(world_from_state(state(inventory=0.0, in_transit=0.0, capacity=15.0))) == 15.0


def test_the_payload_records_the_declared_graph() -> None:
    payload = StructuralCausalModel().to_payload()
    assert "order_quantity" in payload["edges"]
    assert "inventory" in payload["observed"]
    assert payload["admissible_interventions"]
