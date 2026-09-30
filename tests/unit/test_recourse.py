"""Causal recourse: leverage, affordability, and what is excluded."""

from __future__ import annotations

import pytest

from tests.factories import state
from xai_gov.causal.recourse import RecourseSearch, build_recourse_search
from xai_gov.causal.scm import Variable, reorder_rule, world_from_state


def rule(order_quantity: float = 25.0, reorder_point: float = 30.0):  # type: ignore[no-untyped-def]
    return reorder_rule(reorder_point=reorder_point, order_quantity=order_quantity)


def test_a_lever_that_changes_the_decision_is_found() -> None:
    """Just above the reorder point, dropping inventory triggers an order."""
    world = world_from_state(state(inventory=35.0, in_transit=0.0))
    result = RecourseSearch().search(world, rule())
    assert result.actionability_score > 0.0
    assert result.cheapest is not None
    assert result.cheapest.changed_decision is True


def test_leverage_is_measured_in_outcomes_not_interventions() -> None:
    """A hundred routes to the same order is one option, not a hundred."""
    world = world_from_state(state(inventory=35.0, in_transit=0.0))
    result = RecourseSearch().search(world, rule())
    changing = [cf for cf in result.counterfactuals if cf.changed_decision]
    assert len(result.reachable_quantities) <= len(changing)
    assert result.actionability_score <= 1.0


def test_an_unaffordable_recourse_is_excluded() -> None:
    """A world nobody can afford to reach is not recourse."""
    world = world_from_state(state(inventory=35.0, in_transit=0.0))
    generous = RecourseSearch(budget=100.0).search(world, rule())
    stingy = RecourseSearch(budget=2.0).search(world, rule())
    assert stingy.actionability_score <= generous.actionability_score
    assert all(cf.cost <= 2.0 for cf in stingy.counterfactuals)


def test_reductions_are_not_free() -> None:
    """Reversing an operational commitment costs something; pricing reductions
    at zero would make the search prefer them for no practitioner's reason."""
    search = RecourseSearch()
    assert search.cost_of(Variable.CAPACITY, -10.0) == search.cost_of(Variable.CAPACITY, 10.0)
    assert search.cost_of(Variable.CAPACITY, -10.0) > 0.0


def test_unit_costs_order_the_cheapest_recourse() -> None:
    """Inventory is cheaper than capacity, so it should be offered first."""
    world = world_from_state(state(inventory=35.0, in_transit=0.0))
    result = RecourseSearch().search(world, rule())
    cheapest = result.cheapest
    assert cheapest is not None
    assert "capacity" not in cheapest.interventions


def test_no_recourse_exists_when_the_decision_is_insensitive() -> None:
    """Deep in stockout every affordable lever still orders the same amount, so
    the honest answer is that the operator has no leverage here."""
    world = world_from_state(state(inventory=0.0, in_transit=0.0, backlog=100.0, capacity=5.0))
    result = RecourseSearch(steps=(5.0, 10.0)).search(world, rule())
    assert result.cheapest is None or result.actionability_score < 1.0


def test_interventional_effects_cover_every_admissible_lever() -> None:
    world = world_from_state(state(inventory=35.0, in_transit=0.0))
    effects = RecourseSearch().interventional_effects(world, rule())
    expected = {v.value for v in RecourseSearch().model.admissible_interventions()}
    assert set(effects) == expected


def test_effects_are_probed_at_the_scale_the_lever_moves() -> None:
    """The reorder mechanism is a threshold, so a unit probe crosses it almost
    never and would report every lever as having no effect — which makes every
    descriptor uninformative and every recourse hypothesis untestable."""
    world = world_from_state(state(inventory=35.0, in_transit=0.0))
    effects = RecourseSearch().interventional_effects(world, rule())
    assert any(value != 0.0 for value in effects.values())


def test_a_lever_beyond_the_budget_is_not_credited() -> None:
    """An effect reachable only by an unaffordable change is not an effect the
    agent can act on."""
    world = world_from_state(state(inventory=35.0, in_transit=0.0))
    generous = RecourseSearch(budget=100.0).interventional_effects(world, rule())
    stingy = RecourseSearch(budget=1.0).interventional_effects(world, rule())
    assert sum(abs(v) for v in stingy.values()) <= sum(abs(v) for v in generous.values())


def test_effects_are_zero_where_the_decision_does_not_move() -> None:
    """Far above the reorder point a unit change orders nothing either way."""
    world = world_from_state(state(inventory=500.0, in_transit=0.0))
    effects = RecourseSearch().interventional_effects(world, rule())
    assert all(value == 0.0 for value in effects.values())


def test_data_delay_reaches_the_decision() -> None:
    """The causal graph declares data_delay as a parent of the order. A rule
    that ignored it would leave the SCM describing an edge the mechanism does
    not implement — the drift the model-engine check exists to prevent."""
    fresh = world_from_state(state(inventory=40.0, in_transit=10.0, data_delay=0))
    stale = world_from_state(state(inventory=40.0, in_transit=10.0, data_delay=3))
    decide = rule()
    assert decide(fresh) != decide(stale)


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="budget must be positive"):
        RecourseSearch(budget=0.0)
    with pytest.raises(ValueError, match="at least one intervention step"):
        RecourseSearch(steps=())
    with pytest.raises(ValueError, match="unit costs must be positive"):
        RecourseSearch(unit_costs={Variable.INVENTORY: 0.0})


def test_registry_builds_from_configuration() -> None:
    search = build_recourse_search(
        {"budget": 10.0, "steps": [5.0], "unit_costs": {"inventory": 2.0}}
    )
    assert search.budget == 10.0
    assert search.steps == (5.0,)
    assert search.unit_costs[Variable.INVENTORY] == 2.0
    with pytest.raises(ValueError, match="rejected its parameters"):
        build_recourse_search({"budgett": 10.0})
