"""Theorem A1: the decision margin and the sufficient conditions."""

from __future__ import annotations

import pytest

from xai_gov.analysis.theorem_a1 import check_costs, check_losses, verify
from xai_gov.governance.thresholds import (
    BeliefDynamics,
    ThresholdEconomics,
    solve_threshold,
    solve_values,
)


def test_the_threshold_is_where_the_margin_first_turns_nonnegative() -> None:
    solution = solve_values(ThresholdEconomics(intervention_cost=1.0, disruption_loss=10.0))
    threshold = solution.threshold()
    index = solution.grid.index(threshold)
    assert solution.margin(threshold) >= 0.0
    assert all(solution.margin(b) < 0.0 for b in solution.grid[:index])
    assert threshold == solve_threshold(
        ThresholdEconomics(intervention_cost=1.0, disruption_loss=10.0)
    ).threshold


def test_the_margin_is_its_definition() -> None:
    economics = ThresholdEconomics(intervention_cost=2.0, disruption_loss=10.0, discount=0.9)
    solution = solve_values(economics, grid_size=51)
    b = 0.3
    expected = b * 10.0 + 0.9 * (solution.continuation(b) - solution.at(0.0)) - 2.0
    assert solution.margin(b) == pytest.approx(expected)


def test_the_conditions_hold_on_the_declared_grid() -> None:
    result = verify(grid_size=51, costs=(0.25, 1.0, 4.0), losses=(5.0, 10.0, 20.0))
    optimal = result["optimal"]
    assert result["myopic"]["nondecreasing_in_cost"] and result["myopic"]["nonincreasing_in_loss"]
    assert optimal["C1_single_crossing_everywhere"]
    assert optimal["C2_holds_everywhere"] and optimal["C3_holds_everywhere"]
    assert optimal["nondecreasing_in_cost_everywhere"]
    assert optimal["nonincreasing_in_loss_everywhere"]


def test_the_steps_report_their_slack() -> None:
    dynamics = BeliefDynamics()
    by_cost = check_costs((0.5, 1.0), loss=10.0, dynamics=dynamics, discount=0.95, grid_size=51)
    by_loss = check_losses((5.0, 10.0), cost=1.0, dynamics=dynamics, discount=0.95, grid_size=51)
    assert by_cost["C2_steps"][0]["min_slack"] >= 0.0
    assert by_loss["C3_steps"][0]["min_slack"] >= 0.0
    assert len(by_cost["optimal_threshold"]) == 2
