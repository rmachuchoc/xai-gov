"""Theorem 2 as executable claims.

The protocol asserts that the optimal governance policy is of threshold type
and that the threshold moves in declared directions. These tests are that
assertion, checked against the solver. If the theorem is wrong, or the solver
does not implement it, this file fails.
"""

from __future__ import annotations

import pytest

from xai_gov.governance.thresholds import (
    ThresholdEconomics,
    build_economics,
    myopic_threshold,
    solve_threshold,
)


def economics(**kwargs: float) -> ThresholdEconomics:
    defaults: dict[str, float] = {
        "intervention_cost": 1.0,
        "disruption_loss": 10.0,
        "discount": 0.95,
    }
    return ThresholdEconomics(**(defaults | kwargs))


def test_myopic_threshold_is_the_cost_loss_ratio() -> None:
    assert myopic_threshold(economics(intervention_cost=1.0, disruption_loss=10.0)) == 0.1
    assert myopic_threshold(economics(intervention_cost=5.0, disruption_loss=10.0)) == 0.5


def test_a_cost_above_the_loss_it_prevents_never_justifies_intervening() -> None:
    """A finding, not an error: the threshold saturates at 1."""
    assert myopic_threshold(economics(intervention_cost=50.0, disruption_loss=10.0)) == 1.0


def test_the_solved_threshold_is_interior_and_converges() -> None:
    solution = solve_threshold(economics())
    assert solution.converged is True
    assert 0.0 < solution.threshold < 1.0


def test_accounting_for_continuation_value_never_delays_intervention() -> None:
    """The solved policy must act no later than the myopic one: knowing the
    belief will keep rising can only make intervening more attractive."""
    for cost in (0.5, 1.0, 2.0, 4.0):
        solution = solve_threshold(economics(intervention_cost=cost))
        assert solution.anticipation_gain >= -1e-9, cost


def test_theorem_2_the_threshold_rises_with_the_cost_of_intervening() -> None:
    thresholds = [
        solve_threshold(economics(intervention_cost=cost)).threshold
        for cost in (0.25, 0.5, 1.0, 2.0, 4.0)
    ]
    assert thresholds == sorted(thresholds)
    assert thresholds[-1] > thresholds[0]


def test_theorem_2_the_threshold_falls_with_the_severity_of_the_loss() -> None:
    thresholds = [
        solve_threshold(economics(disruption_loss=loss)).threshold
        for loss in (5.0, 10.0, 20.0, 50.0)
    ]
    assert thresholds == sorted(thresholds, reverse=True)
    assert thresholds[0] > thresholds[-1]


def test_a_finer_grid_does_not_move_the_threshold_materially() -> None:
    """The scalar-belief collapse is an approximation; its discretization must
    not be one as well."""
    coarse = solve_threshold(economics(), grid_size=51).threshold
    fine = solve_threshold(economics(), grid_size=401).threshold
    assert abs(coarse - fine) < 0.05


def test_an_uninformative_detector_makes_the_agent_act_sooner() -> None:
    """A vague detector lowers the threshold, and that is the correct economics.

    The value of waiting is the value of what waiting *teaches*. When flags
    carry almost no information about the regime (0.40 versus 0.50), another
    period of observation will not sharpen the belief, so there is nothing to
    wait for and acting early dominates. The intuitive expectation — that a
    poor detector should make the agent more reluctant — has the causality
    backwards: reluctance is only rational when patience is informative.
    """
    informative = solve_threshold(
        economics(), flag_probability_nominal=0.02, flag_probability_disruption=0.95
    )
    vague = solve_threshold(
        economics(), flag_probability_nominal=0.40, flag_probability_disruption=0.50
    )
    assert vague.threshold <= informative.threshold


def test_the_solution_records_what_produced_it() -> None:
    payload = solve_threshold(economics()).to_payload()
    assert payload["method"] == "value_iteration_on_collapsed_belief"
    assert payload["economics"]["disruption_loss"] == 10.0
    assert payload["converged"] is True


def test_invalid_economics_are_refused() -> None:
    with pytest.raises(ValueError, match="disruption_loss"):
        economics(disruption_loss=0.0)
    with pytest.raises(ValueError, match="discount"):
        economics(discount=1.0)
    with pytest.raises(ValueError, match="intervention_cost"):
        economics(intervention_cost=-1.0)


def test_invalid_solver_arguments_are_refused() -> None:
    with pytest.raises(ValueError, match="grid_size"):
        solve_threshold(economics(), grid_size=5)
    with pytest.raises(ValueError, match="persistence"):
        solve_threshold(economics(), persistence=1.0)
    with pytest.raises(ValueError, match="flag probabilities"):
        solve_threshold(
            economics(), flag_probability_nominal=0.9, flag_probability_disruption=0.1
        )


def test_registry_builds_defaults() -> None:
    assert build_economics({}).disruption_loss == 10.0
    with pytest.raises(ValueError, match="rejected its parameters"):
        build_economics({"nonsense": 1.0})
