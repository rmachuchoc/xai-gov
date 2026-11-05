"""Theorem A1: comparative statics of the intervention threshold (decision 8; R2.6).

The myopic threshold b_m = c / (L + c_false) has exact comparative statics: it
rises with the intervention cost c and falls with the disruption loss L, by
inspection.

The optimal threshold is b* = min{ b : D(b) >= 0 }, with the decision margin

    D(b) = b*L + gamma * Delta(b) - c,    Delta(b) = C(b) - V(0),

where V is the solved value function and C(b) the expected value of the next
period after waiting at belief b. Because c and L also move V, they move
Delta, and the comparative statics of b* do not follow from the margin's
explicit terms alone. They follow under three conditions on the continuation
values, which this module checks numerically on a grid:

* (C1) single crossing: D is nondecreasing in b, so b* is unique;
* (C2) cost: for c1 < c2, gamma * (Delta_c2(b) - Delta_c1(b)) <= c2 - c1 at every
  b, so D falls with c and b* is nondecreasing in c;
* (C3) loss: for L1 < L2, b*(L2 - L1) + gamma * (Delta_L2(b) - Delta_L1(b)) >= 0
  at every b, so D rises with L and b* is nonincreasing in L.

(C2) and (C3) are sufficient, not necessary: b* can be monotone where they
fail, and the observed b* is reported beside them.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Any

from xai_gov.governance.thresholds import (
    BeliefDynamics,
    ThresholdEconomics,
    ValueFunction,
    myopic_threshold,
    solve_values,
)

#: The break-even sweep's costs, and losses around the declared 10.
COST_GRID: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
LOSS_GRID: tuple[float, ...] = (2.5, 5.0, 10.0, 20.0, 40.0)
_TOL = 1e-9


def _single_crossing(solution: ValueFunction) -> tuple[bool, float]:
    margins = [solution.margin(b) for b in solution.grid]
    worst = min((right - left for left, right in pairwise(margins)), default=0.0)
    return worst >= -_TOL, worst


def check_costs(
    costs: tuple[float, ...], *, loss: float, dynamics: BeliefDynamics, discount: float,
    grid_size: int = 201,
) -> dict[str, Any]:
    """(C1) at each cost and (C2) between consecutive costs, at a fixed loss."""
    solutions = [
        solve_values(ThresholdEconomics(intervention_cost=c, disruption_loss=loss,
                                        discount=discount), dynamics, grid_size=grid_size)
        for c in costs
    ]
    steps = []
    for (c1, s1), (c2, s2) in pairwise(zip(costs, solutions, strict=True)):
        slack = min(
            (c2 - c1) - discount * (s2.continuation_gap(b) - s1.continuation_gap(b))
            for b in s1.grid
        )
        steps.append({"from": c1, "to": c2, "min_slack": round(slack, 9),
                      "holds": slack >= -_TOL})
    crossings = [_single_crossing(s) for s in solutions]
    thresholds = [s.threshold() for s in solutions]
    return {
        "loss": loss,
        "costs": list(costs),
        "optimal_threshold": [round(t, 6) for t in thresholds],
        "myopic_threshold": [
            round(myopic_threshold(ThresholdEconomics(intervention_cost=c, disruption_loss=loss)),
                  6) for c in costs
        ],
        "C1_single_crossing": all(ok for ok, _ in crossings),
        "C1_min_increment": round(min(w for _, w in crossings), 9),
        "C2_condition": all(step["holds"] for step in steps),
        "C2_steps": steps,
        "optimal_nondecreasing_in_cost": all(b >= a - _TOL for a, b in pairwise(thresholds)),
    }


def check_losses(
    losses: tuple[float, ...], *, cost: float, dynamics: BeliefDynamics, discount: float,
    grid_size: int = 201,
) -> dict[str, Any]:
    """(C1) at each loss and (C3) between consecutive losses, at a fixed cost."""
    solutions = [
        solve_values(ThresholdEconomics(intervention_cost=cost, disruption_loss=loss,
                                        discount=discount), dynamics, grid_size=grid_size)
        for loss in losses
    ]
    steps = []
    for (l1, s1), (l2, s2) in pairwise(zip(losses, solutions, strict=True)):
        slack = min(
            b * (l2 - l1) + discount * (s2.continuation_gap(b) - s1.continuation_gap(b))
            for b in s1.grid
        )
        steps.append({"from": l1, "to": l2, "min_slack": round(slack, 9),
                      "holds": slack >= -_TOL})
    crossings = [_single_crossing(s) for s in solutions]
    thresholds = [s.threshold() for s in solutions]
    return {
        "cost": cost,
        "losses": list(losses),
        "optimal_threshold": [round(t, 6) for t in thresholds],
        "myopic_threshold": [
            round(myopic_threshold(ThresholdEconomics(intervention_cost=cost, disruption_loss=x)),
                  6) for x in losses
        ],
        "C1_single_crossing": all(ok for ok, _ in crossings),
        "C1_min_increment": round(min(w for _, w in crossings), 9),
        "C3_condition": all(step["holds"] for step in steps),
        "C3_steps": steps,
        "optimal_nonincreasing_in_loss": all(b <= a + _TOL for a, b in pairwise(thresholds)),
    }


def verify(
    *, dynamics: BeliefDynamics | None = None, discount: float = 0.95,
    costs: tuple[float, ...] = COST_GRID, losses: tuple[float, ...] = LOSS_GRID,
    grid_size: int = 201,
) -> dict[str, Any]:
    """Check (C1)-(C3) over the cost grid at every loss, and the loss grid at every cost."""
    dynamics = dynamics or BeliefDynamics()
    by_loss = [check_costs(costs, loss=x, dynamics=dynamics, discount=discount,
                           grid_size=grid_size) for x in losses]
    by_cost = [check_losses(losses, cost=c, dynamics=dynamics, discount=discount,
                            grid_size=grid_size) for c in costs]
    myopic_cost = all(
        all(b >= a for a, b in pairwise(row["myopic_threshold"])) for row in by_loss
    )
    myopic_loss = all(
        all(b <= a for a, b in pairwise(row["myopic_threshold"])) for row in by_cost
    )
    return {
        "theorem": "A1, comparative statics of the intervention threshold",
        "dynamics": {
            "persistence": dynamics.persistence,
            "flag_probability_nominal": dynamics.flag_probability_nominal,
            "flag_probability_disruption": dynamics.flag_probability_disruption,
        },
        "discount": discount,
        "belief_grid": grid_size,
        "myopic": {
            "nondecreasing_in_cost": myopic_cost,
            "nonincreasing_in_loss": myopic_loss,
            "status": "exact: c / (L + c_false) by inspection; checked on the grid",
        },
        "optimal": {
            "C1_single_crossing_everywhere": all(
                r["C1_single_crossing"] for r in (*by_loss, *by_cost)
            ),
            "C2_holds_everywhere": all(r["C2_condition"] for r in by_loss),
            "C3_holds_everywhere": all(r["C3_condition"] for r in by_cost),
            "nondecreasing_in_cost_everywhere": all(
                r["optimal_nondecreasing_in_cost"] for r in by_loss
            ),
            "nonincreasing_in_loss_everywhere": all(
                r["optimal_nonincreasing_in_loss"] for r in by_cost
            ),
            "by_loss": by_loss,
            "by_cost": by_cost,
        },
    }
