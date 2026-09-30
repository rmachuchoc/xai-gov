"""The Pareto frontier of governance.

Theorem 3 says no policy can simultaneously guarantee coverage at the nominal
level, hold the intervention budget, and avoid degrading operational return.
This module turns that impossibility into the deliverable it implies.

If the three cannot be had together, then the practical contribution is not an
optimal design — there is none — but the *frontier*: the set of achievable
combinations, and the shape of what each one costs. A reader choosing an
operating point needs to see what they give up, and that is a picture of a
surface rather than a recommendation.

The frontier is computed by non-domination over campaign arms. An arm is on the
frontier when no other arm is at least as good on every objective and strictly
better on one. Two details matter for honesty:

* **Objectives declare their direction.** Service level is maximized, cost
  minimized. Getting this backwards produces a frontier that looks perfectly
  reasonable and is exactly inverted.
* **An arm missing an objective is excluded, not defaulted.** Substituting a
  zero for a KPI a run never produced would place a phantom point on the
  frontier, and the frontier is the one artifact where a fabricated point does
  the most damage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Objective:
    """One axis of the frontier."""

    name: str
    indicator: str
    maximize: bool
    description: str = ""

    def better(self, left: float, right: float) -> bool:
        return left > right if self.maximize else left < right

    def at_least_as_good(self, left: float, right: float) -> bool:
        return left >= right if self.maximize else left <= right

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "indicator": self.indicator,
            "direction": "maximize" if self.maximize else "minimize",
            "description": self.description,
        }


#: The three quantities Theorem 3 says cannot be jointly optimized.
THEOREM_3_OBJECTIVES: tuple[Objective, ...] = (
    Objective(
        name="safety_coverage",
        indicator="guarantees.coverage_error",
        maximize=False,
        description="distance between realized and nominal coverage (CCR)",
    ),
    Objective(
        name="governance_budget",
        indicator="governance.total_intervention_cost",
        maximize=False,
        description="what oversight cost over the run",
    ),
    Objective(
        name="operational_return",
        indicator="operational.service_level",
        maximize=True,
        description="service level attained",
    ),
)


@dataclass(frozen=True, slots=True)
class FrontierPoint:
    """One arm's position in objective space."""

    arm: str
    values: dict[str, float]
    dominated_by: tuple[str, ...] = ()

    @property
    def on_frontier(self) -> bool:
        return not self.dominated_by

    def to_payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "values": {k: round(v, 6) for k, v in sorted(self.values.items())},
            "on_frontier": self.on_frontier,
            "dominated_by": list(self.dominated_by),
        }


@dataclass(frozen=True, slots=True)
class ParetoFrontier:
    """The achievable set, and what it says about Theorem 3."""

    objectives: tuple[Objective, ...]
    points: tuple[FrontierPoint, ...]
    excluded: dict[str, str]

    @property
    def frontier_arms(self) -> tuple[str, ...]:
        return tuple(point.arm for point in self.points if point.on_frontier)

    def best_on(self, objective_name: str) -> str | None:
        """The arm that does best on one objective, ignoring the rest."""
        objective = next(
            (o for o in self.objectives if o.name == objective_name), None
        )
        if objective is None or not self.points:
            return None
        ranked = [p for p in self.points if objective.name in p.values]
        if not ranked:
            return None
        return min(
            ranked,
            key=lambda p: p.values[objective.name] * (-1 if objective.maximize else 1),
        ).arm

    @property
    def single_arm_dominates_all(self) -> bool:
        """Whether one arm is best on every objective at once.

        If this is ever True, Theorem 3 has a counterexample in the data and
        either the theorem or the implementation is wrong. It is surfaced as a
        flag rather than left to be noticed, because a result that contradicts
        the project's own theory must not pass quietly.
        """
        if len(self.points) < 2 or not self.objectives:
            return False
        winners = {self.best_on(o.name) for o in self.objectives}
        winners.discard(None)
        return len(winners) == 1

    def trade_off(self, first: str, second: str) -> float | None:
        """Signed exchange rate between two objectives along the frontier.

        How much of ``second`` one unit of ``first`` costs. The sign is the
        content: a negative rate means buying one objective spends the other,
        which is what a trade-off *is*. Computing it from the spread of each
        objective independently would make every rate positive and silently
        turn every trade-off into an apparent alignment.

        Estimated as the slope between the two arms furthest apart on the first
        objective, which is the frontier's overall exchange rate rather than a
        local one.
        """
        pairs = [
            (p.values[first], p.values[second])
            for p in self.points
            if p.on_frontier and first in p.values and second in p.values
        ]
        if len(pairs) < 2:
            return None
        low, high = min(pairs, key=lambda p: p[0]), max(pairs, key=lambda p: p[0])
        run = high[0] - low[0]
        if abs(run) < 1e-12:
            return None
        return float((high[1] - low[1]) / run)

    def to_payload(self) -> dict[str, Any]:
        return {
            "objectives": [o.to_payload() for o in self.objectives],
            "points": [p.to_payload() for p in self.points],
            "frontier_arms": list(self.frontier_arms),
            "excluded_arms": dict(sorted(self.excluded.items())),
            # The finding the frontier exists to deliver: governance has no
            # universally optimal setting, only a set of informed choices.
            "theorem_3_consistent": not self.single_arm_dominates_all,
            "interpretation": (
                "no arm is best on every objective; the frontier is the set of "
                "informed choices"
                if not self.single_arm_dominates_all
                else "one arm dominates every objective, which contradicts Theorem 3 "
                "and must be investigated before the result is reported"
            ),
        }


def compute_frontier(
    arm_values: dict[str, dict[str, float]],
    objectives: tuple[Objective, ...] = THEOREM_3_OBJECTIVES,
) -> ParetoFrontier:
    """Find the non-dominated arms in objective space."""
    if not objectives:
        raise ValueError("a frontier needs at least one objective")

    required = {objective.name for objective in objectives}
    usable: dict[str, dict[str, float]] = {}
    excluded: dict[str, str] = {}
    for arm, values in arm_values.items():
        missing = sorted(required - set(values))
        if missing:
            # Not defaulted to zero: a phantom point on the frontier is the
            # most damaging fabrication this analysis could make.
            excluded[arm] = f"missing objectives {missing}"
            continue
        usable[arm] = values

    points: list[FrontierPoint] = []
    for arm, values in sorted(usable.items()):
        dominators = tuple(
            sorted(
                other
                for other, other_values in usable.items()
                if other != arm and _dominates(other_values, values, objectives)
            )
        )
        points.append(
            FrontierPoint(
                arm=arm,
                values={o.name: values[o.name] for o in objectives},
                dominated_by=dominators,
            )
        )

    return ParetoFrontier(
        objectives=objectives, points=tuple(points), excluded=excluded
    )


def _dominates(
    left: dict[str, float], right: dict[str, float], objectives: tuple[Objective, ...]
) -> bool:
    """Whether ``left`` dominates ``right``: no worse anywhere, better somewhere."""
    no_worse = all(
        objective.at_least_as_good(left[objective.name], right[objective.name])
        for objective in objectives
    )
    strictly_better = any(
        objective.better(left[objective.name], right[objective.name])
        for objective in objectives
    )
    return no_worse and strictly_better
