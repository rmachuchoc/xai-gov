"""Causal recourse and the actionability score.

The protocol's AS_causal counts governance actions reachable through an
admissible intervention of bounded cost:

    AS = |{ a ∈ 𝒜 : ∃ do(X_S=x'_S) ∈ 𝒞(x_t), c(x_t, x'_S) ≤ δ, π(x^do) = a }| / |𝒜|

Three filters are applied in that formula and each excludes a different kind of
useless explanation. `𝒞(x_t)` excludes worlds that cannot exist. `c ≤ δ`
excludes worlds nobody can afford to reach. And searching for *distinct
resulting actions* rather than for distinct interventions is what makes the
score a measure of leverage: a hundred interventions that all produce the same
order tell the agent it has one option, not a hundred.

The search is exhaustive over a declared grid rather than gradient-based. That
is a deliberate trade: the grid is coarse, but it is auditable, deterministic,
and cannot converge to a local optimum that a reviewer is unable to reproduce.
Every counterfactual reported is one a reader can recompute by hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from xai_gov.causal.scm import (
    CausalWorld,
    DecisionRule,
    StructuralCausalModel,
    Variable,
)

#: Cost per unit of change, by lever. These are declared parameters of the
#: study, not physical truths: acquiring inventory is cheap relative to
#: expanding capacity, and improving data freshness sits between them. They are
#: reported with every descriptor so a reviewer can see what "affordable" meant.
DEFAULT_UNIT_COSTS: dict[Variable, float] = {
    Variable.INVENTORY: 1.0,
    Variable.IN_TRANSIT: 1.0,
    Variable.CAPACITY: 3.0,
    Variable.DATA_DELAY: 5.0,
}


@dataclass(frozen=True, slots=True)
class Counterfactual:
    """One admissible intervention and the action it would have produced."""

    interventions: dict[str, float]
    cost: float
    resulting_quantity: float
    changed_decision: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "interventions": {k: round(v, 6) for k, v in sorted(self.interventions.items())},
            "cost": round(self.cost, 6),
            "resulting_quantity": round(self.resulting_quantity, 6),
            "changed_decision": self.changed_decision,
        }


@dataclass(frozen=True, slots=True)
class RecourseResult:
    """The outcome of a recourse search, with everything needed to audit it."""

    actionability_score: float
    counterfactuals: tuple[Counterfactual, ...]
    reachable_quantities: tuple[float, ...]
    budget: float
    considered: int
    identifiable: bool
    unidentifiable_levers: tuple[str, ...]

    @property
    def cheapest(self) -> Counterfactual | None:
        """The least costly intervention that changes the decision.

        This is the recourse an operator would actually be offered, so it is
        singled out rather than left for a caller to rediscover.
        """
        changing = [cf for cf in self.counterfactuals if cf.changed_decision]
        return min(changing, key=lambda cf: cf.cost) if changing else None

    def to_payload(self) -> dict[str, Any]:
        cheapest = self.cheapest
        return {
            "actionability_score": round(self.actionability_score, 6),
            "budget": self.budget,
            "considered": self.considered,
            "reachable_quantities": [round(q, 6) for q in self.reachable_quantities],
            "cheapest_recourse": cheapest.to_payload() if cheapest else None,
            "identifiable": self.identifiable,
            "unidentifiable_levers": list(self.unidentifiable_levers),
        }


@dataclass(slots=True)
class RecourseSearch:
    """Exhaustive search for admissible, affordable, decision-changing worlds."""

    model: StructuralCausalModel = field(default_factory=StructuralCausalModel)
    unit_costs: dict[Variable, float] = field(default_factory=lambda: dict(DEFAULT_UNIT_COSTS))
    budget: float = 30.0
    steps: tuple[float, ...] = (-20.0, -10.0, -5.0, 5.0, 10.0, 20.0)
    max_levers: int = 1

    def __post_init__(self) -> None:
        if self.budget <= 0.0:
            raise ValueError("recourse budget must be positive")
        if not self.steps:
            raise ValueError("at least one intervention step is required")
        if self.max_levers < 1:
            raise ValueError("max_levers must be at least 1")
        if any(cost <= 0.0 for cost in self.unit_costs.values()):
            raise ValueError("unit costs must be positive")

    def cost_of(self, variable: Variable, delta: float) -> float:
        """Cost of moving one lever by ``delta``.

        Symmetric in the direction of change. Releasing capacity is not free
        merely because it is a reduction: reversing an operational commitment
        costs something, and pricing reductions at zero would make the search
        prefer them for no reason a practitioner would recognize.
        """
        return abs(delta) * self.unit_costs.get(variable, 1.0)

    def search(self, world: CausalWorld, rule: DecisionRule) -> RecourseResult:
        """Enumerate the affordable admissible worlds and score leverage."""
        baseline = rule(world)
        levers = self.model.admissible_interventions()
        unidentifiable = tuple(
            sorted(
                variable.value
                for variable in self.model.actionable
                if not self.model.is_identifiable(variable)
            )
        )

        found: list[Counterfactual] = []
        considered = 0
        for variable in levers:
            current = world.get(variable)
            for step in self.steps:
                considered += 1
                candidate = current + step
                if not self.model.is_admissible(variable, candidate):
                    continue
                cost = self.cost_of(variable, step)
                if cost > self.budget:
                    continue
                intervened = world.with_intervention({variable: candidate})
                resulting = rule(intervened)
                found.append(
                    Counterfactual(
                        interventions={variable.value: candidate},
                        cost=cost,
                        resulting_quantity=resulting,
                        changed_decision=abs(resulting - baseline) > 1e-9,
                    )
                )

        # Leverage is measured in distinct *outcomes*, not distinct
        # interventions: a hundred routes to the same order is one option.
        reachable = tuple(
            sorted({round(cf.resulting_quantity, 6) for cf in found if cf.changed_decision})
        )
        # The denominator is the number of levers available. A node with one
        # lever that works has full actionability; the score answers "how much
        # of my available leverage actually moves this decision", which is the
        # question an operator has.
        denominator = max(len(levers), 1)
        score = min(len(reachable) / denominator, 1.0)

        return RecourseResult(
            actionability_score=score,
            counterfactuals=tuple(found),
            reachable_quantities=reachable,
            budget=self.budget,
            considered=considered,
            identifiable=not unidentifiable,
            unidentifiable_levers=unidentifiable,
        )

    def interventional_effects(
        self, world: CausalWorld, rule: DecisionRule
    ) -> dict[str, float]:
        """The true effect of each lever on the decision.

        Probed at the scale the lever is actually moved, not at a unit step.
        The reorder mechanism is a threshold: a one-unit perturbation almost
        never crosses it, so a unit probe reports every effect as zero, the
        fidelity audit returns 'not verifiable', and no descriptor is ever
        informative. Asking whether a lever affects the decision only means
        something at the magnitude the lever can be moved by.

        The largest effect over the declared steps is taken, which answers the
        question the governance agent is asking: is there *any* affordable
        change to this lever that would alter the decision.
        """
        baseline = rule(world)
        effects: dict[str, float] = {}
        for variable in self.model.admissible_interventions():
            current = world.get(variable)
            deltas: list[float] = []
            for step in self.steps:
                candidate = current + step
                if not self.model.is_admissible(variable, candidate):
                    continue
                if self.cost_of(variable, step) > self.budget:
                    continue
                intervened = world.with_intervention({variable: candidate})
                deltas.append(rule(intervened) - baseline)
            effects[variable.value] = max(deltas, key=abs) if deltas else 0.0
        return effects

    def to_payload(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "steps": list(self.steps),
            "max_levers": self.max_levers,
            "unit_costs": {k.value: v for k, v in sorted(self.unit_costs.items())},
            "model": self.model.to_payload(),
        }


def build_recourse_search(config: dict[str, Any]) -> RecourseSearch:
    """Construct the recourse search from configuration."""
    params = dict(config or {})
    steps: Sequence[float] | None = params.pop("steps", None)
    costs = params.pop("unit_costs", None)
    kwargs: dict[str, Any] = dict(params)
    if steps is not None:
        kwargs["steps"] = tuple(float(value) for value in steps)
    if costs is not None:
        if not isinstance(costs, dict):
            raise ValueError("recourse 'unit_costs' must be a mapping")
        kwargs["unit_costs"] = {Variable(str(k)): float(v) for k, v in costs.items()}
    try:
        return RecourseSearch(**kwargs)
    except TypeError as error:
        raise ValueError(f"recourse search rejected its parameters: {error}") from error
