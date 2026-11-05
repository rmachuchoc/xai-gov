"""The intervention threshold, derived rather than chosen.

This module is where the protocol's Theorem 2 becomes code. The claim is
that the optimal governance policy is of threshold type — intervene if and
only if the belief in disruption reaches some b* — and that b* moves in
declared directions: up with the cost of intervening, down with the severity
of the loss avoided.

Two derivations are provided, and the difference between them is the point.

`myopic_threshold` compares one period of expected loss against one period of
intervention cost. It is exact, closed form, and pessimistic: it ignores that
the belief will keep rising if the regime really has changed, so it waits
longer than optimal. It is therefore an upper bound on b*, which makes it a
useful sanity rail rather than the answer.

`solve_threshold` runs value iteration over the belief simplex collapsed to
the disruption dimension, which is the reduction Theorem 2's proof sketch
relies on. It accounts for continuation value and so intervenes earlier. Both
are computed and both are recorded, because a derived threshold that nobody
can compare against a closed form is a number no reviewer can check.

What is *not* claimed: neither derivation makes the resulting policy optimal
for the full three-regime CPOMDP. The collapse to a scalar belief is an
approximation, stated here and reported in the run record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ThresholdEconomics:
    """The declared costs from which a threshold is derived.

    ``disruption_loss`` is the operational loss incurred by *not* intervening
    while genuinely disrupted, per period, in the same units as the logistics
    cost. ``intervention_cost`` is what one governance action costs. Both are
    experimental factors; a threshold is only as defensible as the two numbers
    it comes from, so they travel with it into every artifact.
    """

    intervention_cost: float = 1.0
    disruption_loss: float = 10.0
    discount: float = 0.95
    false_intervention_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.intervention_cost < 0.0:
            raise ValueError("intervention_cost must be non-negative")
        if self.disruption_loss <= 0.0:
            raise ValueError("disruption_loss must be positive")
        if not 0.0 < self.discount < 1.0:
            raise ValueError("discount must lie in (0, 1)")
        if self.false_intervention_cost < 0.0:
            raise ValueError("false_intervention_cost must be non-negative")

    @property
    def effective_loss(self) -> float:
        """Loss avoided by intervening, net of the cost of being wrong."""
        return self.disruption_loss + self.false_intervention_cost

    def to_payload(self) -> dict[str, Any]:
        return {
            "intervention_cost": self.intervention_cost,
            "disruption_loss": self.disruption_loss,
            "discount": self.discount,
            "false_intervention_cost": self.false_intervention_cost,
        }


def myopic_threshold(economics: ThresholdEconomics) -> float:
    """Single-period threshold: b* = c / (L + c_false).

    Intervene when the expected loss from inaction, b·L, reaches the cost of
    acting, c. Clipped into (0, 1): a cost above the loss it prevents means
    intervention is never worthwhile, which is a finding rather than an error,
    and it is reported as a threshold of 1.
    """
    ratio = economics.intervention_cost / economics.effective_loss
    return min(max(ratio, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class ThresholdSolution:
    """A derived threshold with the evidence behind it."""

    threshold: float
    myopic: float
    iterations: int
    residual: float
    converged: bool
    grid_size: int
    economics: ThresholdEconomics

    @property
    def anticipation_gain(self) -> float:
        """How much earlier the solved policy acts than the myopic one.

        Non-negative whenever the solver is behaving: accounting for
        continuation value can only make intervention more attractive, never
        less. A negative value indicates a bug, and the test suite asserts it.
        """
        return self.myopic - self.threshold

    def to_payload(self) -> dict[str, Any]:
        return {
            "threshold": round(self.threshold, 6),
            "myopic_threshold": round(self.myopic, 6),
            "anticipation_gain": round(self.anticipation_gain, 6),
            "iterations": self.iterations,
            "residual": round(self.residual, 9),
            "converged": self.converged,
            "grid_size": self.grid_size,
            "method": "value_iteration_on_collapsed_belief",
            "economics": self.economics.to_payload(),
        }


@dataclass(frozen=True, slots=True)
class BeliefDynamics:
    """The two-state belief model both solver and runtime filter use."""

    persistence: float = 0.9
    flag_probability_nominal: float = 0.05
    flag_probability_disruption: float = 0.80

    def __post_init__(self) -> None:
        if not 0.0 < self.persistence < 1.0:
            raise ValueError("persistence must lie in (0, 1)")
        if not 0.0 < self.flag_probability_nominal < self.flag_probability_disruption < 1.0:
            raise ValueError("flag probabilities must satisfy 0 < nominal < disruption < 1")

    def predict(self, belief: float) -> float:
        """One step of the latent dynamics, before any observation."""
        return self.persistence * belief + (1.0 - self.persistence) * (1.0 - belief)

    def posterior(self, prior: float, flagged: bool) -> float:
        """Bayes update of the disruption belief on one observation."""
        d = self.flag_probability_disruption
        n = self.flag_probability_nominal
        likelihood_d = d if flagged else 1.0 - d
        likelihood_n = n if flagged else 1.0 - n
        numerator = prior * likelihood_d
        denominator = numerator + (1.0 - prior) * likelihood_n
        return numerator / denominator if denominator > 0.0 else prior

    def flag_probability(self, prior: float) -> float:
        return (
            prior * self.flag_probability_disruption
            + (1.0 - prior) * self.flag_probability_nominal
        )


@dataclass(frozen=True, slots=True)
class ValueFunction:
    """The solved value function on the belief grid, and what it implies."""

    grid: tuple[float, ...]
    values: tuple[float, ...]
    economics: ThresholdEconomics
    dynamics: BeliefDynamics
    iterations: int
    residual: float

    def at(self, belief: float) -> float:
        """Linear interpolation of the value function off the grid."""
        size = len(self.grid)
        position = belief * (size - 1)
        lower = int(position)
        if lower >= size - 1:
            return self.values[size - 1]
        weight = position - lower
        return self.values[lower] * (1.0 - weight) + self.values[lower + 1] * weight

    def continuation(self, belief: float) -> float:
        """C(b): expected value next period after waiting at belief b."""
        prior = self.dynamics.predict(belief)
        flag = self.dynamics.flag_probability(prior)
        return (
            flag * self.at(self.dynamics.posterior(prior, True))
            + (1.0 - flag) * self.at(self.dynamics.posterior(prior, False))
        )

    def continuation_gap(self, belief: float) -> float:
        """Delta(b) = C(b) - V(0): what waiting costs in continuation terms."""
        return self.continuation(belief) - self.at(0.0)

    def margin(self, belief: float) -> float:
        """D(b) = wait - intervene = b*L + gamma*Delta(b) - c. Intervene iff D >= 0."""
        economics = self.economics
        return (
            belief * economics.disruption_loss
            + economics.discount * self.continuation_gap(belief)
            - economics.intervention_cost
        )

    def threshold(self) -> float:
        """The first grid belief at which intervening is preferred."""
        for belief in self.grid:
            if self.margin(belief) >= 0.0:
                return belief
        return 1.0


def solve_values(
    economics: ThresholdEconomics,
    dynamics: BeliefDynamics | None = None,
    *,
    grid_size: int = 201,
    tolerance: float = 1e-9,
    max_iterations: int = 5000,
) -> ValueFunction:
    """Value iteration over the disruption belief.

    The state is b, the belief in disruption. Two actions: wait, paying b·L
    and then updating the belief on the observation that arrives; or
    intervene, paying c and returning the system to the nominal regime.

    The belief update is the exact two-state Bayes filter, so the transition
    the solver reasons about is the same one the runtime filter applies —
    a solver that optimized against a different dynamics than the agent
    experiences would produce a threshold that is precisely wrong.
    """
    if grid_size < 11:
        raise ValueError("grid_size must be at least 11 for a usable threshold")
    dynamics = dynamics or BeliefDynamics()
    grid = tuple(index / (grid_size - 1) for index in range(grid_size))
    solution = ValueFunction(grid, (0.0,) * grid_size, economics, dynamics, 0, float("inf"))
    discount = economics.discount
    residual = float("inf")
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        # Intervening restores the nominal regime, so its continuation is
        # the value at belief zero — which is what makes intervention
        # attractive beyond the single period the myopic rule considers.
        intervene = economics.intervention_cost + discount * solution.at(0.0)
        updated = tuple(
            min(belief * economics.disruption_loss + discount * solution.continuation(belief),
                intervene)
            for belief in grid
        )
        residual = max(abs(new - old) for new, old in zip(updated, solution.values, strict=True))
        solution = ValueFunction(grid, updated, economics, dynamics, iterations, residual)
        if residual < tolerance:
            break
    return solution


def solve_threshold(
    economics: ThresholdEconomics,
    *,
    persistence: float = 0.9,
    flag_probability_nominal: float = 0.05,
    flag_probability_disruption: float = 0.80,
    grid_size: int = 201,
    tolerance: float = 1e-9,
    max_iterations: int = 5000,
) -> ThresholdSolution:
    """The optimal threshold: the first belief at which D(b) >= 0."""
    dynamics = BeliefDynamics(
        persistence=persistence,
        flag_probability_nominal=flag_probability_nominal,
        flag_probability_disruption=flag_probability_disruption,
    )
    solution = solve_values(
        economics, dynamics, grid_size=grid_size, tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return ThresholdSolution(
        threshold=solution.threshold(),
        myopic=myopic_threshold(economics),
        iterations=solution.iterations,
        residual=solution.residual,
        converged=solution.residual < tolerance,
        grid_size=grid_size,
        economics=economics,
    )


def build_economics(config: dict[str, Any]) -> ThresholdEconomics:
    """Construct the threshold economics from configuration."""
    if not config:
        return ThresholdEconomics()
    try:
        return ThresholdEconomics(**config)
    except TypeError as error:
        raise ValueError(f"threshold economics rejected its parameters: {error}") from error
