"""Probabilistic model checking on a finite abstraction.

The twin's real state space is continuous and unbounded, so it is abstracted to
a finite Markov chain over inventory and backlog bands, and the properties are
checked exactly on that abstraction. Two consequences follow, and both are
stated in the run record rather than buried.

**The abstraction is declared.** Bands are configuration, not a hidden
constant. A coarser abstraction verifies faster and proves less; a reader must
be able to see which was used.

**The result transfers only as far as the abstraction is faithful.** Checking
proves a property of the abstract chain. Whether the concrete system inherits
it depends on the abstraction being conservative, which is why the runtime
monitors of `monitors.py` exist: they watch the concrete system for exactly the
divergences the abstraction cannot rule out. Verification plus monitoring is
the pair that licenses the claim; verification alone does not.

The checker is exact — value iteration to a fixpoint on reachability
probabilities — rather than statistical. A statistical model checker would give
a confidence interval on the probability, which is a weaker statement than the
protocol's assurance case needs and would put a second layer of sampling error
underneath a claim that is supposed to be structural.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xai_gov.core.logging import get_logger
from xai_gov.verification.specification import (
    Operator,
    SafetyProperty,
    Specification,
)

_LOG = get_logger("verification.model_checker")


@dataclass(frozen=True, slots=True)
class AbstractState:
    """One cell of the abstraction: an inventory band and a backlog band."""

    inventory_band: int
    backlog_band: int

    def to_key(self) -> tuple[int, int]:
        return (self.inventory_band, self.backlog_band)


@dataclass(slots=True)
class Abstraction:
    """A finite abstraction of the twin's operating state.

    ``inventory_bands`` and ``backlog_bands`` are upper edges; a state falls in
    the first band whose edge it does not exceed, with a final open band above.
    """

    inventory_bands: tuple[float, ...] = (0.0, 5.0, 20.0, 50.0, 100.0)
    backlog_bands: tuple[float, ...] = (0.0, 10.0, 50.0, 100.0)

    def __post_init__(self) -> None:
        for name, bands in (
            ("inventory_bands", self.inventory_bands),
            ("backlog_bands", self.backlog_bands),
        ):
            if len(bands) < 2:
                raise ValueError(f"{name} needs at least two edges to abstract anything")
            if list(bands) != sorted(bands):
                raise ValueError(f"{name} must be ascending")

    @property
    def size(self) -> int:
        return (len(self.inventory_bands) + 1) * (len(self.backlog_bands) + 1)

    def band_of(self, value: float, edges: tuple[float, ...]) -> int:
        for index, edge in enumerate(edges):
            if value <= edge:
                return index
        return len(edges)

    def abstract(self, inventory: float, backlog: float) -> AbstractState:
        return AbstractState(
            inventory_band=self.band_of(inventory, self.inventory_bands),
            backlog_band=self.band_of(backlog, self.backlog_bands),
        )

    def states(self) -> tuple[AbstractState, ...]:
        return tuple(
            AbstractState(inventory_band=i, backlog_band=b)
            for i in range(len(self.inventory_bands) + 1)
            for b in range(len(self.backlog_bands) + 1)
        )

    def is_bad(self, state: AbstractState, predicate: str) -> bool:
        """Whether an abstract state satisfies a bad predicate.

        Conservative by construction: a band counts as bad if *any* concrete
        state in it would. Erring the other way would let the checker certify a
        property that the concrete system violates, which is the one failure
        mode a verification layer must not have.
        """
        if predicate in ("stockout", "inventory_critical"):
            edges = self.inventory_bands
            threshold = 0.0 if predicate == "stockout" else 5.0
            # Band 0 is [-inf, edges[0]]; a band is bad if its lower edge is at
            # or below the threshold.
            lower = -float("inf") if state.inventory_band == 0 else edges[state.inventory_band - 1]
            return lower <= threshold
        if predicate == "backlog_critical":
            edges = self.backlog_bands
            upper = float("inf") if state.backlog_band >= len(edges) else edges[state.backlog_band]
            return upper >= 50.0
        if predicate == "capacity_exhausted":
            # Capacity is not part of this abstraction, so the property cannot
            # be discharged here. Reported rather than silently passed.
            return False
        raise ValueError(f"predicate {predicate!r} is not expressible in this abstraction")

    def expressible(self, predicate: str) -> bool:
        return predicate in ("stockout", "inventory_critical", "backlog_critical")

    def to_payload(self) -> dict[str, Any]:
        return {
            "inventory_bands": list(self.inventory_bands),
            "backlog_bands": list(self.backlog_bands),
            "states": self.size,
        }


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of checking one property."""

    property_id: str
    formula: str
    probability: float
    bound: float
    satisfied: bool
    discharged: bool
    reason: str
    counterexample: tuple[tuple[int, int], ...] = ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.property_id,
            "formula": self.formula,
            "probability": round(self.probability, 6),
            "bound": self.bound,
            "satisfied": self.satisfied,
            "discharged": self.discharged,
            "reason": self.reason,
            "counterexample": [list(step) for step in self.counterexample],
        }


TransitionModel = dict[tuple[int, int], dict[tuple[int, int], float]]


@dataclass(slots=True)
class ModelChecker:
    """Exact reachability checking over the abstraction."""

    abstraction: Abstraction = field(default_factory=Abstraction)
    tolerance: float = 1e-9
    max_iterations: int = 2000

    def check(
        self,
        specification: Specification,
        transitions: TransitionModel,
        *,
        initial: AbstractState,
    ) -> tuple[CheckResult, ...]:
        """Check every property, reporting which could not be discharged."""
        results: list[CheckResult] = []
        for prop in specification:
            if not self.abstraction.expressible(prop.predicate):
                results.append(
                    CheckResult(
                        property_id=prop.property_id,
                        formula=prop.formula(),
                        probability=0.0,
                        bound=prop.bound,
                        satisfied=False,
                        discharged=False,
                        reason=(
                            f"predicate {prop.predicate!r} is not expressible in the "
                            "declared abstraction; it must be discharged by a runtime "
                            "monitor instead"
                        ),
                    )
                )
                continue
            results.append(self._check_one(prop, transitions, initial=initial))
        return tuple(results)

    def _check_one(
        self,
        prop: SafetyProperty,
        transitions: TransitionModel,
        *,
        initial: AbstractState,
    ) -> CheckResult:
        bad = {
            state.to_key()
            for state in self.abstraction.states()
            if self.abstraction.is_bad(state, prop.predicate)
        }
        horizon = None if prop.operator is Operator.EVENTUALLY else prop.horizon
        probability = self._reachability(transitions, bad, horizon=horizon)

        satisfied = probability <= prop.bound + self.tolerance
        counterexample: tuple[tuple[int, int], ...] = ()
        if not satisfied:
            counterexample = self._witness(transitions, bad, initial.to_key())

        return CheckResult(
            property_id=prop.property_id,
            formula=prop.formula(),
            probability=probability,
            bound=prop.bound,
            satisfied=satisfied,
            discharged=True,
            reason=(
                "probability within bound"
                if satisfied
                else f"probability {probability:.4f} exceeds bound {prop.bound}"
            ),
            counterexample=counterexample,
        )

    def _reachability(
        self,
        transitions: TransitionModel,
        bad: set[tuple[int, int]],
        *,
        horizon: int | None,
    ) -> float:
        """Maximum probability of reaching a bad state from any start.

        The maximum over initial states, not the probability from one of them.
        A property that holds from the observed start and fails from a state the
        system can reach is not a safety property, and reporting the former
        would be the most flattering possible reading of the model.
        """
        keys = [state.to_key() for state in self.abstraction.states()]
        values = {key: (1.0 if key in bad else 0.0) for key in keys}

        iterations = self.max_iterations if horizon is None else horizon
        for _ in range(iterations):
            updated: dict[tuple[int, int], float] = {}
            for key in keys:
                if key in bad:
                    updated[key] = 1.0
                    continue
                successors = transitions.get(key, {})
                updated[key] = sum(
                    probability * values[successor]
                    for successor, probability in successors.items()
                    if successor in values
                )
            residual = max(abs(updated[key] - values[key]) for key in keys)
            values = updated
            if horizon is None and residual < self.tolerance:
                break

        reachable = self._reachable_from_any(transitions, keys)
        # Bad states are excluded from the maximum. A bad state reaches itself
        # with probability 1, so including them would make every property whose
        # bad region is reachable at all report a probability of exactly 1 —
        # collapsing a probabilistic bound into "is the bad state reachable",
        # and discarding the quantitative content the bound exists for.
        candidates = [key for key in reachable if key not in bad]
        return max((values[key] for key in candidates), default=0.0)

    def _reachable_from_any(
        self, transitions: TransitionModel, keys: list[tuple[int, int]]
    ) -> set[tuple[int, int]]:
        """States the model can actually occupy.

        Unreachable cells of the abstraction are excluded from the maximum: a
        band the transition model never enters says nothing about the system,
        and letting it drive the verdict would fail properties for cells that
        exist only as an artifact of how the bands were drawn.
        """
        occupied = {key for key in keys if transitions.get(key)}
        for successors in transitions.values():
            occupied.update(successors)
        return occupied or set(keys)

    def _witness(
        self,
        transitions: TransitionModel,
        bad: set[tuple[int, int]],
        start: tuple[int, int],
    ) -> tuple[tuple[int, int], ...]:
        """A shortest path from the start into a bad state.

        A failed check without a counterexample tells an engineer that
        something is wrong but not what; the path is what makes the failure
        actionable.
        """
        frontier: list[tuple[tuple[int, int], ...]] = [(start,)]
        seen = {start}
        while frontier:
            path = frontier.pop(0)
            current = path[-1]
            if current in bad:
                return path
            for successor in transitions.get(current, {}):
                if successor not in seen:
                    seen.add(successor)
                    frontier.append((*path, successor))
        return ()

    def to_payload(self) -> dict[str, Any]:
        return {
            "method": "exact_reachability_on_finite_abstraction",
            "abstraction": self.abstraction.to_payload(),
            "tolerance": self.tolerance,
        }


def estimate_transitions(
    observations: list[tuple[tuple[int, int], tuple[int, int]]],
    *,
    laplace: float = 1.0,
) -> TransitionModel:
    """Estimate the abstract transition model from observed band changes.

    Laplace smoothing spreads mass over every state the run ever visited, not
    only the successors a given source was seen entering. Smoothing over the
    observed successors alone smooths nothing: a source seen going to exactly
    one place still gets a probability of 1, which is precisely the fabricated
    certainty the smoothing exists to prevent. The checker would then verify
    properties of a model that claims to know what the data never showed.
    """
    if laplace < 0.0:
        raise ValueError("laplace smoothing must be non-negative")

    counts: dict[tuple[int, int], dict[tuple[int, int], float]] = {}
    support: set[tuple[int, int]] = set()
    for source, target in observations:
        counts.setdefault(source, {})
        counts[source][target] = counts[source].get(target, 0.0) + 1.0
        support.update((source, target))

    model: TransitionModel = {}
    for source, targets in counts.items():
        smoothed = {
            candidate: targets.get(candidate, 0.0) + laplace for candidate in support
        }
        total = sum(smoothed.values())
        model[source] = {key: value / total for key, value in smoothed.items() if value > 0.0}
    return model
