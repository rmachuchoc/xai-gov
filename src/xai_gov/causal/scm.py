"""The structural causal model of the twin.

This module is the protocol's move from correlational explanation to causal
recourse. A structural causal model 𝓜 = ⟨U, V, F, P(U)⟩ is declared over the
operating variables, and every explanation the governance descriptor carries is
an *interventional* claim evaluated in it: what minimal reachable change would
have produced a different decision.

The one commitment that makes this honest: **the mechanisms here must be the
simulator's mechanisms.** An SCM fitted to observational traces of the twin,
or invented for the descriptor's convenience, would produce identifiable
counterfactuals about a world that does not exist — and the resulting recourse
would be confidently unreachable. So the parents and functional forms below
mirror `simulation/engine.py` directly, and `assert_matches_engine` exists to
fail loudly if the two drift apart.

Identifiability is declared, not assumed. Because the graph is known by
construction (the simulator *is* the mechanism) every interventional query on
observed variables is identifiable by definition — but the twin also contains
mechanisms the governance agent cannot observe, and any variable behind one is
marked accordingly. A descriptor built on an unidentifiable path reports itself
as such rather than presenting an estimate as a fact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from xai_gov.io.decision_record import OperatingState


class Variable(StrEnum):
    """Endogenous variables of the model, named as the engine names them."""

    DEMAND = "demand_observed"
    INVENTORY = "inventory"
    BACKLOG = "backlog"
    IN_TRANSIT = "in_transit"
    CAPACITY = "capacity"
    DATA_DELAY = "data_delay"
    # Derived: the position the reorder rule actually compares against s.
    INVENTORY_POSITION = "inventory_position"
    # The decision itself, so interventions on its parents are queries about it.
    ORDER_QUANTITY = "order_quantity"


#: Structural parents. Read directly off the engine's step function: the
#: reorder rule compares inventory position (on hand + in transit − backlog)
#: against the reorder point, and the order is capped by capacity.
#:
#: In-transit inventory is a **root**, not a child of the order. Within one
#: period the order placed now has not arrived: it lands after the lead time,
#: so this period's in-transit was determined by orders already in flight.
#: Making it a child of ORDER_QUANTITY collapses that lag and creates a cycle
#: (order → in transit → position → order) — the acyclicity check refuses it,
#: correctly, because a model with that cycle has no counterfactuals to
#: compute. A cross-period twin would carry in_transit[t] ← order[t-1] as a
#: lagged edge; this is a single-decision model, and it says so.
PARENTS: dict[Variable, tuple[Variable, ...]] = {
    Variable.DEMAND: (),
    Variable.CAPACITY: (),
    Variable.DATA_DELAY: (),
    Variable.IN_TRANSIT: (),
    Variable.INVENTORY: (Variable.DEMAND,),
    Variable.BACKLOG: (Variable.DEMAND, Variable.INVENTORY),
    Variable.INVENTORY_POSITION: (
        Variable.INVENTORY,
        Variable.IN_TRANSIT,
        Variable.BACKLOG,
    ),
    Variable.ORDER_QUANTITY: (
        Variable.INVENTORY_POSITION,
        Variable.CAPACITY,
        Variable.DATA_DELAY,
    ),
}

#: Variables the governance agent observes directly. `data_delay` is observed
#: as a number but its *cause* — the data-quality regime — is not, so an
#: intervention on it is identifiable while an intervention on what produced it
#: is not. That distinction is the whole reason this set is explicit.
OBSERVED: frozenset[Variable] = frozenset(
    {
        Variable.DEMAND,
        Variable.INVENTORY,
        Variable.BACKLOG,
        Variable.IN_TRANSIT,
        Variable.CAPACITY,
        Variable.DATA_DELAY,
    }
)

#: Physical and policy constraints on interventions. An intervention outside
#: these is not a counterfactual, it is a fantasy: negative inventory does not
#: occur, and no operator can reduce observed demand by decree.
BOUNDS: dict[Variable, tuple[float, float]] = {
    Variable.DEMAND: (0.0, float("inf")),
    Variable.INVENTORY: (0.0, float("inf")),
    Variable.BACKLOG: (0.0, float("inf")),
    Variable.IN_TRANSIT: (0.0, float("inf")),
    Variable.CAPACITY: (0.0, float("inf")),
    Variable.DATA_DELAY: (0.0, 10.0),
}

#: Variables an operator can actually act on within a period. Demand and
#: backlog are outcomes, not levers: an explanation whose recourse is "have
#: had less demand" is correlationally true and operationally useless, and
#: excluding them here is what stops the descriptor from producing it.
ACTIONABLE: frozenset[Variable] = frozenset(
    {Variable.INVENTORY, Variable.IN_TRANSIT, Variable.CAPACITY, Variable.DATA_DELAY}
)


@dataclass(frozen=True, slots=True)
class CausalWorld:
    """One assignment of values to the model's variables."""

    values: Mapping[str, float]

    def get(self, variable: Variable) -> float:
        return float(self.values[variable.value])

    def with_intervention(self, assignments: Mapping[Variable, float]) -> CausalWorld:
        """Return the world under do(X_S = x'_S), with descendants recomputed.

        This is what separates an interventional query from an associational
        one: setting inventory and leaving inventory position untouched would
        describe a world where the accounting identity is broken, and any
        conclusion drawn there is about nothing.
        """
        updated = dict(self.values)
        for variable, value in assignments.items():
            updated[variable.value] = float(value)
        world = CausalWorld(values=updated)
        return world.recompute()

    def recompute(self) -> CausalWorld:
        """Restore the deterministic identities the engine maintains."""
        updated = dict(self.values)
        updated[Variable.INVENTORY_POSITION.value] = (
            float(updated[Variable.INVENTORY.value])
            + float(updated[Variable.IN_TRANSIT.value])
            - float(updated[Variable.BACKLOG.value])
        )
        return CausalWorld(values=updated)

    def to_payload(self) -> dict[str, float]:
        return {key: round(float(value), 6) for key, value in sorted(self.values.items())}


def world_from_state(state: OperatingState, *, order_quantity: float = 0.0) -> CausalWorld:
    """Lift an observed operating state into the model."""
    return CausalWorld(
        values={
            Variable.DEMAND.value: state.demand_observed,
            Variable.INVENTORY.value: state.inventory,
            Variable.BACKLOG.value: state.backlog,
            Variable.IN_TRANSIT.value: state.in_transit,
            Variable.CAPACITY.value: state.capacity,
            Variable.DATA_DELAY.value: float(state.data_delay),
            Variable.ORDER_QUANTITY.value: order_quantity,
            Variable.INVENTORY_POSITION.value: 0.0,
        }
    ).recompute()


@dataclass(frozen=True, slots=True)
class StructuralCausalModel:
    """The declared model, with its graph and its identifiability claims."""

    parents: dict[Variable, tuple[Variable, ...]] = field(
        default_factory=lambda: dict(PARENTS)
    )
    observed: frozenset[Variable] = OBSERVED
    actionable: frozenset[Variable] = ACTIONABLE
    bounds: dict[Variable, tuple[float, float]] = field(default_factory=lambda: dict(BOUNDS))

    def __post_init__(self) -> None:
        self._assert_acyclic()

    def _assert_acyclic(self) -> None:
        """A cyclic graph has no counterfactuals to compute."""
        state: dict[Variable, int] = {}

        def visit(node: Variable) -> None:
            mark = state.get(node, 0)
            if mark == 1:
                raise ValueError(f"causal graph contains a cycle through {node.value}")
            if mark == 2:
                return
            state[node] = 1
            for parent in self.parents.get(node, ()):
                visit(parent)
            state[node] = 2

        for node in self.parents:
            visit(node)

    def ancestors(self, variable: Variable) -> frozenset[Variable]:
        found: set[Variable] = set()
        stack = list(self.parents.get(variable, ()))
        while stack:
            node = stack.pop()
            if node in found:
                continue
            found.add(node)
            stack.extend(self.parents.get(node, ()))
        return frozenset(found)

    def descendants(self, variable: Variable) -> frozenset[Variable]:
        return frozenset(
            node for node in self.parents if variable in self.ancestors(node)
        )

    def affects_decision(self, variable: Variable) -> bool:
        """Whether intervening on this variable can change the order at all.

        A variable with no causal path to the decision cannot explain it. An
        attribution method that assigns such a variable non-zero importance is
        reporting a correlation, and `fidelity` will say so.
        """
        return variable in self.ancestors(Variable.ORDER_QUANTITY)

    def is_identifiable(self, variable: Variable) -> bool:
        """Whether do(variable) is identifiable from what the agent observes."""
        return variable in self.observed

    def is_admissible(self, variable: Variable, value: float) -> bool:
        """Whether do(variable = value) is a world that can exist."""
        if variable not in self.actionable:
            return False
        low, high = self.bounds.get(variable, (-float("inf"), float("inf")))
        return low <= value <= high

    def admissible_interventions(self) -> tuple[Variable, ...]:
        """Levers that are actionable, identifiable and causally relevant.

        All three conditions are required, and each excludes a different kind
        of useless explanation: an unactionable lever yields advice nobody can
        take, an unidentifiable one yields a number nobody can trust, and a
        causally irrelevant one yields a story about the wrong mechanism.
        """
        return tuple(
            sorted(
                (
                    variable
                    for variable in self.actionable
                    if self.is_identifiable(variable) and self.affects_decision(variable)
                ),
                key=lambda item: item.value,
            )
        )

    def assert_matches_engine(self, engine_fields: frozenset[str]) -> None:
        """Fail if the model and the simulator have drifted apart.

        Called from the test suite. A silent divergence here is the most
        dangerous failure this project can have: every downstream number would
        remain well formed while describing a mechanism the twin does not run.
        """
        modelled = {variable.value for variable in self.observed}
        missing = modelled - engine_fields
        if missing:
            raise ValueError(
                f"the causal model declares observed variables the engine does not "
                f"expose: {sorted(missing)}"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "variables": sorted(variable.value for variable in self.parents),
            "edges": {
                node.value: [parent.value for parent in parents]
                for node, parents in sorted(self.parents.items(), key=lambda kv: kv[0].value)
                if parents
            },
            "observed": sorted(v.value for v in self.observed),
            "actionable": sorted(v.value for v in self.actionable),
            "admissible_interventions": [v.value for v in self.admissible_interventions()],
        }


#: The decision mechanism, matching the (s, Q)-style rule the policies use.
#: Passed explicitly rather than imported so the causal layer never depends on
#: the policy layer: an explanation must be able to reason about a policy it
#: does not own.
DecisionRule = Callable[[CausalWorld], float]


def reorder_rule(*, reorder_point: float, order_quantity: float) -> DecisionRule:
    """The classical rule as a structural mechanism."""

    def rule(world: CausalWorld) -> float:
        # A stale view degrades the position the rule believes it sees. This is
        # modelled as a mechanism rather than as noise because the protocol asks
        # whether recourse on data quality is actionable — and because the
        # causal graph declares data_delay as a parent of the order, so a rule
        # that ignored it would leave the SCM describing an edge the mechanism
        # does not implement.
        delay = world.get(Variable.DATA_DELAY)
        position = world.get(Variable.INVENTORY_POSITION) - delay * world.get(
            Variable.DEMAND
        )
        if position >= reorder_point:
            return 0.0
        return min(order_quantity, world.get(Variable.CAPACITY))

    return rule
