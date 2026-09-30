"""Safety properties in probabilistic temporal logic.

A safety property here is not a rule someone wrote in the agent. It is a claim
about the whole system, stated before the run, in a form a machine can check
exhaustively: "the probability that backlog exceeds the critical threshold for
more than k consecutive periods is below eta."

The distinction that matters for the protocol: testing a hundred situations and
finding no violation shows that a hundred situations are safe. Model checking
shows that *no reachable situation* violates the property, or produces the
counterexample that does. The first is evidence, the second is proof, and only
the second supports a claim of safety by construction.

Two operators are supported, and the choice is deliberate rather than a
limitation of ambition. `P_max` bounds the probability that a bad state is ever
reached; `P_bounded` bounds it within a horizon. Together they express every
property the twin's assurance case needs, and both admit exact checking on a
finite abstraction — an unbounded until, by contrast, would require fixpoint
machinery whose result nobody in the review chain could verify by hand.

Properties are declared in configuration, not in code, because the assurance
case is an artifact of the study rather than an implementation detail. A run
records the properties it was checked against, so a reader knows precisely
what "verified" meant for that run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from xai_gov.io.decision_record import OperatingState


class Operator(StrEnum):
    """The temporal operators the checker supports."""

    #: P(eventually bad) <= eta, over the whole horizon.
    EVENTUALLY = "eventually"
    #: P(bad within k steps) <= eta.
    BOUNDED_EVENTUALLY = "bounded_eventually"
    #: P(bad state persists for k consecutive steps) <= eta.
    PERSISTENT = "persistent"


#: State predicates available to properties. Named rather than expressed as
#: arbitrary Python so a specification stays readable to a reviewer who does
#: not read code, and so an unknown predicate fails at load rather than at the
#: moment it would have mattered.
Predicate = Callable[[OperatingState], bool]


def _stockout(state: OperatingState) -> bool:
    return state.inventory <= 0.0


def _backlog_critical(state: OperatingState) -> bool:
    return state.backlog >= 50.0


def _inventory_critical(state: OperatingState) -> bool:
    return state.inventory <= 5.0


def _capacity_exhausted(state: OperatingState) -> bool:
    return state.capacity <= 0.0


PREDICATES: dict[str, Predicate] = {
    "stockout": _stockout,
    "backlog_critical": _backlog_critical,
    "inventory_critical": _inventory_critical,
    "capacity_exhausted": _capacity_exhausted,
}


def available_predicates() -> tuple[str, ...]:
    return tuple(sorted(PREDICATES))


@dataclass(frozen=True, slots=True)
class SafetyProperty:
    """One PCTL-style claim about the system."""

    property_id: str
    operator: Operator
    predicate: str
    bound: float
    horizon: int = 0
    description: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.bound <= 1.0:
            raise ValueError(f"{self.property_id}: bound must be a probability in [0, 1]")
        if self.predicate not in PREDICATES:
            raise ValueError(
                f"{self.property_id}: unknown predicate {self.predicate!r}; "
                f"available: {', '.join(available_predicates())}"
            )
        if self.operator is not Operator.EVENTUALLY and self.horizon < 1:
            raise ValueError(
                f"{self.property_id}: operator {self.operator.value} needs a horizon of "
                "at least 1; without one the claim has no temporal content"
            )

    def holds_for(self, state: OperatingState) -> bool:
        """Whether the bad predicate is false in this state."""
        return not PREDICATES[self.predicate](state)

    def formula(self) -> str:
        """The property in PCTL notation, for the assurance case."""
        if self.operator is Operator.EVENTUALLY:
            return f"P<={self.bound} [ F {self.predicate} ]"
        if self.operator is Operator.BOUNDED_EVENTUALLY:
            return f"P<={self.bound} [ F<={self.horizon} {self.predicate} ]"
        return f"P<={self.bound} [ G<={self.horizon} {self.predicate} ]"

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.property_id,
            "formula": self.formula(),
            "operator": self.operator.value,
            "predicate": self.predicate,
            "bound": self.bound,
            "horizon": self.horizon,
            "description": self.description,
        }


#: The default assurance case. Every property here is one the twin's operator
#: would recognize as a real commitment, not a formality.
DEFAULT_PROPERTIES: tuple[SafetyProperty, ...] = (
    SafetyProperty(
        property_id="SAFE-1",
        operator=Operator.PERSISTENT,
        predicate="backlog_critical",
        bound=0.05,
        horizon=3,
        description=(
            "backlog does not stay above its critical level for three consecutive "
            "periods with probability above 0.05"
        ),
    ),
    SafetyProperty(
        property_id="SAFE-2",
        operator=Operator.BOUNDED_EVENTUALLY,
        predicate="inventory_critical",
        bound=0.20,
        horizon=5,
        description="inventory does not fall to its critical level within five periods",
    ),
    SafetyProperty(
        property_id="SAFE-3",
        operator=Operator.PERSISTENT,
        predicate="stockout",
        bound=0.02,
        horizon=2,
        description="a stockout does not persist across two consecutive periods",
    ),
)


@dataclass(frozen=True, slots=True)
class Specification:
    """The set of properties a run is checked against."""

    properties: tuple[SafetyProperty, ...] = DEFAULT_PROPERTIES

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for prop in self.properties:
            if prop.property_id in seen:
                raise ValueError(f"duplicate property id {prop.property_id!r}")
            seen.add(prop.property_id)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.properties)

    def __len__(self) -> int:
        return len(self.properties)

    def by_id(self, property_id: str) -> SafetyProperty:
        for prop in self.properties:
            if prop.property_id == property_id:
                return prop
        raise KeyError(f"no property {property_id!r} in this specification")

    def violated_by(self, state: OperatingState) -> tuple[SafetyProperty, ...]:
        """Properties whose bad predicate is true in this state.

        Note this is a *state* check, not a temporal one: it says the system is
        currently in a state the property is about, not that the property has
        been violated. Persistent and bounded properties are only violated over
        a trace, which is what the runtime monitors track.
        """
        return tuple(prop for prop in self.properties if not prop.holds_for(state))

    def to_payload(self) -> dict[str, Any]:
        return {
            "properties": [prop.to_payload() for prop in self.properties],
            "count": len(self.properties),
        }


def build_specification(config: Mapping[str, Any] | None) -> Specification:
    """Construct the specification declared in configuration."""
    if not config:
        return Specification()
    declared = config.get("properties")
    if declared is None:
        return Specification()
    if not isinstance(declared, list):
        raise ValueError("specification 'properties' must be a list")

    properties: list[SafetyProperty] = []
    for index, entry in enumerate(declared):
        if not isinstance(entry, dict):
            raise ValueError(f"property {index} must be a mapping")
        params = dict(entry)
        identifier = str(params.pop("id", f"SAFE-{index + 1}"))
        operator_name = str(params.pop("operator", "persistent"))
        try:
            operator = Operator(operator_name)
        except ValueError as error:
            raise ValueError(
                f"{identifier}: unknown operator {operator_name!r}; "
                f"available: {', '.join(op.value for op in Operator)}"
            ) from error
        try:
            properties.append(
                SafetyProperty(
                    property_id=identifier,
                    operator=operator,
                    predicate=str(params.pop("predicate", "stockout")),
                    bound=float(params.pop("bound", 0.05)),
                    horizon=int(params.pop("horizon", 0)),
                    description=str(params.pop("description", "")),
                )
            )
        except TypeError as error:
            raise ValueError(f"{identifier}: rejected its parameters: {error}") from error
        if params:
            raise ValueError(f"{identifier}: unknown keys {sorted(params)}")
    return Specification(properties=tuple(properties))


@dataclass(slots=True)
class TraceWindow:
    """A bounded window of recent states, for temporal predicates.

    Kept explicit rather than folded into the monitor because the shield and
    the monitors read the same history, and two independently maintained copies
    of it would eventually disagree about what happened.
    """

    capacity: int = 16
    states: list[OperatingState] = field(default_factory=list)

    def append(self, state: OperatingState) -> None:
        self.states.append(state)
        if len(self.states) > self.capacity:
            del self.states[: len(self.states) - self.capacity]

    def consecutive_true(self, predicate: str) -> int:
        """How many of the most recent states satisfy the bad predicate."""
        test = PREDICATES[predicate]
        count = 0
        for state in reversed(self.states):
            if not test(state):
                break
            count += 1
        return count

    def recent_true(self, predicate: str, horizon: int) -> int:
        """How many of the last ``horizon`` states satisfy the bad predicate."""
        test = PREDICATES[predicate]
        window = self.states[-horizon:] if horizon > 0 else self.states
        return sum(1 for state in window if test(state))
