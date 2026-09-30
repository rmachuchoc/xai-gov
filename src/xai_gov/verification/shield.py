"""The synthesized shield.

Until now the governance agent could only refuse: every intervention resolved
to holding. That is safe in one direction and unsafe in the other — cancelling
a replenishment during a stockout is itself a way to violate a safety property,
and an agent whose only tool is refusal will do it.

The shield closes that gap. It is a filter that sits between the governance
verdict and execution, and it does two things a rule-based safe mode cannot:

1. **It is derived from the specification, not written by hand.** The set of
   admissible actions in each state is computed from the properties, so adding
   a property changes the shield automatically. A hand-written safe mode drifts
   from the specification the first time either is edited alone.
2. **It substitutes rather than blocks.** When the proposed action would enter
   a state the specification forbids, the shield picks the closest admissible
   action — usually a smaller order, sometimes a larger one when the violation
   is a stockout rather than an overstock.

Minimal intervention is the design rule. The shield changes the action as
little as the specification allows, because every unit of substitution is
performance taken from a policy that may well have been right. That is the
quantity Proposition 2 bounds, and `activation_rate` × mean substitution is
what the run reports as the price of verified safety.

What the shield does not do: override a human escalation. If governance
referred a decision to a person, the shield may constrain what is executed in
the meantime but the escalation stands. Safety machinery that silently
resolves a question posed to a human has removed the oversight it was meant
to support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xai_gov.core.logging import get_logger
from xai_gov.io.decision_record import ActionKind, OperatingState, ShieldOutcome
from xai_gov.verification.specification import (
    PREDICATES,
    SafetyProperty,
    Specification,
    TraceWindow,
)

_LOG = get_logger("verification.shield")


@dataclass(frozen=True, slots=True)
class ActionEnvelope:
    """The admissible range of order quantities in a state."""

    minimum: float
    maximum: float
    binding_property: str | None
    reason: str

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError(
                f"empty envelope [{self.minimum}, {self.maximum}]: the specification "
                "forbids every action in this state, which means the properties are "
                "jointly unsatisfiable here and must be revised"
            )

    def contains(self, quantity: float) -> bool:
        return self.minimum - 1e-9 <= quantity <= self.maximum + 1e-9

    def clamp(self, quantity: float) -> float:
        """The admissible action closest to the proposed one."""
        return min(max(quantity, self.minimum), self.maximum)

    def to_payload(self) -> dict[str, Any]:
        return {
            "minimum": round(self.minimum, 6),
            "maximum": round(self.maximum, 6),
            "binding_property": self.binding_property,
            "reason": self.reason,
        }


@dataclass(slots=True)
class Shield:
    """A correct-by-construction filter synthesized from the specification."""

    specification: Specification = field(default_factory=Specification)
    horizon_lookahead: int = 1
    max_order: float = 200.0
    activations: int = 0
    decisions: int = 0
    substituted_units: float = 0.0
    escalations_preserved: int = 0
    #: Conflicts between properties, counted rather than logged. A run that
    #: emits one warning per conflict produces thousands of lines and reports
    #: the rate nowhere; the rate is what a specification review needs.
    conflicts: int = 0
    conflicting_pairs: dict[str, int] = field(default_factory=dict)
    _window: TraceWindow = field(default_factory=TraceWindow, repr=False)

    def __post_init__(self) -> None:
        if self.horizon_lookahead < 1:
            raise ValueError("horizon_lookahead must be at least 1")
        if self.max_order <= 0.0:
            raise ValueError("max_order must be positive")

    # -- synthesis --------------------------------------------------------
    def envelope(self, state: OperatingState) -> ActionEnvelope:
        """The admissible action range in this state, derived from the spec.

        Each property contributes a constraint, and the envelope is their
        intersection. A property that the state does not engage contributes
        nothing — the shield is inactive by default and earns each restriction.
        """
        minimum = 0.0
        maximum = min(state.capacity, self.max_order)
        binding: str | None = None
        reason = "no property is engaged in this state"
        lower_source: str | None = None
        upper_source: str | None = None

        for prop in self.specification:
            lower, upper, engaged = self._constraint_of(prop, state)
            if not engaged:
                continue
            if lower > minimum:
                minimum, binding, lower_source = lower, prop.property_id, prop.property_id
                reason = f"{prop.property_id} requires ordering at least {lower:.1f}"
            if upper < maximum:
                maximum, binding, upper_source = upper, prop.property_id, prop.property_id
                reason = f"{prop.property_id} caps the order at {upper:.1f}"

        if minimum > maximum:
            # The properties conflict here. Capacity wins, because ordering
            # beyond it is not an action the world can execute. The pair is
            # recorded so a specification review can see which properties are
            # jointly unsatisfiable and in how many reachable states.
            self.conflicts += 1
            pair = "::".join(
                sorted(filter(None, (lower_source, upper_source or "capacity")))
            )
            self.conflicting_pairs[pair] = self.conflicting_pairs.get(pair, 0) + 1
            minimum = maximum
            reason = f"{reason} (conflicting properties resolved in favour of capacity)"

        return ActionEnvelope(
            minimum=minimum, maximum=maximum, binding_property=binding, reason=reason
        )

    def _constraint_of(
        self, prop: SafetyProperty, state: OperatingState
    ) -> tuple[float, float, bool]:
        """One property's contribution: (lower bound, upper bound, engaged)."""
        test = PREDICATES[prop.predicate]
        consecutive = self._window.consecutive_true(prop.predicate)

        if prop.predicate in ("stockout", "inventory_critical"):
            threshold = 0.0 if prop.predicate == "stockout" else 5.0
            # Engaged when the state is at or approaching the bad region within
            # the lookahead: the shield must act before the violation, not
            # report it afterwards.
            projected = state.inventory + state.in_transit - state.demand_observed * (
                self.horizon_lookahead
            )
            if not (test(state) or projected <= threshold):
                return (0.0, float("inf"), False)
            # Order at least enough to clear the threshold with margin.
            shortfall = threshold + state.demand_observed - state.inventory - state.in_transit
            return (max(shortfall, 0.0), float("inf"), True)

        if prop.predicate == "backlog_critical":
            if not test(state) and consecutive == 0:
                return (0.0, float("inf"), False)
            # Backlog is cleared by supply, so this property also demands
            # ordering rather than restraining it. The cap comes from capacity,
            # which the envelope already applies.
            return (min(state.backlog, self.max_order), float("inf"), True)

        if prop.predicate == "capacity_exhausted":
            if not test(state):
                return (0.0, float("inf"), False)
            return (0.0, 0.0, True)

        return (0.0, float("inf"), False)

    # -- filtering --------------------------------------------------------
    def filter(
        self,
        *,
        state: OperatingState,
        action: ActionKind,
        quantity: float,
        escalated: bool = False,
    ) -> tuple[ActionKind, float, ShieldOutcome]:
        """Return the action that may execute, and what the shield did."""
        self.decisions += 1
        self._window.append(state)
        envelope = self.envelope(state)

        if envelope.contains(quantity):
            return action, quantity, ShieldOutcome()

        substituted = envelope.clamp(quantity)
        delta = abs(substituted - quantity)
        self.activations += 1
        self.substituted_units += delta
        if escalated:
            # The escalation stands; the shield only constrains what executes
            # while the human is deciding.
            self.escalations_preserved += 1

        substituted_action = ActionKind.REORDER if substituted > 0.0 else ActionKind.HOLD
        return (
            substituted_action,
            substituted,
            ShieldOutcome(
                activated=True,
                property_id=envelope.binding_property,
                substituted_action=substituted_action,
                substituted_quantity=round(substituted, 6),
            ),
        )

    # -- the price of safety ----------------------------------------------
    @property
    def activation_rate(self) -> float:
        return self.activations / self.decisions if self.decisions else 0.0

    @property
    def mean_substitution(self) -> float:
        """Average units moved per activation.

        Reported per activation rather than per decision: the question
        Proposition 2 asks is how large a correction the shield makes when it
        makes one, and averaging over untouched decisions would hide it.
        """
        return self.substituted_units / self.activations if self.activations else 0.0

    @property
    def conflict_rate(self) -> float:
        """Share of decisions in which the specification was unsatisfiable.

        A specification that is jointly unsatisfiable in reachable states is a
        formal-methods finding rather than a runtime nuisance, so it is reported
        as a rate alongside the activation rate rather than left in a log.
        """
        return self.conflicts / self.decisions if self.decisions else 0.0

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "specification": self.specification.to_payload(),
            "horizon_lookahead": self.horizon_lookahead,
            "decisions": self.decisions,
            "activations": self.activations,
            "activation_rate": round(self.activation_rate, 6),
            "substituted_units": round(self.substituted_units, 6),
            "mean_substitution": round(self.mean_substitution, 6),
            "escalations_preserved": self.escalations_preserved,
            "conflicts": self.conflicts,
            "conflict_rate": round(self.conflict_rate, 6),
            "conflicting_pairs": dict(sorted(self.conflicting_pairs.items())),
        }
        if self.conflicts:
            # Logged at debug, not warning: one line per run turns a campaign's
            # console into hundreds of identical messages and buries the phase
            # summaries. The rate is in the payload above, and the diagnostics
            # layer is what surfaces it once per campaign.
            _LOG.debug(
                "specification unsatisfiable in reachable states",
                extra={
                    "conflicts": self.conflicts,
                    "rate": round(self.conflict_rate, 4),
                    "pairs": sorted(self.conflicting_pairs),
                },
            )
        return payload


@dataclass(slots=True)
class NoShield:
    """The control arm: nothing is filtered.

    Required for RQ8. The price of verified safety is the difference between a
    run with the shield and the same run without it, and without this arm that
    difference cannot be computed.
    """

    decisions: int = 0

    def filter(
        self,
        *,
        state: OperatingState,
        action: ActionKind,
        quantity: float,
        escalated: bool = False,
    ) -> tuple[ActionKind, float, ShieldOutcome]:
        del state, escalated
        self.decisions += 1
        return action, quantity, ShieldOutcome()

    @property
    def activation_rate(self) -> float:
        return 0.0

    @property
    def mean_substitution(self) -> float:
        return 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "specification": None,
            "decisions": self.decisions,
            "activations": 0,
            "conflicts": 0,
            "conflict_rate": 0.0,
        }


ShieldLike = Shield | NoShield
