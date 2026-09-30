"""Runtime monitors.

Model checking proves properties of an abstraction. The executed system is not
the abstraction, so something must watch the concrete run for the divergences
the abstraction cannot rule out. That is what these monitors are, and the pair
— static verification plus dynamic monitoring — is what licenses a claim of
safety by construction rather than safety by absence of observed incidents.

Three things are watched, and the third is the one usually left out.

* **Property violations in the concrete trace.** A persistent property is
  violated over a window of states, so it can only be checked as the run
  unfolds.
* **Properties the abstraction could not discharge.** When the checker reports
  a property as not expressible, the monitor is the only thing standing behind
  it, and the run record says so explicitly.
* **Divergence between the verified model and observed behaviour.** If the
  abstraction said a transition was impossible and it happens, the verification
  result no longer transfers — and the run must record that its own guarantee
  lapsed rather than continuing to report it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xai_gov.core.logging import get_logger
from xai_gov.io.decision_record import OperatingState
from xai_gov.verification.model_checker import Abstraction
from xai_gov.verification.specification import (
    Operator,
    SafetyProperty,
    Specification,
    TraceWindow,
)

_LOG = get_logger("verification.monitors")


#: A set of abstract transitions the verified model contemplated. Named so the
#: monitor's signature stays readable: the type is what carries the meaning,
#: and inlining it made the line unreadable at exactly the point a reader needs
#: to understand what is being compared.
KnownTransitions = set[tuple[tuple[int, int], tuple[int, int]]]


@dataclass(frozen=True, slots=True)
class Violation:
    """One observed violation of a property in the concrete trace."""

    property_id: str
    period: int
    detail: str

    def to_payload(self) -> dict[str, Any]:
        return {"property_id": self.property_id, "period": self.period, "detail": self.detail}


@dataclass(slots=True)
class RuntimeMonitor:
    """Watches the concrete run against the specification."""

    specification: Specification = field(default_factory=Specification)
    abstraction: Abstraction = field(default_factory=Abstraction)
    undischarged: frozenset[str] = frozenset()
    periods: int = 0
    violations: list[Violation] = field(default_factory=list)
    model_divergences: int = 0
    _window: TraceWindow = field(default_factory=lambda: TraceWindow(capacity=32), repr=False)
    _previous: tuple[int, int] | None = field(default=None, repr=False)
    _transitions_seen: set[tuple[tuple[int, int], tuple[int, int]]] = field(
        default_factory=set, repr=False
    )

    def observe(
        self, state: OperatingState, *, known_transitions: KnownTransitions | None = None
    ) -> tuple[Violation, ...]:
        """Fold one period in and return any violations it completes."""
        self.periods += 1
        self._window.append(state)

        current = self.abstraction.abstract(state.inventory, state.backlog).to_key()
        if self._previous is not None:
            transition = (self._previous, current)
            self._transitions_seen.add(transition)
            # A transition the verified model never contemplated means the
            # abstraction was not conservative, and the checking result stops
            # transferring from here on.
            if known_transitions is not None and transition not in known_transitions:
                self.model_divergences += 1
                _LOG.warning(
                    "observed transition absent from the verified model",
                    extra={"period": state.period, "from": self._previous, "to": current},
                )
        self._previous = current

        found: list[Violation] = []
        for prop in self.specification:
            violation = self._check(prop, state)
            if violation is not None:
                found.append(violation)
                self.violations.append(violation)
        return tuple(found)

    def _check(self, prop: SafetyProperty, state: OperatingState) -> Violation | None:
        if prop.operator is Operator.PERSISTENT:
            run = self._window.consecutive_true(prop.predicate)
            if run >= prop.horizon:
                return Violation(
                    property_id=prop.property_id,
                    period=state.period,
                    detail=(
                        f"{prop.predicate} held for {run} consecutive periods, "
                        f"the property allows {prop.horizon - 1}"
                    ),
                )
            return None

        if prop.operator is Operator.BOUNDED_EVENTUALLY:
            # Violated when the bad state occurs more often within the window
            # than the bound allows as a frequency. This is the concrete
            # analogue of the checked probability, and it is deliberately a
            # frequency rather than a single occurrence: one bad period inside
            # a bounded-eventually property is exactly what the bound permits.
            hits = self._window.recent_true(prop.predicate, prop.horizon)
            allowed = max(int(prop.bound * prop.horizon), 0)
            if hits > allowed and len(self._window.states) >= prop.horizon:
                return Violation(
                    property_id=prop.property_id,
                    period=state.period,
                    detail=(
                        f"{prop.predicate} occurred {hits} times in {prop.horizon} periods, "
                        f"above the {allowed} the bound {prop.bound} permits"
                    ),
                )
            return None

        if not prop.holds_for(state):
            return Violation(
                property_id=prop.property_id,
                period=state.period,
                detail=f"{prop.predicate} became true",
            )
        return None

    @property
    def violation_rate(self) -> float:
        return len(self.violations) / self.periods if self.periods else 0.0

    @property
    def guarantee_intact(self) -> bool:
        """Whether the verification result still transfers to this run.

        False as soon as the concrete system leaves the verified model. A run
        that reports its properties as verified while having diverged from the
        model they were verified on is making a claim it cannot support.
        """
        return self.model_divergences == 0

    def by_property(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for violation in self.violations:
            counts[violation.property_id] = counts.get(violation.property_id, 0) + 1
        return dict(sorted(counts.items()))

    def to_payload(self) -> dict[str, Any]:
        return {
            "periods": self.periods,
            "violations": len(self.violations),
            "violation_rate": round(self.violation_rate, 6),
            "by_property": self.by_property(),
            "monitored_only": sorted(self.undischarged),
            "model_divergences": self.model_divergences,
            "guarantee_intact": self.guarantee_intact,
            "first_violations": [v.to_payload() for v in self.violations[:5]],
        }


@dataclass(slots=True)
class NoMonitor:
    """Control arm: nothing is watched."""

    periods: int = 0

    def observe(
        self, state: OperatingState, *, known_transitions: Any = None
    ) -> tuple[Violation, ...]:
        del state, known_transitions
        self.periods += 1
        return ()

    @property
    def violation_rate(self) -> float:
        return 0.0

    @property
    def guarantee_intact(self) -> bool:
        return False

    def to_payload(self) -> dict[str, Any]:
        return {"periods": self.periods, "violations": None, "guarantee_intact": False}


MonitorLike = RuntimeMonitor | NoMonitor
