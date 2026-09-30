"""Autonomy regimes as allocations of decision rights.

H0 through H3 are not permission flags with increasing values. Each is a
different distribution of decision rights between the system and the
organization, and the protocol's contribution rests on treating them that
way: what the system may do, what the organization retains, and — the part
usually left implicit — *when* the organization gets to exercise what it
retains.

    H0  recommend only          organization decides everything, ex ante
    H1  act within a range      veto ex ante on anything out of range
    H2  act, escalate on doubt   veto ex post, may reduce autonomy
    H3  full autonomy            audit afterwards, right of revocation

The temporal distinction is what makes the regimes economically different.
An ex ante veto costs oversight attention on every decision; an ex post veto
costs attention only on the decisions that turned out to matter, but pays for
that saving by allowing some unsafe actions to execute first. That trade is
what RQ2 measures, and it is only visible if the regimes are modeled as
different in kind rather than in degree.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from xai_gov.io.decision_record import ActionKind, GovernanceAction


class AutonomyRegime(StrEnum):
    H0 = "H0"
    H1 = "H1"
    H2 = "H2"
    H3 = "H3"


@dataclass(frozen=True, slots=True)
class RegimeRights:
    """The rights a regime grants and withholds."""

    regime: AutonomyRegime
    may_act_without_approval: bool
    veto_is_ex_ante: bool
    may_escalate: bool
    may_reduce_autonomy: bool
    bounded_quantity: bool
    description: str

    def permits(self, action: ActionKind, *, quantity: float, bound: float) -> bool:
        """Whether the system may execute this action unaided."""
        if not self.may_act_without_approval:
            # Holding requires no authority: doing nothing is always available.
            return action is ActionKind.HOLD
        return not (self.bounded_quantity and quantity > bound)

    def default_action_when_not_permitted(self) -> GovernanceAction:
        """What the regime does with an action it does not authorize.

        H0 escalates, because the organization holds the decision and must be
        asked. H1 vetoes, because the range was agreed in advance and stepping
        outside it is a violation rather than a question. The distinction
        matters for the attention budget: escalation consumes it, a veto does
        not.
        """
        if self.regime is AutonomyRegime.H0:
            return GovernanceAction.ESCALATE
        return GovernanceAction.VETO

    def to_payload(self) -> dict[str, Any]:
        return {
            "regime": self.regime.value,
            "may_act_without_approval": self.may_act_without_approval,
            "veto_is_ex_ante": self.veto_is_ex_ante,
            "may_escalate": self.may_escalate,
            "may_reduce_autonomy": self.may_reduce_autonomy,
            "bounded_quantity": self.bounded_quantity,
            "description": self.description,
        }


_RIGHTS: dict[AutonomyRegime, RegimeRights] = {
    AutonomyRegime.H0: RegimeRights(
        regime=AutonomyRegime.H0,
        may_act_without_approval=False,
        veto_is_ex_ante=True,
        may_escalate=True,
        may_reduce_autonomy=False,
        bounded_quantity=False,
        description="recommend only; the organization decides every action",
    ),
    AutonomyRegime.H1: RegimeRights(
        regime=AutonomyRegime.H1,
        may_act_without_approval=True,
        veto_is_ex_ante=True,
        may_escalate=True,
        may_reduce_autonomy=False,
        bounded_quantity=True,
        description="act within an agreed range; ex ante veto outside it",
    ),
    AutonomyRegime.H2: RegimeRights(
        regime=AutonomyRegime.H2,
        may_act_without_approval=True,
        veto_is_ex_ante=False,
        may_escalate=True,
        may_reduce_autonomy=True,
        bounded_quantity=False,
        description="act with conditional escalation; ex post veto and autonomy reduction",
    ),
    AutonomyRegime.H3: RegimeRights(
        regime=AutonomyRegime.H3,
        may_act_without_approval=True,
        veto_is_ex_ante=False,
        may_escalate=False,
        may_reduce_autonomy=False,
        bounded_quantity=False,
        description="full autonomy under audit; revocation only after the fact",
    ),
}


def rights_for(regime: AutonomyRegime | str) -> RegimeRights:
    """Look up the rights of a regime."""
    key = AutonomyRegime(regime) if isinstance(regime, str) else regime
    return _RIGHTS[key]


def available_regimes() -> tuple[str, ...]:
    return tuple(regime.value for regime in AutonomyRegime)


@dataclass(slots=True)
class AutonomyState:
    """The regime currently in force, and its history of reductions.

    Autonomy is reducible but not, within a run, restorable. That asymmetry is
    deliberate: restoring autonomy is an organizational decision made on
    evidence accumulated over time, not something the agent that just lost
    trust may grant itself. A run that ends at H0 having started at H2 is
    reporting a real outcome, and the protocol's option-value argument is
    about exactly that path.
    """

    initial: AutonomyRegime = AutonomyRegime.H2
    current: AutonomyRegime = AutonomyRegime.H2
    reductions: int = 0

    def __post_init__(self) -> None:
        self.current = self.initial

    @property
    def rights(self) -> RegimeRights:
        return rights_for(self.current)

    @property
    def reduced(self) -> bool:
        return self.current is not self.initial

    def reduce(self) -> bool:
        """Step one regime down. Returns False when already at H0."""
        order = [AutonomyRegime.H3, AutonomyRegime.H2, AutonomyRegime.H1, AutonomyRegime.H0]
        index = order.index(self.current)
        if index >= len(order) - 1:
            return False
        self.current = order[index + 1]
        self.reductions += 1
        return True

    def to_payload(self) -> dict[str, Any]:
        return {
            "initial": self.initial.value,
            "current": self.current.value,
            "reductions": self.reductions,
            "reduced": self.reduced,
            "rights": self.rights.to_payload(),
        }
