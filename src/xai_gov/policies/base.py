"""Decision policies.

A policy proposes an action; it never executes one. The separation is the
whole point of the architecture: proposal (this layer), mediation
(governance), and enforcement (the shield) are distinct authorities, and
only the last two may alter what happens.

Every policy carries an ``id`` and a ``version`` that reach the decision
record, so a result can always be attributed to the exact decision rule
that produced it. Policies are also required to expose ``cost_units`` — a
declared compute budget — because the protocol compares policy families
under matched budget, and a comparison against an unbudgeted baseline is
not evidence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from xai_gov.io.decision_record import ActionKind

if TYPE_CHECKING:  # annotations only: importing these at runtime would make
    # policies depend on the simulation package, which depends on the engine,
    # which depends on governance, which depends on policies.
    import numpy as np

    from xai_gov.io.decision_record import OperatingState
    from xai_gov.simulation.network import NodeSpec


@dataclass(frozen=True, slots=True)
class PolicyProposal:
    """What a policy proposes, and the trace of why.

    ``features`` is the policy's own account of the quantities that drove
    the proposal. It is not an explanation: the causal descriptor of stage
    4 is derived independently and its fidelity is measured against the
    mechanism, not against this dictionary.
    """

    action: ActionKind
    quantity: float
    features: dict[str, float]
    rationale: str

    def __post_init__(self) -> None:
        if self.quantity < 0.0:
            raise ValueError("a proposed quantity cannot be negative")
        if self.action is ActionKind.HOLD and self.quantity != 0.0:
            raise ValueError("a hold must carry zero quantity")


class Policy(ABC):
    """Base class for every decision policy."""

    policy_id: str = "abstract"
    version: str = "0.0.0"
    cost_units: float = 0.0

    @abstractmethod
    def propose(
        self,
        *,
        spec: NodeSpec,
        state: OperatingState,
        rng: np.random.Generator,
    ) -> PolicyProposal:
        """Propose an action for one node in one period."""

    def to_payload(self) -> dict[str, Any]:
        return {"id": self.policy_id, "version": self.version, "cost_units": self.cost_units}


def clamp_to_capacity(quantity: float, state: OperatingState) -> float:
    """Never propose beyond free capacity.

    A proposal that cannot physically be accepted would be silently
    truncated downstream, and the record would then show an action that
    never existed. Truncating here keeps the record honest.
    """
    headroom = max(state.capacity - state.inventory - state.in_transit, 0.0)
    return float(min(quantity, headroom))
