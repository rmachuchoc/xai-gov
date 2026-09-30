"""Governance budget and its shadow price.

The protocol treats the cost of governance as a constraint, not a penalty
term. The difference is not cosmetic: a penalty with a hand-picked weight
produces a policy whose behaviour depends on an uninterpretable constant,
while a constraint with a Lagrange multiplier produces a policy plus a
*price* — κ*, the operational performance the organization forgoes for one
additional unit of oversight capacity. That number is the bridge between the
decision formalism and a management conversation, and it is the reason this
module exists rather than a `lambda_3` in a config file.

The multiplier is learned by dual ascent: overspending raises the price of
intervening, which suppresses intervention until spending returns to budget.
Two properties are enforced. κ is non-negative, because a constraint that is
slack has zero price, never a negative one. And the budget is expressed in
discounted terms so that a finite horizon and an infinite one are comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class GovernanceBudget:
    """Tracks discounted governance spend against its allowance.

    ``allowance`` is B in the protocol's constraint
    E[Σ γ^t c_gov(a_t)] ≤ B. ``dual_rate`` is the step size of the dual
    ascent; larger values track the constraint faster and oscillate more.
    """

    allowance: float = 20.0
    discount: float = 0.95
    dual_rate: float = 0.05
    multiplier: float = 0.0
    spent: float = 0.0
    discounted_spent: float = 0.0
    refused_demand: float = 0.0
    periods: int = 0
    interventions: int = 0
    refusals: int = 0
    _last_cost: float = 0.0
    _multiplier_trace: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.allowance < 0.0:
            raise ValueError("allowance must be non-negative")
        if not 0.0 < self.discount < 1.0:
            raise ValueError("discount must lie in (0, 1)")
        if self.dual_rate <= 0.0:
            raise ValueError("dual_rate must be positive")
        if self.multiplier < 0.0:
            raise ValueError("multiplier must be non-negative")

    # -- accounting -------------------------------------------------------
    def charge(self, cost: float, period: int) -> None:
        """Record one intervention's cost."""
        if cost < 0.0:
            raise ValueError("an intervention cost cannot be negative")
        self.spent += cost
        self.discounted_spent += (self.discount**period) * cost
        self.interventions += 1
        self._last_cost = cost

    def refuse(self, cost: float = 0.0) -> None:
        """Record an intervention the budget declined to fund.

        Counted separately from an approval: a decision left unintervened
        because the agent judged it safe and one left unintervened because the
        budget was exhausted are different events, and conflating them would
        make an exhausted budget look like confidence.

        The refused cost is retained because it is *demand for oversight that
        was turned away*, and the price of a constraint is driven by exactly
        that suppressed demand.
        """
        if cost < 0.0:
            raise ValueError("a refused cost cannot be negative")
        self.refusals += 1
        self.refused_demand += cost
        if cost > 0.0:
            self._last_cost = cost

    def tick(self) -> None:
        self.periods += 1

    # -- the constraint ---------------------------------------------------
    @property
    def remaining(self) -> float:
        return max(self.allowance - self.discounted_spent, 0.0)

    @property
    def exhausted(self) -> bool:
        """True when the budget can no longer fund another intervention.

        Defined against what an intervention actually costs, not against the
        allowance. Discounting means spend converges on the allowance without
        reaching it, so a bare ``discounted_spent >= allowance`` would report a
        budget as solvent while every request to it is being refused — the
        predicate and ``can_afford`` would disagree, which is the kind of
        inconsistency that makes a KPI unreadable.
        """
        if self._last_cost > 0.0:
            return not self.can_afford(self._last_cost)
        return self.remaining <= 0.0

    @property
    def utilization(self) -> float:
        return self.discounted_spent / self.allowance if self.allowance > 0.0 else 1.0

    def can_afford(self, cost: float) -> bool:
        return self.discounted_spent + cost <= self.allowance

    # -- the price --------------------------------------------------------
    def update_multiplier(self, period: int) -> float:
        """One step of dual ascent on the budget constraint.

        The gradient is the *relative* violation of the constraint, counting
        refused interventions as demand that was turned away:

            (discounted spend + refused demand) / B − 1

        Two properties follow, and both are what make κ a price rather than a
        tuning constant. It is scale free, so two arms with different
        allowances are comparable — an absolute gradient would report a larger
        budget as more strained merely because its numbers are bigger. And a
        binding budget keeps generating positive gradient through its
        refusals, whereas a slack one decays to zero. A run that never
        approaches its allowance reports κ* ≈ 0, correctly saying oversight
        capacity was not the binding scarcity.
        """
        del period  # the constraint is on the whole horizon, not on a pace
        if self.allowance <= 0.0:
            # With no allowance at all every request is refused; the price is
            # unbounded in principle, so the multiplier is left to grow.
            self.multiplier += self.dual_rate * max(self.refusals, 1)
        else:
            demand = self.discounted_spent + self.refused_demand
            self.multiplier = max(
                0.0, self.multiplier + self.dual_rate * (demand / self.allowance - 1.0)
            )
        self._multiplier_trace.append(self.multiplier)
        return self.multiplier

    @property
    def shadow_price(self) -> float:
        """κ*: the marginal value of one more unit of oversight capacity.

        Reported as the mean over the second half of the run. The early
        multiplier reflects the dual variable's own transient, not the
        economics of the problem, and averaging it in would understate the
        price.
        """
        if not self._multiplier_trace:
            return 0.0
        tail = self._multiplier_trace[len(self._multiplier_trace) // 2 :]
        return sum(tail) / len(tail)

    def effective_cost(self, cost: float) -> float:
        """Cost as the policy sees it: nominal cost priced by the multiplier.

        This is where the constraint changes behaviour. When the budget is
        slack, κ is zero and interventions cost their face value; as the
        budget binds, the same intervention becomes progressively more
        expensive to justify.
        """
        return cost * (1.0 + self.multiplier)

    def to_payload(self) -> dict[str, Any]:
        return {
            "allowance": self.allowance,
            "discount": self.discount,
            "dual_rate": self.dual_rate,
            "spent": round(self.spent, 6),
            "discounted_spent": round(self.discounted_spent, 6),
            "remaining": round(self.remaining, 6),
            "utilization": round(self.utilization, 6),
            "interventions": self.interventions,
            "refusals": self.refusals,
            "refused_demand": round(self.refused_demand, 6),
            "exhausted": self.exhausted,
            "multiplier": round(self.multiplier, 6),
            "shadow_price": round(self.shadow_price, 6),
        }


def build_budget(config: dict[str, Any]) -> GovernanceBudget:
    """Construct the budget from configuration."""
    if not config:
        return GovernanceBudget()
    try:
        return GovernanceBudget(**config)
    except TypeError as error:
        raise ValueError(f"governance budget rejected its parameters: {error}") from error
