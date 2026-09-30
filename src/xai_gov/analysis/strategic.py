"""Strategic indicators: the value of governance, computed rather than defined.

The protocol names three organizational indicators — return on governed
decision, total value creation, and the value of governance as value of
information — and the frontier analysis is built on coverage, budget and
service level instead. That substitution is right for the impossibility
result, whose three objectives are exactly those, and wrong for the management
audience the protocol addresses. This module closes the gap.

All three are *paired* quantities: each is a difference between a governed arm
and its ungoverned counterpart on the same seed, the same network and the same
demand path. None of them is computable from a single run, which is why they
live in the analysis layer rather than in the per-run KPI layers. A per-run
"ROIDG" would have to invent the counterfactual it divides by.

The value of governance is the one that matters most and is easiest to state
wrongly. It is not the benefit of governance minus its cost — that is total
value creation. It is the value of *having the signal at all*, in the sense of
value of information: the difference in expected outcome between a decision
maker who can see the governance signal and one who cannot. The paired
ungoverned arm supplies exactly that counterfactual.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StrategicValue:
    """The organizational indicators for one governed/ungoverned pair."""

    arm: str
    baseline: str
    pairs: int
    benefit_governed: float
    benefit_ungoverned: float
    cost_of_governance: float
    rgd: float | None
    cost_per_decision: float
    tvc: float | None
    value_of_governance: float
    shadow_price: float | None

    @property
    def governance_pays(self) -> bool | None:
        """Whether governance created net value on this pair.

        None when the return is undefined, which happens when governance cost
        nothing — a free intervention is not infinitely profitable, it is
        outside the ratio's domain, and returning a large number there would be
        an artifact of dividing by zero.
        """
        return None if self.tvc is None else self.tvc > 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "baseline": self.baseline,
            "pairs": self.pairs,
            "benefit_governed": round(self.benefit_governed, 6),
            "benefit_ungoverned": round(self.benefit_ungoverned, 6),
            "cost_of_governance": round(self.cost_of_governance, 6),
            "rgd": None if self.rgd is None else round(self.rgd, 6),
            "cost_per_decision": round(self.cost_per_decision, 6),
            "tvc": None if self.tvc is None else round(self.tvc, 6),
            "value_of_governance": round(self.value_of_governance, 6),
            "shadow_price": (
                None if self.shadow_price is None else round(self.shadow_price, 6)
            ),
            "governance_pays": self.governance_pays,
            "definitions": {
                "rgd": "(B_governed - B_ungoverned) / C_governance",
                "tvc": "RGD - cost per governed decision",
                "value_of_governance": (
                    "E[V | governance signal available] - E[V | unavailable], "
                    "estimated from the paired ungoverned arm"
                ),
            },
        }


def strategic_value(
    *,
    arm: str,
    baseline: str,
    governed: list[dict[str, Any]],
    ungoverned: list[dict[str, Any]],
    benefit_indicator: str = "operational.service_level",
) -> StrategicValue | None:
    """Compute the strategic indicators for one arm against its baseline.

    ``governed`` and ``ungoverned`` are KPI trees for the same replicates, in
    the same order. Pairs where either side lacks the indicator are dropped
    rather than filled: substituting a mean for a missing side would
    manufacture agreement between the arms, and the whole point of these
    indicators is the difference between them.
    """
    pairs = [
        (g, u)
        for g, u in zip(governed, ungoverned, strict=False)
        if _dig(g, benefit_indicator) is not None
        and _dig(u, benefit_indicator) is not None
    ]
    if not pairs:
        return None

    benefit_g = statistics.fmean(_dig(g, benefit_indicator) or 0.0 for g, _ in pairs)
    benefit_u = statistics.fmean(_dig(u, benefit_indicator) or 0.0 for _, u in pairs)
    cost = statistics.fmean(
        _dig(g, "governance.total_intervention_cost") or 0.0 for g, _ in pairs
    )
    per_decision = statistics.fmean(
        _dig(g, "governance.cost_per_decision")
        or (_dig(g, "governance.total_intervention_cost") or 0.0)
        / max(_dig(g, "governance.decisions") or 1.0, 1.0)
        for g, _ in pairs
    )
    shadow = statistics.fmean(
        _dig(g, "governance.shadow_price") or 0.0 for g, _ in pairs
    )

    # Undefined rather than infinite when governance was free. A ratio whose
    # denominator is zero has no value, and a large sentinel would propagate
    # into the frontier as a spuriously excellent arm.
    rgd = (benefit_g - benefit_u) / cost if cost > 1e-9 else None
    tvc = None if rgd is None else rgd - per_decision

    return StrategicValue(
        arm=arm,
        baseline=baseline,
        pairs=len(pairs),
        benefit_governed=benefit_g,
        benefit_ungoverned=benefit_u,
        cost_of_governance=cost,
        rgd=rgd,
        cost_per_decision=per_decision,
        tvc=tvc,
        # The value of the signal itself, in the units of the benefit
        # indicator. Signed on purpose: governance that destroys value has a
        # negative value of information, which is a finding rather than an
        # error, and the campaign has already produced one.
        value_of_governance=benefit_g - benefit_u,
        shadow_price=shadow if shadow > 0.0 else None,
    )


def _dig(tree: Any, path: str) -> float | None:
    node = tree
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, int | float):
        return None
    return float(node)
