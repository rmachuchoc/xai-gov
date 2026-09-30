"""Explanatory fidelity.

An attribution is a claim about what matters. Fidelity checks that claim
against what actually happens when you intervene, and it exists because the
protocol refuses to let an unverified explanation enter a governance decision.

The measure is rank agreement between attributed importance and true
interventional effect, computed as a normalized Kendall tau over the levers.
Rank rather than magnitude, for a specific reason: SHAP values and order
quantities are in different units, so any magnitude comparison would be
measuring the scaling convention rather than the explanation. What we need to
know is whether the explanation *orders the levers correctly* — an operator
acts on the top of that list.

Two failure modes are distinguished, because they call for different responses:

* **Uninformative** — the explanation ranks levers no better than chance.
  The descriptor is dropped and the agent operates with declared uncertainty.
* **Anti-correlated** — the explanation ranks them reliably *backwards*. This
  is worse than useless and is flagged separately, because an operator
  following it would act on the least effective lever available while
  believing they had chosen the best one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class FidelityReport:
    """How well an attribution matches the interventional truth."""

    fidelity: float
    concordant: int
    discordant: int
    compared: int
    floor: float
    top_lever_agrees: bool
    verdict: str

    @property
    def informative(self) -> bool:
        return self.verdict == "informative"

    def to_payload(self) -> dict[str, Any]:
        return {
            "fidelity": round(self.fidelity, 6),
            "concordant": self.concordant,
            "discordant": self.discordant,
            "compared": self.compared,
            "floor": self.floor,
            "top_lever_agrees": self.top_lever_agrees,
            "verdict": self.verdict,
        }


def explanatory_fidelity(
    attributed: Mapping[str, float],
    interventional: Mapping[str, float],
    *,
    floor: float = 0.5,
) -> FidelityReport:
    """Compare an attribution against measured interventional effects.

    Returns a fidelity in [-1, 1]: 1 when the ordering matches exactly, 0 at
    chance, negative when the explanation is reliably backwards.
    """
    shared = sorted(set(attributed) & set(interventional))
    if len(shared) < 2:
        # With fewer than two levers there is no ordering to verify. Reporting
        # a fidelity of 1 here would let a single-lever node claim a perfectly
        # faithful explanation on no evidence at all.
        return FidelityReport(
            fidelity=0.0,
            concordant=0,
            discordant=0,
            compared=0,
            floor=floor,
            top_lever_agrees=False,
            verdict="not_verifiable",
        )

    concordant = discordant = 0
    for index, left in enumerate(shared):
        for right in shared[index + 1 :]:
            attributed_order = abs(attributed[left]) - abs(attributed[right])
            true_order = abs(interventional[left]) - abs(interventional[right])
            product = attributed_order * true_order
            if abs(attributed_order) < 1e-12 or abs(true_order) < 1e-12:
                continue  # a tie carries no ordering information either way
            if product > 0:
                concordant += 1
            else:
                discordant += 1

    compared = concordant + discordant
    if compared == 0:
        return FidelityReport(
            fidelity=0.0,
            concordant=0,
            discordant=0,
            compared=0,
            floor=floor,
            top_lever_agrees=False,
            verdict="not_verifiable",
        )

    fidelity = (concordant - discordant) / compared

    def top(source: Mapping[str, float]) -> str:
        return max(shared, key=lambda key: abs(source[key]))

    top_agrees = top(attributed) == top(interventional)

    if fidelity < -floor:
        verdict = "anti_correlated"
    elif fidelity < floor:
        verdict = "uninformative"
    else:
        verdict = "informative"

    return FidelityReport(
        fidelity=fidelity,
        concordant=concordant,
        discordant=discordant,
        compared=compared,
        floor=floor,
        top_lever_agrees=top_agrees,
        verdict=verdict,
    )
