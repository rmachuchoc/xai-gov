"""Running a campaign and reporting what it found.

This is the piece that turns the instrument into evidence. It executes the
declared arms at many seeds, applies the sealed exclusion criteria, tests each
preregistered hypothesis with anytime-valid statistics, and writes a report
whose every number can be traced to a run identifier.

The report states its own limits. A hypothesis whose evidence has not reached
the threshold is reported as insufficient evidence rather than as no effect;
a campaign whose seal does not match is marked exploratory throughout; and an
excluded run is named with its reason rather than quietly dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xai_gov.analysis.anytime import confidence_sequence, directional_evidence
from xai_gov.analysis.campaign import CampaignRunner, CellResult, plan_campaign
from xai_gov.analysis.frontier import compute_frontier
from xai_gov.analysis.preregistration import Preregistration, verify_seal
from xai_gov.core.hashing import canonical_json
from xai_gov.core.logging import get_logger
from xai_gov.core.settings import Settings

_LOG = get_logger("analysis.report")

CAMPAIGN_REPORT_NAME = "campaign_report.json"


@dataclass(frozen=True, slots=True)
class HypothesisResult:
    """What a campaign concluded about one preregistered claim."""

    hypothesis_id: str
    indicator: str
    treatment: str
    control: str
    direction: str
    pairs: int
    mean_difference: float
    evidence: dict[str, Any]
    interval: dict[str, Any]
    supported: bool
    verdict: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.hypothesis_id,
            "indicator": self.indicator,
            "treatment": self.treatment,
            "control": self.control,
            "direction": self.direction,
            "pairs": self.pairs,
            "mean_difference": round(self.mean_difference, 6),
            "evidence": self.evidence,
            "confidence_sequence": self.interval,
            "supported": self.supported,
            "verdict": self.verdict,
        }


def _orient(differences: list[float], direction: str) -> list[float]:
    """Point the differences the way the sealed hypothesis predicted.

    A 'greater' claim is tested on the raw differences, a 'less' claim on their
    negation, and a two-sided claim on their magnitude. Orienting here rather
    than reinterpreting the verdict afterwards is what keeps the test the one
    that was preregistered.
    """
    if direction == "less":
        return [-value for value in differences]
    if direction == "different":
        return [abs(value) for value in differences]
    return list(differences)


def test_hypothesis(
    runner: CampaignRunner,
    plan: Preregistration,
    hypothesis_id: str,
) -> HypothesisResult:
    """Evaluate one hypothesis against the campaign's paired differences.

    Both directions are tested. A small forward e-value means the bet lost, not
    that the effect runs the other way, so establishing the opposite direction
    requires its own e-value against its own null — and the familywise
    threshold accounts for the seven preregistered hypotheses, which anytime
    validity does not do on its own.
    """
    hypothesis = plan.by_id(hypothesis_id)
    differences = runner.paired(
        hypothesis.treatment, hypothesis.control, hypothesis.indicator
    )

    if len(differences) < 2:
        return HypothesisResult(
            hypothesis_id=hypothesis_id,
            indicator=hypothesis.indicator,
            treatment=hypothesis.treatment,
            control=hypothesis.control,
            direction=hypothesis.direction,
            pairs=len(differences),
            mean_difference=0.0,
            evidence={},
            interval={},
            supported=False,
            verdict=(
                f"only {len(differences)} usable pairs; the indicator "
                f"{hypothesis.indicator!r} was unavailable in one or both arms, so "
                "the hypothesis is untested rather than refuted"
            ),
        )

    # The direction is applied to the differences, not to the verdict. A
    # one-sided test built by reinterpreting a two-sided result after the fact
    # is the optional-stopping error in another costume.
    oriented = _orient(differences, hypothesis.direction)
    evidence = directional_evidence(
        oriented,
        alpha=plan.alpha,
        hypotheses=len(plan.hypotheses),
        two_sided=hypothesis.direction == "different",
        indicator=hypothesis.indicator.split(".")[-1],
    )

    scale = max(max(abs(value) for value in oriented), 1e-9)
    # On the *raw* differences, not the oriented ones. Orientation is an
    # internal device for the one-sided test; a reader comparing the interval
    # against the raw difference in the adjacent column must not find them on
    # opposite sign conventions. Four rows of the first table read as
    # contradictions before this: a positive difference beside an entirely
    # negative interval.
    interval = confidence_sequence(differences, alpha=plan.alpha, bound=scale)
    mean_difference = sum(differences) / len(differences)

    return HypothesisResult(
        hypothesis_id=hypothesis_id,
        indicator=hypothesis.indicator,
        treatment=hypothesis.treatment,
        control=hypothesis.control,
        direction=hypothesis.direction,
        pairs=len(differences),
        mean_difference=mean_difference,
        evidence=evidence.to_payload(),
        interval=interval.to_payload(),
        supported=evidence.supported,
        verdict=evidence.verdict,
    )


@dataclass(slots=True)
class CampaignReport:
    """Everything a campaign produced, in one traceable artifact."""

    plan: Preregistration
    seal_status: dict[str, Any]
    runner_summary: dict[str, Any]
    hypotheses: tuple[HypothesisResult, ...]
    frontier: dict[str, Any]
    cells: tuple[dict[str, Any], ...]
    hierarchical: dict[str, Any] = field(default_factory=dict)
    strategic: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        supported = [h.hypothesis_id for h in self.hypotheses if h.supported]
        refuted = [
            h.hypothesis_id
            for h in self.hypotheses
            if (h.evidence or {}).get("refuted")
        ]
        untested = [h.hypothesis_id for h in self.hypotheses if h.pairs < 2]
        return {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "campaign": self.plan.campaign,
            "plan": self.plan.to_payload(),
            "plan_hash": self.plan.plan_hash,
            "seal": self.seal_status,
            "campaign_summary": self.runner_summary,
            "hypotheses": [h.to_payload() for h in self.hypotheses],
            "frontier": self.frontier,
            "hierarchical": self.hierarchical,
            "strategic": self.strategic,
            "cells": list(self.cells),
            "multiplicity": {
                "hypotheses": len(self.hypotheses),
                "alpha": self.plan.alpha,
                "per_comparison_threshold": round(1.0 / self.plan.alpha, 4),
                "familywise_threshold": round(
                    len(self.hypotheses) / self.plan.alpha, 4
                ),
                "primary": "familywise",
                "procedure": (
                    "union bound over e-values: with k preregistered hypotheses, "
                    "requiring e >= k/alpha controls the familywise error rate at "
                    "alpha. Anytime validity handles optional stopping and says "
                    "nothing about multiplicity, so both are declared separately."
                ),
            },
            "findings": {
                "supported": supported,
                "refuted_in_the_opposite_direction": refuted,
                "inconclusive": [
                    h.hypothesis_id
                    for h in self.hypotheses
                    if not h.supported
                    and not (h.evidence or {}).get("refuted")
                    and h.pairs >= 2
                ],
                "untested": untested,
            },
            # Stated once, prominently, because it governs how every number
            # above may be read.
            "reporting_status": (
                "confirmatory"
                if self.seal_status.get("matches")
                else "exploratory: the plan does not match its seal"
            ),
        }

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(canonical_json(self.to_payload()) + "\n", encoding="utf-8")
        return path


def run_campaign(
    *,
    plan: Preregistration,
    arms: dict[str, Path],
    settings: Settings,
    project_root: Path,
    replicates: int,
    base_seed: int,
    sealed_hash: str | None = None,
    workers: int = 1,
    resume: bool = True,
) -> CampaignReport:
    """Execute a campaign and produce its report."""
    missing = sorted(set(plan.arms()) - set(arms))
    if missing:
        raise ValueError(
            f"the plan names arms with no experiment file: {missing}; "
            f"provided: {sorted(arms)}"
        )
    if replicates < plan.minimum_replicates:
        _LOG.warning(
            "campaign is under-powered against its own plan",
            extra={"replicates": replicates, "required": plan.minimum_replicates},
        )

    cells = plan_campaign(arms=arms, replicates=replicates, base_seed=base_seed)
    runner = CampaignRunner(
        settings=settings, project_root=project_root, workers=workers, resume=resume
    )
    results = runner.run(cells)

    hypotheses = tuple(
        test_hypothesis(runner, plan, hypothesis.hypothesis_id)
        for hypothesis in plan.hypotheses
    )

    # The frontier is computed over arm means of the included cells.
    arm_values: dict[str, dict[str, float]] = {}
    from xai_gov.analysis.frontier import THEOREM_3_OBJECTIVES

    for arm in sorted(arms):
        collected: dict[str, list[float]] = {}
        for result in results:
            if not result.included or result.cell.arm != arm:
                continue
            for objective in THEOREM_3_OBJECTIVES:
                value = result.indicator(objective.indicator)
                if value is not None:
                    collected.setdefault(objective.name, []).append(value)
        if collected:
            arm_values[arm] = {
                name: sum(values) / len(values) for name, values in collected.items()
            }

    seal_status = (
        verify_seal(plan, sealed_hash).to_payload()
        if sealed_hash is not None
        else {
            "matches": False,
            "status": "no seal supplied; this campaign is exploratory",
        }
    )

    return CampaignReport(
        plan=plan,
        seal_status=seal_status,
        runner_summary=runner.summary(),
        hypotheses=hypotheses,
        frontier=compute_frontier(arm_values).to_payload(),
        cells=tuple(result.to_payload() for result in results),
        hierarchical=_decompose_indicators(results),
        strategic=_strategic_indicators(results, arms),
    )


def _decompose_indicators(results: tuple[CellResult, ...]) -> dict[str, Any]:
    """Variance decomposition over the indicators the hypotheses use.

    Answers the question a table of per-arm means implicitly asks and never
    states: how much of the visible spread belongs to the configuration rather
    than to the seed. A pairwise test cannot answer it, because it compares two
    arms without reference to the variance across all of them.
    """
    from xai_gov.analysis.artifacts import CELL_INDICATORS
    from xai_gov.analysis.hierarchical import decompose_all

    values: dict[str, dict[str, list[float]]] = {}
    for result in results:
        if not result.included:
            continue
        for indicator in CELL_INDICATORS:
            value = result.indicator(indicator)
            if value is not None:
                values.setdefault(indicator, {}).setdefault(result.cell.arm, []).append(
                    value
                )
    return decompose_all(values)


def _strategic_indicators(
    results: tuple[CellResult, ...], arms: dict[str, Path]
) -> dict[str, Any]:
    """Return on governed decision, total value creation, value of governance.

    Each is a paired quantity against an ungoverned baseline, so the baseline
    must exist in the campaign. Without it these are not merely unavailable but
    undefined — a per-run return on governance would have to invent the
    counterfactual it divides by — and the report says so rather than omitting
    the layer silently.
    """
    from xai_gov.analysis.strategic import strategic_value

    baseline = next(
        (name for name in ("ungoverned", "unshielded") if name in arms), None
    )
    if baseline is None:
        return {
            "status": "not_available",
            "requires": (
                "an ungoverned baseline arm; these indicators are differences "
                "against one and are undefined without it"
            ),
        }

    by_arm: dict[str, dict[int, dict[str, Any]]] = {}
    for result in results:
        if result.included:
            by_arm.setdefault(result.cell.arm, {})[result.cell.replicate] = result.kpis

    control = by_arm.get(baseline, {})
    computed: dict[str, Any] = {}
    for arm, replicates in sorted(by_arm.items()):
        if arm == baseline:
            continue
        shared = sorted(set(replicates) & set(control))
        value = strategic_value(
            arm=arm,
            baseline=baseline,
            governed=[replicates[r] for r in shared],
            ungoverned=[control[r] for r in shared],
        )
        if value is not None:
            computed[arm] = value.to_payload()

    return {"status": "available", "baseline": baseline, "arms": computed}
