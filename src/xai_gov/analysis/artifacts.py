"""Study artifacts: the intermediate files a reviewer and an analyst need.

The study bundle is one large JSON. That is the right archival object and the
wrong thing to read. This module writes the intermediate layer: per-phase
files, a flat per-cell table, per-arm aggregates, a diagnostics file naming
everything that looks wrong, and a short digest meant to be read by a person.

The diagnostics file is the part that matters most. A study can complete
cleanly and still be uninterpretable — arms that produce identical numbers, a
sensitivity parameter the model never reads, a calibration whose posterior is
its prior. None of those raise an exception, and all of them invalidate a
conclusion drawn from the run. So they are detected and named here rather than
left for someone to notice while writing the paper.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xai_gov.core.hashing import canonical_json

#: Indicators pulled into the flat cell table. Anything a hypothesis or the
#: frontier refers to must be here, or the table cannot be used to check them.
CELL_INDICATORS: tuple[str, ...] = (
    "operational.service_level",
    "operational.fill_rate",
    "operational.stockout_periods",
    "operational.mean_backlog",
    "operational.average_inventory",
    "systemic_risk.bullwhip",
    "systemic_risk.recovery_periods",
    "governance.intervention_rate",
    "governance.escalation_rate",
    "governance.safe_mode_rate",
    "governance.autonomy_reduction_rate",
    "governance.total_intervention_cost",
    "governance.traceability",
    "guarantees.ood_rate",
    "guarantees.coverage_error",
    "guarantees.abs_coverage_error",
    "guarantees.mean_nominal_level",
    "guarantees.change_declarations",
    "guarantees.shield_activation_rate",
    "guarantees.conflict_rate",
    "guarantees.mean_substitution",
    "xai.mean_explanatory_fidelity",
    "xai.informative_rate",
    "xai.mean_actionability",
    "xai.identifiable_rate",
)


def _dig(tree: Any, path: str) -> float | None:
    """Read a dotted path, returning None rather than a fabricated zero."""
    node = tree
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, int | float):
        return None
    return float(node)


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing wrong enough that a conclusion should not be drawn past it."""

    severity: str
    code: str
    detail: str
    implication: str

    def to_payload(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "detail": self.detail,
            "implication": self.implication,
        }


def _arm_indicator(
    bundle: dict[str, Any], treatment: str, control: str, indicator: str
) -> list[float] | None:
    """The two arms' values for an indicator, from the frontier aggregates.

    Returned so a zero difference can be told apart from two arms that are both
    at zero. Reading it from the bundle rather than recomputing keeps the
    diagnosis to what the run actually recorded.
    """
    points = bundle.get("campaign", {}).get("frontier", {}).get("points", [])
    short = indicator.split(".")[-1]
    found: list[float] = []
    for arm in (treatment, control):
        entry = next((p for p in points if p.get("arm") == arm), None)
        if entry is None:
            return None
        matches = [v for k, v in entry.get("values", {}).items() if short in k]
        if not matches:
            return None
        found.append(float(matches[0]))
    return found or None


def diagnose(bundle: dict[str, Any]) -> list[Finding]:
    """Find the failures that complete without erroring.

    Ordered by severity because the first blocking finding usually explains the
    rest, and a reader working down the list should not have to rank it.
    """
    findings: list[Finding] = []
    campaign = bundle.get("campaign", {})

    # An arm whose every cell was excluded for the same reason is not a set of
    # bad runs, it is a property of the arm. Reported as such, because the
    # alternative reads as "0 usable pairs" and invites the conclusion that the
    # comparison was misconfigured when in fact the treatment systematically
    # fails a declared data-quality criterion.
    summary = campaign.get("campaign_summary", {})
    exclusions = summary.get("exclusions", [])
    included_by_arm = summary.get("included_by_arm", {})
    if exclusions:
        by_arm: dict[str, dict[str, int]] = {}
        for entry in exclusions:
            arm = str(entry.get("arm", "?"))
            reason = str(entry.get("reason", "?"))
            by_arm.setdefault(arm, {})[reason] = by_arm.setdefault(arm, {}).get(reason, 0) + 1
        for arm, reasons in sorted(by_arm.items()):
            if included_by_arm.get(arm, 0) == 0 and len(reasons) == 1:
                reason = next(iter(reasons))
                findings.append(
                    Finding(
                        severity="blocking",
                        code="arm_systematically_excluded",
                        detail=(
                            f"{arm}: every one of {reasons[reason]} cells excluded, all "
                            f"for the same reason \u2014 {reason}"
                        ),
                        implication=(
                            "this is a finding about the arm rather than a set of bad "
                            "runs: the treatment systematically fails a declared "
                            "criterion, and any hypothesis using it is answered in the "
                            "negative rather than left untested"
                        ),
                    )
                )

    # A resumed campaign that mixes cells judged under different criteria
    # reports a mixture while looking uniform.
    if summary.get("re_run_for_stale_criteria"):
        findings.append(
            Finding(
                severity="note",
                code="criteria_version_advanced",
                detail=(
                    f"{summary['re_run_for_stale_criteria']} cells re-run because they "
                    f"were judged under older exclusion criteria"
                ),
                implication="the campaign is internally consistent; no action needed",
            )
        )

    # A detector whose realized flag rate sits far below its nominal level has
    # normalized the shift away: its coverage number describes the score's
    # self-normalization rather than the environment. Blocking rather than an
    # exclusion criterion — when it holds for every arm, excluding on it removes
    # the whole study, and the finding is about the conformal layer rather than
    # about any run.
    under_flagging = sorted(
        {
            str(cell.get("arm"))
            for cell in campaign.get("cells", [])
            if cell.get("under_flagging")
        }
    )
    if under_flagging:
        arms_total = len(set(str(c.get("arm")) for c in campaign.get("cells", [])))
        universal = len(under_flagging) >= max(arms_total - 1, 1)
        findings.append(
            Finding(
                severity="blocking",
                code="detector_under_flags",
                detail=(
                    f"realized flag rate far below the nominal level in "
                    f"{len(under_flagging)} of {arms_total} arms: "
                    f"{', '.join(under_flagging[:6])}"
                    + (" …" if len(under_flagging) > 6 else "")
                ),
                implication=(
                    "the conformal layer is not measuring the environment in any "
                    "configuration, so no coverage claim in this run is supported; "
                    "the defect is in the detector rather than in any arm"
                    if universal
                    else "these arms' coverage numbers describe the score's "
                    "self-normalization; treat their guarantees layer as unavailable"
                ),
            )
        )

    # A hypothesis whose paired differences are all exactly zero is not a null
    # result. It means the two arms produced the same number on that indicator,
    # so the comparison had nothing to measure and the ablation is vacuous.
    for result in campaign.get("hypotheses", []):
        pairs = result.get("pairs", 0)
        mean = result.get("mean_difference", 0.0)
        evidence = result.get("evidence") or {}
        forward = evidence.get("forward_e_value")
        reverse = evidence.get("reverse_e_value")
        # Both e-values at exactly one is a sharper signature than the forward
        # one alone: it says the bet lost in *both* directions, which happens
        # only when every paired difference is zero. A two-sided hypothesis has
        # no reverse test, so it falls back to the forward value.
        inert_evidence = forward == 1.0 and (reverse is None or reverse == 1.0)
        if pairs >= 2 and mean == 0.0 and inert_evidence:
            arm_values = _arm_indicator(
                bundle, result["treatment"], result["control"], result["indicator"]
            )
            # Two different failures wear the same zero, and they call for
            # opposite fixes. Both arms sitting at zero means the mechanism had
            # nothing to act on — move the hypothesis to a regime where the
            # quantity is non-zero. Both at the same non-zero value means the
            # treatment does not reach the indicator at all — wire it, or the
            # comparison is structurally empty.
            both_zero = arm_values is not None and all(v == 0.0 for v in arm_values)
            findings.append(
                Finding(
                    severity="blocking",
                    code="indicator_inert" if both_zero else "identical_arms",
                    detail=(
                        f"{result['id']}: every one of {pairs} paired differences on "
                        f"{result['indicator']} is exactly zero between "
                        f"{result['treatment']} and {result['control']}"
                        + (
                            "; the indicator is 0.0 in both arms"
                            if both_zero
                            else f"; both arms sit at {arm_values[0]:.6f}"
                            if arm_values
                            else ""
                        )
                    ),
                    implication=(
                        "the indicator is inert in this scenario, so the mechanism "
                        "has nothing to act on; the hypothesis needs a regime where "
                        "the quantity is non-zero, not a different treatment"
                        if both_zero
                        else "the treatment does not reach this indicator, so the "
                        "hypothesis is untestable as configured rather than "
                        "unsupported; reporting it as a null result would be wrong"
                    ),
                )
            )
        elif pairs < 2:
            findings.append(
                Finding(
                    severity="blocking",
                    code="no_pairs",
                    detail=f"{result['id']}: {pairs} usable pairs on {result['indicator']}",
                    implication="the indicator is absent from one arm; untested, not refuted",
                )
            )

    # A hypothesis whose effect runs opposite to its prediction is a finding,
    # not an error — but it must be surfaced so it is discussed rather than
    # quietly folded into an 'insufficient evidence' line.
    plan = {h["id"]: h for h in campaign.get("plan", {}).get("hypotheses", [])}
    for result in campaign.get("hypotheses", []):
        declared = plan.get(result["id"], {}).get("direction")
        mean = result.get("mean_difference", 0.0)
        if result.get("pairs", 0) < 2 or mean == 0.0:
            continue
        wrong_way = (declared == "greater" and mean < 0) or (declared == "less" and mean > 0)
        if wrong_way:
            findings.append(
                Finding(
                    severity="attention",
                    code="effect_reversed",
                    detail=(
                        f"{result['id']}: predicted {declared}, observed "
                        f"{mean:+.4f} on {result['indicator']}"
                    ),
                    implication=(
                        "the preregistered direction was wrong; report the reversal "
                        "explicitly rather than as absent evidence"
                    ),
                )
            )

    # A one-sided test on a signed error rewards being further from the target
    # in one direction. The claim such a hypothesis is usually making is about
    # the magnitude, and the two are opposite whenever the arms fall on the same
    # side of nominal.
    #
    # An indicator that is already a magnitude is exempt. Matching on the
    # ``_error`` suffix alone flagged ``abs_coverage_error`` — the very
    # indicator this diagnostic exists to recommend — which would have made the
    # fix look like the defect.
    for result in campaign.get("hypotheses", []):
        indicator = str(result.get("indicator", ""))
        leaf = indicator.split(".")[-1]
        direction = plan.get(result["id"], {}).get("direction")
        already_absolute = leaf.startswith(("abs_", "absolute_")) or leaf.endswith(
            ("_magnitude", "_distance", "_rate")
        )
        if direction in ("greater", "less") and leaf.endswith("_error") and not already_absolute:
            findings.append(
                Finding(
                    severity="blocking",
                    code="signed_error_direction",
                    detail=(
                        f"{result['id']}: a '{direction}' test on the signed "
                        f"{indicator}"
                    ),
                    implication=(
                        "a directional test on a signed error rewards deviating from "
                        "the target in one direction; if the claim is that the error "
                        "stays small, test the absolute indicator instead"
                    ),
                )
            )

    # A sensitivity parameter with exactly zero total effect has almost never
    # been shown irrelevant: far more often the model function does not read it,
    # which makes the zero an artifact of the analysis wiring.
    for index in bundle.get("sensitivity", {}).get("indices", []):
        if index.get("total_effect") == 0.0 and index.get("first_order") == 0.0:
            findings.append(
                Finding(
                    severity="attention",
                    code="parameter_unread",
                    detail=f"{index['name']}: S1 and ST are both exactly 0.000",
                    implication=(
                        "an exact zero usually means the sensitivity model never "
                        "reads this parameter; verify before reporting it as "
                        "non-influential"
                    ),
                )
            )

    calibration = bundle.get("calibration")
    if calibration:
        # A gap measured against simulator output is a sim-to-sim gap. Reported
        # as blocking because every conclusion in the study is qualified by it,
        # and a reader who takes it for a sim-to-real gap has been misled by
        # the artifact rather than by the prose.
        provenance = calibration.get("series", {})
        if provenance.get("synthetic"):
            findings.append(
                Finding(
                    severity="blocking",
                    code="synthetic_calibration_reference",
                    detail=(
                        "the calibration reference series came from the simulator: "
                        f"{provenance.get('source', 'unknown source')}"
                    ),
                    implication=(
                        "the reported gap is sim-to-sim and establishes no external "
                        "validity; place a real series under data/demand/ before "
                        "any conclusion is described as transferring"
                    ),
                )
            )
        elif provenance.get("intermittent"):
            findings.append(
                Finding(
                    severity="attention",
                    code="intermittent_reference_series",
                    detail=(
                        f"{provenance.get('zero_fraction', 0):.0%} of the reference "
                        "series periods have no demand"
                    ),
                    implication=(
                        "intermittent demand is a different modelling problem than "
                        "this twin represents; conclusions do not transfer to it"
                    ),
                )
            )

        weak = [name for name, ok in calibration.get("identified", {}).items() if not ok]
        if weak:
            findings.append(
                Finding(
                    severity="attention",
                    code="weakly_identified",
                    detail=f"posterior barely narrower than prior for {sorted(weak)}",
                    implication=(
                        "the data did not constrain these parameters; their posterior "
                        "means are priors and must not be reported as calibrated values"
                    ),
                )
            )
        check = calibration.get("predictive_check", {})
        if check.get("overfitted"):
            findings.append(
                Finding(
                    severity="attention",
                    code="calibration_overfit",
                    detail=(
                        f"held-out distance {check.get('held_out_distance')} against "
                        f"calibrated distance {check.get('calibrated_distance')}"
                    ),
                    implication=(
                        "the simulator matches the statistics it was fitted to far "
                        "better than the held-out ones; no conclusion transfers "
                        "beyond the fitted moments"
                    ),
                )
            )

    if not bundle.get("power", {}).get("adequate", True):
        findings.append(
            Finding(
                severity="attention",
                code="underpowered",
                detail=(
                    f"{bundle['power']['planned_replicates']} replicates against "
                    f"{bundle['power']['required_anytime_valid']} required"
                ),
                implication="unsupported hypotheses may be power failures, not nulls",
            )
        )

    frontier = campaign.get("frontier", {})
    if frontier and not frontier.get("theorem_3_consistent", True):
        findings.append(
            Finding(
                severity="blocking",
                code="theorem_3_violated",
                detail="one arm dominates every objective",
                implication="the theorem or the implementation is wrong; do not report",
            )
        )

    order = {"blocking": 0, "attention": 1, "note": 2}
    return sorted(findings, key=lambda f: (order.get(f.severity, 3), f.code))


def cell_table(bundle: dict[str, Any], kpis_by_run: dict[str, dict[str, Any]]) -> str:
    """A flat CSV of every cell, one indicator per column.

    Written as CSV on purpose: the per-cell layer is the one someone will want
    to open in a spreadsheet or read into pandas without traversing nested JSON.
    Missing indicators are left empty rather than zero-filled, so a gap in the
    table stays visibly a gap.
    """
    header = ["arm", "replicate", "seed", "run_name", "included", "exclusion_reason"]
    header.extend(indicator.replace(".", "_") for indicator in CELL_INDICATORS)
    rows = [",".join(header)]

    for cell in bundle.get("campaign", {}).get("cells", []):
        kpis = kpis_by_run.get(cell["run_name"], {})
        values = [
            str(cell.get("arm", "")),
            str(cell.get("replicate", "")),
            str(cell.get("seed", "")),
            str(cell.get("run_name", "")),
            "1" if cell.get("included") else "0",
            f"\"{cell.get('exclusion_reason') or ''}\"",
        ]
        for indicator in CELL_INDICATORS:
            value = _dig(kpis, indicator)
            values.append("" if value is None else f"{value:.6f}")
        rows.append(",".join(values))
    return "\n".join(rows) + "\n"


def arm_aggregates(
    bundle: dict[str, Any], kpis_by_run: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Per-arm mean, standard deviation and n for every indicator.

    The standard deviation is what tells an identical-arms problem apart from a
    genuine null: two arms with the same mean and zero variance are the same
    arm, while two with the same mean and real spread are a real null.
    """
    by_arm: dict[str, dict[str, list[float]]] = {}
    for cell in bundle.get("campaign", {}).get("cells", []):
        if not cell.get("included"):
            continue
        kpis = kpis_by_run.get(cell["run_name"], {})
        collected = by_arm.setdefault(str(cell["arm"]), {})
        for indicator in CELL_INDICATORS:
            value = _dig(kpis, indicator)
            if value is not None:
                collected.setdefault(indicator, []).append(value)

    aggregates: dict[str, Any] = {}
    for arm, indicators in sorted(by_arm.items()):
        aggregates[arm] = {
            indicator: {
                "n": len(values),
                "mean": round(statistics.fmean(values), 6),
                "sd": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
                "min": round(min(values), 6),
                "max": round(max(values), 6),
            }
            for indicator, values in sorted(indicators.items())
        }
    return aggregates


def digest(bundle: dict[str, Any], findings: list[Finding], aggregates: dict[str, Any]) -> str:
    """A short markdown summary, meant to be read rather than parsed."""
    campaign = bundle.get("campaign", {})
    summary = campaign.get("campaign_summary", {})
    power = bundle.get("power", {})
    lines: list[str] = [
        f"# Study digest — {bundle.get('study', 'unnamed')}",
        "",
        f"- status: **{bundle.get('reporting_status', 'unknown')}**",
        f"- replicates: {bundle.get('replicates')} "
        f"(required for anytime-valid: {power.get('required_anytime_valid')})",
        f"- cells: {summary.get('cells')} run, {summary.get('included')} included, "
        f"{summary.get('excluded')} excluded",
        f"- minimum detectable effect: {power.get('minimum_detectable_effect')}",
        "",
        "## Hypotheses",
        "",
        "| id | indicator | treatment vs control | n | mean diff | "
        "95% conf. sequence | e+ | e− | verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in campaign.get("hypotheses", []):
        evidence = result.get("evidence") or {}
        forward = evidence.get("forward_e_value", 0.0)
        reverse = evidence.get("reverse_e_value")
        interval = result.get("confidence_sequence") or {}
        bounds = (
            f"[{interval['lower']:+.4f}, {interval['upper']:+.4f}]"
            if "lower" in interval
            else "—"
        )
        lines.append(
            f"| {result['id']} | {result['indicator'].split('.')[-1]} | "
            f"{result['treatment']} vs {result['control']} | {result['pairs']} | "
            f"{result['mean_difference']:+.4f} | {bounds} | {forward:.4g} | "
            f"{'—' if reverse is None else f'{reverse:.4g}'} | "
            f"{result.get('verdict', '')} |"
        )

    frontier = campaign.get("frontier", {})
    lines += [
        "",
        "## Frontier",
        "",
        f"- arms on the frontier: {', '.join(frontier.get('frontier_arms', [])) or 'none'}",
        f"- excluded for missing objectives: "
        f"{', '.join(frontier.get('excluded_arms', {})) or 'none'}",
        f"- Theorem 3 consistent: {frontier.get('theorem_3_consistent')}",
        "",
        "## Sensitivity",
        "",
        "| parameter | S1 | ST | influential |",
        "| --- | --- | --- | --- |",
    ]
    for index in bundle.get("sensitivity", {}).get("indices", []):
        lines.append(
            f"| {index['name']} | {index['first_order']:.3f} | "
            f"{index['total_effect']:.3f} | {index['influential']} |"
        )

    calibration = bundle.get("calibration")
    lines += ["", "## Calibration", ""]
    if calibration:
        check = calibration.get("predictive_check", {})
        provenance = calibration.get("series", {})
        lines += [
            f"- reference series: {provenance.get('source', 'unknown')}",
            f"- {provenance.get('gap_interpretation', 'provenance not recorded')}",
            f"- gap on held-out statistics: {check.get('sim_to_real_gap')}",
            f"- overfitted to the fitted moments: {check.get('overfitted')}",
            f"- identified parameters: {calibration.get('identified')}",
        ]
    else:
        lines.append("- skipped: no gap established")

    lines += ["", "## Diagnostics", ""]
    if not findings:
        lines.append("Nothing flagged.")
    else:
        for finding in findings:
            lines += [
                f"**{finding.severity.upper()} · {finding.code}**",
                f"- {finding.detail}",
                f"- {finding.implication}",
                "",
            ]

    lines += [
        "## Per-arm coverage behaviour",
        "",
        "| arm | realized flag rate | nominal level | coverage error | honest |",
        "| --- | --- | --- | --- | --- |",
    ]
    for arm, indicators in aggregates.items():
        realized = indicators.get("guarantees.ood_rate", {})
        nominal = indicators.get("guarantees.mean_nominal_level", {})
        error = indicators.get("guarantees.abs_coverage_error", {})
        if not realized:
            continue
        honest = "yes"
        if nominal.get("mean") and realized.get("mean", 0.0) < nominal["mean"] / 2:
            honest = "UNDER-FLAGS"
        lines.append(
            f"| {arm} | {realized.get('mean', '—')} | {nominal.get('mean', '—')} | "
            f"{error.get('mean', '—')} | {honest} |"
        )

    lines += [
        "",
        "## Per-arm service level and intervention",
        "",
        "| arm | service level | sd | intervention rate | sd | coverage error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm, indicators in aggregates.items():
        service = indicators.get("operational.service_level", {})
        intervention = indicators.get("governance.intervention_rate", {})
        coverage = indicators.get("guarantees.coverage_error", {})
        lines.append(
            f"| {arm} | {service.get('mean', '—')} | {service.get('sd', '—')} | "
            f"{intervention.get('mean', '—')} | {intervention.get('sd', '—')} | "
            f"{coverage.get('mean', '—')} |"
        )
    return "\n".join(lines) + "\n"


def manuscript_tables(bundle: dict[str, Any], aggregates: dict[str, Any]) -> str:
    """The report's tables, in the form a manuscript needs them.

    Distinct from the digest, which is a working summary. These carry the
    rounding, the column order and the qualifiers the report uses, so a section
    can be updated by copying a block rather than by reading numbers off one
    document and typing them into another.

    An effect size is reported standardized as well as raw. Comparing a raw
    difference against a minimum detectable effect expressed in standard
    deviations is a unit error that survives review, because the number looks
    reasonable and is wrong by whatever the dispersion happens to be.
    """
    campaign = bundle.get("campaign", {})
    power = bundle.get("power", {})
    detectable = power.get("minimum_detectable_effect")
    hypotheses = len(campaign.get("hypotheses", []))
    multiplicity = campaign.get("multiplicity", {})
    lines: list[str] = [
        f"# Tables — {bundle.get('study', 'unnamed')}",
        "",
        f"Generated from the study bundle. Status: **{bundle.get('reporting_status')}**; "
        f"{bundle.get('replicates')} replicates; minimum detectable effect "
        f"{detectable if detectable is not None else '—'} (standardized).",
        "",
        f"**Decision rule.** `e+` tests the predicted direction and `e−` tests its "
        f"opposite, each against its own null. A verdict requires an e-value at or "
        f"above the familywise threshold "
        f"{multiplicity.get('familywise_threshold', hypotheses / 0.05):.0f} "
        f"(= k/α for k = {hypotheses} preregistered hypotheses); the per-comparison "
        f"threshold is {multiplicity.get('per_comparison_threshold', 20):.0f}. "
        f"A small `e+` on its own means the bet lost — not that the effect runs the "
        f"other way, which only `e−` can establish.",
        "",
        "## Table 1 · Preregistered hypotheses",
        "",
        "| RQ | Comparison | Indicator | n | Raw diff | Standardized | "
        "95% conf. sequence | e+ | e− | Outcome |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for result in campaign.get("hypotheses", []):
        indicator = str(result.get("indicator", ""))
        raw = float(result.get("mean_difference", 0.0))
        # Standardized against the treatment arm's own dispersion, which is the
        # scale the detectable effect is expressed in.
        spread = (
            aggregates.get(result.get("treatment", ""), {})
            .get(indicator, {})
            .get("sd")
        )
        standardized = (
            f"{raw / spread:+.2f}" if isinstance(spread, int | float) and spread > 1e-9 else "—"
        )
        evidence = result.get("evidence") or {}

        def fmt(value: float | None) -> str:
            if not isinstance(value, int | float):
                return "—"
            return f"{value:.2e}" if value >= 1000 else f"{value:.3g}"

        forward = evidence.get("forward_e_value")
        reverse = evidence.get("reverse_e_value")
        interval = result.get("confidence_sequence") or {}
        bounds = (
            f"[{interval['lower']:+.4f}, {interval['upper']:+.4f}]"
            if "lower" in interval
            else "—"
        )

        # The verdict comes from the two e-values against the familywise
        # threshold, never from a forward value being small. A forward e-value
        # below one means the bet lost; only the reverse e-value can establish
        # that the effect runs the other way, and reading the first as the
        # second was an error this project shipped in an earlier draft.
        if result.get("supported"):
            outcome = "**supported**"
        elif evidence.get("refuted"):
            outcome = "**refuted** (opposite direction)"
        else:
            outcome = "inconclusive"
        lines.append(
            f"| {result['id']} | {result['treatment']} vs {result['control']} | "
            f"{indicator.split('.')[-1]} | {result['pairs']} | {raw:+.4f} | "
            f"{standardized} | {bounds} | {fmt(forward)} | {fmt(reverse)} | "
            f"{outcome} |"
        )

    lines += [
        "",
        "## Table 2 · Per-arm outcomes",
        "",
        "| Arm | Service level | sd | Intervention rate | sd | Coverage error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for arm, indicators in aggregates.items():
        service = indicators.get("operational.service_level", {})
        intervention = indicators.get("governance.intervention_rate", {})
        coverage = indicators.get("guarantees.abs_coverage_error", {})
        lines.append(
            f"| {arm} | {service.get('mean', '—')} | {service.get('sd', '—')} | "
            f"{intervention.get('mean', '—')} | {intervention.get('sd', '—')} | "
            f"{coverage.get('mean', '—')} |"
        )

    strategic = campaign.get("strategic", {})
    if strategic.get("status") == "available":
        lines += [
            "",
            f"## Table 3 · Strategic indicators (baseline: {strategic.get('baseline')})",
            "",
            "| Arm | Value of governance | RGD | TVC | Pays |",
            "| --- | --- | --- | --- | --- |",
        ]
        for arm, value in sorted(strategic.get("arms", {}).items()):
            pays = value.get("governance_pays")
            lines.append(
                f"| {arm} | {value['value_of_governance']:+.4f} | "
                f"{value['rgd'] if value['rgd'] is not None else 'undefined'} | "
                f"{value['tvc'] if value['tvc'] is not None else 'undefined'} | "
                f"{'yes' if pays else ('undefined' if pays is None else 'no')} |"
            )

    hierarchical = campaign.get("hierarchical", {})
    inert = hierarchical.get("indistinguishable_indicators", [])
    if inert:
        lines += [
            "",
            "## Table 4 · Indicators dominated by seed variation",
            "",
            "On these, differences between arms are seed noise whatever a pairwise ",
            "test reports about any individual pair.",
            "",
            "| Indicator | Intraclass correlation |",
            "| --- | --- |",
        ]
        for indicator in inert:
            payload = hierarchical.get("indicators", {}).get(indicator, {})
            lines.append(
                f"| {indicator} | {payload.get('intraclass_correlation', '—')} |"
            )

    return "\n".join(lines) + "\n"


def write_artifacts(
    *,
    bundle: dict[str, Any],
    phases: dict[str, Any],
    kpis_by_run: dict[str, dict[str, Any]],
    directory: Path,
    project_root: Path | None = None,
) -> dict[str, Path]:
    """Write every intermediate artifact and return where each landed."""
    directory.mkdir(parents=True, exist_ok=True)
    phase_dir = directory / "phases"
    phase_dir.mkdir(exist_ok=True)

    written: dict[str, Path] = {}
    for name, payload in phases.items():
        path = phase_dir / f"{name}.json"
        path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        written[name] = path

    findings = diagnose(bundle)
    aggregates = arm_aggregates(bundle, kpis_by_run)

    files = {
        "cells.csv": cell_table(bundle, kpis_by_run),
        "arm_aggregates.json": canonical_json(aggregates) + "\n",
        "diagnostics.json": canonical_json(
            {
                "findings": [f.to_payload() for f in findings],
                "blocking": sum(1 for f in findings if f.severity == "blocking"),
                "attention": sum(1 for f in findings if f.severity == "attention"),
                "safe_to_report": not any(f.severity == "blocking" for f in findings),
            }
        )
        + "\n",
        "digest.md": digest(bundle, findings, aggregates),
        # Tables in the form a manuscript needs them, generated rather than
        # transcribed. Four of the nine catalogued defects were found in
        # transcribed prose, so the transcription step is removed instead of
        # being performed more carefully.
        "tables.md": manuscript_tables(bundle, aggregates),
    }
    for name, content in files.items():
        path = directory / name
        path.write_text(content, encoding="utf-8")
        written[name] = path

    from xai_gov.analysis.figures import write_figures

    written.update(
        write_figures(
            bundle=bundle,
            aggregates=aggregates,
            directory=directory,
            project_root=project_root,
        )
    )
    return written
