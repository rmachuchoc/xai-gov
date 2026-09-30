"""Indicator layers.

The protocol declares six layers. Two are computable from a stage-2 run
(operational, systemic risk) and one is partially computable (governance:
intervention and traceability rates exist even under the control arm). The
remaining layers depend on components that do not exist yet.

The registry reports every layer's status explicitly, and an absent layer
is written as ``{"status": "not_available", "requires": …}`` rather than
omitted or filled with zeros. A KPI file with a silent zero is worse than
one with a stated gap: the first corrupts a meta-analysis, the second is
merely incomplete.
"""

from __future__ import annotations

import math
from typing import Any

from xai_gov.io.decision_record import DecisionRecord, GovernanceAction
from xai_gov.simulation.engine import SimulationResult


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _safe_ratio(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    return numerator / denominator if denominator > 0.0 else default


def operational_layer(result: SimulationResult) -> dict[str, Any]:
    """Service, inventory and cost."""
    traces = result.traces
    demand = sum(t.demand_total for t in traces)
    served = sum(t.served_total for t in traces)
    periods = max(len(traces), 1)

    # Service level counts periods fully served; fill rate counts units.
    # Reporting only one of them hides the difference between a node that
    # is slightly short every period and one that fails outright twice.
    fully_served = sum(
        1 for t in traces if t.demand_total <= 0.0 or t.served_total >= t.demand_total - 1e-9
    )
    return {
        "status": "available",
        "service_level": round(_safe_ratio(fully_served, periods, default=1.0), 6),
        "fill_rate": round(_safe_ratio(served, demand, default=1.0), 6),
        "unmet_demand": round(max(demand - served, 0.0), 6),
        "stockout_periods": sum(1 for t in traces if t.stocked_out_nodes > 0),
        "avg_inventory": round(sum(t.inventory_total for t in traces) / periods, 6),
        "avg_backlog": round(sum(t.backlog_total for t in traces) / periods, 6),
        "max_backlog": round(max((t.backlog_total for t in traces), default=0.0), 6),
        "reorder_events": sum(
            1 for record in result.records if record.final_quantity > 0.0
        ),
        "total_logistics_cost": round(sum(t.logistics_cost for t in traces), 6),
        "holding_cost": round(sum(t.holding_cost for t in traces), 6),
        "backlog_cost": round(sum(t.backlog_cost for t in traces), 6),
        "order_cost": round(sum(t.order_cost for t in traces), 6),
    }


def systemic_risk_layer(
    result: SimulationResult, *, shock_periods: tuple[int, ...] = ()
) -> dict[str, Any]:
    """Bullwhip, cascades and recovery.

    The bullwhip ratio is the variance of orders over the variance of
    demand, both in coefficient-of-variation form so the ratio is scale
    free — a ratio of raw variances would rise with mean demand alone and
    reward nothing.
    """
    traces = result.traces
    demand = result.demand_series
    orders = result.order_series
    periods = max(len(traces), 1)

    demand_mean = sum(demand) / periods
    order_mean = sum(orders) / periods
    demand_cv2 = _safe_ratio(_variance(demand), demand_mean**2) if demand_mean > 0 else 0.0
    order_cv2 = _safe_ratio(_variance(orders), order_mean**2) if order_mean > 0 else 0.0

    layer: dict[str, Any] = {
        "status": "available",
        "bullwhip_ratio": round(_safe_ratio(order_cv2, demand_cv2, default=0.0), 6),
        "demand_cv": round(math.sqrt(demand_cv2), 6),
        "order_cv": round(math.sqrt(order_cv2), 6),
        "max_simultaneous_stockouts": max((t.stocked_out_nodes for t in traces), default=0),
        "cascade_periods": sum(1 for t in traces if t.stocked_out_nodes > 1),
        "safe_mode_activations": sum(1 for record in result.records if record.safe_mode),
    }

    if shock_periods:
        first = min(shock_periods)
        pre = [t for t in traces if t.period < first]
        post = [t for t in traces if t.period >= first]
        pre_fill = _safe_ratio(
            sum(t.served_total for t in pre), sum(t.demand_total for t in pre), default=1.0
        )
        post_fill = _safe_ratio(
            sum(t.served_total for t in post), sum(t.demand_total for t in post), default=1.0
        )
        layer |= {
            "shock_periods": list(shock_periods),
            "fill_rate_pre_shock": round(pre_fill, 6),
            "fill_rate_post_shock": round(post_fill, 6),
            "degradation": round(pre_fill - post_fill, 6),
            "recovery_periods": _recovery_periods(traces, first, pre_fill),
        }
    return layer


def _recovery_periods(traces: list[Any], shock_period: int, target_fill: float) -> int | None:
    """Periods from the shock until the fill rate returns to target.

    ``None`` means recovery did not occur within the horizon, which is a
    finding rather than a missing value; encoding it as the horizon length
    would understate the damage.
    """
    for trace in traces:
        if trace.period < shock_period:
            continue
        if trace.demand_total <= 0.0:
            continue
        if _safe_ratio(trace.served_total, trace.demand_total, default=1.0) >= target_fill - 1e-9:
            return int(trace.period - shock_period)
    return None


def governance_layer(records: list[DecisionRecord]) -> dict[str, Any]:
    """Intervention, escalation and traceability.

    Computable under every arm, including the ungoverned control, where the
    rates are legitimately zero — an important baseline rather than a gap.
    """
    total = len(records)
    if total == 0:
        return {"status": "not_available", "requires": "at least one decision"}
    interventions = sum(
        1
        for record in records
        if record.governance_action is not GovernanceAction.APPROVE
    )
    modified = sum(1 for record in records if record.action_was_modified)
    return {
        "status": "available",
        "decisions": total,
        "intervention_rate": round(interventions / total, 6),
        "escalation_rate": round(sum(1 for r in records if r.escalated) / total, 6),
        "veto_rate": round(
            sum(1 for r in records if r.governance_action is GovernanceAction.VETO) / total, 6
        ),
        "action_modification_rate": round(modified / total, 6),
        "safe_mode_rate": round(sum(1 for r in records if r.safe_mode) / total, 6),
        "autonomy_reduction_rate": round(
            sum(
                1
                for r in records
                if r.governance_action is GovernanceAction.REDUCE_AUTONOMY
            )
            / total,
            6,
        ),
        "shield_activation_rate": round(
            sum(1 for r in records if r.shield.activated) / total, 6
        ),
        # Traceability is 1.0 by construction: the engine cannot execute an
        # action without a sealed record. It is reported rather than assumed
        # so a regression in the writer shows up as a KPI change.
        "traceability": round(total / total, 6),
        "total_intervention_cost": round(sum(r.intervention_cost for r in records), 6),
    }


def _verification_block(records: list[DecisionRecord]) -> dict[str, Any]:
    """Shield activity as seen from the decision log."""
    if not records:
        return {}
    activations = [r for r in records if r.shield.activated]
    if not activations:
        return {"shield_activation_rate": 0.0, "shield_properties": {}}

    by_property: dict[str, int] = {}
    for record in activations:
        key = record.shield.property_id or "unattributed"
        by_property[key] = by_property.get(key, 0) + 1

    # Substitution magnitude, not just frequency. Proposition 2 bounds the
    # return loss by the activation rate times the loss per substitution, so a
    # rate reported without a magnitude cannot be checked against it.
    substituted = [
        abs((r.shield.substituted_quantity or 0.0) - r.proposed_quantity)
        for r in activations
    ]
    return {
        "shield_activation_rate": round(len(activations) / len(records), 6),
        "shield_properties": dict(sorted(by_property.items())),
        "mean_substitution": round(sum(substituted) / len(substituted), 6),
        "max_substitution": round(max(substituted), 6),
    }

def guarantees_layer(
    records: list[DecisionRecord], *, shock_periods: tuple[int, ...] = ()
) -> dict[str, Any]:
    """Realized coverage, detection delay and saturation.

    This is the layer that would have caught the pilot's failure. An OOD rate
    near 1.0 against a nominal level of 0.1 is not an anomalous environment,
    it is a saturated detector, and the coverage error names that gap directly
    rather than leaving it to be inferred from a raw flag count.
    """
    scored = [r for r in records if r.conformal.active]
    if not scored:
        return {
            "status": "not_available",
            "requires": "an active conformal scheme (governance architecture 'cpomdp')",
        }
    total = len(scored)
    flags = sum(1 for r in scored if r.conformal.flagged_ood)
    nominal_level = sum(r.conformal.level for r in scored) / total
    realized = flags / total

    layer: dict[str, Any] = {
        "status": "available",
        "scheme": scored[0].conformal.scheme,
        "scored_decisions": total,
        "warmup_decisions": len(records) - total,
        "ood_rate": round(realized, 6),
        "mean_nominal_level": round(nominal_level, 6),
        # CCR in the protocol's terms: realized miscoverage minus nominal.
        "coverage_error": round(realized - nominal_level, 6),
        # The magnitude, which is what a claim of the form "coverage stays
        # within e of nominal" is actually about. Kept as a separate indicator
        # because a directional test on the signed error rewards undershooting
        # the nominal level, which is worse coverage rather than better: the
        # sign must not be the thing a hypothesis optimizes.
        "abs_coverage_error": round(abs(realized - nominal_level), 6),
        "mean_nonconformity": round(sum(r.conformal.score for r in scored) / total, 6),
        "mean_threshold": round(sum(r.conformal.threshold or 0.0 for r in scored) / total, 6),
        "change_declarations": sum(1 for r in scored if r.conformal.change_declared),
    }

    # A detector that flags nearly everything, or nothing at all, is not
    # measuring the environment. Reported as a finding rather than left for a
    # reader to notice.
    layer["saturated"] = bool(realized > 0.5 or (total > 20 and flags == 0))
    # Under-flagging is the mirror failure and the one the flag count alone
    # cannot see. A detector that fires a few times while its realized rate
    # sits far below its nominal level is not measuring the environment either,
    # and the first campaign passed this check with a coverage error of -0.158
    # against a nominal 0.1. Both directions make the guarantee vacuous, so
    # both are named.
    layer["under_flagging"] = bool(
        total > 20 and realized < nominal_level / 2 and abs(realized - nominal_level) > 0.05
    )
    layer["coverage_honest"] = not (layer["saturated"] or layer["under_flagging"])

    # The verification half of the guarantees layer. Reported alongside
    # coverage because both answer the same question — whether a stated
    # guarantee still holds for this run — and separating them would let a run
    # advertise conformal coverage while its safety properties had lapsed.
    verification = _verification_block(records)
    if verification:
        layer.update(verification)

    if shock_periods:
        first_shock = min(shock_periods)
        declared = [r.period for r in scored if r.conformal.change_declared]
        after = [period for period in declared if period >= first_shock]
        layer["first_shock_period"] = first_shock
        # None means the change was never detected within the horizon: a
        # finding, not a missing value.
        layer["detection_delay"] = (min(after) - first_shock) if after else None
        layer["false_alarms_before_shock"] = sum(1 for p in declared if p < first_shock)
    return layer


def xai_layer(records: list[DecisionRecord]) -> dict[str, Any]:
    """Explanation quality: fidelity, actionability, and honesty about both.

    The two indicators that matter here are not performance measures. Mean
    fidelity says whether the explanations were true; the uninformative rate
    says how often the system knew they were not and declined to rely on them.
    A high uninformative rate is not a failure — it is the descriptor layer
    working. Silently trusting a low-fidelity explanation would be the failure,
    and it would leave no trace in any indicator.
    """
    described = [r for r in records if r.descriptor.method != "none"]
    if not described:
        return {
            "status": "not_available",
            "requires": "a descriptor arm (governance 'explainability')",
        }

    total = len(described)
    informative = [r for r in described if r.descriptor.informative]
    fidelities = [r.descriptor.explanatory_fidelity for r in described]

    layer: dict[str, Any] = {
        "status": "available",
        "method": described[0].descriptor.method,
        "described_decisions": total,
        "mean_explanatory_fidelity": round(sum(fidelities) / total, 6),
        "min_explanatory_fidelity": round(min(fidelities), 6),
        "informative_rate": round(len(informative) / total, 6),
        "uninformative_rate": round(1.0 - len(informative) / total, 6),
        "identifiable_rate": round(
            sum(1 for r in described if r.descriptor.identifiable) / total, 6
        ),
        "mean_actionability": round(
            sum(r.descriptor.actionability_score for r in described) / total, 6
        ),
    }

    # An anti-correlated explanation is worse than none: an operator following
    # it acts on the least effective lever while believing it is the best.
    layer["anti_correlated_decisions"] = sum(
        1 for r in described if r.descriptor.explanatory_fidelity < -0.5
    )

    # Explanation-action consistency: of the decisions the agent intervened on,
    # how many had an explanation it was willing to rely on. A governance layer
    # intervening on uninformative descriptors is acting on the conformal
    # signal alone, and the protocol requires that to be visible.
    intervened = [r for r in described if r.governance_action is not GovernanceAction.APPROVE]
    if intervened:
        layer["explained_intervention_rate"] = round(
            sum(1 for r in intervened if r.descriptor.informative) / len(intervened), 6
        )
    return layer


def per_echelon_layer(records: list[DecisionRecord]) -> dict[str, Any]:
    """Operational indicators broken out by node.

    Aggregate indicators average over echelons, which is defensible for most
    quantities and close to meaningless for amplification: the bullwhip effect
    is defined *between* echelons, so averaging across them discards the
    structure being measured. Reading the breakdown from the decision records
    rather than from the period traces keeps this out of the engine, which has
    no reason to know about echelons.

    The breakdown also carries the mechanism behind the headline collapse:
    knowing which echelon starves first under governance-as-refusal is what
    turns an observed effect into an explained one.
    """
    if not records:
        return {"status": "not_available", "requires": "at least one decision record"}

    by_node: dict[str, list[DecisionRecord]] = {}
    for record in records:
        by_node.setdefault(record.node_id, []).append(record)

    nodes: dict[str, Any] = {}
    for node_id, node_records in sorted(by_node.items()):
        ordered = sorted(node_records, key=lambda r: r.period)
        demand = [r.state.demand_observed for r in ordered]
        orders = [r.final_quantity for r in ordered]
        stockouts = sum(1 for r in ordered if r.state.inventory <= 0.0)
        nodes[node_id] = {
            "periods": len(ordered),
            "mean_demand": round(_mean(demand), 6),
            "mean_order": round(_mean(orders), 6),
            "mean_inventory": round(_mean([r.state.inventory for r in ordered]), 6),
            "mean_backlog": round(_mean([r.state.backlog for r in ordered]), 6),
            "stockout_rate": round(stockouts / len(ordered), 6),
            # Per-node amplification: the order series' dispersion against the
            # demand series' own, both scale free.
            "local_bullwhip": round(_cv_ratio(orders, demand), 6),
            "intervention_rate": round(
                sum(1 for r in ordered if r.governance_action is not GovernanceAction.APPROVE)
                / len(ordered),
                6,
            ),
        }

    # Which echelon fails first. Under governance-as-refusal the collapse
    # propagates upstream, and naming the node where it starts is the
    # difference between reporting the effect and explaining it.
    first_stockout: dict[str, int | None] = {}
    for node_id, node_records in sorted(by_node.items()):
        periods = [r.period for r in sorted(node_records, key=lambda r: r.period)
                   if r.state.inventory <= 0.0]
        first_stockout[node_id] = periods[0] if periods else None

    starved = [(p, n) for n, p in first_stockout.items() if p is not None]
    return {
        "status": "available",
        "nodes": nodes,
        "first_stockout_period": first_stockout,
        "first_node_to_starve": min(starved)[1] if starved else None,
        "echelons": len(nodes),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _cv_ratio(numerator: list[float], denominator: list[float]) -> float:
    """Ratio of squared coefficients of variation, floored at zero.

    Scale free by construction, so a node with larger throughput does not read
    as more amplifying merely for being larger.
    """
    num_mean, den_mean = _mean(numerator), _mean(denominator)
    if abs(num_mean) < 1e-12 or abs(den_mean) < 1e-12:
        return 0.0
    num_cv2 = _variance(numerator) / (num_mean**2)
    den_cv2 = _variance(denominator) / (den_mean**2)
    return num_cv2 / den_cv2 if den_cv2 > 1e-12 else 0.0


def value_layer(operational: dict[str, Any], governance: dict[str, Any]) -> dict[str, Any]:
    """Organizational value.

    RGD and TVC compare a governed arm against its ungoverned counterpart,
    which a single run cannot do: the comparison belongs to the campaign
    layer, and computing it from one run would invent a counterfactual.
    Only the quantities that are well defined within a run are reported.
    """
    return {
        "status": "partial",
        "requires": "paired governed/ungoverned runs for RGD, TVC and VoG (stage 6)",
        "total_logistics_cost": operational.get("total_logistics_cost"),
        "total_governance_cost": governance.get("total_intervention_cost"),
        "cost_per_decision": round(
            _safe_ratio(
                float(operational.get("total_logistics_cost", 0.0)),
                float(governance.get("decisions", 0) or 0),
            ),
            6,
        ),
    }


def compute_kpis(
    result: SimulationResult, *, shock_periods: tuple[int, ...] = ()
) -> dict[str, Any]:
    """Assemble all six layers for one run."""
    operational = operational_layer(result)
    governance = governance_layer(result.records)
    return {
        "operational": operational,
        "systemic_risk": systemic_risk_layer(result, shock_periods=shock_periods),
        "xai": xai_layer(result.records),
        "guarantees": guarantees_layer(result.records, shock_periods=shock_periods),
        "governance": governance,
        "per_echelon": per_echelon_layer(result.records),
        "organizational_value": value_layer(operational, governance),
    }
