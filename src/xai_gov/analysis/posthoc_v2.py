"""Post-hoc summaries of the sealed v2 campaign, read from its stored cells.

Nothing here runs a simulation or touches the confirmatory report: every
quantity is recomputed from the KPI files (and, where needed, the decision
logs) of the cells the sealed campaign already produced, and written to a
separate artifact labelled post hoc. The confirmatory evidence for H1-H6 is
unchanged by anything in this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from xai_gov.analysis.campaign import CampaignCell, CellResult

STATUS = "post hoc: recomputed from the stored cells of the sealed campaign"


def load_cells(report_path: Path, runs_root: Path) -> tuple[CellResult, ...]:
    """Rebuild the campaign's cell results from its report and run folders."""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results: list[CellResult] = []
    for cell in report["cells"]:
        kpis = json.loads((runs_root / cell["run_name"] / "kpis.json").read_text("utf-8"))
        results.append(
            CellResult(
                cell=CampaignCell(
                    arm=cell["arm"], experiment=Path(cell["experiment"]),
                    replicate=int(cell["replicate"]), seed=int(cell["seed"]),
                ),
                run_name=cell["run_name"],
                kpis=kpis,
                included=bool(cell["included"]),
                exclusion_reason=cell.get("exclusion_reason"),
            )
        )
    return tuple(results)


def comparison_key(config: dict[str, Any]) -> str:
    """What must match for an ungoverned arm to be a valid economic baseline."""
    network = (config.get("network") or {}).get("name", "unnamed")
    policy = json.dumps(config.get("policy") or {}, sort_keys=True)
    demand = json.dumps(config.get("demand") or {}, sort_keys=True)
    periods = (config.get("experiment") or {}).get("periods")
    return f"{network}|{policy}|{demand}|{periods}"


def economics(
    results: tuple[CellResult, ...],
    *,
    arm_keys: dict[str, str],
    baselines: dict[str, str],
) -> dict[str, Any]:
    """The Pareto frontier and the strategic indicators, with beta as the return.

    The sealed run computed them with the v1 service level (now alpha) and one
    baseline for every arm. Here beta is the return, and each arm is compared
    with an ungoverned arm on the same network, policy, demand and horizon
    (``baselines`` maps a comparison key to its ungoverned arm). An arm with no
    such baseline is reported as unavailable rather than compared across
    topologies or policies.
    """
    from xai_gov.analysis.frontier import THEOREM_3_OBJECTIVES, compute_frontier
    from xai_gov.analysis.strategic import strategic_value

    by_arm: dict[str, dict[int, CellResult]] = {}
    for result in results:
        if result.included:
            by_arm.setdefault(result.cell.arm, {})[result.cell.replicate] = result

    arm_values: dict[str, dict[str, float]] = {}
    for arm, cells in sorted(by_arm.items()):
        collected: dict[str, list[float]] = {}
        for result in cells.values():
            for objective in THEOREM_3_OBJECTIVES:
                value = result.indicator(objective.indicator)
                if value is not None:
                    collected.setdefault(objective.name, []).append(value)
        if collected:
            arm_values[arm] = {k: sum(v) / len(v) for k, v in collected.items()}

    strategic: dict[str, Any] = {}
    unmatched: dict[str, str] = {}
    baseline_arms = set(baselines.values())
    for arm, cells in sorted(by_arm.items()):
        if arm in baseline_arms:
            continue
        baseline = baselines.get(arm_keys.get(arm, ""))
        if baseline is None or baseline not in by_arm:
            unmatched[arm] = "no ungoverned arm on the same network, policy, demand and horizon"
            continue
        control = by_arm[baseline]
        shared = sorted(set(cells) & set(control))
        computed = strategic_value(
            arm=arm, baseline=baseline,
            governed=[cells[r].kpis for r in shared],
            ungoverned=[control[r].kpis for r in shared],
        )
        if computed is not None:
            strategic[arm] = computed.to_payload()
    return {
        "status": STATUS,
        "return_indicator": "operational.fill_rate",
        "baseline_rule": "same network, policy, demand and horizon; paired by replicate",
        "frontier": compute_frontier(arm_values).to_payload(),
        "strategic": {"arms": strategic, "unmatched": unmatched},
    }


def _decisions_from_log(path: Path) -> dict[str, list[Any]]:
    """(state, proposed, executed) per node, in period order, from a decision log."""
    from xai_gov.io.decision_record import OperatingState

    by_node: dict[str, list[tuple[int, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            entry = json.loads(line)
            if entry.get("kind") != "decision":
                continue
            payload = entry["payload"]
            state = OperatingState(**payload["state"])
            by_node.setdefault(payload["node_id"], []).append(
                (state.period, (state, float(payload["proposed"]["quantity"]),
                                float(payload["final"]["quantity"])))
            )
    return {node: [d for _, d in sorted(items, key=lambda i: i[0])]
            for node, items in sorted(by_node.items())}


def _cell_admissibility(job: tuple[str, str, str, Any, int]) -> tuple[str, int, dict[str, int]]:
    from xai_gov.verification.envelope_log import admissibility_counts

    arm, run_dir, _, specification, lookahead = job
    counts = admissibility_counts(
        _decisions_from_log(Path(run_dir) / "decisions.jsonl"), specification,
        horizon_lookahead=lookahead,
    )
    return arm, 0, counts


def admissibility_by_arm(
    results: tuple[CellResult, ...],
    *,
    runs_root: Path,
    specifications: dict[str, tuple[Any, int]],
    workers: int = 1,
) -> dict[str, Any]:
    """Admissibility failures summed over every run of each arm.

    ``specifications`` gives each arm's declared specification and lookahead,
    so an unshielded arm is judged against the same envelope as its shielded
    counterpart. Run-level shares carry Clopper-Pearson intervals (the run is
    the unit); decision-level rates are pooled and descriptive.
    """
    from concurrent.futures import ProcessPoolExecutor

    from xai_gov.verification.envelope_log import (
        ADMISSIBILITY_DEFINITIONS,
        _rates,
        admissibility_rates,
    )

    jobs = [
        (r.cell.arm, str(runs_root / r.run_name), r.run_name, *specifications[r.cell.arm])
        for r in results if r.included and r.cell.arm in specifications
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outcomes = list(pool.map(_cell_admissibility, jobs, chunksize=32))
    else:
        outcomes = [_cell_admissibility(job) for job in jobs]

    per_arm: dict[str, list[dict[str, int]]] = {}
    for arm, _, counts in outcomes:
        per_arm.setdefault(arm, []).append(counts)
    summary: dict[str, Any] = {}
    for arm, runs in sorted(per_arm.items()):
        totals = {key: sum(run[key] for run in runs) for key in ADMISSIBILITY_DEFINITIONS}
        summary[arm] = {
            **admissibility_rates(totals),
            "runs": len(runs),
            "runs_with_executed_inadmissible": _rates(
                sum(1 for run in runs if run["executed_inadmissible"]), len(runs)
            ),
            "runs_with_infeasible_states": _rates(
                sum(1 for run in runs if run["infeasible_states"]), len(runs)
            ),
        }
    return {"status": STATUS, "arms": summary}


#: How each mediator action becomes an executable replenishment action (R2.7),
#: as implemented in governance/cpomdp.py. Every executed action then passes the
#: arm's shield, which may substitute it.
MEDIATOR_TO_ACTION = {
    "approve": "the policy's proposal (action and quantity) as issued",
    "veto": "hold: order nothing this period",
    "reduce_autonomy": "hold, and the autonomy regime is lowered for later periods",
    "activate_safe_mode": "hold, and safe mode is recorded",
    "escalate (no supervisor review)": "hold while the escalation stands",
    "escalate (supervisor reviewed)": "the proposal if the supervisor approves, otherwise hold",
}

RATE_DEFINITIONS = {
    "intervention_rate": "decisions whose governance action is not approve / all decisions",
    "veto_rate": "decisions vetoed / all decisions",
    "escalation_rate": "decisions escalated / all decisions",
    "shield_activation_rate": (
        "decisions in which the shield replaced the action (proposal or mediator hold "
        "outside the envelope) / all decisions"
    ),
    "decisions": "one per node and period: nodes x 60 periods per run",
}


def governance_rates(results: tuple[CellResult, ...]) -> dict[str, Any]:
    """Mean and sd over runs of each arm's governance rates, with their definitions."""
    import statistics

    keys = ("intervention_rate", "veto_rate", "escalation_rate", "shield_activation_rate")
    by_arm: dict[str, dict[str, list[float]]] = {}
    for result in results:
        if not result.included:
            continue
        layer = result.kpis.get("governance") or {}
        for key in keys:
            if isinstance(layer.get(key), int | float):
                by_arm.setdefault(result.cell.arm, {}).setdefault(key, []).append(
                    float(layer[key])
                )
    return {
        "definitions": RATE_DEFINITIONS,
        "mediator_to_action": MEDIATOR_TO_ACTION,
        "arms": {
            arm: {
                key: {"mean": round(statistics.fmean(v), 6),
                      "sd": round(statistics.stdev(v), 6) if len(v) > 1 else 0.0,
                      "runs": len(v)}
                for key, v in sorted(values.items())
            }
            for arm, values in sorted(by_arm.items())
        },
    }


def epsilon_clipping(results: tuple[CellResult, ...]) -> dict[str, Any]:
    """How often, and for how long, eps_t sat on its clip bounds, per arm (R2.5).

    Read from each run's guarantees layer; arms without an active adaptive
    detector are omitted. The run is the unit for the share of runs with any
    clipping (Clopper-Pearson); frequencies and durations are over runs.
    """
    import statistics

    from xai_gov.verification.envelope_log import _rates

    by_arm: dict[str, list[dict[str, Any]]] = {}
    saturated: dict[str, list[bool]] = {}
    for result in results:
        if not result.included:
            continue
        guarantees = result.kpis.get("guarantees") or {}
        clipping = (guarantees.get("epsilon") or {}).get("clipping")
        if guarantees.get("status") != "available" or not clipping:
            continue
        by_arm.setdefault(result.cell.arm, []).append(clipping)
        saturated.setdefault(result.cell.arm, []).append(bool(guarantees.get("saturated")))

    def mean_sd(values: list[float]) -> dict[str, float]:
        return {"mean": round(statistics.fmean(values), 6),
                "sd": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0}

    arms: dict[str, Any] = {}
    for arm, runs in sorted(by_arm.items()):
        durations = [float(run["mean_duration"]) for run in runs if run["episodes"]]
        arms[arm] = {
            "runs": len(runs),
            "bounds": [runs[0]["min_level"], runs[0]["max_level"]],
            "clipped_frequency": mean_sd([float(run["clipped_frequency"]) for run in runs]),
            "clipped_low_share": round(
                sum(sum(n["clipped_low"] for n in run["per_node"].values()) for run in runs)
                / max(sum(sum(n["periods"] for n in run["per_node"].values()) for run in runs), 1),
                6,
            ),
            "clipped_high_share": round(
                sum(sum(n["clipped_high"] for n in run["per_node"].values()) for run in runs)
                / max(sum(sum(n["periods"] for n in run["per_node"].values()) for run in runs), 1),
                6,
            ),
            "runs_with_clipping": _rates(sum(1 for run in runs if run["episodes"]), len(runs)),
            "episodes_per_run": mean_sd([float(run["episodes"]) for run in runs]),
            "mean_episode_duration": mean_sd(durations) if durations else None,
            "max_episode_duration": max(int(run["max_duration"]) for run in runs),
            "runs_saturated": _rates(sum(saturated[arm]), len(runs)),
        }
    return {
        "status": STATUS,
        "definitions": {
            "clipped": "eps_t at its lower or upper clip bound after the update",
            "episode": "a maximal run of consecutive clipped scored periods at one node",
            "saturated": (
                "the guarantees layer's saturation flag "
                "(flag rate > 0.5, or no flag in more than 20 scored periods)"
            ),
        },
        "arms": arms,
    }
