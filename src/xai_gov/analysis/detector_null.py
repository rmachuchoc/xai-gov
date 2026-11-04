"""The detector null line: false alarms measured, not bounded (decision 6).

The change martingale declares a change when it crosses 1/δ. Ville's inequality
would bound the false-alarm probability by δ if the conformal p-values were
exactly uniform and independent under the null; here they are not, so the bound
is withdrawn and the rate is measured instead. Each arm is run many times on
stationary customer demand, and every declared change is a false alarm.

Reported per arm:

* the fraction of runs with at least one declaration at a demand-facing node
  (the primary rate, since only there is the input stationary by construction),
  and at any node, each with an exact Clopper–Pearson interval;
* the coverage error against the declared target (realized rate − 0.10);
* how often ε_t sat on its clip bounds, how many clip episodes occurred, and
  their mean and longest duration.

Upstream nodes see their customer's orders, which the policy makes
non-stationary even under stationary demand, so their declarations are
reported but are not false alarms in the same sense.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from xai_gov.verification.envelope_log import clopper_pearson

if TYPE_CHECKING:
    from collections.abc import Sequence

    from xai_gov.analysis.campaign import CellResult


@dataclass(frozen=True, slots=True)
class NullLinePlan:
    """What the null line runs: arms, replicates and its own base seed."""

    arms: dict[str, Path]
    replicates: int
    base_seed: int

    @classmethod
    def from_config(cls, config: dict[str, Any], *, root: Path) -> NullLinePlan:
        arms = config.get("arms")
        if not isinstance(arms, dict) or not arms:
            raise ValueError("the null line must declare a non-empty 'arms' mapping")
        replicates = int(config.get("replicates", 200))
        if replicates < 1:
            raise ValueError("the null line needs at least one replicate")
        if "base_seed" not in config:
            raise ValueError("the null line must declare its own base_seed")
        return cls(
            arms={str(arm): root / str(path) for arm, path in sorted(arms.items())},
            replicates=replicates,
            base_seed=int(config["base_seed"]),
        )


def _rate(hits: int, runs: int) -> dict[str, Any]:
    low, high = clopper_pearson(hits, runs)
    return {
        "runs": runs,
        "runs_with_alarm": hits,
        "frequency": round(hits / runs, 6) if runs else None,
        "ci_low": round(low, 6),
        "ci_high": round(high, 6),
    }


def _mean_sd(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "sd": None, "n": 0}
    mean = sum(values) / len(values)
    variance = (
        sum((v - mean) ** 2 for v in values) / (len(values) - 1) if len(values) > 1 else 0.0
    )
    return {"mean": round(mean, 6), "sd": round(math.sqrt(variance), 6), "n": len(values)}


def summarize_arm(
    results: Sequence[CellResult], *, demand_facing: frozenset[str]
) -> dict[str, Any]:
    """The null-line summary of one arm's cells."""
    usable = [r for r in results if (r.kpis.get("guarantees") or {}).get("status") == "available"]
    by_node = [r.kpis["guarantees"].get("change_declared_by_node", {}) for r in usable]
    at_retail = sum(1 for nodes in by_node if any(nodes.get(n, False) for n in demand_facing))
    anywhere = sum(1 for nodes in by_node if any(nodes.values()))

    clipping = [
        r.kpis["guarantees"].get("epsilon", {}).get("clipping") for r in usable
    ]
    clipping = [c for c in clipping if c]
    durations = [c["mean_duration"] for c in clipping if c["episodes"]]
    return {
        "cells": len(results),
        "usable": len(usable),
        "false_alarms_demand_facing": _rate(at_retail, len(usable)),
        "declarations_any_node": _rate(anywhere, len(usable)),
        "coverage_error": _mean_sd(
            [float(r.kpis["guarantees"]["coverage_error"]) for r in usable]
        ),
        "epsilon_clipping": {
            "clipped_frequency": _mean_sd([float(c["clipped_frequency"]) for c in clipping]),
            "runs_with_clipping": sum(1 for c in clipping if c["episodes"]),
            "episodes_per_run": _mean_sd([float(c["episodes"]) for c in clipping]),
            "mean_duration": _mean_sd([float(d) for d in durations]),
            "max_duration": max((int(c["max_duration"]) for c in clipping), default=0),
        },
    }


def summarize_null_line(
    results: Sequence[CellResult], *, demand_facing: dict[str, frozenset[str]]
) -> dict[str, Any]:
    """Per-arm summaries; ``demand_facing`` names each arm's echelon-0 nodes."""
    arms = sorted({r.cell.arm for r in results})
    return {
        "null_line": "detector_null",
        "claim": "false-alarm rate measured on stationary runs; no Ville bound is claimed",
        "arms": {
            arm: summarize_arm(
                [r for r in results if r.cell.arm == arm], demand_facing=demand_facing[arm]
            )
            for arm in arms
        },
    }


def _demand_facing(path: Path, settings: Any) -> frozenset[str]:
    from xai_gov.orchestration.runner import plan_experiment

    nodes = plan_experiment(path, settings=settings).config["network"]["nodes"]
    return frozenset(str(node["node_id"]) for node in nodes if int(node["echelon"]) == 0)


def run_null_line(
    plan: NullLinePlan, *, settings: Any, project_root: Path, workers: int = 1
) -> dict[str, Any]:
    """Run every arm at the plan's replicates and summarize the false alarms."""
    from xai_gov.analysis.campaign import CampaignRunner, plan_campaign

    cells = plan_campaign(arms=plan.arms, replicates=plan.replicates, base_seed=plan.base_seed)
    runner = CampaignRunner(settings=settings, project_root=project_root, workers=workers)
    results = runner.run(cells)
    summary = summarize_null_line(
        results,
        demand_facing={arm: _demand_facing(path, settings) for arm, path in plan.arms.items()},
    )
    summary["replicates"] = plan.replicates
    summary["base_seed"] = plan.base_seed
    summary["arm_configs"] = {arm: path.name for arm, path in plan.arms.items()}
    return summary
