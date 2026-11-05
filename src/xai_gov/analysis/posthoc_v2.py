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
