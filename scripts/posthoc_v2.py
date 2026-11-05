#!/usr/bin/env python
"""Post-hoc summaries of the sealed v2 campaign (no simulation is run).

    python scripts/posthoc_v2.py --outputs-root outputs/v2

Reads the stored cells of the confirmatory campaign and writes
<outputs-root>/campaigns/confirmatory_v2/posthoc_v2.json, labelled post hoc.
The confirmatory report and its seal are left untouched.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from xai_gov.analysis import posthoc_v2
from xai_gov.analysis.campaign import CampaignRunner, plan_campaign
from xai_gov.core.hashing import canonical_json, content_hash
from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings
from xai_gov.io.yaml_loader import load_config
from xai_gov.orchestration.runner import plan_experiment

REPORT = "campaigns/confirmatory_v2/confirmatory_v2_report.json"
CAMPAIGN = "configs/campaigns/confirmatory_v2.yaml"
#: Ungoverned baselines the sealed campaign lacks, run here as exploratory
#: cells at the campaign's own seed so they pair replicate by replicate.
EXTRA_BASELINES = {
    "q_ungoverned": "configs/experiments/q_ungoverned.yaml",
    "five_ungoverned": "configs/experiments/five_ungoverned.yaml",
}
MAX_WORKERS = 6
OUTPUT = "campaigns/confirmatory_v2/posthoc_v2.json"
DETECTOR_REPORT = "campaigns/detector_exploratory_v2/detector_exploratory_v2_report.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="post-hoc summaries of the v2 campaign")
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must lie in [1, {MAX_WORKERS}]")
    root = Path(__file__).resolve().parents[1]

    outputs = Path(args.outputs_root).resolve()
    report_path = outputs / REPORT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = posthoc_v2.load_cells(report_path, outputs / "runs")

    from dataclasses import replace

    settings = replace(load_settings(Paths(root=root)), outputs_root=outputs)
    campaign = load_config(root / CAMPAIGN, root=root)
    plan = report["plan"]
    extra = {arm: root / path for arm, path in EXTRA_BASELINES.items()}
    runner = CampaignRunner(settings=settings, project_root=root, workers=args.workers)
    extra_results = runner.run(
        plan_campaign(arms=extra, replicates=int(plan["replicates"]),
                      base_seed=int(plan["base_seed"]))
    )
    all_results = (*results, *extra_results)

    paths = {str(a): root / str(p) for a, p in campaign["arms"].items()} | extra
    keys = {
        arm: posthoc_v2.comparison_key(plan_experiment(path, settings=settings).config)
        for arm, path in paths.items()
    }
    baselines = {keys[a]: a for a in ("ungoverned", *EXTRA_BASELINES)}

    from xai_gov.verification.specification import build_specification

    specifications = {}
    for arm, path in paths.items():
        verification = plan_experiment(path, settings=settings).config.get("verification") or {}
        specifications[arm] = (
            build_specification(verification.get("specification")),
            int(verification.get("horizon_lookahead", 1)),
        )

    payload = {
        "status": posthoc_v2.STATUS,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_report": REPORT,
        "source_report_hash": content_hash(report),
        "plan_hash": report["plan_hash"],
        "cells": len(results),
        "extra_baselines": {
            "arms": sorted(EXTRA_BASELINES),
            "replicates": int(plan["replicates"]),
            "base_seed": int(plan["base_seed"]),
            "status": "exploratory cells outside the sealed plan, paired by replicate seed",
        },
        "economics": posthoc_v2.economics(all_results, arm_keys=keys, baselines=baselines),
        "governance_rates": posthoc_v2.governance_rates(all_results),
        "epsilon_clipping": {
            "confirmatory": posthoc_v2.epsilon_clipping(results),
            "detector_exploratory": (
                posthoc_v2.epsilon_clipping(
                    posthoc_v2.load_cells(outputs / DETECTOR_REPORT, outputs / "runs")
                )
                if (outputs / DETECTOR_REPORT).is_file()
                else None
            ),
        },
        "admissibility": posthoc_v2.admissibility_by_arm(
            all_results, runs_root=outputs / "runs", specifications=specifications,
            workers=args.workers,
        ),
    }
    path = outputs / OUTPUT
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(f"  {len(results)} cells -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
