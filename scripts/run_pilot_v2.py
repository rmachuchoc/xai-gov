#!/usr/bin/env python
"""The short pilot on the new engine (step 3 of the revision).

    python scripts/run_pilot_v2.py --outputs-root /path/outside/outputs

Runs every arm of the v2 confirmatory campaign at a few replicates from the
pilot's own base seed, and writes the paired differences of each hypothesis to
specs/v2/pilot_v2.json. Their dispersion feeds the power simulation; nothing
here is a result. The pilot seed is distinct from the v1 campaign seed and from
the confirmatory seed fixed when the plan is sealed, so no pilot cell can be
reused by the confirmatory campaign.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from xai_gov.analysis.campaign import CampaignRunner, plan_campaign
from xai_gov.analysis.preregistration import build_preregistration
from xai_gov.core.hashing import canonical_json
from xai_gov.core.logging import configure_logging
from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings
from xai_gov.io.yaml_loader import load_config

PILOT_SEED = 20261105
V1_CAMPAIGN_SEED = 20260703
MAX_WORKERS = 6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the v2 pilot on the new engine")
    parser.add_argument("--campaign", default="configs/campaigns/confirmatory_v2.yaml")
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--outputs-root", required=True,
                        help="where the pilot's run artifacts go; never the outputs/ folder")
    parser.add_argument("--report", default="specs/v2/pilot_v2.json")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must lie in [1, {MAX_WORKERS}]")

    root = Path(__file__).resolve().parents[1]
    settings = load_settings(Paths(root=root))
    outputs = Path(args.outputs_root).resolve()
    protected = settings.outputs_root.resolve()
    if outputs == protected or protected in outputs.parents:
        parser.error("the pilot must not write into the configured outputs/ folder")
    (outputs / "runs").mkdir(parents=True, exist_ok=True)
    settings = replace(settings, outputs_root=outputs)
    configure_logging(
        level=args.log_level, console=True, jsonl=True,
        log_dir=settings.paths.logs, filename="pilot_v2.log", force=True,
    )

    config = load_config(root / args.campaign, root=root)
    plan = build_preregistration(config)
    arms = {str(arm): root / str(path) for arm, path in config["arms"].items()}

    started = time.monotonic()
    cells = plan_campaign(arms=arms, replicates=args.replicates, base_seed=PILOT_SEED)
    runner = CampaignRunner(settings=settings, project_root=root, workers=args.workers)
    runner.run(cells)

    hypotheses = {}
    for hypothesis in plan.hypotheses:
        differences = runner.paired(hypothesis.treatment, hypothesis.control, hypothesis.indicator)
        sd = statistics.stdev(differences) if len(differences) > 1 else math.nan
        hypotheses[hypothesis.hypothesis_id] = {
            "treatment": hypothesis.treatment,
            "control": hypothesis.control,
            "indicator": hypothesis.indicator,
            "direction": hypothesis.direction,
            "pairs": len(differences),
            "mean_difference": round(statistics.fmean(differences), 6) if differences else None,
            "sd_difference": round(sd, 6),
            "min_difference": round(min(differences), 6) if differences else None,
            "max_difference": round(max(differences), 6) if differences else None,
            "differences": [round(d, 8) for d in differences],
        }
        print(f"  {hypothesis.hypothesis_id}: n={len(differences):3d}  "
              f"mean={hypotheses[hypothesis.hypothesis_id]['mean_difference']}  sd={sd:.4f}")

    excluded = [r.to_payload() for r in runner.results if not r.included]
    report = {
        "pilot": "v2 short pilot on the serial engine with backorders",
        "status": "pilot: dispersion for the power simulation, not a result",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "campaign": plan.campaign,
        "base_seed": PILOT_SEED,
        "seeds_not_to_reuse": sorted({PILOT_SEED, V1_CAMPAIGN_SEED}),
        "replicates": args.replicates,
        "arms": {arm: path.name for arm, path in arms.items()},
        "cells": len(runner.results),
        "excluded": excluded,
        "hypotheses": hypotheses,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    path = root / args.report
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(report), encoding="utf-8")
    print(f"  {len(runner.results)} cells, {len(excluded)} excluded, "
          f"{report['elapsed_seconds']} s  ->  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
