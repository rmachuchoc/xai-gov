#!/usr/bin/env python
"""Simulated power of the v2 confirmatory procedure (step 3 of the revision).

    python scripts/power_v2.py                    # 2000 paths per case
    python scripts/power_v2.py --sims 300         # a quick look

Reads the pilot's paired differences (specs/v2/pilot_v2.json), simulates the
predictable aGRAPA e-process at the familywise threshold 2k/alpha = 240 for a
declared effect of 0.05 on beta, and writes the power curves and the required
number of replicates to specs/v2/power_v2.json. The campaign's n is the largest
requirement across hypotheses, since all arms share replicates.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xai_gov.analysis.agrapa import describe
from xai_gov.analysis.power_sim import PowerSpec, plan_replicates, simulate_hypothesis
from xai_gov.core.hashing import canonical_json, content_hash
from xai_gov.io.yaml_loader import load_config

POWER_SEED = 20261106
MAX_WORKERS = 6


def _run(job: tuple[PowerSpec, dict[str, Any]]) -> dict[str, Any]:
    spec, options = job
    return simulate_hypothesis(spec, **options)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="simulate the power of the v2 procedure")
    parser.add_argument("--pilot", default="specs/v2/pilot_v2.json")
    parser.add_argument("--campaign", default="configs/campaigns/confirmatory_v2.yaml")
    parser.add_argument("--effect", type=float, default=0.05)
    parser.add_argument("--target", type=float, default=0.80)
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--n-max", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--report", default="specs/v2/power_v2.json")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must lie in [1, {MAX_WORKERS}]")

    root = Path(__file__).resolve().parents[1]
    pilot_text = (root / args.pilot).read_text(encoding="utf-8")
    pilot = json.loads(pilot_text)
    campaign = load_config(root / args.campaign, root=root)
    hypotheses = campaign["hypotheses"]
    options = {
        "alpha": float(campaign.get("alpha", 0.05)), "hypotheses": len(hypotheses),
        "target": args.target, "sims": args.sims, "n_max": args.n_max,
    }
    jobs = []
    for index, hypothesis in enumerate(hypotheses):
        entry = pilot["hypotheses"][hypothesis["id"]]
        spec = PowerSpec(
            hypothesis_id=hypothesis["id"], differences=tuple(entry["differences"]),
            two_sided=hypothesis["direction"] == "different", effect=args.effect,
        )
        jobs.append((spec, {**options, "seed": POWER_SEED + 1000 * index}))

    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
        results = list(pool.map(_run, jobs))
    plan = plan_replicates(results)

    report = {
        "status": "power simulation of the confirmatory procedure",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "procedure": describe(),
        "pilot": args.pilot,
        "pilot_hash": content_hash(pilot),
        "pilot_base_seed": pilot["base_seed"],
        "power_seed": POWER_SEED,
        "effect": args.effect,
        "options": options,
        "plan": plan,
        "hypotheses": results,
        "elapsed_seconds": round(time.monotonic() - started, 1),
    }
    path = root / args.report
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(report), encoding="utf-8")

    for result in results:
        gens = result["generators"]
        print(f"  {result['hypothesis']}  sd={result['pilot_sd']:.4f}  "
              f"n(bootstrap)={gens['bootstrap']['required_n']}  "
              f"n(normal)={gens['normal']['required_n']}  "
              f"null rejection={gens['bootstrap']['null_rejection_rate']:.4f}"
              f"/{gens['normal']['null_rejection_rate']:.4f} (bound {result['null_bound']})")
    print(f"  replicates: {plan['replicates']}  (binding {plan['binding_hypothesis']})  "
          f"{report['elapsed_seconds']} s  ->  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
