#!/usr/bin/env python
"""Re-run the v1 descriptor comparison (RQ3) on the v2 engine, as exploratory.

    python scripts/run_descriptor_exploratory_v2.py --outputs-root outputs/v2
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from xai_gov.analysis.preregistration import build_preregistration
from xai_gov.analysis.report import run_campaign
from xai_gov.core.hashing import canonical_json
from xai_gov.core.logging import configure_logging
from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings
from xai_gov.io.yaml_loader import load_config

MAX_WORKERS = 6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v2 descriptor comparison (exploratory)")
    parser.add_argument("--campaign", default="configs/campaigns/descriptor_exploratory_v2.yaml")
    parser.add_argument("--outputs-root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must lie in [1, {MAX_WORKERS}]")

    root = Path(__file__).resolve().parents[1]
    outputs = Path(args.outputs_root).resolve()
    (outputs / "runs").mkdir(parents=True, exist_ok=True)
    settings = replace(load_settings(Paths(root=root)), outputs_root=outputs)
    configure_logging(level="WARNING", console=True, jsonl=True, log_dir=settings.paths.logs,
                      filename="descriptor_exploratory_v2.log", force=True)
    config = load_config(root / args.campaign, root=root)
    plan = build_preregistration({k: v for k, v in config.items() if k != "arms"})
    arms = {str(k): root / str(v) for k, v in config["arms"].items()}
    assert plan.replicates is not None and plan.base_seed is not None
    report = run_campaign(plan=plan, arms=arms, settings=settings, project_root=root,
                          replicates=plan.replicates, base_seed=plan.base_seed,
                          sealed_hash=None, workers=args.workers)
    payload = report.to_payload()
    directory = outputs / "campaigns" / plan.campaign
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "descriptor_exploratory_v2_report.json"
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    for h in payload["hypotheses"]:
        cs = h["confidence_sequence"]
        print(f"  {h['id']}: pairs={h['pairs']} mean diff={h['mean_difference']:+.4f} "
              f"CS=[{cs['lower']:+.4f}, {cs['upper']:+.4f}]  {h['verdict']}")
    print(f"  status: {payload['reporting_status']}\n  report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
