#!/usr/bin/env python
"""Run the sealed v2 confirmatory campaign (step 5 of the revision).

    python scripts/run_confirmatory_v2.py --outputs-root DIR [--workers 6]

Refuses to run unless the plan matches its seal in specs/v2. Replicates and the
base seed come from the sealed plan and cannot be overridden here. Writes the
campaign report, with the evidence for H1-H6, the envelope per arm and the
scenario the sealed decision rules select, to DIR/campaigns/confirmatory_v2/.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from xai_gov.analysis.plan_v2 import decision_scenario
from xai_gov.analysis.preregistration import build_preregistration, load_seal
from xai_gov.analysis.report import run_campaign
from xai_gov.core.hashing import canonical_json
from xai_gov.core.logging import configure_logging
from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings
from xai_gov.io.yaml_loader import load_config

MAX_WORKERS = 6
REPORT_NAME = "confirmatory_v2_report.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the sealed v2 campaign")
    parser.add_argument("--campaign", default="configs/campaigns/confirmatory_v2.yaml")
    parser.add_argument("--seal", default="specs/v2/confirmatory_v2.seal.json")
    parser.add_argument("--outputs-root", required=True,
                        help="where run artifacts and the report go")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must lie in [1, {MAX_WORKERS}]")

    root = Path(__file__).resolve().parents[1]
    settings = load_settings(Paths(root=root))
    outputs = Path(args.outputs_root).resolve()
    (outputs / "runs").mkdir(parents=True, exist_ok=True)
    settings = replace(settings, outputs_root=outputs)
    configure_logging(level=args.log_level, console=True, jsonl=True,
                      log_dir=settings.paths.logs, filename="confirmatory_v2.log", force=True)

    config = load_config(root / args.campaign, root=root)
    plan = build_preregistration({k: v for k, v in config.items() if k != "arms"})
    seal_path = root / args.seal
    if not seal_path.is_file():
        print(f"no seal at {seal_path}; run scripts/seal_v2.py first", file=sys.stderr)
        return 2
    _, sealed_hash = load_seal(seal_path)
    if sealed_hash != plan.plan_hash:
        print("the plan does not match its seal; refusing to run a confirmatory campaign",
              file=sys.stderr)
        return 2
    if plan.replicates is None or plan.base_seed is None:
        print("the sealed plan fixes no replicates or base seed", file=sys.stderr)
        return 2

    arms = {str(k): root / str(v) for k, v in config["arms"].items()}
    print(f"confirmatory_v2: {len(arms)} arms x {plan.replicates} replicates, "
          f"seed {plan.base_seed}, seal {sealed_hash[:16]}")
    report = run_campaign(
        plan=plan, arms=arms, settings=settings, project_root=root,
        replicates=plan.replicates, base_seed=plan.base_seed,
        sealed_hash=sealed_hash, workers=args.workers,
    )
    payload = report.to_payload()
    payload["decision_scenario"] = decision_scenario(payload["hypotheses"])

    directory = outputs / "campaigns" / plan.campaign
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / REPORT_NAME
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")

    for hypothesis in payload["hypotheses"]:
        evidence = hypothesis["evidence"]
        print(f"  {hypothesis['id']}: pairs={hypothesis['pairs']}  "
              f"mean diff={hypothesis['mean_difference']:+.4f}  "
              f"e={evidence.get('forward_e_value')}  {hypothesis['verdict']}")
    scenario = payload["decision_scenario"]
    print(f"  scenario {scenario['rule']}: {scenario['when']} -> {scenario['claim']}")
    print(f"  status: {payload['reporting_status']}\n  report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
