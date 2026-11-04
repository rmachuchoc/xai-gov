#!/usr/bin/env python
"""Seal the v2 confirmatory plan (step 4 of the revision).

    python scripts/seal_v2.py            # check, then seal
    python scripts/seal_v2.py --check    # check only

Runs every check of `analysis.plan_v2` and, only if all pass, writes the seal to
specs/v2/confirmatory_v2.seal.json. An existing seal is never overwritten: a
changed plan must be sealed as a recorded amendment (--amend).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from xai_gov.analysis.plan_v2 import check_plan, failures
from xai_gov.analysis.preregistration import build_preregistration
from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings
from xai_gov.io.yaml_loader import load_config
from xai_gov.orchestration.runner import plan_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="check and seal the v2 plan")
    parser.add_argument("--campaign", default="configs/campaigns/confirmatory_v2.yaml")
    parser.add_argument("--seal", default="specs/v2/confirmatory_v2.seal.json")
    parser.add_argument("--check", action="store_true", help="run the checks only")
    parser.add_argument("--amend", action="store_true",
                        help="seal a changed plan as an amendment, keeping the old seal")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    settings = load_settings(Paths(root=root))
    config = load_config(root / args.campaign, root=root)
    plan = build_preregistration({k: v for k, v in config.items() if k != "arms"})
    horizons = {
        str(arm): plan_experiment(root / str(path), settings=settings).periods
        for arm, path in config["arms"].items()
    }
    checks = check_plan(plan, root=root, horizons=horizons)
    for check in checks:
        print(f"  [{'ok' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    failed = failures(checks)
    if failed:
        print(f"\n{len(failed)} check(s) failed; the plan is not sealed.", file=sys.stderr)
        return 2
    if args.check:
        print(f"\nall {len(checks)} checks pass; plan hash {plan.plan_hash[:16]}")
        return 0
    digest = plan.seal(root / args.seal, amend=args.amend)
    print(f"\nsealed {plan.campaign}: {digest}\n  -> {args.seal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
