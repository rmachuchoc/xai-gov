#!/usr/bin/env python
"""Run the detector null line (decision 6 of the revision).

    python scripts/run_detector_null.py                  # 200 replicates per arm
    python scripts/run_detector_null.py --replicates 20  # a quick check

Every arm runs on stationary customer demand, so each change the martingale
declares at the retailer is a false alarm. The report gives the false-alarm
rate with a Clopper-Pearson interval, the coverage error against the target,
and how often and for how long ε_t was clipped. No Ville bound is claimed.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from xai_gov.analysis.detector_null import NullLinePlan, run_null_line
from xai_gov.core.hashing import canonical_json
from xai_gov.core.logging import configure_logging
from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings
from xai_gov.io.yaml_loader import load_config

#: The machine budget: at most 40% of 16 cores.
MAX_WORKERS = 6


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the detector null line")
    parser.add_argument("--config", default="configs/analysis/detector_null.yaml")
    parser.add_argument("--replicates", type=int, default=None,
                        help="override the replicates declared in the config")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", default=None,
                        help="report path (default: <outputs>/analysis/detector_null.json)")
    parser.add_argument("--outputs-root", default=None,
                        help="where run artifacts go (default: the configured outputs root)")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must lie in [1, {MAX_WORKERS}]")

    root = Path(__file__).resolve().parents[1]
    settings = load_settings(Paths(root=root))
    if args.outputs_root:
        outputs = Path(args.outputs_root).resolve()
        (outputs / "runs").mkdir(parents=True, exist_ok=True)
        settings = replace(settings, outputs_root=outputs)
    configure_logging(
        level=args.log_level, console=True, jsonl=True,
        log_dir=settings.paths.logs, filename="detector_null.log", force=True,
    )
    config = load_config(root / args.config, root=root)
    if args.replicates is not None:
        config["replicates"] = args.replicates
    plan = NullLinePlan.from_config(config, root=root)

    summary = run_null_line(plan, settings=settings, project_root=root, workers=args.workers)
    summary["generated_at"] = datetime.now(UTC).isoformat(timespec="seconds")

    output = Path(args.output) if args.output else (
        settings.outputs_root / "analysis" / "detector_null.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(canonical_json(summary), encoding="utf-8")

    for arm, entry in summary["arms"].items():
        alarm = entry["false_alarms_demand_facing"]
        clip = entry["epsilon_clipping"]
        print(
            f"  {arm:<18} false alarms {alarm['runs_with_alarm']}/{alarm['runs']} "
            f"[{alarm['ci_low']:.3f}, {alarm['ci_high']:.3f}]   "
            f"coverage error {entry['coverage_error']['mean']}   "
            f"clipped {clip['clipped_frequency']['mean']}  max run {clip['max_duration']}"
        )
    print(f"  report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
