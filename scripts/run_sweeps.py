#!/usr/bin/env python
"""Run the two parameter sweeps the coauthor reviews asked for.

    python scripts/run_sweeps.py                      # both sweeps
    python scripts/run_sweeps.py --only break-even    # one of them

Kept separate from `run_study.py` because a sweep re-runs a paired campaign at
every level, so its cost is the study's cost times the number of levels. Making
it a phase of the main study would mean every routine run paid for an analysis
that changes only when the declared economics change.

The break-even sweep answers: at what unit intervention cost does substitution
begin to create net value? The severity sweep answers: what disruption severity
justifies that cost? The second also tests Theorem 2's prediction that the
optimal threshold decreases in severity, so the curve's monotonicity is a
result rather than a diagnostic.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xai_gov.analysis.sweeps import (
    BREAK_EVEN_COSTS,
    SEVERITY_LEVELS,
    ParameterSweep,
)
from xai_gov.core.hashing import canonical_json
from xai_gov.core.logging import configure_logging
from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings

SWEEP_REPORT_NAME = "sweeps.json"
MAX_WORKERS = 6


def report(result: Any) -> None:
    payload = result.to_payload()
    print(f"\n  {payload['parameter']}  ({payload['comparison']})")
    print(f"  indicator: {payload['indicator']}")
    for level in payload["levels"]:
        difference = level["difference"]
        shown = "—" if difference is None else f"{difference:+.4f}"
        ci = level.get("ci95")
        interval = "" if ci is None else f"  [{ci[0]:+.4f}, {ci[1]:+.4f}]"
        print(f"    {level['label']:<24} {shown:>10}{interval}   "
              f"pairs={level.get('paired_replicates')}")
    if not payload["has_data"]:
        # No crossing and no monotonicity are reported when nothing was
        # measured: printing them would present the absence of data as a
        # property of the curve.
        print(f"    NO DATA    {payload['usable_levels']} of "
              f"{len(payload['levels'])} levels usable")
        print(f"    {payload['interpretation']}")
        return
    print(f"    crossing   {payload['crossing']}")
    crossing_ci = payload.get("crossing_interval") or {}
    print(f"    crossing CI {crossing_ci.get('ci95')}  "
          f"(resamples without a crossing: {crossing_ci.get('share_without_crossing')})")
    print(f"    monotone   {payload['monotone']}")
    print(f"    {payload['interpretation']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="run the parameter sweeps")
    parser.add_argument(
        "--only", choices=("break-even", "severity"), help="run one sweep only"
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=100,
        help=(
            "replicates per level (default 100). Each level's paired difference "
            "and the crossing carry a bootstrap interval over these replicates."
        ),
    )
    parser.add_argument(
        "--service-value",
        type=float,
        default=100.0,
        help=(
            "value of one unit of service level in the units the oversight cost "
            "is denominated in (default 100). Declared rather than assumed: the "
            "indicator is a fraction and the cost is a total, so a crossing "
            "depends on this price and must be reported with it."
        ),
    )
    # v2: a seed of its own, distinct from the v1 campaign (20260703) and from
    # every seed of the v2 plan, so no sweep cell coincides with a sealed cell.
    parser.add_argument("--base-seed", type=int, default=20261108)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--outputs-root", required=True,
                        help="where the sweep runs and report go (v2: outputs/v2)")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= MAX_WORKERS:
        parser.error(f"--workers must lie in [1, {MAX_WORKERS}]")

    root = Path(__file__).resolve().parents[1]
    settings = load_settings(Paths(root=root))
    outputs = Path(args.outputs_root).resolve()
    (outputs / "runs").mkdir(parents=True, exist_ok=True)
    settings = replace(settings, outputs_root=outputs)
    configure_logging(
        level=args.log_level, console=True, jsonl=True,
        log_dir=settings.paths.logs, filename="sweeps.log", force=True,
    )

    experiments = root / "configs" / "experiments"
    treatment = experiments / "high_vol_shielded.yaml"
    control = experiments / "high_vol_governed_h2.yaml"
    # The ungoverned arm is required, not optional: both questions the reviews
    # asked are economic, and net value is a difference against it. Without the
    # baseline the sweep could only compare two governed arms on an operational
    # indicator, which answers whether the shield helps rather than whether it
    # pays for itself.
    baseline = experiments / "high_vol_ungoverned.yaml"
    for path in (treatment, control, baseline):
        if not path.exists():
            print(f"missing experiment: {path}", file=sys.stderr)
            return 2

    print(f"XAI-Gov sweeps  ({root})")
    print(f"replicates per level  {args.replicates}")
    print(f"comparison            {treatment.stem} vs {control.stem}")
    print(f"baseline              {baseline.stem}")
    print(
        f"value                 (gain over baseline x {args.service_value:g}) "
        "minus oversight cost"
    )

    sweep = ParameterSweep(
        settings=settings,
        project_root=root,
        treatment=treatment,
        control=control,
        baseline=baseline,
        service_value=args.service_value,
        replicates=args.replicates,
        base_seed=args.base_seed,
        workers=args.workers,
    )

    results: dict[str, Any] = {}
    empty: list[str] = []

    if args.only != "severity":
        print("\n[1] Break-even in the unit intervention cost")
        break_even = sweep.run(
            parameter="intervention_cost",
            levels=list(BREAK_EVEN_COSTS),
            override="governance.economics.intervention_cost",
        )
        report(break_even)
        results["break_even"] = break_even.to_payload()
        if not break_even.has_data:
            empty.append("break_even")

    if args.only != "break-even":
        print("\n[2] Severity threshold")
        severity = sweep.run(
            parameter="volatility_multiplier",
            levels=list(SEVERITY_LEVELS),
            override="demand.volatility_multiplier",
            labels=["mild 1.5", "moderate 3.0", "severe 6.0", "extreme 10.0"],
        )
        report(severity)
        results["severity"] = severity.to_payload()
        if not severity.has_data:
            empty.append("severity")
        monotone = results["severity"]["monotone"]
        if monotone is False:
            print(
                "\n  NOTE: the severity curve is not monotone, which contradicts "
                "Theorem 2's prediction. Locate the violated assumption before "
                "reporting a threshold.",
                file=sys.stderr,
            )

    directory = settings.outputs_root / "sweeps"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / SWEEP_REPORT_NAME
    path.write_text(
        canonical_json(
            {
                "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "replicates_per_level": args.replicates,
                "service_value": args.service_value,
                "treatment": treatment.stem,
                "control": control.stem,
                "sweeps": results,
                "qualification": (
                    "Any crossing below is conditional on service_value, the declared "
                    "price of one unit of service level: the operational gain is a "
                    "fraction and the oversight cost is a total, so the two are only "
                    "comparable through that price. The intervention cost also carries "
                    "essentially all variance in the modelled outcome (Sobol total "
                    "index 1.000), so a break-even threshold on it is informative and "
                    "fragile at once. Report both qualifications, never the threshold "
                    "alone."
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nsweep report  {path}")
    if empty:
        # A sweep that measured nothing must not exit clean: the report would
        # sit beside the study bundle looking like a result.
        print(
            f"\nsweeps with no usable data: {', '.join(empty)}. The indicator was "
            "absent from the runs, so no threshold is supported.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
