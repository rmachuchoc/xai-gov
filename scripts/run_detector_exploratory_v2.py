#!/usr/bin/env python
"""Re-run the v1 detector ablations on the v2 engine, as exploratory (R1.4).

    python scripts/run_detector_exploratory_v2.py --outputs-root outputs/v2

Runs RQ6-RQ6d with coverage measured against the fixed target 0.10, and
recomputes the v1 claim that excluding the warm-up explains about one seventh
of the calibration gap: the share of the contaminated arm's |coverage error|
removed by excluding the warm-up scores, (|CE_contaminated| - |CE_shielded|) /
|CE_contaminated|, with a paired bootstrap interval over replicates.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from xai_gov.analysis.preregistration import build_preregistration
from xai_gov.analysis.report import run_campaign
from xai_gov.core.hashing import canonical_json
from xai_gov.core.logging import configure_logging
from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings
from xai_gov.io.yaml_loader import load_config

MAX_WORKERS = 6
BOOTSTRAP_SEED = 20261110


def warmup_share(contaminated: list[float], shielded: list[float], *, resamples: int = 2000,
                 seed: int = BOOTSTRAP_SEED) -> dict[str, object]:
    """Share of the contaminated gap removed by excluding the warm-up, with a CI."""
    c = np.asarray(contaminated, dtype=float)
    s = np.asarray(shielded, dtype=float)

    def share(ci: np.ndarray, si: np.ndarray) -> float:
        base = float(ci.mean())
        return float((base - si.mean()) / base) if base > 0 else float("nan")

    rng = np.random.default_rng(seed)
    draws = [
        share(c[idx], s[idx])
        for idx in (rng.integers(0, len(c), size=len(c)) for _ in range(resamples))
    ]
    return {
        "share": round(share(c, s), 6),
        "ci95": [round(float(np.nanquantile(draws, 0.025)), 6),
                 round(float(np.nanquantile(draws, 0.975)), 6)],
        "mean_abs_ce_contaminated": round(float(c.mean()), 6),
        "mean_abs_ce_shielded": round(float(s.mean()), 6),
        "pairs": len(c),
        "v1_claim": "about one seventh (0.143)",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v2 detector ablations (exploratory)")
    parser.add_argument("--campaign", default="configs/campaigns/detector_exploratory_v2.yaml")
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
                      filename="detector_exploratory_v2.log", force=True)
    config = load_config(root / args.campaign, root=root)
    plan = build_preregistration({k: v for k, v in config.items() if k != "arms"})
    arms = {str(k): root / str(v) for k, v in config["arms"].items()}
    assert plan.replicates is not None and plan.base_seed is not None

    report = run_campaign(plan=plan, arms=arms, settings=settings, project_root=root,
                          replicates=plan.replicates, base_seed=plan.base_seed,
                          sealed_hash=None, workers=args.workers)
    payload = report.to_payload()

    by_arm: dict[str, dict[int, float]] = {}
    for cell in payload["cells"]:
        if not cell["included"]:
            continue
        kpis = (outputs / "runs" / cell["run_name"] / "kpis.json").read_text("utf-8")
        guarantees = json.loads(kpis).get("guarantees", {})
        if guarantees.get("status") == "available":
            by_arm.setdefault(cell["arm"], {})[cell["replicate"]] = float(
                guarantees["abs_coverage_error"]
            )
    shared = sorted(set(by_arm["contaminated"]) & set(by_arm["shielded"]))
    payload["warmup_share"] = warmup_share(
        [by_arm["contaminated"][r] for r in shared], [by_arm["shielded"][r] for r in shared]
    )
    payload["coverage_by_arm"] = {
        arm: {"mean_abs_coverage_error": round(float(np.mean(list(v.values()))), 6),
              "replicates": len(v)}
        for arm, v in sorted(by_arm.items())
    }

    directory = outputs / "campaigns" / plan.campaign
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "detector_exploratory_v2_report.json"
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    for hypothesis in payload["hypotheses"]:
        print(f"  {hypothesis['id']}: mean diff={hypothesis['mean_difference']:+.4f}  "
              f"{hypothesis['verdict']}")
    for arm, entry in payload["coverage_by_arm"].items():
        print(f"  {arm:16s} |CE| = {entry['mean_abs_coverage_error']:.4f}")
    print(f"  warm-up share: {payload['warmup_share']}")
    print(f"  status: {payload['reporting_status']}\n  report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
