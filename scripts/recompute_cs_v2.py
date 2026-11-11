#!/usr/bin/env python
"""Betting confidence sequences for every v2 campaign, from the stored cells.

    python scripts/recompute_cs_v2.py --outputs-root outputs/v2

The campaign reports are not rewritten: the post-hoc analysis cites the
confirmatory report by hash, and the report is the sealed record of what was
run. Next to each report this writes ``confidence_sequences_v2.json`` with the
interval of every hypothesis under the betting construction of
``analysis.anytime.confidence_sequence``, the empirical-Bernstein interval it
replaces, and a check that every supported hypothesis has an interval that
excludes zero. No run is simulated again; the differences are read from the
runs the report names.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from xai_gov.analysis.anytime import CS_PROCEDURE, confidence_sequence
from xai_gov.analysis.campaign import CellResult
from xai_gov.analysis.posthoc_v2 import load_cells
from xai_gov.core.hashing import canonical_json, content_hash

REPORTS = (
    "campaigns/confirmatory_v2/confirmatory_v2_report.json",
    "campaigns/descriptor_exploratory_v2/descriptor_exploratory_v2_report.json",
    "campaigns/detector_exploratory_v2/detector_exploratory_v2_report.json",
)
OUTPUT_NAME = "confidence_sequences_v2.json"
#: Predeclared in the plan: indicators lie in [0, 1], so differences lie in [-1, 1].
SCALE = 1.0


def paired(results: tuple[CellResult, ...], treatment: str, control: str,
           indicator: str) -> list[float]:
    """Treatment minus control by replicate, as `CampaignRunner.paired` does."""
    by_replicate: dict[int, dict[str, float]] = {}
    for result in results:
        if not result.included or result.cell.arm not in (treatment, control):
            continue
        value = result.indicator(indicator)
        if value is not None:
            by_replicate.setdefault(result.cell.replicate, {})[result.cell.arm] = value
    return [v[treatment] - v[control] for _, v in sorted(by_replicate.items())
            if treatment in v and control in v]


def recompute(report_path: Path, runs_root: Path) -> dict[str, Any]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    results = load_cells(report_path, runs_root)
    alpha = float(report["plan"]["alpha"])
    rows = []
    for h in report["hypotheses"]:
        differences = paired(results, h["treatment"], h["control"], h["indicator"])
        if len(differences) != h["pairs"]:
            raise SystemExit(f"{h['id']}: {len(differences)} pairs, report has {h['pairs']}")
        mean = sum(differences) / len(differences)
        if not math.isclose(mean, h["mean_difference"], abs_tol=1e-6):
            raise SystemExit(f"{h['id']}: mean {mean} differs from the report")
        interval = confidence_sequence(differences, alpha=alpha, bound=SCALE)
        coherent = (not h["supported"]) or interval.excludes_zero
        rows.append({
            "id": h["id"], "treatment": h["treatment"], "control": h["control"],
            "indicator": h["indicator"], "pairs": len(differences),
            "supported": h["supported"], "verdict": h["verdict"],
            "confidence_sequence": interval.to_payload(),
            "replaced_empirical_bernstein": {
                k: h["confidence_sequence"][k] for k in ("lower", "upper", "width")
            },
            "coherent_with_test": coherent,
        })
    if not all(r["coherent_with_test"] for r in rows):
        raise SystemExit(f"{report_path}: a supported hypothesis has an interval with zero")
    return {
        "campaign": report["campaign"],
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "procedure": CS_PROCEDURE,
        "alpha": alpha,
        "scale": SCALE,
        "source_report": str(report_path.relative_to(runs_root.parent)),
        "source_report_hash": content_hash(report),
        "note": ("anytime-valid by Ville's inequality applied to the two-sided aGRAPA "
                 "e-process at each candidate mean; the interval is the running "
                 "intersection over replicates, unadjusted for multiplicity"),
        "hypotheses": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="v2 betting confidence sequences")
    parser.add_argument("--outputs-root", required=True)
    args = parser.parse_args(argv)
    outputs = Path(args.outputs_root).resolve()
    for relative in REPORTS:
        report_path = outputs / relative
        payload = recompute(report_path, outputs / "runs")
        target = report_path.parent / OUTPUT_NAME
        target.write_text(canonical_json(payload) + "\n", encoding="utf-8")
        print(f"{payload['campaign']}: {target}")
        for r in payload["hypotheses"]:
            cs, old = r["confidence_sequence"], r["replaced_empirical_bernstein"]
            print(f"  {r['id']:5s} mean {cs['mean']:+.4f}  CS [{cs['lower']:+.4f}, "
                  f"{cs['upper']:+.4f}]  (EB [{old['lower']:+.4f}, {old['upper']:+.4f}])  "
                  f"{r['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
