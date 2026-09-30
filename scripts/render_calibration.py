#!/usr/bin/env python
"""Build the calibration table the manuscript needs.

Reads the calibration reports produced by `calibrate_all.py` and emits a LaTeX
table. Generated rather than transcribed: four of the nine defects this project
catalogued were found in transcribed prose, so the transcription step is
removed instead of being performed more carefully.

The matching *figure* is rendered by `render_figures.py` as fig5-calibration,
at the same print geometry and 300 dpi as the other four. Two renderers for one
report would eventually disagree about a number, so this one draws nothing.

    python scripts/render_calibration.py \\
        --reports outputs/calibration/calibration_weekly_both.json \\
                  outputs/calibration/calibration_weekly_ar1.json

What it reports, and why in this shape. The deliverable is not a gap but a
*distribution* of gaps across thirty real series, so the table carries min,
median and max rather than a single number that invites being read as the gap.
And the identification column is reported beside the fit column because the
finding is the tension between them: the twin's own process is identified and
wrong, while the process that fits is not identified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# The narrative order: the twin's own process first, then the alternative, so
# the table reads as the comparison it is rather than as two unrelated rows.
PROCESS_LABEL = {
    "iid": "Level plus independent noise (the twin's own)",
    "ar1": "Level plus persistent noise, AR(1)",
}


def load(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Collect per-process summaries, later reports winning on conflict.

    A later report is a rerun at a larger proposal budget, so it supersedes an
    earlier one for the same process. Merging silently in the other direction
    would let a thinner run overwrite a better one.
    """
    merged: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run calibrate_all.py first")
        payload = json.loads(path.read_text(encoding="utf-8"))
        by_process = payload.get("by_process", {})
        series = payload.get("series", {})
        for process, summary in by_process.items():
            if process.startswith("_"):
                continue
            gaps = sorted(
                s["processes"][process]["sim_to_real_gap"]
                for s in series.values()
                if process in s.get("processes", {})
            )
            merged[process] = {
                **summary,
                "series_count": len(gaps),
                "gaps": gaps,
                "proposals": payload.get("proposals"),
                "cadence": payload.get("cadence"),
                "dropped": payload.get("dropped_statistics", []),
            }
    if not merged:
        raise ValueError("no per-process summaries found in the given reports")
    return merged


def latex_table(merged: dict[str, dict[str, Any]]) -> str:
    """The calibration table, in the manuscript's own conventions."""
    order = [p for p in ("iid", "ar1") if p in merged]
    cadence = merged[order[0]]["cadence"]
    count = merged[order[0]]["series_count"]

    rows: list[str] = []
    for process in order:
        s = merged[process]
        gap = s["gap"]
        identified = s["identified_series"]
        overfit = s["overfitted_series"]
        rows.append(
            f"{PROCESS_LABEL[process]} & {gap['min']:.2f} & {gap['median']:.2f} & "
            f"{gap['max']:.2f} & {identified}/{count} & {overfit}/{count}\\\\"
        )

    body = "\n".join(rows)
    return f"""\\begin{{table}}[H]
\\small
\\caption{{Calibration against {count} real store-by-category demand series at
{cadence} cadence, M5 competition data. The gap is the posterior predictive
distance on statistics held out of the fit; the distribution is reported rather
than a single value because the point of running every series is that the gap
is a range. Series are rescaled to the twin's operating magnitude before
calibrating: the absolute level is an artifact of how many items the
aggregation swept up, while the shape --- dispersion and autocorrelation --- is
what the twin's mechanism can or cannot reproduce, and the rescaling is affine
so it leaves both unchanged.\\label{{tab:calibration}}}}
\\begin{{tabularx}}{{\\textwidth}}{{XS[table-format=1.2]S[table-format=1.2]S[table-format=1.2]cc}}
\\toprule
\\textbf{{Generating process}} & {{\\textbf{{Min}}}} & {{\\textbf{{Median}}}} &
{{\\textbf{{Max}}}} & \\textbf{{Identified}} & \\textbf{{Overfit}}\\\\
\\midrule
{body}
\\bottomrule
\\end{{tabularx}}

\\noindent{{\\small The two rows are in tension and that tension is the finding:
the twin's own process is identified and wrong, reproducing the moments of real
demand and failing its dynamics in every series, while the process that halves
the gap leaves its persistence parameter unidentified. Raising the proposal
budget tenfold did not narrow it, and removing a statistic redundant given the
others made the fit worse, so the non-identifiability is a property of the
summary rather than of the sampling.}}
\\end{{table}}"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/calibration"))
    args = parser.parse_args(argv)

    merged = load(args.reports)
    args.out.mkdir(parents=True, exist_ok=True)

    table_path = args.out / "table-calibration.tex"
    table_path.write_text(latex_table(merged) + "\n", encoding="utf-8")

    for process in ("iid", "ar1"):
        if process not in merged:
            continue
        s = merged[process]
        gap = s["gap"]
        print(
            f"{process:<5} gap {gap['min']:.2f} / {gap['median']:.2f} / {gap['max']:.2f}"
            f"   identified {s['identified_series']}/{s['series_count']}"
            f"   overfit {s['overfitted_series']}/{s['series_count']}"
        )
    print(f"\ntable   {table_path}")
    print(
        "\nThe matching figure is fig5-calibration from render_figures.py; run "
        "`make figures` for the 300 dpi PNG."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
