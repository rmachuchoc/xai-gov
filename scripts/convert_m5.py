#!/usr/bin/env python
"""Convert an M5 sales file into the demand series the calibration reads.

    python scripts/convert_m5.py sales_train_evaluation.csv --item FOODS_3_090 \
        --store CA_3 --out data/demand/demand.csv

M5 arrives in *wide* format: one row per (item, store) series and one column
per day, named d_1 … d_1941. The calibration phase wants a long series, one row
per period, so the conversion is a transpose plus a column selection.

Why a script rather than a loader feature: the choice of which series to
calibrate against is a modelling decision that belongs in the record, not a
default buried in code. Running this leaves a file whose provenance is a
command someone can read.

The M5 data is not redistributed here. Download `sales_train_evaluation.csv`
from the M5 Forecasting Accuracy competition on Kaggle; see
`data/demand/README.md` for alternatives that need no account.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="convert M5 wide format to a demand series")
    parser.add_argument("source", type=Path, help="M5 sales_train_evaluation.csv")
    parser.add_argument("--item", help="item_id to extract, e.g. FOODS_3_090")
    parser.add_argument("--store", help="store_id to extract, e.g. CA_3")
    parser.add_argument(
        "--aggregate",
        action="store_true",
        help=(
            "sum every matching series instead of requiring exactly one. Useful for "
            "a store or department total, and it changes what is being calibrated: "
            "an aggregate is smoother than any single SKU, so the fitted dispersion "
            "will understate item-level volatility."
        ),
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=0,
        help="keep only the last N days (0 keeps all). The calibration needs 30 or more.",
    )
    parser.add_argument("--out", type=Path, default=Path("data/demand/demand.csv"))
    args = parser.parse_args(argv)

    if not args.source.exists():
        print(f"not found: {args.source}", file=sys.stderr)
        return 2

    with args.source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            print("the source has no header row", file=sys.stderr)
            return 2
        day_columns = [name for name in reader.fieldnames if name.startswith("d_")]
        if not day_columns:
            print(
                "no d_* columns found; this does not look like an M5 wide file",
                file=sys.stderr,
            )
            return 2
        # Sorted numerically, not lexically: d_10 must not precede d_2, and a
        # lexical sort would silently reorder the series into nonsense.
        day_columns.sort(key=lambda name: int(name[2:]))

        matched: list[list[float]] = []
        for row in reader:
            if args.item and row.get("item_id") != args.item:
                continue
            if args.store and row.get("store_id") != args.store:
                continue
            matched.append([float(row.get(day) or 0.0) for day in day_columns])

    if not matched:
        print("no rows matched the given item and store", file=sys.stderr)
        return 1
    if len(matched) > 1 and not args.aggregate:
        print(
            f"{len(matched)} series matched. Narrow the selection, or pass "
            "--aggregate to sum them and accept the smoothing that implies.",
            file=sys.stderr,
        )
        return 1

    series = (
        [sum(values) for values in zip(*matched, strict=True)]
        if len(matched) > 1
        else matched[0]
    )
    if args.periods > 0:
        series = series[-args.periods :]

    if len(series) < 30:
        print(
            f"only {len(series)} observations; the held-out predictive check needs "
            "at least 30 to be a check",
            file=sys.stderr,
        )
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["period", "demand"])
        for index, value in enumerate(series, start=1):
            writer.writerow([index, f"{value:g}"])

    nonzero = sum(1 for value in series if value > 0)
    print(f"wrote {args.out}  ({len(series)} periods, {nonzero} nonzero)")
    if nonzero < len(series) * 0.5:
        # Intermittent demand is a different modelling problem, and calibrating
        # a normal-dispersion process against it would fit the zeros.
        print(
            "WARNING: more than half the periods are zero. This is intermittent "
            "demand, and the twin's demand process is not intermittent — pick a "
            "faster-moving series or model the intermittency explicitly.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
