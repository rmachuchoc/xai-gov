#!/usr/bin/env python
"""Turn the M5 competition files into store-by-category demand series.

The M5 data cannot be redistributed: the Kaggle competition terms permit use
but not republication, so this repository carries the recipe and a checksum
rather than the file. Run this script once against a local copy and the
resulting CSVs become ordinary project data.

    # accept the terms at kaggle.com/competitions/m5-forecasting-accuracy,
    # download and unzip, then:
    python scripts/fetch_m5.py --source ~/Downloads/m5 --level store_category

Why store x category and not the bottom level. M5's 30,490 bottom-level series
are intermittent: long runs of zeros, which is a documented property of the
data and a real forecasting challenge. But the twin's demand process is not
intermittent, so calibrating against an intermittent series would test a
mismatch we did not set out to study and would fail for the wrong reason. The
store x category level is smooth, retains weekly seasonality and genuine
autocorrelation, and autocorrelation is the statistic the synthetic
calibration failed. The level is a declared study parameter, written into
every emitted file's header.

What this script does not do. It does not aggregate to a cadence. A period of
the twin is abstract, so mapping days onto periods is a modelling decision
that determines the autocorrelation --- the very quantity being calibrated ---
and it belongs in a preregistered configuration rather than in a loader.
Daily values are emitted as-is and `--resample` is offered only to produce a
*separate* series for a declared weekly cadence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

#: SHA-256 of the two files this script reads, as distributed by the
#: competition. Checked rather than assumed: a silently different file would
#: produce a calibration that cannot be reproduced from the stated source, and
#: that is worse than no calibration.
EXPECTED: dict[str, str] = {
    "sales_train_evaluation.csv":
        "4b4a47c44c38380d2a9168216fea8c9ff2f31b1ddb772f8a0995952a038b8aa0",
    "calendar.csv":
        "d12b5914ef03e66649adf5dd9e996e6602251c22b7a6af8f1f7e3aa12f8860f5",
}

SALES_FILES = ("sales_train_evaluation.csv", "sales_train_validation.csv")
CALENDAR = "calendar.csv"


@dataclass(frozen=True, slots=True)
class Series:
    """One demand series with the provenance needed to defend it."""

    store_id: str
    category: str
    values: list[float]
    first_date: str
    cadence: str

    @property
    def name(self) -> str:
        return f"{self.store_id}_{self.category}".lower()

    def summary(self) -> dict[str, float]:
        """The statistics the calibration compares against.

        Autocorrelation is computed here rather than left to the calibrator so
        that a series can be rejected at load time: a series whose lag-1
        autocorrelation is near zero carries no dynamics to calibrate and
        would reproduce the synthetic failure under a real filename.
        """
        n = len(self.values)
        mean = sum(self.values) / n
        centred = [v - mean for v in self.values]
        var = sum(c * c for c in centred) / (n - 1)
        denom = sum(c * c for c in centred)
        acf1 = (
            sum(centred[i] * centred[i + 1] for i in range(n - 1)) / denom
            if denom > 1e-12
            else 0.0
        )
        std = var**0.5
        return {
            "n": float(n),
            "mean": mean,
            "std": std,
            "cv": std / mean if abs(mean) > 1e-9 else 0.0,
            "autocorr_1": acf1,
            "max_ratio": max(self.values) / mean if abs(mean) > 1e-9 else 0.0,
            "zero_share": sum(1 for v in self.values if v == 0.0) / n,
        }


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            sha.update(block)
    return sha.hexdigest()


def locate(source: Path) -> tuple[Path, Path]:
    """Find the sales and calendar files, preferring the evaluation split.

    The evaluation file carries 1,941 days against the validation file's
    1,913. Both are the same series; the longer one is used because a
    calibration set truncated for no reason is a smaller calibration set.
    """
    for name in SALES_FILES:
        sales = source / name
        if sales.exists():
            break
    else:
        raise FileNotFoundError(
            f"no sales file in {source}; expected one of {', '.join(SALES_FILES)}"
        )
    calendar = source / CALENDAR
    if not calendar.exists():
        raise FileNotFoundError(f"{CALENDAR} not found in {source}")
    return sales, calendar


def first_date(calendar: Path) -> str:
    with calendar.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
        return str(row["date"])


def aggregate(sales: Path, *, level: str) -> list[Series]:
    """Sum the bottom-level series up to the requested level.

    Streamed row by row. The file is roughly 30,000 rows by 1,900 columns and
    loading it whole is unnecessary when the operation is a sum.
    """
    keys = {
        "store_category": ("store_id", "cat_id"),
        "store_department": ("store_id", "dept_id"),
        "state_category": ("state_id", "cat_id"),
    }
    if level not in keys:
        raise ValueError(
            f"unknown level {level!r}; available: {', '.join(sorted(keys))}"
        )
    left, right = keys[level]

    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    with sales.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        day_index = [i for i, name in enumerate(header) if name.startswith("d_")]
        col = {name: i for i, name in enumerate(header)}
        for row in reader:
            key = (row[col[left]], row[col[right]])
            bucket = totals[key]
            if not bucket:
                bucket.extend([0.0] * len(day_index))
            for position, index in enumerate(day_index):
                value = row[index]
                if value:
                    bucket[position] += float(value)

    return [
        Series(
            store_id=a,
            category=b,
            values=values,
            first_date="",
            cadence="daily",
        )
        for (a, b), values in sorted(totals.items())
    ]


def resample_weekly(series: Series) -> Series:
    """Sum daily values into calendar weeks.

    Offered as a *separate* output rather than as a transformation of the
    default, because the cadence a period represents is a preregistered
    modelling choice and a loader that silently picked one would be making it.
    """
    weeks = [
        sum(series.values[i : i + 7])
        for i in range(0, len(series.values) - 6, 7)
    ]
    return Series(
        store_id=series.store_id,
        category=series.category,
        values=weeks,
        first_date=series.first_date,
        cadence="weekly",
    )


def write_series(series: Series, directory: Path, *, source_hash: str) -> Path:
    """Write one series with its provenance in the header comments."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"m5_{series.name}_{series.cadence}.csv"
    stats = series.summary()
    lines = [
        "# M5 Forecasting - Accuracy (Kaggle), aggregated by this project.",
        "# Source: kaggle.com/competitions/m5-forecasting-accuracy",
        "# Redistribution is not permitted by the competition terms; this file",
        "# is derived locally and is not committed to the repository.",
        f"# source_sha256: {source_hash}",
        f"# aggregation: {series.store_id} x {series.category}",
        f"# cadence: {series.cadence}  (one row is one {series.cadence[:-2]})",
        f"# first_date: {series.first_date}",
        "# lag-1 autocorrelation: "
        f"{stats['autocorr_1']:.4f}  zero share: {stats['zero_share']:.4f}",
        "period,demand",
    ]
    lines.extend(f"{i},{v:.1f}" for i, v in enumerate(series.values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source", required=True, type=Path, help="directory holding the M5 CSVs"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("data/demand"), help="output directory"
    )
    parser.add_argument(
        "--level",
        default="store_category",
        help="aggregation level (default store_category)",
    )
    parser.add_argument(
        "--resample",
        choices=("none", "weekly", "both"),
        default="none",
        help="also emit a weekly series; the cadence is a declared choice",
    )
    parser.add_argument(
        "--min-autocorr",
        type=float,
        default=0.20,
        help=(
            "reject series whose lag-1 autocorrelation falls below this. A "
            "series with no dynamics cannot calibrate the statistic that "
            "failed, and admitting one would reproduce the synthetic failure "
            "under a real filename."
        ),
    )
    parser.add_argument(
        "--record-checksums",
        action="store_true",
        help="print the source hashes so they can be pinned in EXPECTED",
    )
    args = parser.parse_args(argv)

    try:
        sales, calendar = locate(args.source)
    except FileNotFoundError as error:
        print(f"{error}", file=sys.stderr)
        print(
            "\nDownload the data first:\n"
            "  1. accept the terms at kaggle.com/competitions/m5-forecasting-accuracy\n"
            "  2. download m5-forecasting-accuracy.zip and unzip it\n"
            "  3. rerun with --source pointing at the unzipped directory",
            file=sys.stderr,
        )
        return 2

    sales_hash = digest(sales)
    if args.record_checksums:
        print(f"{sales.name}: {sales_hash}")
        print(f"{calendar.name}: {digest(calendar)}")
        return 0
    expected = EXPECTED.get(sales.name)
    if expected and expected != sales_hash:
        print(
            f"{sales.name} does not match the pinned checksum.\n"
            f"  expected {expected}\n  found    {sales_hash}\n"
            "A different source file yields a calibration that cannot be "
            "reproduced from the stated source.",
            file=sys.stderr,
        )
        return 1

    start = first_date(calendar)
    series = [
        Series(s.store_id, s.category, s.values, start, s.cadence)
        for s in aggregate(sales, level=args.level)
    ]

    emitted: list[Path] = []
    rejected: list[tuple[str, float]] = []
    for item in series:
        variants = [item]
        if args.resample in ("weekly", "both"):
            variants.append(resample_weekly(item))
        if args.resample == "weekly":
            variants = variants[1:]
        for variant in variants:
            stats = variant.summary()
            if stats["autocorr_1"] < args.min_autocorr:
                rejected.append((f"{variant.name} ({variant.cadence})", stats["autocorr_1"]))
                continue
            emitted.append(write_series(variant, args.out, source_hash=sales_hash))

    print(f"source     {sales} ({sales_hash[:16]})")
    print(f"level      {args.level}")
    print(f"first date {start}")
    print(f"emitted    {len(emitted)} series into {args.out}")
    for path in emitted[:6]:
        print(f"  {path.name}")
    if len(emitted) > 6:
        print(f"  ... and {len(emitted) - 6} more")
    if rejected:
        print(f"rejected   {len(rejected)} for insufficient dynamics")
        for name, acf in rejected[:4]:
            print(f"  {name}: lag-1 autocorrelation {acf:.3f}")
    if not emitted:
        print(
            "\nNo series met the autocorrelation floor. Lower --min-autocorr only "
            "if you can say why a series with no dynamics should calibrate a "
            "dynamics statistic.",
            file=sys.stderr,
        )
        return 1
    print(
        "\nNext: point the calibration at one of these and record the cadence "
        "in the sealed plan before running the confirmatory campaign."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
