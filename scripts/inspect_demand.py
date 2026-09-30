#!/usr/bin/env python
"""Summarize the demand series before the cadence is sealed.

The cadence decision determines the autocorrelation, which is the statistic
the calibration is being fixed to reproduce. So it has to be made on evidence
about the series and *before* any calibration runs --- deciding afterwards
would be choosing the cadence that made the gap look best.

    python scripts/inspect_demand.py
    python scripts/inspect_demand.py --weekly   # the same series, resampled

What to look for. The twin's demand process is a level plus noise with no
seasonal term, so it can reproduce a series whose autocorrelation is slow
drift and cannot reproduce one whose autocorrelation is a cycle. The two are
distinguished by shape rather than by size: a cycle makes the correlation
*rise* at the seasonal lag, while drift decays monotonically.

The seasonal lag depends on the cadence --- one week is lag 7 in daily data and
lag 1 in weekly data --- so the comparison lag moves with the resampling. An
earlier version of this script checked lag 7 in both cadences and read seven
weeks of persistence as a weekly cycle, which is the same class of error the
manuscript catalogs: a quantity computed correctly on the wrong scale.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from xai_gov.analysis.data import load_series


def autocorr(values: list[float], lag: int) -> float:
    n = len(values)
    if n <= lag + 1:
        return 0.0
    mean = statistics.fmean(values)
    centred = [v - mean for v in values]
    denom = sum(c * c for c in centred)
    if denom <= 1e-12:
        return 0.0
    return sum(centred[i] * centred[i + lag] for i in range(n - lag)) / denom


def weekly(values: list[float]) -> list[float]:
    return [sum(values[i : i + 7]) for i in range(0, len(values) - 6, 7)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=Path("data/demand"))
    parser.add_argument(
        "--weekly", action="store_true", help="resample to weeks before summarizing"
    )
    args = parser.parse_args(argv)

    paths = sorted(p for p in args.dir.glob("*.csv") if p.suffix == ".csv")
    if not paths:
        print(f"no series in {args.dir}; run `make m5 M5_SOURCE=...` first")
        return 1

    rows: list[tuple[str, int, float, float, float, float, float]] = []
    # The seasonal lag depends on the cadence: one week is lag 7 in daily data
    # and lag 1 in weekly data, so the annual cycle at lag 52 is the only
    # seasonal structure a weekly series can show. Checking lag 7 in both
    # cadences reads seven weeks of slow persistence as a weekly cycle --- a
    # quantity computed correctly on the wrong scale.
    seasonal_lag = 52 if args.weekly else 7
    long_lag = 104 if args.weekly else 30
    for path in paths:
        values = list(load_series(path).values)
        if args.weekly:
            values = weekly(values)
        rows.append(
            (
                path.stem.replace("m5_", "").replace("_daily", ""),
                len(values),
                statistics.fmean(values),
                statistics.stdev(values) / statistics.fmean(values),
                autocorr(values, 1),
                autocorr(values, seasonal_lag),
                autocorr(values, long_lag),
            )
        )

    cadence = "weekly" if args.weekly else "daily"
    unit = "weeks" if args.weekly else "days"
    print(f"{len(rows)} series at {cadence} cadence from {args.dir}\n")
    print(
        f"{'series':<22}{'n':>6}{'mean':>10}{'cv':>8}"
        f"{'acf1':>8}{f'acf{seasonal_lag}':>8}{f'acf{long_lag}':>8}"
    )
    print(f"{'':22}{'':6}{'':10}{'':8}{'1 ' + unit[:1]:>8}{'season':>8}{'long':>8}")
    print("-" * 70)
    for name, n, mean, cv, a1, a7, a30 in rows:
        print(f"{name:<22}{n:>6}{mean:>10.1f}{cv:>8.3f}{a1:>8.3f}{a7:>8.3f}{a30:>8.3f}")

    def spread(index: int) -> tuple[float, float, float]:
        column = sorted(r[index] for r in rows)
        return column[0], statistics.median(column), column[-1]

    print("-" * 70)
    for label, index in (
        ("cv", 3), ("acf1", 4), (f"acf{seasonal_lag}", 5), (f"acf{long_lag}", 6)
    ):
        low, mid, high = spread(index)
        print(f"{label:<8} min {low:>7.3f}   median {mid:>7.3f}   max {high:>7.3f}")

    _, median_acf1, _ = spread(4)
    _, median_seasonal, _ = spread(5)
    _, median_long, _ = spread(6)
    print()
    # A cycle shows as a *bump* at the seasonal lag: the correlation rises
    # above its lag-1 value instead of decaying. Slow drift shows as monotone
    # decay. The twin's level-plus-noise process tracks drift through its EWMA
    # level and has no term that can reproduce a bump.
    if median_seasonal > median_acf1:
        print(
            f"Autocorrelation rises from {median_acf1:.3f} at lag 1 to "
            f"{median_seasonal:.3f} at lag {seasonal_lag}, which is a cycle rather\n"
            "than decay. The twin has no seasonal term, so calibrating at this\n"
            "cadence asks it to reproduce structure it cannot represent."
        )
    elif median_acf1 > median_seasonal > median_long:
        print(
            f"Autocorrelation decays monotonically, {median_acf1:.3f} at lag 1 to "
            f"{median_seasonal:.3f} at lag {seasonal_lag}\nto {median_long:.3f} at lag "
            f"{long_lag}. That is slow drift rather than a cycle --- structure the\n"
            "twin's level-plus-noise process can track through its adaptive level.\n"
            "This cadence is calibratable."
        )
    else:
        print(
            f"The pattern is neither a clean cycle nor a clean decay "
            f"({median_acf1:.3f}, {median_seasonal:.3f}, {median_long:.3f}).\n"
            "Inspect individual series before sealing: an aggregate verdict over a\n"
            "mixture of shapes is not a verdict."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
