"""The M5 adapter: aggregation, provenance, and the autocorrelation floor.

The calibration failure this adapter exists to fix was in autocorrelation, not
in the moments. So the tests that matter are the ones asserting that a series
with no dynamics cannot get through, and that the aggregation level and cadence
travel with the file rather than living in someone's memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from fetch_m5 import (
    Series,
    aggregate,
    resample_weekly,
    write_series,
)

DAYS = 28


def m5_fixture(tmp_path: Path, rows: list[tuple[str, str, list[float]]]) -> Path:
    """A minimal sales file with the columns the real one carries."""
    header = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    header += [f"d_{i + 1}" for i in range(DAYS)]
    lines = [",".join(header)]
    for index, (store, cat, values) in enumerate(rows):
        state = store.split("_")[0]
        dept = f"{cat}_1"
        cells = [f"item_{index}", f"item_{index}", dept, cat, store, state]
        cells = [f"row_{index}", *cells[1:]]
        lines.append(",".join(cells + [f"{v:g}" for v in values]))
    path = tmp_path / "sales_train_evaluation.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def seasonal(n: int, base: float = 40.0) -> list[float]:
    """A series with weekly structure, so its lag-1 autocorrelation is real."""
    return [base + 12.0 * ((i % 7) in (5, 6)) - 4.0 * ((i % 7) == 2) for i in range(n)]


def test_bottom_level_series_sum_to_the_aggregate(tmp_path: Path) -> None:
    """The aggregation is a sum, and a sum that loses units is not one."""
    sales = m5_fixture(
        tmp_path,
        [
            ("CA_1", "FOODS", [1.0] * DAYS),
            ("CA_1", "FOODS", [2.0] * DAYS),
            ("CA_1", "HOBBIES", [5.0] * DAYS),
        ],
    )
    series = {s.name: s for s in aggregate(sales, level="store_category")}
    assert series["ca_1_foods"].values == [3.0] * DAYS
    assert series["ca_1_hobbies"].values == [5.0] * DAYS


def test_empty_cells_count_as_zero_not_as_missing(tmp_path: Path) -> None:
    """M5 leaves no gaps, but a blank must not abort a 30,000-row stream."""
    header = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    header += [f"d_{i + 1}" for i in range(3)]
    path = tmp_path / "sales_train_evaluation.csv"
    path.write_text(
        ",".join(header) + "\nrow_0,item_0,FOODS_1,FOODS,CA_1,CA,4,,6\n",
        encoding="utf-8",
    )
    series = aggregate(path, level="store_category")[0]
    assert series.values == [4.0, 0.0, 6.0]


def test_an_unknown_level_is_refused_by_name(tmp_path: Path) -> None:
    sales = m5_fixture(tmp_path, [("CA_1", "FOODS", [1.0] * DAYS)])
    with pytest.raises(ValueError, match="unknown level"):
        aggregate(sales, level="item")


def test_every_declared_level_works(tmp_path: Path) -> None:
    sales = m5_fixture(
        tmp_path,
        [("CA_1", "FOODS", [1.0] * DAYS), ("TX_1", "FOODS", [3.0] * DAYS)],
    )
    for level, expected in (
        ("store_category", 2),
        ("store_department", 2),
        ("state_category", 2),
    ):
        assert len(aggregate(sales, level=level)) == expected, level


def test_a_seasonal_series_has_real_autocorrelation() -> None:
    """The statistic the whole exercise exists to calibrate."""
    series = Series("CA_1", "FOODS", seasonal(200), "2011-01-29", "daily")
    assert series.summary()["autocorr_1"] > 0.2


def test_a_white_noise_series_has_none() -> None:
    """The case the floor rejects: a series that would reproduce the synthetic
    failure under a real filename."""
    import random

    rng = random.Random(3)
    flat = [40.0 + rng.gauss(0, 8) for _ in range(300)]
    series = Series("CA_1", "FOODS", flat, "2011-01-29", "daily")
    assert abs(series.summary()["autocorr_1"]) < 0.2


def test_the_zero_share_exposes_intermittency() -> None:
    """Bottom-level M5 series are intermittent, and an intermittent series
    fails the twin's assumptions for a reason the study did not set out to
    test. The share is reported so that choice stays visible."""
    sparse = [0.0] * 90 + [5.0] * 10
    series = Series("CA_1", "HOBBIES", sparse, "2011-01-29", "daily")
    assert series.summary()["zero_share"] == 0.9


def test_weekly_resampling_sums_whole_weeks_only(tmp_path: Path) -> None:
    """A trailing partial week would enter the series as a low outlier and
    read as a demand collapse."""
    daily = Series("CA_1", "FOODS", [1.0] * 17, "2011-01-29", "daily")
    weekly = resample_weekly(daily)
    assert weekly.values == [7.0, 7.0]
    assert weekly.cadence == "weekly"


def test_the_written_file_carries_its_provenance(tmp_path: Path) -> None:
    """Aggregation level and cadence are study parameters. A CSV that does not
    state them leaves the reader unable to tell which decision produced it."""
    series = Series("CA_1", "FOODS", seasonal(60), "2011-01-29", "daily")
    path = write_series(series, tmp_path, source_hash="abc123")
    text = path.read_text(encoding="utf-8")
    assert "source_sha256: abc123" in text
    assert "aggregation: CA_1 x FOODS" in text
    assert "cadence: daily" in text
    assert "lag-1 autocorrelation" in text
    assert "Redistribution is not permitted" in text


def test_the_written_file_is_loadable_by_the_project(tmp_path: Path) -> None:
    """The adapter's output must satisfy the loader the calibration uses, or
    the two halves meet only at run time."""
    from xai_gov.analysis.data import load_series

    series = Series("CA_1", "FOODS", seasonal(80), "2011-01-29", "daily")
    path = write_series(series, tmp_path, source_hash="abc123")
    loaded = load_series(path)
    assert len(loaded.values) == 80
    assert loaded.values[0] == pytest.approx(40.0)


def test_the_cadence_is_in_the_filename(tmp_path: Path) -> None:
    """Two cadences of one series are two study configurations, so they must
    not be able to overwrite each other."""
    base = Series("CA_1", "FOODS", seasonal(70), "2011-01-29", "daily")
    daily = write_series(base, tmp_path, source_hash="h")
    weekly = write_series(resample_weekly(base), tmp_path, source_hash="h")
    assert daily != weekly
    assert "daily" in daily.name and "weekly" in weekly.name
