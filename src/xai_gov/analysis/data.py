"""Real demand series for calibration.

The calibration phase was validated against a series the simulator itself
produced, which proves the machinery works and establishes nothing about
external validity. This module is the seam where a real series enters.

Two design choices matter.

The loader is **format-explicit rather than clever**. It reads a single column
of numbers from a CSV and says exactly which column it took, because a
calibration run whose input series nobody can identify is not reproducible in
the sense the rest of the project means it.

And the synthetic fallback is **labelled as synthetic in the run record**. A
study that calibrates against simulator output and reports a sim-to-real gap
has computed a sim-to-sim gap, and the artifact must say which one it holds.
The fallback exists so the pipeline runs without the data file present, not so
the distinction can be skipped.

Obtaining the data: the M5 competition series (Walmart hierarchical daily unit
sales) is the reference this project's protocol names. Place a CSV under
``data/demand/`` with one row per period and the unit count in a named column;
``M5_NOTES`` below records what to extract and why.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: What to extract from M5, and the reasons, so the choice is auditable.
#:
#: The twin models a single node's demand over 60 periods. M5 is daily unit
#: sales per item per store, so a *single item-store series* is the matching
#: granularity — aggregating across items would produce a smooth series whose
#: dispersion is an artifact of aggregation rather than a property of demand,
#: and the project's whole question concerns behaviour under volatility.
#:
#: Series with long zero runs (intermittent demand) are a different modelling
#: problem than the one the twin represents, so the loader reports the zero
#: fraction and the caller decides rather than silently accepting one.
M5_NOTES = (
    "one item-store series, not an aggregate: aggregation smooths the "
    "dispersion the study is about. Check the reported zero fraction — a "
    "highly intermittent series is a different modelling problem than this "
    "twin represents."
)


@dataclass(frozen=True, slots=True)
class DemandSeries:
    """An observed demand series and its provenance."""

    values: tuple[float, ...]
    source: str
    synthetic: bool
    column: str | None = None
    path: str | None = None

    def __post_init__(self) -> None:
        if len(self.values) < 20:
            raise ValueError(
                f"a calibration series needs at least 20 periods, got "
                f"{len(self.values)}; the summary statistics are not "
                f"estimable on fewer"
            )

    @property
    def zero_fraction(self) -> float:
        """Share of periods with no demand.

        Reported rather than acted upon. A series that is mostly zeros is
        intermittent demand, which this twin does not model, and the caller
        should know before the posterior is interpreted.
        """
        return sum(1 for value in self.values if value <= 0.0) / len(self.values)

    @property
    def intermittent(self) -> bool:
        return self.zero_fraction > 0.25

    def to_payload(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "synthetic": self.synthetic,
            "periods": len(self.values),
            "column": self.column,
            "path": self.path,
            "zero_fraction": round(self.zero_fraction, 6),
            "intermittent": self.intermittent,
            # Stated in the artifact, not inferred by a reader: a gap computed
            # against simulator output is not a sim-to-real gap.
            "gap_interpretation": (
                "sim-to-sim: the reference series came from the simulator, so "
                "no external validity is established"
                if self.synthetic
                else "sim-to-real: the reference series is observed data"
            ),
        }


def load_series(
    path: Path, *, column: str | None = None, periods: int | None = None
) -> DemandSeries:
    """Read a demand series from a CSV.

    ``column`` names the column to read; when omitted the loader takes the
    last numeric column and records which one, because guessing silently is
    how a calibration ends up fitted to a row index.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"no demand series at {path}. Place a CSV there with one row per "
            f"period, or run with the synthetic fallback and accept that the "
            f"reported gap is sim-to-sim. {M5_NOTES}"
        )

    # Leading `#` lines are skipped rather than parsed. The M5 adapter writes
    # the aggregation level, the cadence and the source hash into the file's
    # own header, because those are study parameters and a CSV that does not
    # carry them leaves a reader unable to tell which decision produced it.
    # A DictReader handed those lines would take the first one as the header.
    with path.open(encoding="utf-8", newline="") as handle:
        lines = [line for line in handle if not line.lstrip().startswith("#")]
    rows = list(csv.DictReader(lines))
    if not rows:
        raise ValueError(f"{path} holds no rows")

    fields = list(rows[0])
    if column is not None:
        if column not in fields:
            raise ValueError(
                f"{path} has no column {column!r}; available: {', '.join(fields)}"
            )
        chosen = column
    else:
        numeric = [
            field
            for field in fields
            if _is_numeric(rows[0].get(field))
        ]
        if not numeric:
            raise ValueError(f"{path} has no numeric column to read")
        chosen = numeric[-1]

    values: list[float] = []
    for index, row in enumerate(rows):
        raw = row.get(chosen)
        if raw is None or raw.strip() == "":
            continue
        try:
            values.append(float(raw))
        except ValueError as error:
            raise ValueError(
                f"{path} row {index + 2}, column {chosen!r}: {raw!r} is not a number"
            ) from error

    if periods is not None:
        values = values[:periods]

    return DemandSeries(
        values=tuple(values),
        source=f"observed series from {path.name}, column {chosen!r}",
        synthetic=False,
        column=chosen,
        path=str(path),
    )


def synthetic_series(*, base: float = 22.0, noise: float = 4.5, periods: int = 120,
                     seed: int = 20260703) -> DemandSeries:
    """A simulator-generated series, labelled as such.

    Present so the pipeline runs without the data file. Every artifact it
    produces carries ``synthetic: true`` and the gap is labelled sim-to-sim,
    because a study that calibrates against its own output and reports a
    sim-to-real gap has measured nothing about the world.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    values = tuple(float(max(0.0, rng.normal(base, noise))) for _ in range(periods))
    return DemandSeries(
        values=values,
        source=f"synthetic reference (base {base}, noise {noise}, seed {seed})",
        synthetic=True,
    )


def resolve_series(
    root: Path, *, filename: str = "demand.csv", column: str | None = None,
    periods: int | None = None,
) -> DemandSeries:
    """Load the real series if present, else the labelled synthetic one."""
    candidate = root / "data" / "demand" / filename
    if candidate.exists():
        return load_series(candidate, column=column, periods=periods)
    return synthetic_series()


def _is_numeric(value: str | None) -> bool:
    if value is None or value.strip() == "":
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True
