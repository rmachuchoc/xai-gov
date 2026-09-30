"""Shared builders for tests.

Test modules must not import from one another: a fixture reached across module
boundaries makes the importing test fail for reasons that live in a file it is
not about. Anything two test modules both need lives here.
"""

from __future__ import annotations

from xai_gov.io.decision_record import OperatingState


def state(**kwargs: float) -> OperatingState:
    """An operating state with sensible defaults, overridable per field."""
    defaults: dict[str, float] = {
        "period": 0,
        "inventory": 40.0,
        "backlog": 0.0,
        "in_transit": 10.0,
        "capacity": 60.0,
        "demand_observed": 20.0,
        "data_delay": 0,
    }
    merged = defaults | kwargs
    return OperatingState(
        period=int(merged["period"]),
        inventory=merged["inventory"],
        backlog=merged["backlog"],
        in_transit=merged["in_transit"],
        capacity=merged["capacity"],
        demand_observed=merged["demand_observed"],
        data_delay=int(merged["data_delay"]),
    )
