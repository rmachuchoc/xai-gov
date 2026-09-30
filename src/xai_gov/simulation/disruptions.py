"""Disruption schedule.

Disruptions are declared, not sampled: a campaign must be able to state
that severe shift occurred at period 12 and compare cells that differ only
in that. Each disruption names the mechanism it perturbs, so its effect is
inspectable rather than folded into a generic "shock" multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DisruptionKind(StrEnum):
    CAPACITY_LOSS = "capacity_loss"
    LEAD_TIME_SPIKE = "lead_time_spike"
    SUPPLY_HALT = "supply_halt"
    DEMAND_SURGE = "demand_surge"


@dataclass(frozen=True, slots=True)
class Disruption:
    """One scheduled perturbation of the twin."""

    kind: DisruptionKind
    start: int
    duration: int
    magnitude: float
    node_id: str | None = None

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("disruption start must be non-negative")
        if self.duration < 1:
            raise ValueError("disruption duration must be at least one period")
        if self.magnitude < 0.0:
            raise ValueError("disruption magnitude must be non-negative")

    @property
    def end(self) -> int:
        """Exclusive end period."""
        return self.start + self.duration

    def active_at(self, period: int, node_id: str) -> bool:
        if not self.start <= period < self.end:
            return False
        return self.node_id is None or self.node_id == node_id

    def to_payload(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "start": self.start,
            "duration": self.duration,
            "magnitude": self.magnitude,
            "node_id": self.node_id,
        }


@dataclass(frozen=True)
class DisruptionSchedule:
    """The disruptions of a scenario, queried per period and node."""

    disruptions: tuple[Disruption, ...] = ()

    def active(self, period: int, node_id: str) -> tuple[Disruption, ...]:
        return tuple(d for d in self.disruptions if d.active_at(period, node_id))

    def capacity_multiplier(self, period: int, node_id: str) -> float:
        """Product of capacity losses in force. 1.0 when unperturbed."""
        multiplier = 1.0
        for disruption in self.active(period, node_id):
            if disruption.kind is DisruptionKind.CAPACITY_LOSS:
                multiplier *= max(1.0 - disruption.magnitude, 0.0)
        return multiplier

    def lead_time_multiplier(self, period: int, node_id: str) -> float:
        multiplier = 1.0
        for disruption in self.active(period, node_id):
            if disruption.kind is DisruptionKind.LEAD_TIME_SPIKE:
                multiplier *= 1.0 + disruption.magnitude
        return multiplier

    def demand_multiplier(self, period: int, node_id: str) -> float:
        multiplier = 1.0
        for disruption in self.active(period, node_id):
            if disruption.kind is DisruptionKind.DEMAND_SURGE:
                multiplier *= 1.0 + disruption.magnitude
        return multiplier

    def supply_halted(self, period: int, node_id: str) -> bool:
        return any(
            d.kind is DisruptionKind.SUPPLY_HALT for d in self.active(period, node_id)
        )

    @property
    def shock_periods(self) -> tuple[int, ...]:
        return tuple(sorted({d.start for d in self.disruptions}))

    def to_payload(self) -> dict[str, Any]:
        return {"disruptions": [d.to_payload() for d in self.disruptions]}


def build_schedule(config: list[dict[str, Any]] | None) -> DisruptionSchedule:
    """Construct a schedule from a configuration list."""
    if not config:
        return DisruptionSchedule()
    accepted = set(Disruption.__dataclass_fields__)
    built: list[Disruption] = []
    for index, entry in enumerate(config):
        if not isinstance(entry, dict):
            raise ValueError(f"disruption at position {index} is not a mapping")
        unknown = sorted(set(entry) - accepted)
        if unknown:
            raise ValueError(f"disruption at position {index}: unknown keys {unknown}")
        kind = entry.get("kind")
        if kind is None:
            raise ValueError(f"disruption at position {index} declares no 'kind'")
        built.append(
            Disruption(
                kind=DisruptionKind(str(kind)),
                start=int(entry["start"]),
                duration=int(entry.get("duration", 1)),
                magnitude=float(entry.get("magnitude", 0.0)),
                node_id=entry.get("node_id"),
            )
        )
    return DisruptionSchedule(disruptions=tuple(built))
