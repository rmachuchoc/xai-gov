"""The digital twin: network, state, demand, disruptions, engine.

`DigitalTwin` and its result types are resolved lazily (PEP 562). Importing
them eagerly here would mean that any module touching
`xai_gov.simulation.network` also pulls in the engine, and the engine sits
above the policy and governance layers. The lazy re-export keeps the public
name `xai_gov.simulation.DigitalTwin` working without inverting the layering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from xai_gov.simulation.demand import DemandProcess, available_regimes, build_demand
from xai_gov.simulation.disruptions import Disruption, DisruptionKind, DisruptionSchedule
from xai_gov.simulation.network import EXTERNAL_SOURCE, NetworkSpec, NodeSpec, build_network
from xai_gov.simulation.state import NodeState

if TYPE_CHECKING:
    from xai_gov.simulation.engine import DigitalTwin, PeriodTrace, SimulationResult

_LAZY = {"DigitalTwin", "PeriodTrace", "SimulationResult"}

__all__ = [
    "EXTERNAL_SOURCE",
    "DemandProcess",
    "DigitalTwin",
    "Disruption",
    "DisruptionKind",
    "DisruptionSchedule",
    "NetworkSpec",
    "NodeSpec",
    "NodeState",
    "PeriodTrace",
    "SimulationResult",
    "available_regimes",
    "build_demand",
    "build_network",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        from xai_gov.simulation import engine

        return getattr(engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
