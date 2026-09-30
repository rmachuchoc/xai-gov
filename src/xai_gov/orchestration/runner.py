"""Experiment composition and execution.

The runner is the only place that turns a configuration into a run. It
composes the twin from declared factors, executes it, computes the KPI
layers, and writes every artifact through `RunWriter` so the traceability
contract is exercised on every path — including failure, where the run is
aborted without a completed manifest rather than left ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xai_gov.core.hashing import content_hash
from xai_gov.core.logging import get_logger
from xai_gov.core.seeds import SeedBundle
from xai_gov.core.settings import Settings
from xai_gov.governance.registry import build_governance
from xai_gov.io.writers import RunWriter
from xai_gov.io.yaml_loader import load_config
from xai_gov.kpis.layers import compute_kpis
from xai_gov.policies.registry import build_policy
from xai_gov.simulation.demand import build_demand
from xai_gov.simulation.disruptions import build_schedule
from xai_gov.simulation.engine import DigitalTwin, SimulationResult
from xai_gov.simulation.network import build_network

_LOG = get_logger("orchestration.runner")


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """A validated, fully resolved experiment."""

    name: str
    periods: int
    master_seed: int
    config: dict[str, Any]

    @property
    def config_hash(self) -> str:
        return content_hash(self.config)


def _require_mapping(config: dict[str, Any], key: str) -> dict[str, Any]:
    node = config.get(key, {})
    if not isinstance(node, dict):
        raise ValueError(f"experiment section {key!r} must be a mapping")
    return node


def plan_experiment(path: Path, *, settings: Settings) -> ExperimentPlan:
    """Load and validate an experiment configuration."""
    config = load_config(path, root=settings.paths.root)
    experiment = _require_mapping(config, "experiment")
    name = str(experiment.get("name") or path.stem)
    periods = int(experiment.get("periods", 30))
    seed = experiment.get("master_seed")
    if seed is None:
        raise ValueError(
            f"experiment {name!r} declares no master_seed; a run without a declared "
            "seed is not reproducible and is refused"
        )
    return ExperimentPlan(
        name=name, periods=periods, master_seed=int(seed), config=config
    )


def build_twin(plan: ExperimentPlan) -> DigitalTwin:
    """Assemble the twin from a plan."""
    config = plan.config
    network = build_network(_require_mapping(config, "network"))
    demand = build_demand(_require_mapping(config, "demand"))
    policy = build_policy(_require_mapping(config, "policy"))

    # The conformal block lives at the root of the experiment configuration so
    # that a detector ablation can be composed by adding one include, without
    # rewriting the governance block it belongs to.
    governance_config = dict(_require_mapping(config, "governance"))
    for hoisted in ("conformal", "explainability", "verification", "oversight"):
        if hoisted in config:
            governance_config.setdefault(hoisted, _require_mapping(config, hoisted))
    # The supervisor's seed is derived from the run's master seed, so a
    # rerun reproduces the human's judgements too. A supervisor seeded from
    # the clock would make governed runs irreproducible while every other
    # component stayed deterministic — the worst kind of partial determinism,
    # because the artifacts would still look reproducible.
    governance = build_governance(governance_config, seed=plan.master_seed)

    raw_disruptions = config.get("disruptions", [])
    if raw_disruptions and not isinstance(raw_disruptions, list):
        raise ValueError("experiment section 'disruptions' must be a list")
    schedule = build_schedule(raw_disruptions)

    return DigitalTwin(
        network=network,
        demand=demand,
        policy=policy,
        governance=governance,
        seeds=SeedBundle(master_seed=plan.master_seed),
        periods=plan.periods,
        schedule=schedule,
        data_delay=int(_require_mapping(config, "data_quality").get("delay", 0)),
    )


def summarize(plan: ExperimentPlan, twin: DigitalTwin, result: SimulationResult) -> dict[str, Any]:
    """The end-of-run summary written next to the KPIs."""
    return {
        "experiment": plan.name,
        "periods": plan.periods,
        "master_seed": plan.master_seed,
        "config_hash": plan.config_hash,
        "network": twin.network.to_payload(),
        "demand": twin.demand.to_payload(),
        "policy": twin.policy.to_payload(),
        "governance": twin.governance.to_payload(),
        "disruptions": twin.schedule.to_payload(),
        "decisions": len(result.records),
        "final_states": result.final_states,
        "periods_traced": [trace.to_payload() for trace in result.traces],
    }


def run_experiment(
    path: Path, *, settings: Settings, plan: ExperimentPlan | None = None
) -> dict[str, Any]:
    """Execute one experiment and return its manifest.

    ``plan`` lets a campaign substitute the seed while leaving the arm's file
    untouched. Editing the file to change a seed would make a replicate
    indistinguishable from a different experiment in the artifacts, and the two
    are not the same thing.
    """
    plan = plan if plan is not None else plan_experiment(path, settings=settings)
    twin = build_twin(plan)

    # Shock periods come from both the demand regime and the disruption
    # schedule: a structural change with no disruption is still a shock, and
    # the systemic-risk layer must not miss it.
    shock_periods = tuple(
        sorted(set(twin.demand.shift_periods) | set(twin.schedule.shock_periods))
    )

    _LOG.info(
        "experiment planned",
        extra={
            "experiment": plan.name,
            "periods": plan.periods,
            "seed": plan.master_seed,
            "policy": twin.policy.policy_id,
            "governance": twin.governance.agent_id,
            "config_hash": plan.config_hash[:12],
        },
    )

    with RunWriter(
        runs_root=settings.runs_root,
        experiment=plan.name,
        config=plan.config,
        master_seed=plan.master_seed,
        source_root=settings.paths.root / "src",
        write_events_csv=settings.runtime.write_events_csv,
        write_decision_log=settings.runtime.write_decision_log,
    ) as writer:
        for disruption in twin.schedule.disruptions:
            writer.record_event("disruption_scheduled", disruption.to_payload())

        result = twin.run()
        for record in result.records:
            writer.record_decision(record)

        writer.write_summary(summarize(plan, twin, result))
        writer.write_kpis(compute_kpis(result, shock_periods=shock_periods))
        return writer.close()
