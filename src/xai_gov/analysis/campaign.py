"""Campaign execution: factorial cells, replicates, and honest exclusion.

A campaign is a set of experiment arms run at many seeds. Two design decisions
carry most of the weight.

**Replicates are seeds, and the seeds are shared across arms.** Cell *k* of
every arm uses the same seed, so the arms see the same demand path and the same
disruptions. The comparison is then paired, and the variance the design already
removed is not paid for again in the analysis.

**Exclusions are declared before the run and applied mechanically.** The
criteria live in the preregistration; this module applies them and records every
excluded run with its reason. A campaign that drops runs without saying which
ones is not reporting a result, it is reporting a selection.

The grid is not enumerated exhaustively. The full factorial of the protocol's
eleven factors is intractable by construction, so a campaign names the arms it
compares and the analysis states what it therefore cannot say.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xai_gov.core.logging import get_logger
from xai_gov.core.seeds import derive_seed
from xai_gov.core.settings import Settings

_LOG = get_logger("analysis.campaign")


@dataclass(frozen=True, slots=True)
class CampaignCell:
    """One arm at one replicate."""

    arm: str
    experiment: Path
    replicate: int
    seed: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "experiment": self.experiment.name,
            "replicate": self.replicate,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class CellResult:
    """The outcome of one cell, included or not."""

    cell: CampaignCell
    run_name: str
    kpis: dict[str, Any]
    included: bool
    exclusion_reason: str | None = None

    def indicator(self, path: str) -> float | None:
        """Read a dotted indicator path out of the KPI tree.

        Returns None when the indicator is absent or the layer reported itself
        unavailable. A missing indicator must not silently become zero: that is
        the failure the KPI layers were built to prevent, and reintroducing it
        here would undo them.
        """
        node: Any = self.kpis
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return None
            node = node[part]
        if isinstance(node, bool) or not isinstance(node, int | float):
            return None
        return float(node)

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.cell.to_payload(),
            "run_name": self.run_name,
            "included": self.included,
            "exclusion_reason": self.exclusion_reason,
            # Carried per cell so the diagnostic can tell a single anomalous
            # arm apart from a detector that under-flags everywhere.
            "under_flagging": bool(
                (self.kpis.get("guarantees") or {}).get("under_flagging")
            ),
        }


def plan_campaign(
    *,
    arms: dict[str, Path],
    replicates: int,
    base_seed: int,
) -> tuple[CampaignCell, ...]:
    """Build the cell list, pairing arms on shared seeds.

    Cell *k* of every arm draws the same seed. Independent seeds per arm would
    make each comparison an unpaired one and throw away the variance reduction
    the shared twin was built to provide.
    """
    if replicates < 1:
        raise ValueError("a campaign needs at least one replicate")
    if not arms:
        raise ValueError("a campaign needs at least one arm")

    cells: list[CampaignCell] = []
    for replicate in range(replicates):
        seed = derive_seed(base_seed, f"replicate::{replicate}")
        for arm, experiment in sorted(arms.items()):
            cells.append(
                CampaignCell(
                    arm=arm, experiment=experiment, replicate=replicate, seed=seed
                )
            )
    return tuple(cells)


#: Fingerprint of the exclusion logic. Bumped whenever a criterion is added or
#: changed, because a resumed campaign must not mix cells judged under
#: different criteria: the reused cells' artifacts predate the new key, the
#: lookup returns None, and the criterion silently does not apply to them.
EXCLUSION_CRITERIA_VERSION = 3


def apply_exclusions(
    manifest: dict[str, Any], kpis: dict[str, Any]
) -> tuple[bool, str | None]:
    """Apply the preregistered exclusion criteria to one run.

    Each criterion here corresponds to one line of the sealed plan. They are
    mechanical on purpose: a criterion applied by judgement after seeing the
    outcome is not an exclusion criterion, it is a choice.
    """
    if manifest.get("status") != "completed":
        return False, "run did not complete its declared horizon"
    if not manifest.get("chain_verified", False):
        return False, "decision log failed hash-chain verification"

    guarantees = kpis.get("guarantees", {})
    if isinstance(guarantees, dict) and guarantees.get("saturated") is True:
        # A saturated detector is not measuring the environment, so its
        # governance behaviour says nothing about the treatment.
        return False, "conformal detector reported saturation"
    # Under-flagging is deliberately *not* an exclusion criterion, though it is
    # just as serious. Exclusion removes anomalous cells from an analysis that
    # otherwise proceeds; when a condition holds for every arm it is not an
    # anomaly but a property of the pipeline, and excluding on it destroys the
    # study rather than protecting it. The first campaign to apply it that way
    # lost 400 of 440 cells and produced no comparison at all. It is reported as
    # a blocking diagnostic instead, which says the run must not be reported
    # while leaving its numbers available to diagnose.
    return True, None


@dataclass(slots=True)
class CampaignRunner:
    """Executes a campaign and collects its cells."""

    settings: Settings
    project_root: Path
    workers: int = 1
    resume: bool = True
    #: A pre-resolved plan that replaces the one loaded from the arm's file.
    #: Parameter sweeps use it to vary one declared value without editing the
    #: file, so the swept campaign stays reproducible from configuration while
    #: the file on disk keeps describing the unswept arm.
    plan_override: Any = None
    results: list[CellResult] = field(default_factory=list)
    reused: int = 0
    stale: int = 0

    def run(self, cells: tuple[CampaignCell, ...]) -> tuple[CellResult, ...]:
        """Execute every cell, recording inclusions and exclusions alike.

        Cells are independent by construction — each carries its own seed and
        its own run directory, and the directory allocator claims names
        atomically — so they parallelize without coordination. What does not
        parallelize is the ordering of run directory names, so a cell is
        identified by (arm, replicate) rather than by the order it finished in.
        """
        if self.workers < 1:
            raise ValueError("workers must be at least 1")

        collected: list[CellResult] = []
        pending: list[CampaignCell] = []
        for cell in cells:
            cached = self._completed(cell) if self.resume else None
            if cached is not None:
                collected.append(cached)
                self.reused += 1
            else:
                pending.append(cell)

        if self.reused or self.stale:
            _LOG.info(
                "resuming campaign",
                extra={
                    "reused": self.reused,
                    "stale": self.stale,
                    "pending": len(pending),
                },
            )

        if self.workers == 1 or len(pending) < 2 or self.plan_override is not None:
            # A plan override cannot cross a process boundary as configuration,
            # so an overridden sweep runs in-process rather than silently
            # executing the unswept arm in each worker.
            for index, cell in enumerate(pending):
                collected.append(self._execute(cell))
                self._progress(index + 1, len(pending))
        else:
            collected.extend(self._execute_parallel(pending))

        # Sorted so the collection order does not depend on which worker
        # finished first: an analysis whose paired differences depend on
        # scheduling is not reproducible.
        self.results = sorted(
            collected, key=lambda r: (r.cell.replicate, r.cell.arm)
        )
        # Aggregated, not one line per cell. A campaign that logs every
        # exclusion individually buries its own phase summaries under hundreds
        # of identical lines, and the counts are what a reader needs.
        excluded = [r for r in self.results if not r.included]
        if excluded:
            by_reason: dict[str, int] = {}
            for result in excluded:
                key = str(result.exclusion_reason)
                by_reason[key] = by_reason.get(key, 0) + 1
            _LOG.warning(
                "cells excluded",
                extra={"count": len(excluded), "by_reason": dict(sorted(by_reason.items()))},
            )
        return tuple(self.results)

    def _progress(self, done: int, total: int) -> None:
        if total and (done % 10 == 0 or done == total):
            _LOG.info("campaign progress", extra={"completed": done, "total": total})

    def _execute(self, cell: CampaignCell) -> CellResult:
        """Run one cell in this process."""
        from xai_gov.io.writers import KPIS_NAME, read_json
        from xai_gov.orchestration.runner import plan_experiment, run_experiment

        plan = self.plan_override or plan_experiment(cell.experiment, settings=self.settings)
        # The campaign's seed overrides the experiment's own, so a cell is
        # identified by (arm, replicate) rather than by whatever seed the arm
        # file happened to declare.
        seeded = _reseed(plan, cell.seed)
        manifest = run_experiment(cell.experiment, settings=self.settings, plan=seeded)
        directory = self.settings.runs_root / manifest["run_name"]
        kpis = read_json(directory / KPIS_NAME)
        included, reason = apply_exclusions(manifest, kpis)
        self._write_receipt(cell, manifest["run_name"], str(manifest.get("config_hash", "")))
        return CellResult(
            cell=cell,
            run_name=manifest["run_name"],
            kpis=kpis,
            included=included,
            exclusion_reason=reason,
        )

    def _execute_parallel(self, cells: list[CampaignCell]) -> list[CellResult]:
        """Run cells across processes.

        Processes rather than threads because the work is pure-Python compute,
        which threads cannot overlap. Each worker rebuilds its own agent from
        configuration, so nothing stateful crosses the boundary.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed

        collected: list[CellResult] = []
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            futures = {
                pool.submit(
                    _run_cell_worker,
                    experiment=str(cell.experiment),
                    seed=cell.seed,
                    arm=cell.arm,
                    replicate=cell.replicate,
                    project_root=str(self.project_root),
                    outputs_root=str(self.settings.outputs_root),
                ): cell
                for cell in cells
            }
            for index, future in enumerate(as_completed(futures)):
                cell = futures[future]
                run_name, kpis, included, reason, config_hash = future.result()
                self._write_receipt(cell, run_name, config_hash)
                collected.append(
                    CellResult(
                        cell=cell,
                        run_name=run_name,
                        kpis=kpis,
                        included=included,
                        exclusion_reason=reason,
                    )
                )
                self._progress(index + 1, len(cells))
        return collected

    # -- resumability -----------------------------------------------------
    def _receipt_path(self, cell: CampaignCell) -> Path:
        return (
            self.settings.outputs_root
            / "campaigns"
            / "_cells"
            / f"{cell.arm}__{cell.replicate:04d}__{cell.seed}.json"
        )

    def _write_receipt(self, cell: CampaignCell, run_name: str, config_hash: str) -> None:
        """Record that a cell completed, so a rerun can skip it.

        The receipt names the run rather than duplicating its KPIs: the run
        directory is the source of truth, and a second copy of the numbers
        would be a second thing that can disagree.

        It also records the resolved configuration hash and the exclusion
        criteria version. Both are needed and for different reasons. The
        criteria version catches a change in how a cell is *judged*; the config
        hash catches a change in what the cell *is*. Without the second, adding
        a parameter to the detector silently reuses runs whose detector never
        had it — which turned an ablation into a comparison of two identical
        configurations and produced a paired difference of exactly zero across
        forty replicates.
        """
        from xai_gov.core.hashing import canonical_json

        path = self._receipt_path(cell)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            canonical_json(
                {
                    **cell.to_payload(),
                    "run_name": run_name,
                    "config_hash": config_hash,
                    "criteria_version": EXCLUSION_CRITERIA_VERSION,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    def _completed(self, cell: CampaignCell) -> CellResult | None:
        """A previously completed cell, if its artifacts are still verifiable.

        A receipt alone is not enough. The run directory must still exist and
        its decision log must still verify — otherwise a resumed campaign would
        inherit a result nobody can check, which is worse than re-running it.
        """
        import json

        from xai_gov.io.hashchain import ChainVerificationError, verify_file
        from xai_gov.io.writers import DECISION_LOG_NAME, KPIS_NAME, read_json
        from xai_gov.orchestration.runner import plan_experiment

        receipt = self._receipt_path(cell)
        if not receipt.exists():
            return None
        try:
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            # A cell judged under older criteria is re-run rather than reused.
            # Reusing it would let a new criterion apply to some cells and not
            # others, and the report would look uniform while being a mixture.
            if int(payload.get("criteria_version", 0)) != EXCLUSION_CRITERIA_VERSION:
                self.stale += 1
                return None
            # And a cell whose configuration has changed is a different cell.
            # The resolved hash covers every parameter the run reads, so a new
            # detector field invalidates the runs that predate it instead of
            # being silently compared against them.
            plan = plan_experiment(cell.experiment, settings=self.settings)
            current_hash = getattr(plan, "config_hash", None) or _config_hash(plan)
            if str(payload.get("config_hash", "")) != str(current_hash):
                self.stale += 1
                return None
            run_name = str(payload["run_name"])
            directory = self.settings.runs_root / run_name
            verify_file(directory / DECISION_LOG_NAME)
            kpis = read_json(directory / KPIS_NAME)
        except (OSError, KeyError, ValueError, ChainVerificationError):
            return None

        included, reason = apply_exclusions(
            {"status": "completed", "chain_verified": True}, kpis
        )
        return CellResult(
            cell=cell,
            run_name=run_name,
            kpis=kpis,
            included=included,
            exclusion_reason=reason,
        )

    def paired(self, treatment: str, control: str, indicator: str) -> list[float]:
        """Paired differences on one indicator, treatment minus control.

        A replicate contributes only when *both* arms produced a usable value.
        Dropping the pair rather than the single observation is what keeps the
        comparison paired; substituting a mean for the missing side would
        manufacture agreement between the arms.
        """
        by_replicate: dict[int, dict[str, float]] = {}
        for result in self.results:
            if not result.included or result.cell.arm not in (treatment, control):
                continue
            value = result.indicator(indicator)
            if value is None:
                continue
            by_replicate.setdefault(result.cell.replicate, {})[result.cell.arm] = value

        return [
            values[treatment] - values[control]
            for _, values in sorted(by_replicate.items())
            if treatment in values and control in values
        ]

    def summary(self) -> dict[str, Any]:
        included = [r for r in self.results if r.included]
        excluded = [r for r in self.results if not r.included]
        by_arm: dict[str, int] = {}
        for result in included:
            by_arm[result.cell.arm] = by_arm.get(result.cell.arm, 0) + 1
        return {
            "cells": len(self.results),
            "included": len(included),
            "excluded": len(excluded),
            "reused_from_previous_run": self.reused,
            "re_run_for_stale_criteria": self.stale,
            "criteria_version": EXCLUSION_CRITERIA_VERSION,
            "workers": self.workers,
            "included_by_arm": dict(sorted(by_arm.items())),
            # Every dropped run is named with its reason. A campaign that drops
            # runs silently is reporting a selection, not a result.
            "exclusions": [
                {
                    "arm": r.cell.arm,
                    "replicate": r.cell.replicate,
                    "run_name": r.run_name,
                    "reason": r.exclusion_reason,
                }
                for r in excluded
            ],
        }


def _config_hash(plan: Any) -> str:
    """The resolved configuration's fingerprint.

    Read from the plan when it carries one, computed otherwise, so a campaign
    can tell whether a cached run was produced by the configuration it is about
    to compare against.
    """
    from xai_gov.core.hashing import content_hash

    config = getattr(plan, "config", None)
    return content_hash(config) if config is not None else ""


def _reseed(plan: Any, seed: int) -> Any:
    """Return the plan with the campaign's seed substituted."""
    from dataclasses import replace

    return replace(plan, master_seed=seed)


def _run_cell_worker(
    *,
    experiment: str,
    seed: int,
    arm: str,
    replicate: int,
    project_root: str,
    outputs_root: str,
) -> tuple[str, dict[str, Any], bool, str | None, str]:
    """Run one cell in a worker process.

    Arguments are primitives rather than objects. A worker that received a
    constructed agent would be sharing state that the parent had already
    stepped, and the run would no longer be reproducible from its seed.
    """
    from dataclasses import replace as dataclass_replace
    from pathlib import Path as _Path

    from xai_gov.core.logging import configure_logging
    from xai_gov.core.paths import Paths
    from xai_gov.core.settings import load_settings
    from xai_gov.io.writers import KPIS_NAME, read_json
    from xai_gov.orchestration.runner import plan_experiment, run_experiment

    root = _Path(project_root)
    settings = dataclass_replace(
        load_settings(Paths(root=root)), outputs_root=_Path(outputs_root)
    )
    # Workers log at WARNING: one process per core each narrating every period
    # makes the console unreadable and the log file useless.
    configure_logging(
        level="WARNING", console=False, jsonl=True, log_dir=settings.paths.logs,
        filename="campaign_workers.log", force=True,
    )
    del arm, replicate

    path = settings.paths.resolve(experiment)
    plan = _reseed(plan_experiment(path, settings=settings), seed)
    manifest = run_experiment(path, settings=settings, plan=plan)
    kpis = read_json(settings.runs_root / manifest["run_name"] / KPIS_NAME)
    included, reason = apply_exclusions(manifest, kpis)
    return manifest["run_name"], kpis, included, reason, str(manifest.get("config_hash", ""))
