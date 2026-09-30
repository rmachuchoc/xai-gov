"""Parameter sweeps: break-even cost and the severity threshold.

Two questions the coauthor reviews asked, and they share machinery because
both are the same shape: re-run the campaign at each level of one declared
parameter and find where a sign changes.

**Why this cannot be a rescale.** The intervention cost enters the *derived*
threshold, so lowering it changes when the agent intervenes and not merely
what it pays. Dividing an already-completed arm's total cost by a new unit
price would answer a different question — one where the policy is frozen and
only the invoice moves. The sweep therefore executes a campaign per level,
which is why it is a module and not an arithmetic helper.

**Why severity is the more interesting of the two.** Theorem 2 predicts the
sign: the optimal intervention threshold decreases in shock severity, so
substitution should become worthwhile past some severity and the crossing
should move monotonically. That makes the severity sweep a test of the theorem
against data rather than an exploration, and `monotone` reports whether the
prediction held. A non-monotone curve means the simulator violates one of the
theorem's assumptions, which is a finding that needs locating rather than a
result to report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from xai_gov.core.logging import get_logger
from xai_gov.core.settings import Settings

_LOG = get_logger("analysis.sweeps")


@dataclass(frozen=True, slots=True)
class SweepPoint:
    """One level of the swept parameter and what the campaign produced there."""

    level: float
    label: str
    treatment_value: float | None
    control_value: float | None
    included_cells: int
    #: Every run this level was computed from, by arm. Recorded so a published
    #: bundle can back each point of the curve with the decision logs that
    #: produced it; a sweep whose report names no runs cannot be audited, only
    #: re-executed.
    runs: tuple[tuple[str, tuple[str, ...]], ...] = ()

    @property
    def difference(self) -> float | None:
        """Treatment minus control, or None when either side is missing.

        None rather than zero: a level whose runs did not produce the indicator
        has no difference, and a zero there would be read as no effect.
        """
        if self.treatment_value is None or self.control_value is None:
            return None
        return self.treatment_value - self.control_value

    def to_payload(self) -> dict[str, Any]:
        difference = self.difference
        return {
            "level": self.level,
            "label": self.label,
            "treatment": (
                None if self.treatment_value is None else round(self.treatment_value, 6)
            ),
            "control": (
                None if self.control_value is None else round(self.control_value, 6)
            ),
            "difference": None if difference is None else round(difference, 6),
            "included_cells": self.included_cells,
            "runs": [
                {"arm": arm, "run_name": name}
                for arm, names in self.runs
                for name in names
            ],
        }


@dataclass(frozen=True, slots=True)
class SweepResult:
    """A swept parameter, its curve, and where the curve crosses zero."""

    parameter: str
    indicator: str
    treatment: str
    control: str
    points: tuple[SweepPoint, ...]
    replicates: int

    @property
    def usable(self) -> tuple[SweepPoint, ...]:
        return tuple(p for p in self.points if p.difference is not None)

    @property
    def crossing(self) -> float | None:
        """The parameter level at which the difference changes sign.

        Estimated by linear interpolation between the two bracketing levels
        rather than reported as the nearest swept level: a crossing quoted at
        grid resolution overstates precision.

        None means either that no crossing exists in the swept range or that
        there was nothing to cross. Those are different claims, so callers must
        consult ``has_data`` before reading this as a finding.
        """
        usable = self.usable
        for left, right in pairwise(usable):
            a, b = left.difference, right.difference
            if a is None or b is None or (a > 0) == (b > 0):
                continue
            if abs(b - a) < 1e-12:
                return right.level
            return left.level + (right.level - left.level) * (-a) / (b - a)
        return None

    @property
    def has_data(self) -> bool:
        """Whether any level produced a comparable pair.

        Checked before any conclusion is drawn. A sweep whose every level came
        back empty produced no evidence, and the first version of this module
        reported "the sign is constant over everything tested" from zero
        observations — a confident claim built on nothing, which is worse than
        a missing number because it reads as a result.
        """
        return len(self.usable) >= 2

    @property
    def monotone(self) -> bool | None:
        """Whether the difference moves in one direction across the sweep.

        The property Theorem 2 predicts for the severity sweep. None when
        fewer than three usable points exist, since two points are monotone by
        construction and reporting True there would be reporting the sample
        size.
        """
        usable = self.usable
        if len(usable) < 3:
            return None
        deltas = [
            (right.difference or 0.0) - (left.difference or 0.0)
            for left, right in pairwise(usable)
        ]
        positive = [d for d in deltas if d > 1e-9]
        negative = [d for d in deltas if d < -1e-9]
        return not (positive and negative)

    def to_payload(self) -> dict[str, Any]:
        crossing = self.crossing
        return {
            "parameter": self.parameter,
            "indicator": self.indicator,
            "comparison": f"{self.treatment} minus {self.control}",
            "replicates_per_level": self.replicates,
            "levels": [p.to_payload() for p in self.points],
            "usable_levels": len(self.usable),
            "has_data": self.has_data,
            "crossing": None if crossing is None else round(crossing, 6),
            "crossing_method": "linear interpolation between bracketing levels",
            "monotone": self.monotone,
            "interpretation": _interpret(
                self.parameter, crossing, self.monotone, has_data=self.has_data,
                usable=len(self.usable), levels=len(self.points),
            ),
        }


def _interpret(
    parameter: str,
    crossing: float | None,
    monotone: bool | None,
    *,
    has_data: bool,
    usable: int,
    levels: int,
) -> str:
    """State what the curve licenses, in the terms the question was asked in."""
    if not has_data:
        return (
            f"only {usable} of {levels} levels produced a comparable pair, so this "
            "sweep yielded no evidence about " + parameter + ". The indicator was "
            "unavailable in the runs — check that it is a per-run KPI and not a "
            "paired quantity computed in the analysis layer"
        )
    if crossing is None:
        return (
            f"the difference does not change sign anywhere in the swept range of "
            f"{parameter}; the finding is that the sign is constant over everything "
            "tested, not that a crossing exists outside it"
        )
    base = f"the sign changes at {parameter} = {crossing:.4g}"
    if monotone is False:
        return (
            f"{base}, but the curve is not monotone, so the crossing is not unique "
            "and a single threshold should not be quoted from it"
        )
    return f"{base}, and the curve is monotone, so the crossing is a usable threshold"


@dataclass(slots=True)
class ParameterSweep:
    """Re-runs a paired comparison across levels of one declared parameter."""

    settings: Settings
    project_root: Path
    treatment: Path
    control: Path
    #: The ungoverned arm. Total value creation is a difference against it, so
    #: without it the sweep can only compare two governed arms on an operational
    #: indicator — which answers whether the shield helps, not whether it pays.
    #: The reviews asked the second question.
    baseline: Path | None = None
    indicator: str = "operational.service_level"
    #: What one point of the operational indicator is worth, in the same units
    #: the oversight cost is denominated in. Required for the economic mode and
    #: declared rather than assumed: the indicator is a fraction in [0, 1] and
    #: the cost is a total in the tens, so subtracting one from the other
    #: compares a proportion to a currency amount. The first economic run did
    #: exactly that and produced a net value of -161.9 for a quantity whose
    #: operational term cannot leave [-1, 1] — a crossing computed from it is
    #: the point where an arbitrary scale ratio happens to flip sign.
    service_value: float = 100.0
    #: Per-run KPI, not a paired quantity. Total value creation is computed in
    #: the analysis layer from a treatment/baseline pair, so it does not exist
    #: in any single run's KPI tree; reading it here returned None at every
    #: level and the sweep reported a conclusion from zero observations.
    replicates: int = 10
    base_seed: int = 20260703
    workers: int = 1
    #: Results per (arm, override, level), independent of which indicator is
    #: read. Keying on the indicator as well meant the economic mode ran every
    #: arm twice -- once for service, once for cost -- producing duplicate run
    #: directories whose numbers agreed only because the simulator is
    #: deterministic. A bundle built from them could not say which copy backed
    #: which figure.
    _results: dict[str, tuple[Any, ...]] = field(default_factory=dict, repr=False)

    def run(
        self,
        *,
        parameter: str,
        levels: list[float],
        override: str,
        labels: list[str] | None = None,
    ) -> SweepResult:
        """Execute the comparison at every level.

        ``override`` is a dotted path into the experiment configuration, so the
        sweep edits the same declarative surface a human would rather than
        reaching into constructed objects. A sweep that mutated an agent in
        place would not be reproducible from configuration.
        """
        if not levels:
            raise ValueError("a sweep needs at least one level")
        if self.replicates < 2:
            raise ValueError(
                "a sweep needs at least two replicates per level; one run per level "
                "cannot separate the parameter from the seed"
            )

        names = labels or [f"{parameter}={level:g}" for level in levels]
        if len(names) != len(levels):
            raise ValueError("labels must match levels one to one")

        points: list[SweepPoint] = []
        for level, label in zip(levels, names, strict=True):
            _LOG.info("sweep level", extra={"parameter": parameter, "level": level})
            treatment_value, treatment_n = self._evaluate(self.treatment, override, level)
            control_value, control_n = self._evaluate(self.control, override, level)
            if self.baseline is not None:
                # Economic mode. Each arm's value is its own operational gain
                # over the ungoverned baseline minus what oversight cost it, so
                # the swept difference is a difference of net values rather than
                # of service levels. This is what makes a crossing an economic
                # break-even rather than an operational one.
                base_value, base_n = self._evaluate(self.baseline, override, level)
                treatment_cost, _ = self._evaluate(
                    self.treatment, override, level, indicator=_COST
                )
                control_cost, _ = self._evaluate(
                    self.control, override, level, indicator=_COST
                )
                treatment_value = _net_value(
                    treatment_value, base_value, treatment_cost, self.service_value
                )
                control_value = _net_value(
                    control_value, base_value, control_cost, self.service_value
                )
                treatment_n += base_n
            arms = [self.treatment, self.control]
            if self.baseline is not None:
                arms.append(self.baseline)
            points.append(
                SweepPoint(
                    level=float(level),
                    label=label,
                    treatment_value=treatment_value,
                    control_value=control_value,
                    included_cells=treatment_n + control_n,
                    runs=tuple(
                        (arm.stem, self._run_names(arm, override, level)) for arm in arms
                    ),
                )
            )

        return SweepResult(
            parameter=parameter,
            indicator=self.indicator,
            treatment=self.treatment.stem,
            control=self.control.stem,
            points=tuple(points),
            replicates=self.replicates,
        )

    def _evaluate(
        self,
        experiment: Path,
        override: str,
        level: float,
        *,
        indicator: str | None = None,
    ) -> tuple[float | None, int]:
        """Mean indicator value for one arm at one parameter level."""
        wanted = indicator or self.indicator
        results = self._run_level(experiment, override, level)
        values = [
            float(value)
            for result in results
            if result.included and (value := result.indicator(wanted)) is not None
        ]
        return (sum(values) / len(values) if values else None), len(values)

    def _run_names(self, experiment: Path, override: str, level: float) -> tuple[str, ...]:
        """The runs behind one arm at one level, in replicate order."""
        return tuple(
            str(result.run_name) for result in self._run_level(experiment, override, level)
        )

    def _run_level(self, experiment: Path, override: str, level: float) -> tuple[Any, ...]:
        """Execute one arm at one level, once, however many indicators are read."""
        from xai_gov.analysis.campaign import CampaignRunner, plan_campaign
        from xai_gov.orchestration.runner import plan_experiment

        key = f"{experiment.stem}|{override}|{level:g}"
        cached = self._results.get(key)
        if cached is not None:
            return cached

        base = plan_experiment(experiment, settings=self.settings)
        patched = _with_override(base, override, level)

        cells = plan_campaign(
            arms={experiment.stem: experiment},
            replicates=self.replicates,
            base_seed=self.base_seed,
        )
        runner = CampaignRunner(
            settings=self.settings,
            project_root=self.project_root,
            workers=self.workers,
            # Resumption is disabled inside a sweep: every level shares the
            # arm's configuration hash except for the swept value, and a cache
            # keyed on the file rather than the patch would serve one level's
            # runs to another. That is precisely the defect that silently
            # falsified three earlier ablations.
            resume=False,
            plan_override=patched,
        )
        results = tuple(runner.run(cells))
        self._results[key] = results
        return results


#: Per-run oversight cost, used to turn an operational gain into a net value.
_COST = "governance.total_intervention_cost"


def _net_value(
    arm: float | None,
    baseline: float | None,
    cost: float | None,
    service_value: float,
) -> float | None:
    """An arm's operational gain, priced, net of what oversight cost.

    ``service_value`` converts a fraction of service level into the units the
    cost is denominated in. Without it the two terms are incommensurable and
    their difference is governed by whichever happens to be numerically larger,
    not by the economics.

    None when any term is missing rather than treating an absent cost as free:
    a run whose cost the KPI layer did not report is not a run that spent
    nothing, and the difference between those two readings is the whole
    question being swept.
    """
    if arm is None or baseline is None or cost is None:
        return None
    if service_value <= 0.0:
        raise ValueError(
            "service_value must be positive; it is the price that makes the "
            "operational gain and the oversight cost comparable"
        )
    return (arm - baseline) * service_value - cost


def _with_override(plan: Any, path: str, value: float) -> Any:
    """Return the plan with one dotted configuration path replaced."""
    from copy import deepcopy
    from dataclasses import replace

    config = deepcopy(getattr(plan, "config", {}) or {})
    node: Any = config
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(node.get(part), dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value
    return replace(plan, config=config)


#: The two sweeps the reviews asked for, declared rather than passed as
#: literals so the manuscript and the code name the same levels.
BREAK_EVEN_COSTS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
SEVERITY_LEVELS: tuple[float, ...] = (1.5, 3.0, 6.0, 10.0)
