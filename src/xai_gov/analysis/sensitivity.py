"""Global sensitivity analysis.

The protocol's economics carry declared parameters: the cost of an
intervention, the loss from a disruption, the governance budget, the weights of
the reward. Every one of them is a choice, and a conclusion that depends on a
choice nobody can justify is a calibration artifact wearing the clothes of a
finding.

Sobol indices separate the two. They decompose the variance of an outcome
across its inputs, answering how much of the spread in a result each parameter
is responsible for — including through its interactions with the others. That
last part is what one-at-a-time sensitivity misses: varying each parameter
alone around a base point explores a cross through the space and reports
nothing about the corners, which is where interactions live.

Two indices are computed and the gap between them is informative. The
first-order index S1 is the variance a parameter explains on its own; the total
index ST includes everything it explains in combination with others. A large
ST with a small S1 says the parameter matters only alongside something else,
which is a different kind of finding and calls for a different response.

Saltelli sampling is used because plain Monte Carlo would need far more model
evaluations for the same precision, and each evaluation here is a full
simulation run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class Parameter:
    """One input whose influence is being measured."""

    name: str
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low >= self.high:
            raise ValueError(f"{self.name}: low must be strictly below high")

    def scale(self, unit: float) -> float:
        return self.low + unit * (self.high - self.low)

    def to_payload(self) -> dict[str, Any]:
        return {"name": self.name, "low": self.low, "high": self.high}


@dataclass(frozen=True, slots=True)
class SobolIndices:
    """First-order and total-effect indices for one parameter."""

    name: str
    first_order: float
    total_effect: float

    @property
    def interaction_share(self) -> float:
        """How much of the parameter's influence is only in combination.

        The gap between total and first-order effect. A parameter that matters
        alone and a parameter that matters only alongside another require
        different responses, and reporting one number would hide which is which.
        """
        return max(self.total_effect - self.first_order, 0.0)

    @property
    def influential(self) -> bool:
        """Whether the parameter meaningfully drives the outcome.

        The 0.05 convention: below it, the parameter explains less variance
        than the estimator's own noise at practical sample sizes, so declaring
        influence would be reading the sampler.
        """
        return self.total_effect >= 0.05

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "first_order": round(self.first_order, 6),
            "total_effect": round(self.total_effect, 6),
            "interaction_share": round(self.interaction_share, 6),
            "influential": self.influential,
        }


@dataclass(frozen=True, slots=True)
class SensitivityReport:
    """The outcome of one sensitivity analysis."""

    indices: tuple[SobolIndices, ...]
    evaluations: int
    base_samples: int
    output_variance: float

    @property
    def dominant(self) -> str | None:
        if not self.indices:
            return None
        return max(self.indices, key=lambda i: i.total_effect).name

    @property
    def robust(self) -> bool:
        """Whether the conclusion survives the parameter choices.

        A parameter dominates when it does two things at once: carries a
        majority of the output variance, *and* carries substantially more than
        an equal share of it. Both conditions are needed. A bare 0.5 cutoff
        would call a two-parameter model with equal contributors
        non-robust — each necessarily explains about half — which says nothing
        about fragility and everything about the parameter count.

        A result driven overwhelmingly by one declared cost is a finding about
        that cost, not about governance.
        """
        if not self.indices:
            return True
        largest = max(index.total_effect for index in self.indices)
        equal_share = 1.0 / len(self.indices)
        return not (largest >= 0.5 and largest >= 1.5 * equal_share)

    def to_payload(self) -> dict[str, Any]:
        return {
            "method": "saltelli_sobol",
            "evaluations": self.evaluations,
            "base_samples": self.base_samples,
            "output_variance": round(self.output_variance, 9),
            "dominant_parameter": self.dominant,
            "robust_to_parameter_choice": self.robust,
            "indices": [index.to_payload() for index in self.indices],
        }


def saltelli_sample(
    parameters: Sequence[Parameter], *, base_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Draw the A, B and AB matrices of the Saltelli scheme."""
    if base_samples < 8:
        raise ValueError(
            "base_samples must be at least 8; the estimator is meaningless on fewer"
        )
    if not parameters:
        raise ValueError("sensitivity analysis needs at least one parameter")

    rng = np.random.default_rng(seed)
    k = len(parameters)
    unit_a = rng.random((base_samples, k))
    unit_b = rng.random((base_samples, k))

    def scaled(matrix: np.ndarray) -> np.ndarray:
        out = np.empty_like(matrix)
        for column, parameter in enumerate(parameters):
            out[:, column] = parameter.low + matrix[:, column] * (
                parameter.high - parameter.low
            )
        return out

    a = scaled(unit_a)
    b = scaled(unit_b)
    # AB_i is A with column i taken from B: the swap is what isolates the
    # contribution of parameter i.
    ab = []
    for column in range(k):
        swapped = unit_a.copy()
        swapped[:, column] = unit_b[:, column]
        ab.append(scaled(swapped))
    return a, b, ab


def sobol_analysis(
    model: Callable[[dict[str, float]], float],
    parameters: Sequence[Parameter],
    *,
    base_samples: int = 64,
    seed: int = 0,
) -> SensitivityReport:
    """Estimate Sobol indices for ``model`` over ``parameters``."""
    a, b, ab_matrices = saltelli_sample(parameters, base_samples=base_samples, seed=seed)
    names = [parameter.name for parameter in parameters]

    def evaluate(matrix: np.ndarray) -> np.ndarray:
        return np.array(
            [model(dict(zip(names, row, strict=True))) for row in matrix], dtype=float
        )

    ya = evaluate(a)
    yb = evaluate(b)
    yab = [evaluate(matrix) for matrix in ab_matrices]

    variance = float(np.var(np.concatenate([ya, yb]), ddof=1))
    evaluations = len(ya) + len(yb) + sum(len(y) for y in yab)

    if variance <= 1e-12:
        # A constant output has no variance to apportion. Reporting zeros is
        # correct and reporting anything else would be dividing by noise.
        return SensitivityReport(
            indices=tuple(
                SobolIndices(name=name, first_order=0.0, total_effect=0.0)
                for name in names
            ),
            evaluations=evaluations,
            base_samples=base_samples,
            output_variance=variance,
        )

    indices: list[SobolIndices] = []
    for column, name in enumerate(names):
        y_ab = yab[column]
        # Saltelli estimators. Clipped into [0, 1]: the estimators are unbiased
        # but not bounded, and a negative index at small sample sizes is
        # estimator noise rather than a negative contribution to variance.
        first = float(np.mean(yb * (y_ab - ya)) / variance)
        total = float(np.mean((ya - y_ab) ** 2) / (2.0 * variance))
        indices.append(
            SobolIndices(
                name=name,
                first_order=min(max(first, 0.0), 1.0),
                total_effect=min(max(total, 0.0), 1.0),
            )
        )

    return SensitivityReport(
        indices=tuple(indices),
        evaluations=evaluations,
        base_samples=base_samples,
        output_variance=variance,
    )


@dataclass(slots=True)
class ParameterSpace:
    """The declared ranges the protocol's economics are varied over."""

    parameters: list[Parameter] = field(
        default_factory=lambda: [
            Parameter(name="intervention_cost", low=0.25, high=4.0),
            Parameter(name="disruption_loss", low=5.0, high=40.0),
            Parameter(name="allowance", low=4.0, high=40.0),
            Parameter(name="escalation_belief", low=0.4, high=0.8),
        ]
    )

    def names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    def to_payload(self) -> dict[str, Any]:
        return {"parameters": [p.to_payload() for p in self.parameters]}


def build_parameter_space(config: dict[str, Any] | None) -> ParameterSpace:
    """Construct the parameter space from configuration."""
    if not config:
        return ParameterSpace()
    declared = config.get("parameters")
    if not isinstance(declared, list) or not declared:
        raise ValueError("sensitivity 'parameters' must be a non-empty list")
    parameters: list[Parameter] = []
    for index, entry in enumerate(declared):
        if not isinstance(entry, dict):
            raise ValueError(f"parameter {index} must be a mapping")
        try:
            parameters.append(
                Parameter(
                    name=str(entry["name"]),
                    low=float(entry["low"]),
                    high=float(entry["high"]),
                )
            )
        except KeyError as error:
            raise ValueError(f"parameter {index}: missing key {error}") from error
    return ParameterSpace(parameters=parameters)
