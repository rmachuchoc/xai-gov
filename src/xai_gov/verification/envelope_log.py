"""The envelope log: how often each safety property is violated in a run.

The shield's envelope is a *specification-derived operating approximation*, not
a PCTL guarantee: it is synthesized from the properties, but nothing proves the
closed loop satisfies their probability bounds. So the bounds are checked
empirically instead. For every property this module counts, in the concrete
trace of each node, the start periods at which the property's path formula is
violated, and reports the frequency with an exact Clopper–Pearson interval
against the declared bound. It also counts infeasible states: states where the
properties admit no order at all (lower bound above upper bound).

Everything is computed from the decision records, so it is defined identically
for every arm, governed or not, shielded or not.

Path semantics, for a property with horizon h on one node's trace s_0..s_{T-1}:

* persistent (``G<=h bad``): violated at start t if ``bad`` holds at every one of
  t, …, t+h−1. Starts t = 0 … T−h.
* bounded eventually (``F<=h bad``): violated at start t if ``bad`` holds at some
  of t, …, t+h−1. Starts t = 0 … T−h.
* eventually (``F bad``): violated at start t if ``bad`` holds at some s ≥ t
  before the horizon ends; right-censored at the end of the run.

Overlapping starts on one trace are not independent, so the pooled interval of a
single run is descriptive only. The campaign's interval takes the replicate as
the unit (`replicate_trial` and `campaign_envelope`): each replicate contributes
one Bernoulli trial, the violation indicator at one node and one start drawn
uniformly with a seed derived from the replicate's own seed. Replicates are
i.i.d., so the Clopper–Pearson interval over them is exact, and the trial's
success probability is the expected pooled frequency.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from xai_gov.verification.specification import (
    PREDICATES,
    Operator,
    SafetyProperty,
    Specification,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from xai_gov.io.decision_record import DecisionRecord, OperatingState

#: How the envelope is described in every artifact (decision 3 of the revision).
CLAIM = "specification-derived operating approximation"
CONFIDENCE = 0.95


# -- exact binomial interval -----------------------------------------------------
def _log_pmf(k: int, n: int, p: float) -> float:
    return (
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        + k * math.log(p) + (n - k) * math.log1p(-p)
    )


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X ≤ k) for X ~ Binomial(n, p), summed in log space."""
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    logs = [_log_pmf(i, n, p) for i in range(k + 1)]
    top = max(logs)
    return min(math.exp(top) * sum(math.exp(v - top) for v in logs), 1.0)


def _bisect(target: float, function: Any, *, increasing: bool) -> float:
    low, high = 0.0, 1.0
    for _ in range(200):
        middle = (low + high) / 2.0
        value = function(middle)
        if (value < target) == increasing:
            low = middle
        else:
            high = middle
        if high - low < 1e-12:
            break
    return (low + high) / 2.0


def clopper_pearson(
    successes: int, trials: int, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """The exact (Clopper–Pearson) interval for a binomial proportion.

    The lower limit solves P(X ≥ x | p) = α/2 and the upper P(X ≤ x | p) = α/2.
    With no trials the interval is the whole of [0, 1]: nothing was learned.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if trials < 0 or not 0 <= successes <= max(trials, 0):
        raise ValueError("need 0 <= successes <= trials")
    if trials == 0:
        return 0.0, 1.0
    alpha = 1.0 - confidence
    lower = 0.0
    if successes > 0:
        # P(X >= x | p) = 1 - cdf(x - 1) rises with p.
        lower = _bisect(
            alpha / 2.0,
            lambda p: 1.0 - binomial_cdf(successes - 1, trials, p),
            increasing=True,
        )
    upper = 1.0
    if successes < trials:
        # cdf(x | p) falls with p.
        upper = _bisect(
            alpha / 2.0, lambda p: binomial_cdf(successes, trials, p), increasing=False
        )
    return lower, upper


def verdict(ci_low: float, ci_high: float, bound: float) -> str:
    """Where the interval sits relative to the declared bound."""
    if ci_high <= bound:
        return "within_bound"
    if ci_low > bound:
        return "exceeds_bound"
    return "inconclusive"


# -- path formulas on one trace ------------------------------------------------
def violation_by_start(
    trace: Sequence[OperatingState], prop: SafetyProperty
) -> list[bool]:
    """Whether the property is violated from each evaluable start period."""
    bad = [PREDICATES[prop.predicate](state) for state in trace]
    if prop.operator is Operator.EVENTUALLY:
        flags: list[bool] = []
        seen = False
        for value in reversed(bad):
            seen = seen or value
            flags.append(seen)
        return flags[::-1]
    horizon = prop.horizon
    starts = len(bad) - horizon + 1
    if starts <= 0:
        return []
    combine = all if prop.operator is Operator.PERSISTENT else any
    return [combine(bad[t : t + horizon]) for t in range(starts)]


def property_trials(
    trace: Sequence[OperatingState], prop: SafetyProperty
) -> tuple[int, int]:
    """(violations, evaluable starts) of one property on one node's trace."""
    flags = violation_by_start(trace, prop)
    return sum(flags), len(flags)


def _rates(violations: int, trials: int, bound: float | None = None) -> dict[str, Any]:
    low, high = clopper_pearson(violations, trials)
    payload: dict[str, Any] = {
        "violations": violations,
        "trials": trials,
        "frequency": round(violations / trials, 6) if trials else None,
        "ci_low": round(low, 6),
        "ci_high": round(high, 6),
    }
    if bound is not None:
        payload["bound"] = bound
        payload["verdict"] = verdict(low, high, bound) if trials else "no_trials"
    return payload


def traces_by_node(records: Sequence[DecisionRecord]) -> dict[str, list[OperatingState]]:
    """Each node's observed states in period order."""
    traces: dict[str, list[tuple[int, OperatingState]]] = {}
    for record in records:
        traces.setdefault(record.node_id, []).append((record.period, record.state))
    return {
        node: [state for _, state in sorted(entries, key=lambda item: item[0])]
        for node, entries in sorted(traces.items())
    }


def infeasible_counts(
    traces: dict[str, list[OperatingState]],
    specification: Specification,
    *,
    horizon_lookahead: int = 1,
) -> dict[str, tuple[int, int]]:
    """(infeasible states, states) per node, replaying the envelope.

    A fresh shield per node sees the node's states in order, exactly as the
    runtime shield does, and counts the states whose properties conflict.
    """
    from xai_gov.verification.shield import Shield

    counts: dict[str, tuple[int, int]] = {}
    for node, trace in traces.items():
        shield = Shield(specification=specification, horizon_lookahead=horizon_lookahead)
        window = shield.window_for(node)
        for state in trace:
            window.append(state)
            shield.envelope(state, node_id=node)
        counts[node] = (shield.conflicts, len(trace))
    return counts


def envelope_layer(
    records: Sequence[DecisionRecord],
    specification: Specification | None = None,
    *,
    horizon_lookahead: int = 1,
) -> dict[str, Any]:
    """Violation frequencies with Clopper–Pearson intervals, and infeasibility."""
    specification = specification or Specification()
    traces = traces_by_node(records)
    if not traces:
        return {"status": "not_available", "requires": "at least one decision record"}

    properties: dict[str, Any] = {}
    for prop in specification:
        flags = {node: violation_by_start(trace, prop) for node, trace in traces.items()}
        violations = sum(sum(f) for f in flags.values())
        trials = sum(len(f) for f in flags.values())
        properties[prop.property_id] = {
            "formula": prop.formula(),
            # Pooled over starts and nodes: descriptive only, since overlapping
            # starts on one trace are dependent. The campaign's interval uses
            # one trial per replicate drawn from ``by_start``.
            **_rates(violations, trials, prop.bound),
            "per_node": {
                node: _rates(sum(f), len(f), prop.bound) for node, f in flags.items()
            },
            "by_start": {node: [int(v) for v in f] for node, f in flags.items()},
        }

    infeasible = infeasible_counts(traces, specification, horizon_lookahead=horizon_lookahead)
    infeasible_total = sum(v for v, _ in infeasible.values())
    states_total = sum(n for _, n in infeasible.values())
    return {
        "status": "available",
        "claim": CLAIM,
        "confidence": CONFIDENCE,
        "properties": properties,
        "infeasible": {
            **_rates(infeasible_total, states_total),
            "per_node": {node: _rates(v, n) for node, (v, n) in infeasible.items()},
        },
    }


# -- the campaign: one trial per replicate --------------------------------------
def replicate_trial(by_start: dict[str, list[int]], *, seed: int) -> int | None:
    """One Bernoulli trial: a node and a start drawn uniformly, with ``seed``.

    Every evaluable (node, start) pair is equally likely, so the trial's success
    probability equals the run's expected pooled violation frequency.
    """
    import random

    pairs = [(node, t) for node, flags in sorted(by_start.items()) for t in range(len(flags))]
    if not pairs:
        return None
    node, t = random.Random(seed).choice(pairs)
    return int(by_start[node][t])


def campaign_envelope(
    cells: Sequence[tuple[int, dict[str, Any]]],
) -> dict[str, Any]:
    """Per-property intervals over replicates for one arm.

    ``cells`` holds (replicate seed, envelope layer) pairs. The trial seed is
    derived from the replicate seed and the property id, so it does not depend
    on the arm and two arms of one replicate draw the same (node, start) where
    their nodes coincide.
    """
    from xai_gov.core.seeds import derive_seed

    available = [(seed, layer) for seed, layer in cells if layer.get("status") == "available"]
    if not available:
        return {"status": "not_available", "requires": "envelope layers from the cells"}
    property_ids = sorted(available[0][1]["properties"])
    properties: dict[str, Any] = {}
    for property_id in property_ids:
        bound = float(available[0][1]["properties"][property_id]["bound"])
        trials = [
            replicate_trial(
                layer["properties"][property_id]["by_start"],
                seed=derive_seed(seed, f"envelope::{property_id}"),
            )
            for seed, layer in available
        ]
        drawn = [t for t in trials if t is not None]
        pooled_v = sum(layer["properties"][property_id]["violations"] for _, layer in available)
        pooled_n = sum(layer["properties"][property_id]["trials"] for _, layer in available)
        properties[property_id] = {
            "unit": "replicate",
            **_rates(sum(drawn), len(drawn), bound),
            "pooled_descriptive": {
                "violations": pooled_v,
                "trials": pooled_n,
                "frequency": round(pooled_v / pooled_n, 6) if pooled_n else None,
            },
        }
    infeasible_v = sum(layer["infeasible"]["violations"] for _, layer in available)
    infeasible_n = sum(layer["infeasible"]["trials"] for _, layer in available)
    runs_with_infeasible = sum(1 for _, layer in available if layer["infeasible"]["violations"])
    return {
        "status": "available",
        "claim": CLAIM,
        "replicates": len(available),
        "properties": properties,
        "infeasible": {
            "frequency": round(infeasible_v / infeasible_n, 6) if infeasible_n else None,
            "states": infeasible_n,
            "infeasible_states": infeasible_v,
            "runs_with_infeasible_states": _rates(runs_with_infeasible, len(available)),
        },
    }
