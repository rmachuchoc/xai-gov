"""The envelope log: violation frequencies against the bounds, and infeasibility."""

from __future__ import annotations

import pytest

from tests.factories import state
from xai_gov.verification.envelope_log import (
    CLAIM,
    clopper_pearson,
    envelope_layer,
    infeasible_counts,
    property_trials,
    verdict,
)
from xai_gov.verification.specification import Operator, SafetyProperty, Specification


# -- Clopper–Pearson -------------------------------------------------------------
def test_clopper_pearson_matches_the_published_value_for_five_of_ten() -> None:
    low, high = clopper_pearson(5, 10)
    assert low == pytest.approx(0.187086, abs=1e-5)
    assert high == pytest.approx(0.812914, abs=1e-5)


@pytest.mark.parametrize("n", [1, 10, 60, 500])
def test_clopper_pearson_has_closed_forms_at_the_edges(n: int) -> None:
    assert clopper_pearson(0, n) == pytest.approx((0.0, 1.0 - 0.025 ** (1.0 / n)), abs=1e-8)
    assert clopper_pearson(n, n) == pytest.approx((0.025 ** (1.0 / n), 1.0), abs=1e-8)


def test_clopper_pearson_contains_the_estimate_and_narrows_with_n() -> None:
    small = clopper_pearson(3, 30)
    large = clopper_pearson(300, 3000)
    assert small[0] < 0.1 < small[1]
    assert large[0] < 0.1 < large[1]
    assert large[1] - large[0] < small[1] - small[0]


def test_no_trials_means_the_whole_unit_interval() -> None:
    assert clopper_pearson(0, 0) == (0.0, 1.0)


def test_impossible_counts_are_refused() -> None:
    with pytest.raises(ValueError):
        clopper_pearson(4, 3)


@pytest.mark.parametrize(
    ("low", "high", "expected"),
    [(0.0, 0.04, "within_bound"), (0.06, 0.2, "exceeds_bound"), (0.01, 0.09, "inconclusive")],
)
def test_the_verdict_compares_the_interval_with_the_bound(
    low: float, high: float, expected: str
) -> None:
    assert verdict(low, high, 0.05) == expected


# -- path semantics ----------------------------------------------------------------
def _trace(*stocked_out: bool) -> list:  # type: ignore[type-arg]
    return [state(period=t, inventory=0.0 if bad else 50.0) for t, bad in enumerate(stocked_out)]


def _prop(operator: Operator, horizon: int) -> SafetyProperty:
    return SafetyProperty(
        property_id="P", operator=operator, predicate="stockout", bound=0.1, horizon=horizon
    )


def test_persistent_counts_starts_where_the_bad_state_holds_throughout() -> None:
    trace = _trace(True, True, True, False, True)
    # Starts 0..3 with h=2: [T,T] [T,T] [T,F] [F,T] -> two violations.
    assert property_trials(trace, _prop(Operator.PERSISTENT, 2)) == (2, 4)


def test_bounded_eventually_counts_starts_where_the_bad_state_occurs_at_all() -> None:
    trace = _trace(False, False, False, True, False, False)
    # Starts 0..3 with h=3: windows ending at 2, 3, 4, 5 -> the last three contain t=3.
    assert property_trials(trace, _prop(Operator.BOUNDED_EVENTUALLY, 3)) == (3, 4)


def test_eventually_is_right_censored_at_the_end_of_the_run() -> None:
    trace = _trace(False, True, False, False)
    prop = SafetyProperty(property_id="P", operator=Operator.EVENTUALLY, predicate="stockout",
                          bound=0.1)
    assert property_trials(trace, prop) == (2, 4)


def test_a_trace_shorter_than_the_horizon_has_no_trials() -> None:
    assert property_trials(_trace(True), _prop(Operator.PERSISTENT, 3)) == (0, 0)


# -- infeasibility ----------------------------------------------------------------
def test_a_stockout_with_no_capacity_is_an_infeasible_state() -> None:
    """Stockout demands an order; exhausted capacity caps it at zero."""
    starved = state(inventory=0.0, in_transit=0.0, capacity=0.0, demand_observed=10.0)
    healthy = state(inventory=80.0, capacity=120.0)
    counts = infeasible_counts({"n": [healthy, starved, healthy]}, Specification(
        properties=(
            SafetyProperty(property_id="S", operator=Operator.PERSISTENT,
                           predicate="stockout", bound=0.02, horizon=2),
            SafetyProperty(property_id="C", operator=Operator.PERSISTENT,
                           predicate="capacity_exhausted", bound=0.02, horizon=1),
        )
    ))
    assert counts == {"n": (1, 3)}


# -- the layer ----------------------------------------------------------------------
def test_the_layer_names_its_claim_and_reports_every_property() -> None:
    from xai_gov.core.seeds import SeedBundle
    from xai_gov.governance.agent import NoGovernanceAgent
    from xai_gov.policies.families import HeuristicSQPolicy
    from xai_gov.simulation.demand import StableDemand
    from xai_gov.simulation.engine import DigitalTwin
    from xai_gov.simulation.network import EXTERNAL_SOURCE, NetworkSpec, NodeSpec

    network = NetworkSpec(nodes=(
        NodeSpec(node_id="r", echelon=0, supplier_id="p", initial_inventory=20.0,
                 capacity=120.0, reorder_point=25.0, order_quantity=35.0, lead_time_mean=2.0),
        NodeSpec(node_id="p", echelon=1, supplier_id=EXTERNAL_SOURCE, initial_inventory=60.0,
                 capacity=300.0, reorder_point=60.0, order_quantity=90.0, lead_time_mean=3.0),
    ))
    result = DigitalTwin(
        network=network, demand=StableDemand(name="s", mean=12.0, dispersion=0.3),
        policy=HeuristicSQPolicy(), governance=NoGovernanceAgent(),
        seeds=SeedBundle(master_seed=3), periods=40,
    ).run()
    layer = envelope_layer(result.records)

    assert layer["status"] == "available"
    assert layer["claim"] == CLAIM == "specification-derived operating approximation"
    assert sorted(layer["properties"]) == ["SAFE-1", "SAFE-2", "SAFE-3"]
    for entry in layer["properties"].values():
        assert set(entry["per_node"]) == {"p", "r"}
        assert entry["trials"] == sum(node["trials"] for node in entry["per_node"].values())
        assert 0.0 <= entry["ci_low"] <= (entry["frequency"] or 0.0) <= entry["ci_high"] <= 1.0
        assert entry["verdict"] in ("within_bound", "exceeds_bound", "inconclusive")
    assert layer["infeasible"]["trials"] == 80


def test_no_records_means_not_available() -> None:
    assert envelope_layer([])["status"] == "not_available"
