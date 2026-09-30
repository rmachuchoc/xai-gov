"""Policies propose; they never execute, and never exceed capacity."""

from __future__ import annotations

import numpy as np
import pytest

from xai_gov.io.decision_record import ActionKind, OperatingState
from xai_gov.policies.base import PolicyProposal
from xai_gov.policies.families import HeuristicSQPolicy, OpaqueDROPolicy, RandomBoundedPolicy
from xai_gov.policies.registry import available_policies, build_policy
from xai_gov.simulation.network import EXTERNAL_SOURCE, NodeSpec

SPEC = NodeSpec(
    node_id="wh",
    echelon=0,
    supplier_id=EXTERNAL_SOURCE,
    initial_inventory=50.0,
    capacity=100.0,
    reorder_point=30.0,
    order_quantity=40.0,
    lead_time_mean=2.0,
)


def state(**kwargs: float) -> OperatingState:
    defaults: dict[str, float] = {
        "period": 1, "inventory": 50.0, "backlog": 0.0, "in_transit": 0.0,
        "capacity": 100.0, "demand_observed": 10.0,
    }
    merged = defaults | kwargs
    return OperatingState(
        period=int(merged["period"]),
        inventory=merged["inventory"],
        backlog=merged["backlog"],
        in_transit=merged["in_transit"],
        capacity=merged["capacity"],
        demand_observed=merged["demand_observed"],
    )


def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_hold_must_carry_zero_quantity() -> None:
    with pytest.raises(ValueError, match="hold must carry zero"):
        PolicyProposal(action=ActionKind.HOLD, quantity=5.0, features={}, rationale="x")


def test_negative_quantity_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        PolicyProposal(action=ActionKind.REORDER, quantity=-1.0, features={}, rationale="x")


def test_heuristic_holds_above_the_reorder_point() -> None:
    proposal = HeuristicSQPolicy().propose(spec=SPEC, state=state(inventory=50.0), rng=rng())
    assert proposal.action is ActionKind.HOLD
    assert proposal.quantity == 0.0


def test_heuristic_reorders_at_the_reorder_point() -> None:
    proposal = HeuristicSQPolicy().propose(spec=SPEC, state=state(inventory=20.0), rng=rng())
    assert proposal.action is ActionKind.REORDER
    assert proposal.quantity == 40.0


def test_heuristic_counts_in_transit_stock() -> None:
    """The classic error is ignoring the pipeline and double-ordering."""
    proposal = HeuristicSQPolicy().propose(
        spec=SPEC, state=state(inventory=20.0, in_transit=40.0), rng=rng()
    )
    assert proposal.action is ActionKind.HOLD


def test_no_policy_exceeds_free_capacity() -> None:
    tight = state(inventory=95.0, in_transit=0.0, capacity=100.0, backlog=50.0)
    policies = (HeuristicSQPolicy(), OpaqueDROPolicy(), RandomBoundedPolicy(hold_probability=0.0))
    for policy in policies:
        proposal = policy.propose(spec=SPEC, state=tight, rng=rng())
        assert proposal.quantity <= 5.0 + 1e-9, policy.policy_id


def test_dro_expedites_only_under_backlog() -> None:
    policy = OpaqueDROPolicy()
    plain = policy.propose(spec=SPEC, state=state(inventory=0.0, demand_observed=20.0), rng=rng())
    assert plain.action is ActionKind.REORDER

    urgent = OpaqueDROPolicy().propose(
        spec=SPEC, state=state(inventory=0.0, backlog=15.0, demand_observed=20.0), rng=rng()
    )
    assert urgent.action is ActionKind.EXPEDITE


def test_dro_safety_stock_grows_with_the_ambiguity_radius() -> None:
    observed = state(inventory=10.0, demand_observed=20.0)
    narrow = OpaqueDROPolicy(ambiguity_radius=0.0).propose(spec=SPEC, state=observed, rng=rng())
    wide = OpaqueDROPolicy(ambiguity_radius=1.0).propose(spec=SPEC, state=observed, rng=rng())
    assert wide.quantity > narrow.quantity


def test_dro_rejects_impossible_parameters() -> None:
    with pytest.raises(ValueError, match="service_quantile"):
        OpaqueDROPolicy(service_quantile=1.0)
    with pytest.raises(ValueError, match="ambiguity_radius"):
        OpaqueDROPolicy(ambiguity_radius=-0.1)


def test_registry_builds_and_reports_planned_families() -> None:
    assert set(available_policies()) == {"heuristic_sQ", "opaque_dro", "random_bounded"}
    assert build_policy({"name": "heuristic_sQ"}).policy_id == "heuristic_sQ"
    with pytest.raises(NotImplementedError, match="stage 5"):
        build_policy({"name": "marl_heterogeneous"})
    with pytest.raises(ValueError, match="unknown policy"):
        build_policy({"name": "telepathy"})
    with pytest.raises(ValueError, match="rejected its parameters"):
        build_policy({"name": "opaque_dro", "params": {"nonsense": 1}})


def test_erfinv_matches_known_quantiles() -> None:
    from xai_gov.policies.families import _erfinv

    # z at the 95th percentile of a standard normal is 1.6449.
    z = float(np.sqrt(2.0) * _erfinv(2.0 * 0.95 - 1.0))
    assert abs(z - 1.6449) < 1e-3
