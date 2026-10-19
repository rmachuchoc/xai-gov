"""Tabular Q-learning: deterministic, learns from its own proposals."""

from __future__ import annotations

import numpy as np
import pytest

from tests.factories import state
from xai_gov.policies.qlearning import (
    ACTION_FRACTIONS,
    QLearningPolicy,
    band,
    state_key,
)
from xai_gov.policies.registry import available_policies, build_policy
from xai_gov.simulation.network import NodeSpec

SPEC = NodeSpec(
    node_id="retail",
    echelon=0,
    supplier_id="dc",
    initial_inventory=40.0,
    capacity=200.0,
    reorder_point=30.0,
    order_quantity=25.0,
    lead_time_mean=2.0,
)


def policy(**kwargs: object) -> QLearningPolicy:
    return QLearningPolicy(pretrain_episodes=150, **kwargs)  # type: ignore[arg-type]


def test_bands_are_monotone() -> None:
    assert band(-50.0, (0.0, 10.0)) == 0
    assert band(5.0, (0.0, 10.0)) == 1
    assert band(50.0, (0.0, 10.0)) == 2


def test_state_uses_inventory_position_not_on_hand() -> None:
    """Ordering on on-hand alone re-orders while a shipment is in transit."""
    assert state_key(10.0, 30.0, 0.0) != state_key(10.0, 0.0, 0.0)


def test_pretraining_is_deterministic() -> None:
    assert np.array_equal(policy().pretrain(SPEC), policy().pretrain(SPEC))


def test_a_depleted_node_orders_at_least_as_much_as_a_stocked_one() -> None:
    """The mechanism a replenishment learner must acquire from its surrogate."""
    q = policy().pretrain(SPEC)
    low = ACTION_FRACTIONS[int(np.argmax(q[state_key(0.0, 0.0, 0.0)]))]
    high = ACTION_FRACTIONS[int(np.argmax(q[state_key(90.0, 0.0, 0.0)]))]
    assert low >= high


def test_decisions_are_reproducible_from_the_seed() -> None:
    def run() -> list[float]:
        p, rng = policy(), np.random.default_rng(7)
        states = [state(inventory=float(i), in_transit=0.0) for i in range(0, 60, 5)]
        return [p.propose(spec=SPEC, state=s, rng=rng).quantity for s in states]

    assert run() == run()


def test_online_learning_updates_the_proposed_action() -> None:
    """Credit goes to what was proposed, whether or not it was executed."""
    p, rng = policy(epsilon=0.0), np.random.default_rng(1)
    p.propose(spec=SPEC, state=state(inventory=0.0, in_transit=0.0), rng=rng)
    before = p._learners["retail"].q.copy()
    p.propose(spec=SPEC, state=state(inventory=0.0, in_transit=0.0, backlog=30.0), rng=rng)
    assert p._learners["retail"].updates == 1
    assert not np.array_equal(before, p._learners["retail"].q)


def test_frozen_policy_does_not_learn_online() -> None:
    p, rng = policy(online=False), np.random.default_rng(1)
    for _ in range(3):
        p.propose(spec=SPEC, state=state(inventory=0.0), rng=rng)
    assert p._learners["retail"].updates == 0


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="alpha"):
        QLearningPolicy(alpha=0.0)
    with pytest.raises(ValueError, match="gamma"):
        QLearningPolicy(gamma=1.0)
    with pytest.raises(ValueError, match="epsilon"):
        QLearningPolicy(epsilon=1.5)


def test_the_registry_builds_it() -> None:
    assert "qlearning_tabular" in available_policies()
    built = build_policy({"name": "qlearning_tabular", "params": {"pretrain_episodes": 10}})
    assert isinstance(built, QLearningPolicy)
