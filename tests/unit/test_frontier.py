"""The Pareto frontier: Theorem 3 as a deliverable."""

from __future__ import annotations

import pytest

from xai_gov.analysis.frontier import (
    THEOREM_3_OBJECTIVES,
    Objective,
    compute_frontier,
)


def arm(coverage: float, budget: float, service: float) -> dict[str, float]:
    return {
        "safety_coverage": coverage,
        "governance_budget": budget,
        "operational_return": service,
    }


def test_a_dominated_arm_is_off_the_frontier() -> None:
    frontier = compute_frontier(
        {
            "good": arm(0.02, 5.0, 0.99),
            "worse": arm(0.10, 20.0, 0.90),
        }
    )
    assert frontier.frontier_arms == ("good",)
    point = next(p for p in frontier.points if p.arm == "worse")
    assert point.dominated_by == ("good",)


def test_arms_that_trade_off_are_both_on_the_frontier() -> None:
    """The shape Theorem 3 predicts: each arm buys one objective with another."""
    frontier = compute_frontier(
        {
            "safe": arm(0.01, 30.0, 0.90),
            "cheap": arm(0.15, 3.0, 0.92),
            "productive": arm(0.08, 18.0, 0.99),
        }
    )
    assert set(frontier.frontier_arms) == {"safe", "cheap", "productive"}


def test_directions_are_respected() -> None:
    """Getting these backwards produces a frontier that looks reasonable and is
    exactly inverted."""
    coverage = next(o for o in THEOREM_3_OBJECTIVES if o.name == "safety_coverage")
    service = next(o for o in THEOREM_3_OBJECTIVES if o.name == "operational_return")
    assert coverage.maximize is False
    assert service.maximize is True
    assert coverage.better(0.01, 0.05) is True
    assert service.better(0.99, 0.90) is True


def test_an_arm_missing_an_objective_is_excluded_not_defaulted() -> None:
    """A phantom point on the frontier is the most damaging fabrication this
    analysis could make."""
    frontier = compute_frontier(
        {
            "complete": arm(0.02, 5.0, 0.99),
            "partial": {"safety_coverage": 0.01},
        }
    )
    assert frontier.frontier_arms == ("complete",)
    assert "partial" in frontier.excluded
    assert "missing objectives" in frontier.excluded["partial"]


def test_a_dominating_arm_contradicts_theorem_3_and_says_so() -> None:
    """A result contradicting the project's own theory must not pass quietly."""
    frontier = compute_frontier(
        {
            "best": arm(0.01, 1.0, 0.99),
            "other": arm(0.10, 20.0, 0.80),
        }
    )
    assert frontier.single_arm_dominates_all is True
    payload = frontier.to_payload()
    assert payload["theorem_3_consistent"] is False
    assert "must be investigated" in payload["interpretation"]


def test_a_genuine_frontier_is_consistent_with_theorem_3() -> None:
    frontier = compute_frontier(
        {
            "safe": arm(0.01, 30.0, 0.90),
            "cheap": arm(0.15, 3.0, 0.92),
        }
    )
    payload = frontier.to_payload()
    assert payload["theorem_3_consistent"] is True
    assert "informed choices" in payload["interpretation"]


def test_the_trade_off_rate_carries_its_sign() -> None:
    """The sign is the content. Tighter coverage costs budget, so moving along
    the frontier toward better coverage must show budget going up — a rate
    computed from independent spreads would report every trade-off as an
    apparent alignment."""
    frontier = compute_frontier(
        {
            "safe": arm(0.01, 30.0, 0.90),
            "cheap": arm(0.15, 3.0, 0.92),
        }
    )
    rate = frontier.trade_off("safety_coverage", "governance_budget")
    assert rate is not None
    # Coverage error rises from 0.01 to 0.15 while budget falls from 30 to 3.
    assert rate < 0.0


def test_aligned_objectives_show_a_positive_rate() -> None:
    """Not every pair trades off; when two objectives move together the rate
    must say so rather than being forced negative."""
    frontier = compute_frontier(
        {
            "safe": arm(0.01, 3.0, 0.90),
            "loose": arm(0.15, 30.0, 0.92),
        }
    )
    rate = frontier.trade_off("safety_coverage", "governance_budget")
    assert rate is not None
    assert rate > 0.0


def test_a_single_point_has_no_trade_to_quantify() -> None:
    frontier = compute_frontier({"only": arm(0.02, 5.0, 0.99)})
    assert frontier.trade_off("safety_coverage", "governance_budget") is None


def test_the_best_arm_on_one_objective_is_findable() -> None:
    frontier = compute_frontier(
        {
            "safe": arm(0.01, 30.0, 0.90),
            "cheap": arm(0.15, 3.0, 0.92),
        }
    )
    assert frontier.best_on("safety_coverage") == "safe"
    assert frontier.best_on("governance_budget") == "cheap"
    assert frontier.best_on("nonexistent") is None


def test_no_objectives_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one objective"):
        compute_frontier({"a": arm(0.0, 0.0, 0.0)}, objectives=())


def test_a_custom_objective_set_is_honoured() -> None:
    objectives = (Objective(name="cost", indicator="c", maximize=False),)
    frontier = compute_frontier(
        {"cheap": {"cost": 1.0}, "dear": {"cost": 9.0}}, objectives=objectives
    )
    assert frontier.frontier_arms == ("cheap",)
