"""The budget constraint, its shadow price, and decision rights."""

from __future__ import annotations

import pytest

from xai_gov.governance.autonomy import (
    AutonomyRegime,
    AutonomyState,
    available_regimes,
    rights_for,
)
from xai_gov.governance.budget import GovernanceBudget, build_budget
from xai_gov.io.decision_record import ActionKind, GovernanceAction


def test_spending_is_tracked_in_discounted_terms() -> None:
    budget = GovernanceBudget(allowance=10.0, discount=0.9)
    budget.charge(1.0, period=0)
    budget.charge(1.0, period=10)
    assert budget.spent == 2.0
    assert budget.discounted_spent < 2.0


def test_an_exhausted_budget_agrees_with_what_it_can_afford() -> None:
    """Discounting means spend converges on the allowance without reaching it,
    so a bare comparison against the allowance would call a budget solvent
    while refusing every request made to it."""
    budget = GovernanceBudget(allowance=2.0)
    for period in range(5):
        if budget.can_afford(1.0):
            budget.charge(1.0, period=period)
        else:
            budget.refuse(1.0)
    assert budget.can_afford(1.0) is False
    assert budget.exhausted is True
    assert budget.refusals > 0


def test_refusals_are_counted_separately_from_approvals() -> None:
    """A decision left alone because the budget ran out is not the same event
    as one left alone out of confidence."""
    budget = GovernanceBudget(allowance=0.0)
    budget.refuse(1.0)
    payload = budget.to_payload()
    assert payload["refusals"] == 1
    assert payload["interventions"] == 0
    assert payload["refused_demand"] == 1.0


def test_turned_away_demand_drives_the_price_up() -> None:
    """The price of a constraint comes from the demand it suppressed, so a
    budget that keeps refusing must keep generating gradient."""
    binding = GovernanceBudget(allowance=2.0, dual_rate=0.1)
    for period in range(30):
        if binding.can_afford(1.0):
            binding.charge(1.0, period=period)
        else:
            binding.refuse(1.0)
        binding.update_multiplier(period)
    assert binding.refusals > 0
    assert binding.shadow_price > 0.0


def test_the_price_is_comparable_across_allowances() -> None:
    """The gradient is relative, not absolute: a larger budget must not read as
    more strained merely because its numbers are bigger."""
    tight = GovernanceBudget(allowance=4.0, dual_rate=0.1)
    ample = GovernanceBudget(allowance=200.0, dual_rate=0.1)
    for period in range(40):
        for budget in (tight, ample):
            if budget.can_afford(1.0):
                budget.charge(1.0, period=period)
            else:
                budget.refuse(1.0)
            budget.update_multiplier(period)
    assert tight.shadow_price > ample.shadow_price
    assert ample.shadow_price == pytest.approx(0.0)


def test_the_multiplier_rises_only_when_the_budget_binds() -> None:
    lavish = GovernanceBudget(allowance=1000.0, dual_rate=0.1)
    scarce = GovernanceBudget(allowance=1.0, dual_rate=0.1)
    for period in range(30):
        lavish.charge(1.0, period=period)
        lavish.update_multiplier(period)
        if scarce.can_afford(1.0):
            scarce.charge(1.0, period=period)
        else:
            scarce.refuse(1.0)
        scarce.update_multiplier(period)
    assert lavish.multiplier == pytest.approx(0.0)
    assert scarce.multiplier > 0.0


def test_the_shadow_price_is_zero_when_oversight_is_not_scarce() -> None:
    budget = GovernanceBudget(allowance=1000.0)
    for period in range(40):
        budget.update_multiplier(period)
    assert budget.shadow_price == pytest.approx(0.0)


def test_the_multiplier_prices_interventions_upward() -> None:
    budget = GovernanceBudget(allowance=1.0, dual_rate=0.2)
    assert budget.effective_cost(1.0) == 1.0
    for period in range(20):
        if budget.can_afford(1.0):
            budget.charge(1.0, period=period)
        else:
            budget.refuse(1.0)
        budget.update_multiplier(period)
    assert budget.effective_cost(1.0) > 1.0


def test_the_multiplier_never_goes_negative() -> None:
    """A slack constraint has zero price, never a negative one."""
    budget = GovernanceBudget(allowance=100.0, dual_rate=1.0)
    for period in range(50):
        budget.update_multiplier(period)
    assert budget.multiplier >= 0.0


def test_invalid_budget_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="allowance"):
        GovernanceBudget(allowance=-1.0)
    with pytest.raises(ValueError, match="dual_rate"):
        GovernanceBudget(dual_rate=0.0)
    with pytest.raises(ValueError, match="cannot be negative"):
        GovernanceBudget().charge(-1.0, period=0)
    with pytest.raises(ValueError, match="cannot be negative"):
        GovernanceBudget().refuse(-1.0)


def test_registry_builds_defaults() -> None:
    assert build_budget({}).allowance == 20.0
    with pytest.raises(ValueError, match="rejected its parameters"):
        build_budget({"allowence": 5.0})


# -- decision rights ------------------------------------------------------
def test_every_regime_is_available() -> None:
    assert available_regimes() == ("H0", "H1", "H2", "H3")


def test_h0_authorizes_nothing_but_holding() -> None:
    rights = rights_for("H0")
    assert rights.permits(ActionKind.HOLD, quantity=0.0, bound=100.0) is True
    assert rights.permits(ActionKind.REORDER, quantity=1.0, bound=100.0) is False
    # And it asks rather than refuses: the organization holds the decision.
    assert rights.default_action_when_not_permitted() is GovernanceAction.ESCALATE


def test_h1_authorizes_within_an_agreed_range_and_vetoes_outside_it() -> None:
    rights = rights_for("H1")
    assert rights.permits(ActionKind.REORDER, quantity=30.0, bound=40.0) is True
    assert rights.permits(ActionKind.REORDER, quantity=50.0, bound=40.0) is False
    # A veto, not an escalation: the range was agreed in advance, so stepping
    # outside it is a violation rather than a question.
    assert rights.default_action_when_not_permitted() is GovernanceAction.VETO


def test_only_h2_may_reduce_autonomy() -> None:
    assert rights_for("H2").may_reduce_autonomy is True
    for regime in ("H0", "H1", "H3"):
        assert rights_for(regime).may_reduce_autonomy is False


def test_h3_does_not_escalate() -> None:
    assert rights_for("H3").may_escalate is False
    assert rights_for("H3").permits(ActionKind.REORDER, quantity=1e6, bound=1.0) is True


def test_the_ex_ante_ex_post_distinction_is_modelled() -> None:
    """The regimes differ in kind, not degree: when the organization exercises
    what it retained is what makes their oversight cost differ."""
    assert rights_for("H0").veto_is_ex_ante is True
    assert rights_for("H1").veto_is_ex_ante is True
    assert rights_for("H2").veto_is_ex_ante is False
    assert rights_for("H3").veto_is_ex_ante is False


def test_autonomy_steps_down_one_regime_at_a_time() -> None:
    state = AutonomyState(initial=AutonomyRegime.H3)
    assert state.reduce() is True
    assert state.current is AutonomyRegime.H2
    assert state.reduce() is True
    assert state.current is AutonomyRegime.H1
    assert state.reduce() is True
    assert state.current is AutonomyRegime.H0
    assert state.reduce() is False
    assert state.reductions == 3


def test_autonomy_is_not_self_restorable() -> None:
    """Restoring trust is an organizational decision, not something the agent
    that just lost it may grant itself."""
    state = AutonomyState(initial=AutonomyRegime.H2)
    state.reduce()
    assert state.reduced is True
    assert not hasattr(state, "restore")
