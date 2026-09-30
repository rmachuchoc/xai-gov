"""Explanatory fidelity: catching explanations that are decorative or worse."""

from __future__ import annotations

from xai_gov.causal.fidelity import explanatory_fidelity


def test_a_perfectly_ordered_explanation_scores_one() -> None:
    truth = {"inventory": 25.0, "capacity": 10.0, "data_delay": 1.0}
    attributed = {"inventory": 3.0, "capacity": 2.0, "data_delay": 0.5}
    report = explanatory_fidelity(attributed, truth)
    assert report.fidelity == 1.0
    assert report.verdict == "informative"
    assert report.top_lever_agrees is True


def test_a_reversed_explanation_is_flagged_anti_correlated() -> None:
    """Worse than useless: an operator following it acts on the least effective
    lever while believing it is the best."""
    truth = {"inventory": 25.0, "capacity": 10.0, "data_delay": 1.0}
    backwards = {"inventory": 0.5, "capacity": 2.0, "data_delay": 3.0}
    report = explanatory_fidelity(backwards, truth)
    assert report.fidelity == -1.0
    assert report.verdict == "anti_correlated"
    assert report.informative is False
    assert report.top_lever_agrees is False


def test_magnitude_scaling_does_not_change_the_verdict() -> None:
    """Ranks, not magnitudes: attributions and order quantities are in
    different units, so a magnitude comparison would measure the scaling
    convention rather than the explanation."""
    truth = {"a": 20.0, "b": 5.0, "c": 1.0}
    small = {"a": 0.003, "b": 0.002, "c": 0.001}
    large = {"a": 3000.0, "b": 2000.0, "c": 1000.0}
    assert explanatory_fidelity(small, truth).fidelity == 1.0
    assert explanatory_fidelity(large, truth).fidelity == 1.0


def test_sign_is_ignored_only_magnitude_of_effect_is_ranked() -> None:
    """A lever that strongly reduces the order matters as much as one that
    strongly increases it."""
    truth = {"a": -20.0, "b": 5.0}
    attributed = {"a": 9.0, "b": 1.0}
    assert explanatory_fidelity(attributed, truth).fidelity == 1.0


def test_a_single_lever_cannot_claim_fidelity() -> None:
    """With one lever there is no ordering to verify, and reporting 1 would let
    a node claim a faithful explanation on no evidence."""
    report = explanatory_fidelity({"inventory": 1.0}, {"inventory": 5.0})
    assert report.verdict == "not_verifiable"
    assert report.informative is False
    assert report.fidelity == 0.0


def test_all_ties_are_not_verifiable() -> None:
    """When nothing moves the decision there is no ordering to be right about."""
    report = explanatory_fidelity({"a": 1.0, "b": 2.0}, {"a": 0.0, "b": 0.0})
    assert report.verdict == "not_verifiable"
    assert report.compared == 0


def test_a_middling_explanation_is_uninformative_not_wrong() -> None:
    truth = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
    partly = {"a": 4.0, "b": 3.0, "c": 1.0, "d": 2.0}
    report = explanatory_fidelity(partly, truth, floor=0.9)
    assert report.verdict == "uninformative"
    assert 0.0 < report.fidelity < 0.9


def test_the_floor_is_what_separates_the_verdicts() -> None:
    truth = {"a": 4.0, "b": 3.0, "c": 2.0, "d": 1.0}
    partly = {"a": 4.0, "b": 3.0, "c": 1.0, "d": 2.0}
    assert explanatory_fidelity(partly, truth, floor=0.5).verdict == "informative"
    assert explanatory_fidelity(partly, truth, floor=0.9).verdict == "uninformative"


def test_only_shared_levers_are_compared() -> None:
    report = explanatory_fidelity({"a": 3.0, "b": 1.0, "z": 9.0}, {"a": 30.0, "b": 10.0})
    assert report.fidelity == 1.0
    assert report.compared == 1
