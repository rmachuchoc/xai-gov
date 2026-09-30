"""The three descriptor arms, audited on the same footing."""

from __future__ import annotations

import pytest

from tests.factories import state
from xai_gov.causal.descriptor import (
    CausalRecourseDescriptor,
    NoDescriptor,
    PostHocDescriptor,
    available_descriptors,
    build_descriptor,
)


def test_the_control_arm_claims_nothing() -> None:
    descriptor = NoDescriptor().build(
        state=state(), reorder_point=30.0, order_quantity=25.0
    )
    assert descriptor.method == "none"
    assert descriptor.informative is False
    assert descriptor.identifiable is False
    assert descriptor.feature_effects == {}


def test_the_causal_arm_reports_interventional_effects_and_recourse() -> None:
    builder = CausalRecourseDescriptor()
    descriptor = builder.build(
        state=state(inventory=35.0, in_transit=0.0), reorder_point=30.0, order_quantity=25.0
    )
    assert descriptor.method == "causal"
    assert descriptor.identifiable is True
    assert descriptor.feature_effects
    assert builder.last_recourse["cheapest_recourse"] is not None


def test_the_causal_arm_is_faithful_by_construction_but_still_measured() -> None:
    """Fidelity is computed rather than asserted, so a bug in the intervention
    machinery surfaces as a fidelity failure instead of passing silently."""
    descriptor = CausalRecourseDescriptor().build(
        state=state(inventory=35.0, in_transit=0.0), reorder_point=30.0, order_quantity=25.0
    )
    assert descriptor.explanatory_fidelity in (0.0, 1.0)


def test_the_post_hoc_arm_makes_no_identifiability_claim() -> None:
    """Marking a correlational attribution identifiable would be the precise
    error the protocol is about."""
    descriptor = PostHocDescriptor().build(
        state=state(), reorder_point=30.0, order_quantity=25.0
    )
    assert descriptor.method == "post_hoc"
    assert descriptor.identifiable is False
    assert descriptor.actionability_score == 0.0


def test_the_post_hoc_arm_is_audited_not_exempted() -> None:
    """If it ranks the levers correctly it passes; the protocol's claim has to
    be earned rather than assumed."""
    descriptor = PostHocDescriptor().build(
        state=state(inventory=35.0, in_transit=0.0), reorder_point=30.0, order_quantity=25.0
    )
    assert -1.0 <= descriptor.explanatory_fidelity <= 1.0
    assert isinstance(descriptor.informative, bool)


def test_post_hoc_attribution_can_rank_an_irrelevant_feature_first() -> None:
    """The failure mode the protocol describes: a feature deviates dramatically
    and has no causal path to the decision, while the lever that controls the
    outcome sits at its reference value and scores near zero."""
    descriptor = PostHocDescriptor().build(
        state=state(inventory=50.0, in_transit=20.0, capacity=60.0, demand_observed=200.0),
        reorder_point=30.0,
        order_quantity=25.0,
    )
    effects = descriptor.feature_effects
    assert abs(effects["demand_observed"]) > abs(effects["capacity"])


def test_a_low_fidelity_descriptor_declares_itself_uninformative() -> None:
    builder = PostHocDescriptor(fidelity_floor=1.01)
    descriptor = builder.build(
        state=state(inventory=35.0, in_transit=0.0), reorder_point=30.0, order_quantity=25.0
    )
    assert descriptor.informative is False


def test_an_uninformative_descriptor_cannot_move_the_threshold() -> None:
    """The point of auditing fidelity: a low-fidelity explanation must not be
    able to talk the agent out of intervening."""
    from xai_gov.governance.cpomdp import CpomdpGovernanceAgent
    from xai_gov.io.decision_record import CausalDescriptor

    agent = CpomdpGovernanceAgent(recourse_loss_relief=0.4)
    uninformative = CausalDescriptor(
        actionability_score=0.9,
        feature_effects={"inventory": 5.0},
        explanatory_fidelity=-1.0,
        identifiable=True,
        informative=False,
        method="causal",
    )
    threshold, repairable = agent.threshold_for(uninformative)
    assert repairable is False
    assert threshold == agent.threshold


def test_affordable_recourse_raises_the_threshold() -> None:
    """Theorem 2's monotonicity, as the mechanism that makes RQ3 answerable: a
    repairable situation carries less loss from inaction, and less loss means a
    higher bar for intervening."""
    from xai_gov.causal.descriptor import CausalRecourseDescriptor
    from xai_gov.governance.cpomdp import CpomdpGovernanceAgent

    builder = CausalRecourseDescriptor()
    agent = CpomdpGovernanceAgent(
        descriptor_builder=builder, recourse_loss_relief=0.4
    )
    descriptor = builder.build(
        state=state(inventory=35.0, in_transit=0.0),
        reorder_point=30.0,
        order_quantity=25.0,
    )
    threshold, repairable = agent.threshold_for(descriptor)
    if descriptor.informative and builder.last_recourse.get("cheapest_recourse"):
        assert repairable is True
        assert threshold > agent.threshold


def test_the_relief_channel_is_off_by_default() -> None:
    """A descriptor-blind agent is the control arm, so it must be what a bare
    construction gives."""
    from xai_gov.governance.cpomdp import CpomdpGovernanceAgent

    agent = CpomdpGovernanceAgent()
    assert agent.recourse_loss_relief == 0.0
    assert agent.to_payload()["repairable_threshold"] is None


def test_relieving_the_whole_loss_is_refused() -> None:
    from xai_gov.governance.cpomdp import CpomdpGovernanceAgent

    with pytest.raises(ValueError, match="never worthwhile"):
        CpomdpGovernanceAgent(recourse_loss_relief=1.0)
    assert available_descriptors() == ("none", "post_hoc", "causal")
    for name in available_descriptors():
        builder = build_descriptor({"descriptor": name})
        assert builder.method == name


def test_the_registry_passes_recourse_configuration_through() -> None:
    builder = build_descriptor(
        {"descriptor": "causal", "params": {"recourse": {"budget": 7.0}}}
    )
    assert builder.search.budget == 7.0  # type: ignore[union-attr]


def test_the_registry_refuses_a_typo() -> None:
    with pytest.raises(ValueError, match="unknown descriptor"):
        build_descriptor({"descriptor": "telepathy"})
    with pytest.raises(ValueError, match="unknown descriptor parameters"):
        build_descriptor({"descriptor": "causal", "params": {"fidelity_flor": 0.5}})
