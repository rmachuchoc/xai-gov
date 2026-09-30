"""Under-flagging: the failure the flag count cannot see."""

from __future__ import annotations

from xai_gov.analysis.campaign import apply_exclusions


def guarantees(**kwargs: object) -> dict:  # type: ignore[type-arg]
    return {"guarantees": {"status": "available", **kwargs}}


HEALTHY = {"status": "completed", "chain_verified": True}


def test_a_healthy_detector_is_included() -> None:
    included, reason = apply_exclusions(
        HEALTHY, guarantees(saturated=False, under_flagging=False)
    )
    assert included is True
    assert reason is None


def test_under_flagging_does_not_exclude() -> None:
    """Exclusion removes anomalous cells from an analysis that otherwise
    proceeds. When a condition holds for every arm it is not an anomaly but a
    property of the pipeline, and excluding on it destroys the study: applying
    it that way lost 400 of 440 cells and produced no comparison at all. It is
    reported as a blocking diagnostic instead."""
    included, reason = apply_exclusions(HEALTHY, guarantees(under_flagging=True))
    assert included is True
    assert reason is None


def test_under_flagging_is_a_blocking_diagnostic() -> None:
    from xai_gov.analysis.artifacts import diagnose

    bundle = {
        "campaign": {
            "campaign_summary": {"included_by_arm": {}, "exclusions": []},
            "plan": {"hypotheses": []},
            "hypotheses": [],
            "frontier": {},
            "cells": [
                {"arm": "a", "under_flagging": True},
                {"arm": "b", "under_flagging": True},
            ],
        }
    }
    finding = next(f for f in diagnose(bundle) if f.code == "detector_under_flags")
    assert finding.severity == "blocking"
    assert "in the detector rather than in any arm" in finding.implication


def test_one_under_flagging_arm_is_scoped_to_that_arm() -> None:
    """A single arm failing is about that arm; every arm failing is about the
    conformal layer, and the two call for different responses."""
    from xai_gov.analysis.artifacts import diagnose

    bundle = {
        "campaign": {
            "campaign_summary": {"included_by_arm": {}, "exclusions": []},
            "plan": {"hypotheses": []},
            "hypotheses": [],
            "frontier": {},
            "cells": [
                {"arm": "a", "under_flagging": True},
                {"arm": "b", "under_flagging": False},
                {"arm": "c", "under_flagging": False},
            ],
        }
    }
    finding = next(f for f in diagnose(bundle) if f.code == "detector_under_flags")
    assert "self-normalization" in finding.implication


def test_saturation_still_excludes() -> None:
    included, reason = apply_exclusions(HEALTHY, guarantees(saturated=True))
    assert included is False
    assert "saturation" in str(reason)


def test_saturation_and_under_flagging_are_handled_differently() -> None:
    """Saturation is an anomaly a cell can have on its own, so it excludes.
    Under-flagging held for every arm, so it reports instead."""
    over_included, _ = apply_exclusions(HEALTHY, guarantees(saturated=True))
    under_included, _ = apply_exclusions(HEALTHY, guarantees(under_flagging=True))
    assert over_included is False
    assert under_included is True


def test_the_criteria_version_is_bumped_with_the_criteria() -> None:
    """A resumed campaign must not reuse cells judged under older criteria: the
    reused artifacts predate the new key, the lookup returns None, and the
    criterion silently does not apply to them."""
    from xai_gov.analysis.campaign import EXCLUSION_CRITERIA_VERSION

    assert EXCLUSION_CRITERIA_VERSION >= 3


def test_the_criterion_is_declared_in_the_default_plan() -> None:
    """An exclusion applied but not preregistered is a choice, not a criterion."""
    from xai_gov.analysis.preregistration import Hypothesis, Preregistration

    plan = Preregistration(
        campaign="c",
        hypotheses=(
            Hypothesis(
                hypothesis_id="H1",
                question="?",
                indicator="operational.service_level",
                treatment="a",
                control="b",
                direction="greater",
                predicted_effect=0.1,
            ),
        ),
    )
    # Under-flagging is deliberately absent: it is a blocking diagnostic, not
    # an exclusion, and a criterion listed but not applied would be worse than
    # one applied but not listed.
    assert not any("under-flag" in c for c in plan.exclusion_criteria)
    assert any("saturation" in c for c in plan.exclusion_criteria)
