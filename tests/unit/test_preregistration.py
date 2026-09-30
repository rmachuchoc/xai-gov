"""Preregistration: a promise the analysis can check."""

from __future__ import annotations

import pytest

from xai_gov.analysis.preregistration import (
    Hypothesis,
    Preregistration,
    build_preregistration,
    load_seal,
    verify_seal,
)


def hypothesis(**kwargs: object) -> Hypothesis:
    defaults: dict[str, object] = {
        "hypothesis_id": "H1",
        "question": "does governance help?",
        "indicator": "operational.service_level",
        "treatment": "governed",
        "control": "ungoverned",
        "direction": "greater",
        "predicted_effect": 0.05,
    }
    return Hypothesis(**(defaults | kwargs))  # type: ignore[arg-type]


def test_a_comparison_against_itself_is_refused() -> None:
    """Such a comparison cannot fail, so it commits to nothing."""
    with pytest.raises(ValueError, match="cannot fail"):
        hypothesis(treatment="governed", control="governed")


def test_an_unknown_direction_is_refused() -> None:
    with pytest.raises(ValueError, match="direction must be"):
        hypothesis(direction="upwards")


def test_a_plan_with_no_hypotheses_is_refused() -> None:
    with pytest.raises(ValueError, match="commits to nothing"):
        Preregistration(campaign="empty", hypotheses=())


def test_duplicate_hypothesis_ids_are_refused() -> None:
    with pytest.raises(ValueError, match="duplicate hypothesis id"):
        Preregistration(campaign="c", hypotheses=(hypothesis(), hypothesis()))


def test_the_hash_changes_when_the_plan_changes() -> None:
    """The seal is what makes the promise checkable."""
    original = Preregistration(campaign="c", hypotheses=(hypothesis(),))
    edited = Preregistration(
        campaign="c", hypotheses=(hypothesis(predicted_effect=0.5),)
    )
    assert original.plan_hash != edited.plan_hash


def test_the_hash_is_stable_across_identical_plans() -> None:
    first = Preregistration(campaign="c", hypotheses=(hypothesis(),))
    second = Preregistration(campaign="c", hypotheses=(hypothesis(),))
    assert first.plan_hash == second.plan_hash


def test_a_matching_seal_is_confirmatory() -> None:
    plan = Preregistration(campaign="c", hypotheses=(hypothesis(),))
    check = verify_seal(plan, plan.plan_hash)
    assert check.matches is True
    assert "confirmatory" in check.status


def test_an_edited_plan_becomes_exploratory() -> None:
    """The distinction the seal exists to preserve: adding an analysis is fine,
    quietly editing the primary hypothesis and calling it confirmatory is not."""
    sealed = Preregistration(campaign="c", hypotheses=(hypothesis(),)).plan_hash
    edited = Preregistration(campaign="c", hypotheses=(hypothesis(direction="less"),))
    check = verify_seal(edited, sealed)
    assert check.matches is False
    assert "exploratory" in check.status


def test_a_sealed_plan_cannot_be_overwritten(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A preregistration that can be silently replaced is not one."""
    plan = Preregistration(campaign="c", hypotheses=(hypothesis(),))
    path = tmp_path / "plan.json"
    plan.seal(path)
    with pytest.raises(FileExistsError, match="commits to nothing"):
        plan.seal(path)


def test_an_amendment_keeps_the_superseded_seal(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Amending a protocol and rewriting history differ by whether the first
    commitment survives in the record."""
    import json

    original = Preregistration(campaign="c", hypotheses=(hypothesis(),))
    path = tmp_path / "plan.json"
    first = original.seal(path)

    revised = Preregistration(
        campaign="c", hypotheses=(hypothesis(direction="less"),)
    )
    second = revised.seal(path, amend=True)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["plan_hash"] == second != first
    assert payload["amendments"] == 1
    assert payload["superseded"][0]["plan_hash"] == first


def test_a_sealed_plan_round_trips(tmp_path) -> None:  # type: ignore[no-untyped-def]
    plan = Preregistration(campaign="c", hypotheses=(hypothesis(),))
    path = tmp_path / "plan.json"
    digest = plan.seal(path)
    payload, loaded = load_seal(path)
    assert loaded == digest
    assert payload["campaign"] == "c"


def test_the_plan_lists_every_arm_it_names() -> None:
    plan = Preregistration(
        campaign="c",
        hypotheses=(hypothesis(), hypothesis(hypothesis_id="H2", treatment="shielded")),
    )
    assert plan.arms() == ("governed", "shielded", "ungoverned")


def test_the_registry_builds_and_names_what_it_rejects() -> None:
    plan = build_preregistration(
        {
            "campaign": "c",
            "hypotheses": [
                {
                    "id": "H1",
                    "indicator": "operational.service_level",
                    "treatment": "a",
                    "control": "b",
                }
            ],
        }
    )
    assert plan.by_id("H1").direction == "greater"
    with pytest.raises(ValueError, match="non-empty 'hypotheses'"):
        build_preregistration({"campaign": "c"})
    with pytest.raises(ValueError, match="unknown keys"):
        build_preregistration(
            {
                "campaign": "c",
                "hypotheses": [
                    {"indicator": "i", "treatment": "a", "control": "b", "typo": 1}
                ],
            }
        )
