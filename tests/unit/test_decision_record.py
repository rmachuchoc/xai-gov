"""The decision record is the auditable unit; its contract is strict."""

from __future__ import annotations

import pytest

from xai_gov.io.decision_record import (
    ActionKind,
    CausalDescriptor,
    ConformalSignal,
    DecisionRecord,
    GovernanceAction,
    OperatingState,
    RiskRegime,
    ShieldOutcome,
)


def make_record(**overrides: object) -> DecisionRecord:
    base: dict[str, object] = {
        "period": 3,
        "node_id": "wh-01",
        "state": OperatingState(
            period=3, inventory=40.0, backlog=2.0, in_transit=15.0,
            capacity=100.0, demand_observed=12.0,
        ),
        "policy_id": "heuristic_sQ",
        "policy_version": "1.0.0",
        "proposed_action": ActionKind.HOLD,
        "proposed_quantity": 0.0,
        "descriptor": CausalDescriptor(
            actionability_score=0.4, explanatory_fidelity=0.8, method="causal_recourse",
        ),
        "conformal": ConformalSignal(
            score=1.2, threshold=1.0, level=0.1, flagged_ood=True, scheme="aci",
        ),
        "belief": {"nominal": 0.2, "drift": 0.3, "disruption": 0.5},
        "governance_action": GovernanceAction.VETO,
        "governance_rationale": "hold vetoed under disruption belief above threshold",
        "autonomy_regime": "H2",
        "escalated": False,
        "safe_mode": False,
        "intervention_cost": 1.0,
        "shield": ShieldOutcome(),
        "final_action": ActionKind.REORDER,
        "final_quantity": 20.0,
    }
    return DecisionRecord(**(base | overrides))  # type: ignore[arg-type]


def test_belief_must_be_a_distribution() -> None:
    with pytest.raises(ValueError, match="belief"):
        make_record(belief={"nominal": 0.2, "disruption": 0.5})


def test_modification_is_detected() -> None:
    assert make_record().action_was_modified is True
    unchanged = make_record(
        governance_action=GovernanceAction.APPROVE,
        final_action=ActionKind.HOLD,
        final_quantity=0.0,
    )
    assert unchanged.action_was_modified is False


def test_belief_accessor_defaults_to_zero() -> None:
    record = make_record(belief={"nominal": 1.0})
    assert record.belief_of(RiskRegime.DISRUPTION) == 0.0


def test_payload_carries_every_audit_field() -> None:
    payload = make_record().to_payload()
    for key in (
        "period", "state", "policy", "proposed", "descriptor", "conformal",
        "belief", "governance", "shield", "final",
    ):
        assert key in payload
    assert payload["governance"]["rationale"]
    assert payload["conformal"]["threshold"] == 1.0


def test_flat_row_is_csv_friendly() -> None:
    row = make_record().to_flat_row()
    assert row["belief_disruption"] == 0.5
    assert all(not isinstance(value, dict | list) for value in row.values())


def test_absent_conformal_threshold_is_null_not_a_sentinel() -> None:
    """A sentinel infinity is not JSON-serializable, so it could never be
    sealed into the log — and it would satisfy every threshold comparison."""
    from xai_gov.core.hashing import canonical_json
    from xai_gov.governance.agent import NEUTRAL_CONFORMAL

    assert NEUTRAL_CONFORMAL.threshold is None
    assert NEUTRAL_CONFORMAL.active is False
    assert NEUTRAL_CONFORMAL.exceeds_threshold is False
    assert "null" in canonical_json(NEUTRAL_CONFORMAL.to_payload())


def test_active_scheme_must_declare_a_threshold() -> None:
    with pytest.raises(ValueError, match="declares no threshold"):
        ConformalSignal(score=1.0, threshold=None, level=0.1, flagged_ood=True, scheme="aci")


def test_inactive_scheme_cannot_raise_a_flag() -> None:
    with pytest.raises(ValueError, match="yet a flag is raised"):
        ConformalSignal(score=0.0, threshold=None, level=0.0, flagged_ood=True, scheme="none")


def test_non_finite_values_are_refused_and_located() -> None:
    from xai_gov.core.hashing import NonSerializableValueError, canonical_json

    with pytest.raises(NonSerializableValueError, match=r"conformal\.threshold"):
        canonical_json({"conformal": {"threshold": float("inf")}})


def test_record_is_immutable() -> None:
    record = make_record()
    with pytest.raises(AttributeError):
        record.final_quantity = 999.0  # type: ignore[misc]
