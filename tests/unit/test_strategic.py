"""Strategic indicators: paired, signed, and undefined where they must be."""

from __future__ import annotations

from xai_gov.analysis.strategic import strategic_value


def kpis(service: float, cost: float = 10.0, decisions: float = 30.0) -> dict:  # type: ignore[type-arg]
    return {
        "operational": {"service_level": service},
        "governance": {
            "total_intervention_cost": cost,
            "decisions": decisions,
            "shadow_price": 0.4,
        },
    }


def test_governance_that_helps_has_positive_value_of_information() -> None:
    value = strategic_value(
        arm="governed",
        baseline="ungoverned",
        governed=[kpis(0.90), kpis(0.92)],
        ungoverned=[kpis(0.70, cost=0.0), kpis(0.72, cost=0.0)],
    )
    assert value is not None
    assert value.value_of_governance > 0.0
    assert value.rgd is not None and value.rgd > 0.0


def test_governance_that_harms_has_negative_value_of_information() -> None:
    """A signed indicator on purpose: the campaign has already produced an arm
    where governance destroys value, and that is a finding."""
    value = strategic_value(
        arm="governed",
        baseline="ungoverned",
        governed=[kpis(0.06), kpis(0.05)],
        ungoverned=[kpis(0.61, cost=0.0), kpis(0.62, cost=0.0)],
    )
    assert value is not None
    assert value.value_of_governance < 0.0
    assert value.governance_pays is False


def test_free_governance_leaves_the_return_undefined() -> None:
    """A ratio with a zero denominator has no value; a large sentinel would
    propagate into the frontier as a spuriously excellent arm."""
    value = strategic_value(
        arm="a",
        baseline="b",
        governed=[kpis(0.9, cost=0.0), kpis(0.9, cost=0.0)],
        ungoverned=[kpis(0.7, cost=0.0), kpis(0.7, cost=0.0)],
    )
    assert value is not None
    assert value.rgd is None
    assert value.tvc is None
    assert value.governance_pays is None
    # The value of information is still defined: it needs no denominator.
    assert value.value_of_governance > 0.0


def test_pairs_missing_the_indicator_are_dropped_not_filled() -> None:
    value = strategic_value(
        arm="a",
        baseline="b",
        governed=[kpis(0.9), {"operational": {}}, kpis(0.9)],
        ungoverned=[kpis(0.7), kpis(0.7), kpis(0.7)],
    )
    assert value is not None
    assert value.pairs == 2


def test_no_usable_pairs_yields_nothing() -> None:
    assert (
        strategic_value(
            arm="a", baseline="b", governed=[{"operational": {}}], ungoverned=[kpis(0.7)]
        )
        is None
    )


def test_the_payload_carries_its_definitions() -> None:
    value = strategic_value(
        arm="a", baseline="b", governed=[kpis(0.9), kpis(0.9)],
        ungoverned=[kpis(0.7), kpis(0.7)],
    )
    assert value is not None
    payload = value.to_payload()
    # The definition states the expectation difference rather than naming the
    # concept: a reader checking the number needs the formula, not the label.
    assert "E[V |" in payload["definitions"]["value_of_governance"]
    assert "paired ungoverned arm" in payload["definitions"]["value_of_governance"]
    assert payload["baseline"] == "b"
