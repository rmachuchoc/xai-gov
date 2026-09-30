"""The schema contract between the analysis payload and the figure renderers.

A figure renderer reads keys out of the study bundle, and nothing in the type
system connects the key it reads to the key the analysis writes. When those
drift, the renderer finds nothing and reports that the bundle carries no data
for it — a message that is true of where it looked and false of the campaign.
That shape survived three consecutive runs of this study looking like an empty
result, so the contract is pinned here instead of being left to inspection.

These tests deliberately assert on payload *keys* rather than on rendered
output: the defect was never in the drawing, and the print renderer needs
matplotlib, which the library does not.
"""

from __future__ import annotations

from xai_gov.analysis.anytime import directional_evidence
from xai_gov.analysis.strategic import strategic_value


def kpis(service: float, cost: float = 10.0) -> dict:  # type: ignore[type-arg]
    return {
        "operational": {"service_level": service},
        "governance": {
            "total_intervention_cost": cost,
            "decisions": 30.0,
            "shadow_price": 0.4,
        },
    }


def test_the_evidence_payload_carries_the_keys_the_figure_reads() -> None:
    """fig4-evidence reads forward_e_value and reverse_e_value.

    An earlier renderer read 'evidence_for'/'evidence_against' with a nested
    'e_value' — a schema of its own invention that never existed in any bundle.
    """
    payload = directional_evidence(
        [0.5] * 20, alpha=0.05, hypotheses=8, indicator="service_level"
    ).to_payload()

    assert "forward_e_value" in payload
    assert "reverse_e_value" in payload
    assert "familywise_threshold" in payload
    # And the names the renderer must *not* rely on.
    assert "evidence_for" not in payload
    assert "e_value" not in payload


def test_the_familywise_threshold_travels_with_the_evidence() -> None:
    """The figure draws the threshold the verdicts were taken against, so it
    must come from the payload rather than be recomputed from a count."""
    payload = directional_evidence(
        [0.5] * 20, alpha=0.05, hypotheses=8, indicator="x"
    ).to_payload()
    assert payload["familywise_threshold"] == 160.0
    assert payload["per_comparison_threshold"] == 20.0


def test_a_two_sided_hypothesis_has_no_reverse_value() -> None:
    """fig4 must skip that arm rather than plot a missing number as zero."""
    payload = directional_evidence(
        [0.5] * 20, alpha=0.05, hypotheses=8, two_sided=True, indicator="x"
    ).to_payload()
    assert payload["reverse_e_value"] is None


def test_the_strategic_payload_carries_the_keys_fig3_reads() -> None:
    value = strategic_value(
        arm="shielded",
        baseline="ungoverned",
        governed=[kpis(0.67), kpis(0.67)],
        ungoverned=[kpis(0.61, cost=0.0), kpis(0.61, cost=0.0)],
    )
    assert value is not None
    payload = value.to_payload()
    assert "value_of_governance" in payload
    assert "tvc" in payload


def test_an_undefined_tvc_stays_none_in_the_payload() -> None:
    """fig3 plots a point only when both coordinates exist; a sentinel here
    would place a phantom arm on the chart."""
    value = strategic_value(
        arm="a",
        baseline="b",
        governed=[kpis(0.9, cost=0.0), kpis(0.9, cost=0.0)],
        ungoverned=[kpis(0.7, cost=0.0), kpis(0.7, cost=0.0)],
    )
    assert value is not None
    assert value.to_payload()["tvc"] is None


def test_every_figure_declares_what_it_reads() -> None:
    """The skip message names the expected input, so a reader can tell a
    renderer that looked in the wrong place from a campaign that produced
    nothing."""
    from pathlib import Path

    source = Path(__file__).resolve().parents[2] / "scripts" / "render_figures.py"
    text = source.read_text(encoding="utf-8")

    # Read as text rather than imported: the module needs matplotlib, and the
    # contract under test is the declaration, not the drawing.
    assert "_EXPECTED_INPUTS" in text
    assert "forward_e_value" in text
    assert 'h["evidence_for"]' not in text
