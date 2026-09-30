"""Bidirectional evidence, and the multiplicity the anytime guarantee omits.

Two errors this module exists to prevent, and the project shipped both.

The first: reading a forward e-value below one as evidence that the effect runs
the other way. In the standard framework a large e-value is evidence against
its own null; a small one means the bet lost and licenses no claim about the
opposite direction. Establishing that requires a second e-value against a
second null, which is the same betting martingale on negated differences.

The second: treating anytime validity as though it covered multiplicity. It
does not. Validity under optional stopping and familywise error control are
different guarantees, and seven preregistered hypotheses need both declared.
"""

from __future__ import annotations

import numpy as np
import pytest

from xai_gov.analysis.anytime import directional_evidence


def rising(mean: float, *, n: int = 40, sd: float = 0.05, seed: int = 3) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(rng.normal(mean, sd)) for _ in range(n)]


def test_a_predicted_effect_is_supported_in_the_forward_direction() -> None:
    evidence = directional_evidence(rising(0.6), alpha=0.05, hypotheses=7)
    assert evidence.forward >= evidence.familywise_threshold
    assert evidence.supported is True
    assert evidence.refuted is False


def test_a_small_forward_value_alone_does_not_refute() -> None:
    """The error, prevented. A lost bet is not a claim about the other side."""
    evidence = directional_evidence(rising(-0.02, sd=0.3), alpha=0.05, hypotheses=7)
    assert evidence.forward < 1.0
    # Small forward, small reverse: nothing established either way.
    assert evidence.supported is False
    assert evidence.refuted is False
    assert "inconclusive" in evidence.verdict


def test_the_opposite_direction_needs_its_own_e_value() -> None:
    """A large effect against the prediction is established by the reverse
    test, not inferred from the forward one being small."""
    evidence = directional_evidence(rising(-0.55), alpha=0.05, hypotheses=7)
    assert evidence.forward < 1.0
    assert evidence.reverse is not None
    assert evidence.reverse >= evidence.familywise_threshold
    assert evidence.refuted is True
    assert "opposite to the prediction" in evidence.verdict


def test_both_nulls_are_named() -> None:
    """A reader has to be able to see which hypothesis each number is about."""
    evidence = directional_evidence(rising(0.5), indicator="service_level")
    assert "service_level" in evidence.forward_null
    assert evidence.reverse_null is not None
    assert "service_level" in evidence.reverse_null


def test_the_payload_states_what_a_small_forward_value_means() -> None:
    payload = directional_evidence(rising(0.1)).to_payload()
    assert "the bet lost" in payload["note"]
    assert "only the reverse e-value" in payload["note"]


# -- multiplicity ---------------------------------------------------------
def test_the_familywise_threshold_scales_with_the_hypothesis_count() -> None:
    """Anytime validity handles optional stopping and nothing else. With k
    hypotheses, a union bound over e-values gives familywise control at alpha
    by requiring e >= k/alpha."""
    single = directional_evidence(rising(0.5), alpha=0.05, hypotheses=1)
    family = directional_evidence(rising(0.5), alpha=0.05, hypotheses=7)
    assert single.familywise_threshold == pytest.approx(20.0)
    assert family.familywise_threshold == pytest.approx(140.0)
    assert family.threshold == single.threshold == pytest.approx(20.0)


def test_an_effect_between_the_thresholds_is_reported_as_such() -> None:
    """Neither hidden nor promoted: a result that clears the per-comparison bar
    and not the familywise one is exactly what a reader needs told."""
    evidence = directional_evidence(rising(0.28, sd=0.35, seed=11), alpha=0.05, hypotheses=7)
    if evidence.threshold <= evidence.forward < evidence.familywise_threshold:
        assert evidence.supported is False
        assert "per-comparison only" in evidence.verdict


def test_a_two_sided_claim_gets_no_reverse_test() -> None:
    """Its differences are magnitudes, so there is no opposite direction to bet
    on and inventing one would be reporting a test that was never run."""
    evidence = directional_evidence(
        [abs(value) for value in rising(0.5)], two_sided=True
    )
    assert evidence.reverse is None
    assert evidence.reverse_null is None
    assert evidence.refuted is False


def test_invalid_arguments_are_refused() -> None:
    with pytest.raises(ValueError, match="hypotheses must be at least 1"):
        directional_evidence(rising(0.5), hypotheses=0)
    with pytest.raises(ValueError, match="empty set of differences"):
        directional_evidence([])


def test_the_two_directions_use_the_same_scale() -> None:
    """A reverse test on a different normalization would not be comparable to
    the forward one, and the pair is meant to be read together."""
    values = rising(0.4)
    forward_only = directional_evidence(values, two_sided=True).forward
    both = directional_evidence(values)
    assert both.forward == pytest.approx(forward_only)
