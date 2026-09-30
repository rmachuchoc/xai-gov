"""Power analysis: what the campaign can and cannot see."""

from __future__ import annotations

import pytest

from xai_gov.analysis.power import (
    SEQUENTIAL_PENALTY,
    analyse_power,
    required_replicates,
)


def test_a_medium_effect_needs_more_than_thirty_paired_replicates() -> None:
    """The protocol's n >= 30 is a convention until it is checked; this is the
    check, and it comes out at 32."""
    assert required_replicates(effect_size=0.5) == 32


def test_a_larger_effect_needs_fewer_replicates() -> None:
    assert required_replicates(effect_size=1.0) < required_replicates(effect_size=0.5)


def test_a_smaller_effect_needs_many_more() -> None:
    assert required_replicates(effect_size=0.2) > 150


def test_the_sequential_test_costs_power() -> None:
    """The right to stop at any time is not free, and quoting the fixed-n
    number while running an anytime-valid analysis would promise a sensitivity
    the campaign does not have."""
    analysis = analyse_power(planned=30, effect_size=0.5)
    assert analysis.required_sequential > analysis.required_paired
    assert analysis.required_sequential == pytest.approx(
        analysis.required_paired * SEQUENTIAL_PENALTY, rel=0.05
    )


def test_the_protocols_thirty_replicates_fall_short() -> None:
    """The finding this module exists to produce, and it goes in the paper.

    The protocol's n >= 30 is short of what a medium effect needs even for a
    fixed-sample paired test (32), and well short once the analysis is
    anytime-valid (40). The convention was carried over from unpaired
    two-sample guidance; checking it against the design actually being run is
    what turns a citation into a number.
    """
    analysis = analyse_power(planned=30, effect_size=0.5)
    assert analysis.required_paired == 32
    assert analysis.required_sequential == 40
    assert analysis.adequate is False
    # And the campaign would only see effects this large or bigger.
    assert analysis.detectable_effect > 0.5


def test_forty_replicates_are_adequate() -> None:
    """Which is why the study runs at 40 rather than at the convention."""
    assert analyse_power(planned=40, effect_size=0.5).adequate is True


def test_a_generous_sample_is_adequate() -> None:
    assert analyse_power(planned=100, effect_size=0.5).adequate is True


def test_the_minimum_detectable_effect_is_reported() -> None:
    """The number a reader needs when a hypothesis comes back unsupported."""
    small = analyse_power(planned=10, effect_size=0.5)
    large = analyse_power(planned=200, effect_size=0.5)
    assert small.detectable_effect > large.detectable_effect
    assert large.detectable_effect < 0.5


def test_invalid_parameters_are_refused() -> None:
    with pytest.raises(ValueError, match="effect_size"):
        required_replicates(effect_size=0.0)
    with pytest.raises(ValueError, match="alpha"):
        required_replicates(alpha=1.0)
    with pytest.raises(ValueError, match="power"):
        required_replicates(power=0.0)


def test_the_payload_explains_the_paired_design() -> None:
    payload = analyse_power(planned=30).to_payload()
    assert "paired design" in payload["note"]
    assert payload["planned_replicates"] == 30
