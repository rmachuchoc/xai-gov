"""Seed derivation is the basis of bit-for-bit reproducibility."""

from __future__ import annotations

from xai_gov.core.seeds import SeedBundle, derive_seed


def test_derivation_is_deterministic_and_label_specific() -> None:
    assert derive_seed(42, "demand") == derive_seed(42, "demand")
    assert derive_seed(42, "demand") != derive_seed(42, "lead_time")
    assert derive_seed(42, "demand") != derive_seed(43, "demand")


def test_generators_are_stable_regardless_of_construction_order() -> None:
    first = SeedBundle(master_seed=7)
    a1 = first.generator("demand").random()
    b1 = first.generator("lead_time").random()

    second = SeedBundle(master_seed=7)
    b2 = second.generator("lead_time").random()
    a2 = second.generator("demand").random()

    assert (a1, b1) == (a2, b2)


def test_adding_a_component_does_not_shift_existing_streams() -> None:
    baseline = SeedBundle(master_seed=99).generator("demand").random()
    with_extra = SeedBundle(master_seed=99)
    with_extra.generator("new_component").random()
    assert with_extra.generator("demand").random() == baseline
