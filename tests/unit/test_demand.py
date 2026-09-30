"""Demand regimes: shifts must be declared, draws must be seeded."""

from __future__ import annotations

import numpy as np
import pytest

from xai_gov.simulation.demand import (
    DoubleShockDemand,
    HighVolatilityDemand,
    StableDemand,
    StructuralChangeDemand,
    available_regimes,
    build_demand,
)


def rng(seed: int = 3) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_stable_demand_is_exchangeable() -> None:
    process = StableDemand(name="stable", mean=10.0)
    assert process.exchangeable
    assert process.shift_periods == ()
    assert process.intensity(0) == process.intensity(25)


def test_shifting_regimes_declare_their_breaks() -> None:
    double = DoubleShockDemand(name="double_shock", mean=10.0)
    assert double.shift_periods == (10, 20)
    assert not double.exchangeable

    structural = StructuralChangeDemand(name="structural_change", mean=10.0, change_period=15)
    assert structural.shift_periods == (15,)


def test_double_shock_intensity_steps_at_the_declared_periods() -> None:
    process = DoubleShockDemand(
        name="double_shock", mean=10.0, first_multiplier=2.0, second_multiplier=0.5
    )
    assert process.intensity(9) == 10.0
    assert process.intensity(10) == 20.0
    assert process.intensity(20) == 5.0


def test_draws_are_non_negative_and_reproducible() -> None:
    process = StableDemand(name="stable", mean=10.0, dispersion=0.3)
    first = [process.draw(p, rng()) for p in range(5)]
    second = [process.draw(p, rng()) for p in range(5)]
    assert first == second
    assert all(value >= 0.0 for value in first)


def test_zero_dispersion_is_deterministic() -> None:
    process = StableDemand(name="stable", mean=7.0, dispersion=0.0)
    assert process.draw(0, rng()) == 7.0


def test_high_volatility_keeps_the_mean_and_moves_the_spread() -> None:
    process = HighVolatilityDemand(name="high_volatility", mean=10.0, dispersion=0.1)
    generator = rng(11)
    draws = [process.draw(p, generator) for p in range(400)]
    assert process.intensity(0) == process.intensity(3)
    # The mean is preserved within sampling error while the spread is not.
    assert abs(float(np.mean(draws)) - 10.0) < 1.0
    assert float(np.std(draws)) > 0.1


def test_structural_change_moves_both_mean_and_dispersion() -> None:
    process = StructuralChangeDemand(
        name="structural_change", mean=10.0, dispersion=0.1,
        change_period=10, new_mean_multiplier=2.0, new_dispersion=0.5,
    )
    generator = rng(5)
    before = [process.draw(p, generator) for p in range(9) for _ in range(40)]
    after = [process.draw(p, generator) for p in range(10, 20) for _ in range(40)]
    assert float(np.mean(after)) > float(np.mean(before)) * 1.5


def test_registry_lists_every_regime() -> None:
    assert set(available_regimes()) == {
        "stable", "growing", "double_shock", "high_volatility", "structural_change",
    }


def test_build_demand_rejects_unknown_regime_and_unknown_key() -> None:
    with pytest.raises(ValueError, match="unknown demand regime"):
        build_demand({"regime": "chaotic"})
    with pytest.raises(ValueError, match="does not accept"):
        build_demand({"regime": "stable", "shock_period": 4})
