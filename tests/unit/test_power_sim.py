"""Simulated power of the aGRAPA procedure."""

from __future__ import annotations

import pytest

from xai_gov.analysis.agrapa import run_e_process
from xai_gov.analysis.power_sim import (
    PowerSpec,
    crossing_times,
    plan_replicates,
    power_curve,
    required_n,
    simulate_hypothesis,
)

PILOT = tuple(0.5 + 0.06 * ((i % 7) - 3) / 3 for i in range(30))


def test_required_n_is_the_target_quantile_of_the_crossing_times() -> None:
    crossings: list[int | None] = [10, 20, 30, 40, None]
    assert required_n(crossings, target=0.6, n_max=100) == 30
    assert required_n(crossings, target=0.8, n_max=100) == 40
    assert required_n(crossings, target=1.0, n_max=100) is None
    assert required_n(crossings, target=0.8, n_max=35) is None


def test_the_power_curve_is_monotone() -> None:
    curve = power_curve([5, 9, None, 3, 12], [1, 4, 8, 10, 20])
    values = list(curve.values())
    assert values == sorted(values)
    assert curve[20] == pytest.approx(0.8)


def test_the_simulation_runs_the_real_procedure() -> None:
    """A zero-variance pilot gives identical paths: every crossing equals the
    one run_e_process gives on that constant stream."""
    spec = PowerSpec(hypothesis_id="H", differences=(0.3, 0.3, 0.3), two_sided=False)
    crossings = crossing_times(spec, shift=0.05, generator="bootstrap", sims=5,
                               n_max=400, threshold=240.0, seed=1)
    expected = run_e_process([0.05] * 400, threshold=240.0).first_crossing
    assert expected is not None
    assert crossings == [expected] * 5


def test_the_pilot_mean_is_removed_before_the_declared_effect_is_added() -> None:
    """Power is for the declared effect, not for whatever the pilot showed."""
    spec = PowerSpec(hypothesis_id="H", differences=PILOT, two_sided=False)
    result = simulate_hypothesis(spec, sims=100, n_max=600, seed=3)
    assert 200 < result["required_n"] < 600


def test_two_sided_is_simulated_in_both_directions() -> None:
    spec = PowerSpec(hypothesis_id="H", differences=PILOT, two_sided=True)
    result = simulate_hypothesis(spec, sims=60, n_max=600, seed=4)
    shifts = [entry["shift"] for entry in result["generators"]["bootstrap"]["by_shift"]]
    assert shifts == [0.05, -0.05]


def test_the_null_rejection_rate_stays_below_one_over_the_threshold() -> None:
    spec = PowerSpec(hypothesis_id="H", differences=PILOT, two_sided=False)
    result = simulate_hypothesis(spec, sims=300, n_max=300, seed=5)
    for generator in result["generators"].values():
        assert generator["null_rejection_rate"] <= result["null_bound"]


def test_the_campaign_takes_the_largest_requirement() -> None:
    plan = plan_replicates([
        {"hypothesis": "H1", "required_n": 250}, {"hypothesis": "H5", "required_n": 357},
    ])
    assert (plan["replicates"], plan["binding_hypothesis"]) == (357, "H5")
    unmet = plan_replicates([{"hypothesis": "H1", "required_n": None}])
    assert unmet["replicates"] is None and unmet["unmet_within_n_max"] == ["H1"]


def test_a_pilot_of_one_pair_is_refused() -> None:
    with pytest.raises(ValueError, match="two pairs"):
        PowerSpec(hypothesis_id="H", differences=(0.1,), two_sided=False)
