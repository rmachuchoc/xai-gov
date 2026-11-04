"""The predictable aGRAPA e-process."""

from __future__ import annotations

import numpy as np
import pytest

from xai_gov.analysis.agrapa import (
    BET_CAP,
    AgrapaEProcess,
    familywise_threshold,
    run_e_process,
)


def test_the_first_bet_is_zero_and_the_second_is_the_plug_in() -> None:
    process = AgrapaEProcess()
    assert process.next_bet() == 0.0
    process.update(0.5)
    # mu_1 = 0.5 / 2, sigma2_1 = (1 + (0.5 - 0.25)^2) / 2.
    assert process.next_bet() == pytest.approx(0.25 / (0.53125 + 0.0625))


def test_a_bet_is_fixed_before_its_own_observation() -> None:
    """Predictable: changing x_t leaves lambda_t unchanged."""
    history = [0.1, -0.05, 0.2, 0.08]
    bets = []
    for last in (-0.9, 0.0, 0.9):
        process = AgrapaEProcess()
        for value in [*history, last]:
            process.update(value)
        bets.append(process.bets[-1])
    assert bets[0] == bets[1] == bets[2]


def test_bets_stay_in_zero_to_one_half() -> None:
    rng = np.random.default_rng(0)
    process = AgrapaEProcess()
    for value in rng.uniform(-1.0, 1.0, 500):
        process.update(float(value))
    assert all(0.0 <= bet <= BET_CAP for bet in process.bets)
    strong = AgrapaEProcess()
    for _ in range(50):
        strong.update(0.9)
    assert strong.bets[-1] == BET_CAP


def test_a_difference_outside_the_declared_range_is_refused() -> None:
    with pytest.raises(ValueError, match="declared range"):
        AgrapaEProcess(scale=1.0).update(1.2)
    AgrapaEProcess(scale=2.0).update(1.2)


def test_the_scale_is_the_declared_range_not_the_data() -> None:
    narrow = run_e_process([0.04, 0.06, 0.05] * 10, scale=1.0)
    wide = run_e_process([0.04, 0.06, 0.05] * 10, scale=0.1)
    assert narrow.final != wide.final


@pytest.mark.parametrize("mean", [0.0, -0.05])
def test_the_type_i_error_is_below_one_over_the_threshold(mean: float) -> None:
    """Ville: P(sup K >= c) <= 1/c under H0, checked on bounded null streams."""
    rng = np.random.default_rng(42)
    threshold = 20.0
    crossings = 0
    runs = 1000
    for _ in range(runs):
        stream = np.clip(rng.normal(mean, 0.3, 80), -1.0, 1.0)
        path = run_e_process([float(v) for v in stream], threshold=threshold)
        crossings += path.first_crossing is not None
    assert crossings / runs <= 1.0 / threshold


def test_a_real_effect_crosses_and_the_crossing_is_recorded() -> None:
    """With R = 1 and lambda <= 1/2 the log wealth grows at most ~lambda*0.05
    per replicate, so 240 needs more than ln(240)/0.025 ~ 220 replicates."""
    rng = np.random.default_rng(1)
    stream = [float(v) for v in np.clip(rng.normal(0.05, 0.04, 400), -1, 1)]
    path = run_e_process(stream, threshold=240.0)
    assert path.first_crossing is not None
    assert path.peak >= 240.0
    assert path.peak >= path.final
    assert path.first_crossing > 220


def test_two_sided_detects_either_direction_and_stays_valid_in_noise() -> None:
    rng = np.random.default_rng(5)
    up = [float(v) for v in np.clip(rng.normal(0.05, 0.04, 400), -1, 1)]
    assert run_e_process(up, two_sided=True, threshold=240.0).first_crossing is not None
    assert run_e_process([-v for v in up], two_sided=True, threshold=240.0).first_crossing
    crossings = 0
    for _ in range(500):
        noise = [float(v) for v in np.clip(rng.normal(0.0, 0.3, 80), -1, 1)]
        crossings += run_e_process(noise, two_sided=True, threshold=20.0).peak >= 20.0
    assert crossings / 500 <= 1.0 / 20.0


def test_the_familywise_threshold_for_six_hypotheses_is_240() -> None:
    assert familywise_threshold(hypotheses=6, alpha=0.05) == pytest.approx(240.0)
