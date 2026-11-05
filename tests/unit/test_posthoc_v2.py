"""Post-hoc economics: beta as the return, like-for-like baselines."""

from __future__ import annotations

from pathlib import Path

from xai_gov.analysis.campaign import CampaignCell, CellResult
from xai_gov.analysis.posthoc_v2 import comparison_key, economics


def config(network: str, policy: str, periods: int = 60) -> dict:  # type: ignore[type-arg]
    return {"network": {"name": network}, "policy": {"family": policy},
            "demand": {"regime": "high_volatility"}, "experiment": {"periods": periods}}


def test_a_baseline_must_share_network_policy_demand_and_horizon() -> None:
    base = comparison_key(config("three_echelon", "opaque_dro"))
    assert comparison_key(config("three_echelon", "opaque_dro")) == base
    assert comparison_key(config("five_echelon", "opaque_dro")) != base
    assert comparison_key(config("three_echelon", "qlearning")) != base
    assert comparison_key(config("three_echelon", "opaque_dro", periods=40)) != base


def cell(arm: str, replicate: int, beta: float, cost: float) -> CellResult:
    return CellResult(
        cell=CampaignCell(arm=arm, experiment=Path(f"{arm}.yaml"), replicate=replicate, seed=1),
        run_name=f"{arm}-{replicate}",
        kpis={"operational": {"fill_rate": beta},
              "governance": {"total_intervention_cost": cost, "decisions": 180}},
        included=True,
    )


def test_economics_uses_beta_and_the_matching_baseline_only() -> None:
    results = (
        *(cell("projection", r, 0.6, 10.0) for r in range(3)),
        *(cell("ungoverned", r, 0.7, 0.0) for r in range(3)),
        *(cell("five_projection", r, 0.5, 10.0) for r in range(3)),
    )
    keys = {"projection": "A", "ungoverned": "A", "five_projection": "B"}
    payload = economics(results, arm_keys=keys, baselines={"A": "ungoverned"})
    strategic = payload["strategic"]
    assert payload["return_indicator"] == "operational.fill_rate"
    assert strategic["arms"]["projection"]["baseline"] == "ungoverned"
    assert strategic["arms"]["projection"]["rgd"] == round((0.6 - 0.7) / 10.0, 6)
    assert "five_projection" in strategic["unmatched"]
    assert "five_projection" not in strategic["arms"]
