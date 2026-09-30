"""Verification and oversight, end to end.

The two claims this stage adds are RQ8 (what verified safety costs) and RQ7
(whether the human-machine team is genuinely complementary). Both are answered
from paired runs — same network, demand, seed, policy and descriptor, differing
only in the arm under test — because an unpaired comparison of safety machinery
measures the seed as much as the machinery.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from xai_gov.core.paths import Paths
from xai_gov.core.settings import load_settings
from xai_gov.io.hashchain import read_chain
from xai_gov.io.writers import DECISION_LOG_NAME, KPIS_NAME, SUMMARY_NAME, read_json
from xai_gov.orchestration.runner import run_experiment

pytestmark = pytest.mark.integration

EXPERIMENTS = Path("configs/experiments")


@pytest.fixture
def settings(project_root: Path, tmp_path: Path):  # type: ignore[no-untyped-def]
    base = load_settings(Paths(root=project_root))
    outputs = tmp_path / "outputs"
    (outputs / "runs").mkdir(parents=True)
    return replace(base, outputs_root=outputs)


def run(name: str, settings, project_root: Path) -> dict:  # type: ignore[no-untyped-def, type-arg]
    manifest = run_experiment(project_root / EXPERIMENTS / name, settings=settings)
    directory = settings.runs_root / manifest["run_name"]
    return {
        "manifest": manifest,
        "kpis": read_json(directory / KPIS_NAME),
        "summary": read_json(directory / SUMMARY_NAME),
        "log": directory / DECISION_LOG_NAME,
    }


def test_the_shielded_arm_runs_and_records_its_specification(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """The properties travel into the artifact as formulas: a reader must be
    able to see what 'verified' meant for this run."""
    artifacts = run("high_vol_shielded.yaml", settings, project_root)
    assert artifacts["manifest"]["status"] == "completed"
    shield = artifacts["summary"]["governance"]["shield"]
    formulas = [p["formula"] for p in shield["specification"]["properties"]]
    assert any(formula.startswith("P<=") for formula in formulas)
    assert shield["decisions"] > 0


def test_the_shield_can_raise_a_quantity_governance_zeroed(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """The stage's central capability: the agent can now repair rather than
    only refuse. A veto that would starve the node is corrected upward."""
    artifacts = run("high_vol_shielded.yaml", settings, project_root)
    repairs = [
        entry.payload
        for entry in read_chain(artifacts["log"])
        if entry.kind == "decision"
        and entry.payload["shield"]["activated"]
        and entry.payload["final"]["quantity"] > entry.payload["proposed"]["quantity"]
    ]
    if repairs:
        assert repairs[0]["shield"]["property_id"]
        assert "shield substituted" in repairs[0]["governance"]["rationale"]


def test_every_substitution_names_the_property_that_forced_it(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    artifacts = run("high_vol_shielded.yaml", settings, project_root)
    for entry in read_chain(artifacts["log"]):
        if entry.kind == "decision" and entry.payload["shield"]["activated"]:
            assert entry.payload["shield"]["property_id"] is not None


def test_the_price_of_verified_safety_is_computable(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """RQ8. The unshielded arm is instrumented identically, so the difference
    is attributable to the shield alone."""
    shielded = run("high_vol_shielded.yaml", settings, project_root)
    unshielded = run("high_vol_unshielded.yaml", settings, project_root)

    for key in ("network", "demand", "policy"):
        assert shielded["summary"][key] == unshielded["summary"][key]
    assert shielded["summary"]["master_seed"] == unshielded["summary"]["master_seed"]

    assert unshielded["summary"]["governance"]["shield"]["activations"] == 0
    # Monitors run in both arms: instrumenting only one would make them
    # incomparable in exactly the dimension being measured.
    assert unshielded["summary"]["governance"]["monitor"]["periods"] > 0


def test_the_monitor_reports_whether_the_guarantee_holds(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """Verification proves a property of an abstraction; the monitor is what
    watches whether the concrete run stayed inside it."""
    artifacts = run("high_vol_shielded.yaml", settings, project_root)
    monitor = artifacts["summary"]["governance"]["monitor"]
    assert monitor["periods"] > 0
    assert isinstance(monitor["guarantee_intact"], bool)
    assert isinstance(monitor["violations"], int)


def test_the_guarantees_layer_carries_shield_activity(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """Coverage and verification answer the same question — whether a stated
    guarantee still holds — so they are reported together."""
    artifacts = run("high_vol_shielded.yaml", settings, project_root)
    layer = artifacts["kpis"]["guarantees"]
    assert layer["status"] == "available"
    assert "shield_activation_rate" in layer
    if layer["shield_activation_rate"] > 0.0:
        # A rate without a magnitude cannot be checked against Proposition 2.
        assert layer["mean_substitution"] > 0.0
        assert layer["shield_properties"]


def test_the_delegated_arm_spends_human_attention(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    artifacts = run("high_vol_delegated.yaml", settings, project_root)
    oversight = artifacts["summary"]["governance"]["oversight"]
    assert oversight["supervisor"] is not None
    assert oversight["deferral"]["policy"] == "learned"
    supervisor = oversight["supervisor"]
    assert supervisor["attention_spent"] <= supervisor["attention_budget"]


def test_an_ungoverned_arm_carries_no_supervisor(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """A supervisor who is never consulted is not a neutral default but a claim
    that oversight exists."""
    artifacts = run("high_vol_shielded.yaml", settings, project_root)
    assert artifacts["summary"]["governance"]["oversight"]["supervisor"] is None


def test_complementarity_is_measured_and_may_be_negative(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """RQ7. The protocol requires this to be able to fail, so the test asserts
    that the verdict is reported — not that it is favourable."""
    artifacts = run("high_vol_delegated.yaml", settings, project_root)
    ledger = artifacts["summary"]["governance"]["oversight"]["complementarity"]
    assert isinstance(ledger["strict_complementarity"], bool)
    assert set(ledger["team_accuracy"]) <= {"nominal", "drift", "disruption"}
    for regime, rate in ledger["team_accuracy"].items():
        assert 0.0 <= rate <= 1.0, regime


def test_learned_and_threshold_oversight_are_a_valid_ablation(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """Same supervisor, different deferral policy: whatever separates them is
    what the learning adds."""
    learned = run("high_vol_delegated.yaml", settings, project_root)
    fixed = run("high_vol_threshold_oversight.yaml", settings, project_root)

    learned_oversight = learned["summary"]["governance"]["oversight"]
    fixed_oversight = fixed["summary"]["governance"]["oversight"]
    assert learned_oversight["deferral"]["policy"] == "learned"
    assert fixed_oversight["deferral"]["policy"] == "threshold"
    assert (
        learned_oversight["supervisor"]["attention_budget"]
        == fixed_oversight["supervisor"]["attention_budget"]
    )


def test_the_supervisor_is_reproducible_across_reruns(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    """A supervisor seeded from the clock would make governed runs
    irreproducible while the artifacts still looked reproducible."""
    first = run("high_vol_delegated.yaml", settings, project_root)
    second = run("high_vol_delegated.yaml", settings, project_root)
    assert (
        first["summary"]["governance"]["oversight"]["supervisor"]["realized_accuracy"]
        == second["summary"]["governance"]["oversight"]["supervisor"]["realized_accuracy"]
    )


def test_every_shipped_experiment_still_runs(settings, project_root: Path) -> None:  # type: ignore[no-untyped-def]
    for path in sorted((project_root / EXPERIMENTS).glob("*.yaml")):
        manifest = run_experiment(path, settings=settings)
        assert manifest["status"] == "completed", path.name
        assert manifest["chain_verified"] is True, path.name
