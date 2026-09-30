"""Runtime monitors: what verification cannot cover."""

from __future__ import annotations

from tests.factories import state
from xai_gov.verification.monitors import NoMonitor, RuntimeMonitor
from xai_gov.verification.specification import (
    Operator,
    SafetyProperty,
    Specification,
)


def persistent(horizon: int = 2) -> Specification:
    return Specification(
        (
            SafetyProperty(
                property_id="P",
                operator=Operator.PERSISTENT,
                predicate="stockout",
                bound=0.02,
                horizon=horizon,
            ),
        )
    )


def test_a_healthy_run_reports_no_violations() -> None:
    monitor = RuntimeMonitor(specification=persistent())
    for period in range(10):
        assert monitor.observe(state(period=period, inventory=60.0)) == ()
    assert monitor.violation_rate == 0.0


def test_a_persistent_property_is_violated_over_a_trace_not_a_state() -> None:
    """One bad period is not a violation of a persistence property; that is
    what makes it a temporal claim."""
    monitor = RuntimeMonitor(specification=persistent(horizon=3))
    assert monitor.observe(state(period=0, inventory=0.0)) == ()
    assert monitor.observe(state(period=1, inventory=0.0)) == ()
    violations = monitor.observe(state(period=2, inventory=0.0))
    assert violations
    assert violations[0].property_id == "P"
    assert "3 consecutive" in violations[0].detail


def test_recovery_clears_the_run() -> None:
    monitor = RuntimeMonitor(specification=persistent(horizon=3))
    monitor.observe(state(period=0, inventory=0.0))
    monitor.observe(state(period=1, inventory=0.0))
    monitor.observe(state(period=2, inventory=50.0))
    assert monitor.observe(state(period=3, inventory=0.0)) == ()


def test_violations_are_counted_by_property() -> None:
    monitor = RuntimeMonitor(specification=persistent(horizon=1))
    for period in range(4):
        monitor.observe(state(period=period, inventory=0.0))
    assert monitor.by_property() == {"P": 4}


def test_an_unmodelled_transition_lapses_the_guarantee() -> None:
    """A run that reports its properties as verified while having diverged from
    the model they were verified on is making a claim it cannot support."""
    monitor = RuntimeMonitor(specification=persistent())
    monitor.observe(state(period=0, inventory=80.0), known_transitions=set())
    monitor.observe(state(period=1, inventory=0.0), known_transitions=set())
    assert monitor.model_divergences == 1
    assert monitor.guarantee_intact is False


def test_a_run_inside_the_verified_model_keeps_its_guarantee() -> None:
    monitor = RuntimeMonitor(specification=persistent())
    abstraction = monitor.abstraction
    healthy = abstraction.abstract(80.0, 0.0).to_key()
    known = {(healthy, healthy)}
    monitor.observe(state(period=0, inventory=80.0), known_transitions=known)
    monitor.observe(state(period=1, inventory=80.0), known_transitions=known)
    assert monitor.model_divergences == 0
    assert monitor.guarantee_intact is True


def test_undischarged_properties_are_named_in_the_payload() -> None:
    monitor = RuntimeMonitor(
        specification=persistent(), undischarged=frozenset({"CAP"})
    )
    assert monitor.to_payload()["monitored_only"] == ["CAP"]


def test_the_control_arm_reports_no_guarantee() -> None:
    """Absence of monitoring is not evidence of safety, so it reports False
    rather than an unqualified True."""
    monitor = NoMonitor()
    assert monitor.observe(state()) == ()
    assert monitor.guarantee_intact is False
