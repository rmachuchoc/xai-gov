"""The conformal detector: composition of the four mechanisms.

One object per node assembles score stream, calibrator, level controller,
change martingale and — optionally — risk control, and emits the
`ConformalSignal` that the governance agent's belief filter consumes.

Order of operations within a period, which is where correctness lives:

1. score the new observation against the forecast made before seeing it;
2. read the threshold in force from the calibration set at the current level;
3. decide whether the score exceeds it;
4. compute the conformal p-value and place the martingale's bets;
5. feed the outcome back into the level controller;
6. only then add the score to the calibration set.

Step 6 comes last on purpose. Adding the score before thresholding lets the
observation influence the threshold it is judged against, which inflates
apparent calibration — the same leak the score stream avoids on the
forecasting side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from xai_gov.conformal.adaptive import LevelController, build_level_controller
from xai_gov.conformal.calibrator import Calibrator, build_calibrator
from xai_gov.conformal.martingale import (
    ChangeDetector,
    build_martingale,
    conformal_p_value,
)
from xai_gov.conformal.risk_control import ConformalRiskController, build_risk_controller
from xai_gov.conformal.scores import ScoreStream, build_score
from xai_gov.io.decision_record import ConformalSignal

SCHEME_NONE = "none"


@dataclass(slots=True)
class ConformalDetector:
    """Per-node adaptive conformal detector."""

    score_stream: ScoreStream
    calibrator: Calibrator
    level_controller: LevelController
    change_detector: ChangeDetector
    risk_controller: ConformalRiskController | None = None
    warmup: int = 10
    #: Periods whose scores warm the forecaster but are kept out of the
    #: calibration set. A learned forecaster starts with a scale it invented
    #: from its first observation, so its early residuals are large and its
    #: scores shrink as it converges. Admitting those into calibration means
    #: the threshold is a quantile of the model's own learning curve, and a
    #: later score — produced by a forecaster that now works — can never reach
    #: it. The first campaign flagged zero times in 45 periods for exactly this
    #: reason, at every window and every scale mode, which is why neither
    #: ablation moved the coverage error.
    burn_in: int = 12
    scheme_name: str = "aci"
    _calibration_snapshot: list[float] = field(default_factory=list, repr=False)
    periods: int = 0
    flags: int = 0
    burned: int = 0

    def __post_init__(self) -> None:
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative")
        if self.warmup < 2:
            raise ValueError(
                "warmup must be at least 2; a threshold from one score is not a "
                "quantile"
            )

    def observe(self, observation: float) -> ConformalSignal:
        """Score one observation and emit the signal for this period."""
        self.periods += 1
        score = self.score_stream.observe(observation)
        level = self.level_controller.level

        candidate = self.calibrator.threshold(level)
        # During burn-in the score trains the forecaster and is discarded: it
        # measures the model's inexperience rather than the environment.
        if self.periods <= self.burn_in:
            self.burned += 1
            self._calibration_snapshot = self.calibrator.scores()
            return ConformalSignal(
                score=round(score, 10),
                threshold=None,
                level=round(level, 10),
                flagged_ood=False,
                scheme=SCHEME_NONE,
                martingale_value=1.0,
                change_declared=False,
            )

        # During warmup no threshold is justified, so nothing is flagged and
        # the signal declares itself inactive. Flagging against an
        # uncalibrated threshold is what produces the saturated detector the
        # project set out to eliminate.
        if candidate is None or self.calibrator.size < self.warmup:
            self.calibrator.observe(score)
            self._calibration_snapshot = self.calibrator.scores()
            return ConformalSignal(
                score=round(score, 10),
                threshold=None,
                level=round(level, 10),
                flagged_ood=False,
                scheme=SCHEME_NONE,
                martingale_value=1.0,
                change_declared=False,
            )

        threshold: float = candidate
        # A risk-controlled threshold overrides the coverage-derived one when
        # it is certified and stricter: safety dominates efficiency.
        if self.risk_controller is not None:
            calibration = self.risk_controller.calibrate()
            if calibration.admissible and calibration.threshold is not None:
                threshold = min(threshold, calibration.threshold)
        exceeded = score > threshold

        p_value = conformal_p_value(score, self._calibration_snapshot)
        change = self.change_detector.update(p_value)

        self.level_controller.update(exceeded=exceeded)
        self.flags += int(exceeded)

        self.calibrator.observe(score)
        self._calibration_snapshot = self.calibrator.scores()

        return ConformalSignal(
            score=round(score, 10),
            threshold=round(threshold, 10),
            level=round(self.level_controller.level, 10),
            flagged_ood=exceeded,
            scheme=self.scheme_name,
            martingale_value=round(min(self.change_detector.value, 1e12), 6),
            change_declared=change or self.change_detector.change_declared,
        )

    def record_safety_loss(self, score: float, loss: float) -> None:
        """Feed a realized safety outcome to the risk controller."""
        if self.risk_controller is not None:
            self.risk_controller.observe(score, loss)

    @property
    def flag_rate(self) -> float:
        return self.flags / self.periods if self.periods else 0.0

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "scheme": self.scheme_name,
            "score": self.score_stream.score_fn.name,
            "periods": self.periods,
            "burn_in": self.burn_in,
            "burned": self.burned,
            "flags": self.flags,
            "flag_rate": round(self.flag_rate, 6),
            "calibrator": self.calibrator.to_payload(),
            "level_controller": self.level_controller.to_payload(),
            "change_detector": self.change_detector.to_payload(),
        }
        if self.risk_controller is not None:
            payload["risk_control"] = self.risk_controller.to_payload()
        return payload


def build_detector(config: dict[str, Any]) -> ConformalDetector:
    """Construct a detector from its configuration block."""
    score_name = str(config.get("score", "studentized_residual"))
    forecaster_params = config.get("forecaster", {})
    if not isinstance(forecaster_params, dict):
        raise ValueError("conformal 'forecaster' must be a mapping")

    from xai_gov.conformal.scores import build_forecaster

    forecaster = build_forecaster(forecaster_params)

    level_config = config.get("level", {})
    if not isinstance(level_config, dict):
        raise ValueError("conformal 'level' must be a mapping")
    controller = build_level_controller(level_config)

    return ConformalDetector(
        score_stream=ScoreStream(score_fn=build_score(score_name), forecaster=forecaster),
        calibrator=build_calibrator(config.get("calibrator", {})),
        level_controller=controller,
        change_detector=build_martingale(config.get("martingale", {})),
        risk_controller=build_risk_controller(config.get("risk_control", {})),
        warmup=int(config.get("warmup", 10)),
        burn_in=int(config.get("burn_in", 12)),
        scheme_name=str(level_config.get("scheme", "aci")),
    )


class DetectorBank:
    """One detector per node, built lazily from a shared configuration."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._detectors: dict[str, ConformalDetector] = {}

    def for_node(self, node_id: str) -> ConformalDetector:
        if node_id not in self._detectors:
            self._detectors[node_id] = build_detector(self._config)
        return self._detectors[node_id]

    def to_payload(self) -> dict[str, Any]:
        return {
            node_id: detector.to_payload()
            for node_id, detector in sorted(self._detectors.items())
        }

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(sorted(self._detectors))
