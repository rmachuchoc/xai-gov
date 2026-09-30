"""Adaptive conformal inference, change martingales and risk control."""

from __future__ import annotations

from xai_gov.conformal.adaptive import AciState, FixedLevel, build_level_controller
from xai_gov.conformal.calibrator import (
    Calibrator,
    SplitCalibrator,
    WeightedCalibrator,
    build_calibrator,
    conformal_quantile,
)
from xai_gov.conformal.detector import ConformalDetector, DetectorBank, build_detector
from xai_gov.conformal.martingale import (
    MixturePowerMartingale,
    NoMartingale,
    conformal_p_value,
)
from xai_gov.conformal.risk_control import ConformalRiskController, hoeffding_bound
from xai_gov.conformal.scores import (
    AbsoluteResidual,
    EwmaForecaster,
    RidgeForecaster,
    ScoreStream,
    StudentizedResidual,
    available_scores,
    build_forecaster,
    build_score,
)

__all__ = [
    "AbsoluteResidual",
    "AciState",
    "Calibrator",
    "ConformalDetector",
    "ConformalRiskController",
    "DetectorBank",
    "EwmaForecaster",
    "FixedLevel",
    "MixturePowerMartingale",
    "NoMartingale",
    "RidgeForecaster",
    "ScoreStream",
    "SplitCalibrator",
    "StudentizedResidual",
    "WeightedCalibrator",
    "available_scores",
    "build_calibrator",
    "build_detector",
    "build_forecaster",
    "build_level_controller",
    "build_score",
    "conformal_p_value",
    "conformal_quantile",
    "hoeffding_bound",
]
