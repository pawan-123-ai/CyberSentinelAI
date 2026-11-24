"""
security/risk_analyzer.py

Risk scoring engine for CyberSentinel AI.

Responsibilities:
    - Convert a Prediction (from ai/predictor.py) into a concrete
      0-100 risk score and a SeverityLevel (Safe/Low/Medium/High/Critical).
    - Weight the score by the inherent danger of the predicted attack
      class (a DDoS classification is treated as more dangerous than a
      Port Scan classification at equal confidence).
    - Apply a bounded "traffic intensity" bonus when flow-level
      features (packet rate, byte rate) suggest an active flood,
      applied only to non-Normal classifications so legitimate
      high-throughput traffic is never penalized.
    - Maintain a smoothed rolling risk score plus a short history
      buffer, feeding the dashboard's "Risk Score" card and future
      risk-trend graph.

This module deliberately does NOT:
    - Persist anything to the database (that's security/threat_detector.py).
    - Create or manage user-facing Alert records (that's security/alert_manager.py).
    - Decide UI colors/styling for severity levels (that's the UI layer,
      which can map SeverityLevel -> ThemeConfig colors as it sees fit).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

from ai.predictor import NORMAL_CLASS_NAME, Prediction
from config.settings import AIConfig, UIConfig
from utils.logger import get_logger, get_threat_logger

logger = get_logger(__name__)
threat_logger = get_threat_logger()


class SeverityLevel(str, Enum):
    """
    Discrete risk severity bands derived from a numeric risk score.
    Inherits from str so values serialize cleanly (e.g., to MongoDB
    or JSON) without extra conversion.
    """
    SAFE = "Safe"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# CRITICAL is a band above AIConfig.RISK_SCORE_HIGH for the most
# extreme scores (e.g., high-confidence DDoS with an intensity bonus).
# Kept local to this module rather than added to AIConfig since it's
# an internal scoring-band detail, not a cross-cutting app setting.
RISK_SCORE_CRITICAL: int = 90

# Base severity weight (0-100) per attack class, representing how
# dangerous a *fully confident* classification of that type is. The
# final risk score scales this by the model's actual confidence.
# Every entry in AIConfig.ATTACK_CLASSES must have a weight here.
ATTACK_SEVERITY_WEIGHTS: Dict[str, float] = {
    "Normal": 0.0,
    "Probe": 35.0,
    "Port Scan": 45.0,
    "R2L": 65.0,
    "U2R": 75.0,
    "DoS": 85.0,
    "DDoS": 100.0,
}

# Fallback weight used only if a predicted class is somehow missing
# from ATTACK_SEVERITY_WEIGHTS (should not happen with a fixed class
# list, but guards against silent under-scoring of an unknown class).
_DEFAULT_UNKNOWN_CLASS_WEIGHT: float = 50.0

# Traffic-intensity bonus thresholds and point values. These add a
# small amount of extra risk when a flow's packet/byte rate is
# unusually high, on top of the base class-weighted score.
PACKET_RATE_BONUS_THRESHOLD: float = 500.0     # packets/sec
BYTE_RATE_BONUS_THRESHOLD: float = 100_000.0   # bytes/sec
INTENSITY_BONUS_POINTS: float = 5.0            # points added per triggered threshold

# Smoothing factor for the rolling "current risk level" shown on the
# dashboard, so a single noisy packet doesn't make the Risk Score
# card jump wildly. Higher = more reactive, lower = smoother.
EMA_SMOOTHING_ALPHA: float = 0.3


@dataclass
class RiskAssessment:
    """
    The structured result of scoring a single Prediction. This is the
    data contract consumed by security/threat_detector.py (to decide
    whether/how to act) and security/alert_manager.py (to decide
    whether/how to notify the user).
    """
    timestamp: datetime
    predicted_class: str
    confidence: float
    risk_score: int                  # 0-100
    severity: SeverityLevel
    contributing_factors: List[str]
    prediction: Prediction = field(repr=False)

    def to_dict(self) -> dict:
        """
        Converts this assessment into a plain dictionary, suitable
        for merging into a threat record before persistence via
        DatabaseManager.insert_threat_record() (performed by
        security/threat_detector.py, not this module).
        """
        return {
            "timestamp": self.timestamp,
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "risk_score": self.risk_score,
            "severity": self.severity.value,
            "contributing_factors": self.contributing_factors,
        }


class RiskAnalyzer:
    """
    Scores Prediction instances into RiskAssessments and maintains a
    smoothed, rolling "current risk level" suitable for a live
    dashboard stat card, plus a short history buffer for a future
    risk-trend graph.

    Usage:
        analyzer = RiskAnalyzer()
        assessment = analyzer.analyze(prediction)
        current_level = analyzer.get_current_risk_score()
    """

    def __init__(self, history_size: int = UIConfig.MAX_GRAPH_DATA_POINTS) -> None:
        """
        Args:
            history_size: Maximum number of recent risk scores to
                retain for trend display. Defaults to
                UIConfig.MAX_GRAPH_DATA_POINTS so the buffer length
                naturally matches the dashboard's graph point count.
        """
        self._lock = threading.Lock()
        self._history: Deque[int] = deque(maxlen=history_size)
        self._smoothed_score: float = 0.0
        self._has_smoothed_value: bool = False

        self._missing_weight_warned: set[str] = set()

        logger.info("RiskAnalyzer initialized (history_size=%d).", history_size)

    def analyze(self, prediction: Prediction) -> RiskAssessment:
        """
        Scores a single Prediction into a RiskAssessment and folds the
        resulting score into the rolling smoothed average and history
        buffer.

        Args:
            prediction: The AI prediction to score.

        Returns:
            A populated RiskAssessment instance.
        """
        with self._lock:
            base_weight = self._resolve_base_weight(prediction.predicted_class)
            raw_score = base_weight * prediction.confidence

            intensity_bonus, intensity_reasons = self._compute_intensity_bonus(prediction)
            total_score = raw_score + intensity_bonus

            risk_score = int(round(min(max(total_score, 0.0), 100.0)))
            severity = self._severity_from_score(risk_score)

            contributing_factors = self._build_contributing_factors(
                prediction, base_weight, intensity_reasons
            )

            self._record_score(risk_score)

        assessment = RiskAssessment(
            timestamp=datetime.now(timezone.utc),
            predicted_class=prediction.predicted_class,
            confidence=prediction.confidence,
            risk_score=risk_score,
            severity=severity,
            contributing_factors=contributing_factors,
            prediction=prediction,
        )

        if severity in (SeverityLevel.HIGH, SeverityLevel.CRITICAL):
            threat_logger.warning(
                "%s risk (%d/100): %s from %s:%s -> %s:%s (confidence %.1f%%)",
                severity.value,
                risk_score,
                prediction.predicted_class,
                prediction.source_packet.src_ip,
                prediction.source_packet.src_port,
                prediction.source_packet.dst_ip,
                prediction.source_packet.dst_port,
                prediction.confidence * 100,
            )

        return assessment

    def analyze_batch(self, predictions: List[Prediction]) -> List[RiskAssessment]:
        """
        Scores multiple Predictions in sequence. Provided for API
        symmetry with ai.predictor.ThreatPredictor.predict_batch(),
        since app/core/engine.py may process short bursts of
        predictions together between UI refresh ticks.

        Args:
            predictions: A list of predictions to score.

        Returns:
            A list of RiskAssessment instances, in the same order as
            the input. Empty list if predictions is empty.
        """
        return [self.analyze(prediction) for prediction in predictions]

    def get_current_risk_score(self) -> int:
        """
        Returns the smoothed, rolling "current risk level" for
        display on the dashboard's Risk Score stat card. Uses an
        exponential moving average so a single noisy packet doesn't
        cause the displayed value to jump erratically.

        Returns:
            An integer risk score from 0-100, or 0 if no assessments
            have been made yet.
        """
        with self._lock:
            if not self._has_smoothed_value:
                return 0
            return int(round(self._smoothed_score))

    def get_current_severity(self) -> SeverityLevel:
        """
        Returns:
            The SeverityLevel corresponding to the current smoothed
            risk score, for the dashboard's overall status indicator.
        """
        return self._severity_from_score(self.get_current_risk_score())

    def get_risk_history(self) -> List[int]:
        """
        Returns:
            A list of the most recent risk scores (oldest first), up
            to history_size entries, for plotting a risk-trend graph.
        """
        with self._lock:
            return list(self._history)

    def reset(self) -> None:
        """
        Clears the rolling smoothed score and history buffer. Should
        be called when monitoring is stopped and later restarted, so
        a new session starts from a clean risk baseline rather than
        carrying over the previous session's trend.
        """
        with self._lock:
            self._history.clear()
            self._smoothed_score = 0.0
            self._has_smoothed_value = False
        logger.info("RiskAnalyzer state reset.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_base_weight(self, predicted_class: str) -> float:
        """
        Looks up the base severity weight for a predicted class,
        warning (once per class) if an unrecognized class name is
        encountered, and falling back to a mid-range default weight
        rather than under-scoring an unknown/unexpected classification.

        Args:
            predicted_class: The class name predicted by the AI model.

        Returns:
            The base severity weight (0-100) for the class.
        """
        weight = ATTACK_SEVERITY_WEIGHTS.get(predicted_class)
        if weight is None:
            if predicted_class not in self._missing_weight_warned:
                logger.warning(
                    "No severity weight defined for predicted class '%s'; "
                    "using default weight %.0f.",
                    predicted_class,
                    _DEFAULT_UNKNOWN_CLASS_WEIGHT,
                )
                self._missing_weight_warned.add(predicted_class)
            return _DEFAULT_UNKNOWN_CLASS_WEIGHT
        return weight

    @staticmethod
    def _compute_intensity_bonus(prediction: Prediction) -> Tuple[float, List[str]]:
        """
        Computes a bounded bonus added to the base score when a flow's
        packet/byte rate is unusually high, but ONLY for traffic
        already classified as non-Normal. This avoids penalizing
        legitimate high-throughput Normal traffic (e.g., a large file
        download or video stream) purely for being fast.

        Args:
            prediction: The prediction being scored.

        Returns:
            A tuple of (bonus_points, human_readable_reasons).
        """
        if prediction.predicted_class == NORMAL_CLASS_NAME:
            return 0.0, []

        bonus = 0.0
        reasons: List[str] = []
        features = prediction.feature_vector

        if features.packet_rate >= PACKET_RATE_BONUS_THRESHOLD:
            bonus += INTENSITY_BONUS_POINTS
            reasons.append(
                f"Elevated packet rate observed ({features.packet_rate:.0f} pkt/s)."
            )

        if features.byte_rate >= BYTE_RATE_BONUS_THRESHOLD:
            bonus += INTENSITY_BONUS_POINTS
            reasons.append(
                f"Elevated byte rate observed ({features.byte_rate:.0f} B/s)."
            )

        return bonus, reasons

    @staticmethod
    def _build_contributing_factors(
        prediction: Prediction,
        base_weight: float,
        intensity_reasons: List[str],
    ) -> List[str]:
        """
        Builds a short, human-readable list of reasons behind the risk
        score, intended for display on the Threat Detection page or
        in exported logs.

        Args:
            prediction: The prediction being scored.
            base_weight: The resolved base severity weight used.
            intensity_reasons: Any traffic-intensity bonus reasons.

        Returns:
            A list of human-readable contributing factor strings.
        """
        if prediction.predicted_class == NORMAL_CLASS_NAME:
            return ["Traffic pattern matches expected benign behavior."]

        factors = [
            f"Classified as '{prediction.predicted_class}' with "
            f"{prediction.confidence * 100:.1f}% model confidence.",
            f"Base severity weight for '{prediction.predicted_class}': "
            f"{base_weight:.0f}/100.",
        ]
        factors.extend(intensity_reasons)
        return factors

    @staticmethod
    def _severity_from_score(risk_score: int) -> SeverityLevel:
        """
        Maps a numeric risk score to a discrete SeverityLevel band,
        using AIConfig's medium/high thresholds plus a local critical
        threshold for the most extreme scores.

        Args:
            risk_score: A risk score from 0-100.

        Returns:
            The corresponding SeverityLevel.
        """
        if risk_score <= 0:
            return SeverityLevel.SAFE
        if risk_score >= RISK_SCORE_CRITICAL:
            return SeverityLevel.CRITICAL
        if risk_score >= AIConfig.RISK_SCORE_HIGH:
            return SeverityLevel.HIGH
        if risk_score >= AIConfig.RISK_SCORE_MEDIUM:
            return SeverityLevel.MEDIUM
        return SeverityLevel.LOW

    def _record_score(self, risk_score: int) -> None:
        """
        Folds a new risk score into the history buffer and updates
        the exponentially smoothed "current risk level".

        Must be called while holding self._lock.

        Args:
            risk_score: The freshly computed risk score to record.
        """
        self._history.append(risk_score)

        if not self._has_smoothed_value:
            self._smoothed_score = float(risk_score)
            self._has_smoothed_value = True
        else:
            self._smoothed_score = (
                EMA_SMOOTHING_ALPHA * risk_score
                + (1 - EMA_SMOOTHING_ALPHA) * self._smoothed_score
            )


# ==========================================================================
# Module-level singleton accessor
# ==========================================================================

_risk_analyzer_instance: Optional[RiskAnalyzer] = None


def get_risk_analyzer() -> RiskAnalyzer:
    """
    Returns the shared, application-wide RiskAnalyzer instance,
    creating it on first call.

    Mirrors the singleton pattern used by utils.logger.get_logger(),
    database.db_manager.get_database_manager(), ai.model_loader.get_model_bundle(),
    and ai.predictor.get_predictor() — all modules should call this
    rather than instantiating RiskAnalyzer directly, so the whole
    application shares one rolling risk score and history buffer.

    Returns:
        The singleton RiskAnalyzer instance.
    """
    global _risk_analyzer_instance
    if _risk_analyzer_instance is None:
        _risk_analyzer_instance = RiskAnalyzer()
    return _risk_analyzer_instance