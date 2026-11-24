"""
security/threat_detector.py

Per-packet threat detection pipeline for CyberSentinel AI.

Responsibilities:
    - Tie together ai/feature_extractor.py, ai/predictor.py, and
      security/risk_analyzer.py into a single process_packet() call
      that turns a raw CapturedPacket into a full DetectionResult
      (features -> prediction -> risk score).
    - process_packet() matches network.packet_capture.PacketCallback's
      signature exactly, so it can be registered directly with
      PacketCapture.register_callback() by app/core/engine.py.
    - Persist threat-worthy detections (and a corresponding security
      log entry) via database.db_manager.DatabaseManager.
    - Maintain a small in-memory buffer of recent detections for
      instant UI display (Recent Alerts / Recent Activity panels)
      without requiring a database round-trip.
    - Dispatch every DetectionResult to registered callbacks, giving
      the UI layer and the future security/alert_manager.py a single,
      decoupled extension point for live updates and notifications.

This module deliberately contains no AI model logic (ai/predictor.py)
and no scoring logic (security/risk_analyzer.py) of its own — it only
orchestrates the pipeline and decides what gets persisted.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Deque, List, Optional

from ai.feature_extractor import FeatureExtractor
from ai.predictor import Prediction, ThreatPredictor, get_predictor
from database.db_manager import DatabaseManager, get_database_manager
from network.packet_capture import CapturedPacket
from security.risk_analyzer import RiskAnalyzer, RiskAssessment, SeverityLevel, get_risk_analyzer
from utils.logger import get_logger

logger = get_logger(__name__)

# Maps risk severity to a standard logging level string, used when
# writing security log entries to the "security_logs" collection so
# the future Logs page can filter/color entries by level.
_SEVERITY_TO_LOG_LEVEL: dict[SeverityLevel, str] = {
    SeverityLevel.SAFE: "INFO",
    SeverityLevel.LOW: "INFO",
    SeverityLevel.MEDIUM: "WARNING",
    SeverityLevel.HIGH: "ERROR",
    SeverityLevel.CRITICAL: "CRITICAL",
}

# Default size of the in-memory recent-detections ring buffer.
_DEFAULT_RECENT_DETECTIONS_LIMIT: int = 100


@dataclass
class DetectionResult:
    """
    The complete output of running one packet through the full
    detection pipeline: its extracted-feature-based AI prediction and
    its computed risk assessment, bundled with the originating packet.

    This is the data contract dispatched to every registered callback
    (UI updates, future alert_manager) and is what gets summarized
    for database persistence when is_threat is True.
    """
    timestamp: datetime
    packet: CapturedPacket = field(repr=False)
    prediction: Prediction = field(repr=False)
    risk_assessment: RiskAssessment = field(repr=False)
    is_threat: bool
    persisted: bool = False
    threat_record_id: Optional[str] = None

    def to_summary_dict(self) -> dict:
        """
        Converts this result into a lightweight, flat dictionary
        suitable for UI display (e.g., a "Recent Activity" table row)
        without needing to reach into the nested prediction/risk
        objects.

        Returns:
            A flat dictionary of the most UI-relevant fields.
        """
        return {
            "timestamp": self.timestamp,
            "src_ip": self.packet.src_ip,
            "dst_ip": self.packet.dst_ip,
            "src_port": self.packet.src_port,
            "dst_port": self.packet.dst_port,
            "protocol": self.packet.protocol,
            "packet_size": self.packet.size,
            "predicted_class": self.prediction.predicted_class,
            "confidence": round(self.prediction.confidence, 4),
            "risk_score": self.risk_assessment.risk_score,
            "severity": self.risk_assessment.severity.value,
            "is_threat": self.is_threat,
            "persisted": self.persisted,
        }


@dataclass
class DetectionStats:
    """
    Rolling counters owned by ThreatDetector itself, distinct from
    ai.predictor.PredictionStats (which tracks AI-level confidence
    stats). These specifically track pipeline/persistence outcomes.
    """
    total_processed: int = 0
    total_threats_detected: int = 0
    total_threats_persisted: int = 0

    def to_dict(self) -> dict:
        """Returns a plain-dict snapshot for display or logging."""
        return {
            "total_processed": self.total_processed,
            "total_threats_detected": self.total_threats_detected,
            "total_threats_persisted": self.total_threats_persisted,
        }

    def reset(self) -> None:
        """Clears all counters, e.g. when a monitoring session restarts."""
        self.total_processed = 0
        self.total_threats_detected = 0
        self.total_threats_persisted = 0


# Type alias for functions that consume a completed DetectionResult.
DetectionCallback = Callable[[DetectionResult], None]


class ThreatDetector:
    """
    Orchestrates the full per-packet detection pipeline: feature
    extraction -> AI prediction -> risk scoring -> persistence ->
    callback dispatch.

    Usage:
        detector = ThreatDetector()
        detector.register_callback(my_ui_update_handler)
        packet_capture.register_callback(detector.process_packet)
    """

    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        predictor: Optional[ThreatPredictor] = None,
        risk_analyzer: Optional[RiskAnalyzer] = None,
        db_manager: Optional[DatabaseManager] = None,
        recent_detections_limit: int = _DEFAULT_RECENT_DETECTIONS_LIMIT,
    ) -> None:
        """
        Args:
            feature_extractor: Optional explicit FeatureExtractor. If
                None, a dedicated new instance is created (flow state
                should not be shared across unrelated detector instances).
            predictor: Optional explicit ThreatPredictor. Defaults to
                the shared application singleton via ai.predictor.get_predictor().
            risk_analyzer: Optional explicit RiskAnalyzer. Defaults to
                the shared application singleton via
                security.risk_analyzer.get_risk_analyzer().
            db_manager: Optional explicit DatabaseManager. Defaults to
                the shared application singleton via
                database.db_manager.get_database_manager().
            recent_detections_limit: Maximum number of recent
                DetectionResults to retain in memory for instant UI access.
        """
        self._feature_extractor = feature_extractor or FeatureExtractor()
        self._predictor = predictor or get_predictor()
        self._risk_analyzer = risk_analyzer or get_risk_analyzer()
        self._db_manager = db_manager or get_database_manager()

        self._callbacks: List[DetectionCallback] = []
        self._lock = threading.Lock()

        self.stats = DetectionStats()
        self._recent_detections: Deque[DetectionResult] = deque(maxlen=recent_detections_limit)

        logger.info("ThreatDetector initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_callback(self, callback: DetectionCallback) -> None:
        """
        Registers a function to be invoked with every completed
        DetectionResult (threat or not), regardless of whether it was
        persisted. Multiple callbacks may be registered (e.g., a
        dashboard live-feed handler and a future alert_manager handler).

        Args:
            callback: A callable accepting a single DetectionResult argument.
        """
        self._callbacks.append(callback)
        logger.debug(
            "Registered detection callback: %s",
            getattr(callback, "__qualname__", callback),
        )

    def process_packet(self, packet: CapturedPacket) -> DetectionResult:
        """
        Runs a single captured packet through the full detection
        pipeline: feature extraction, AI prediction, risk scoring,
        conditional persistence, and callback dispatch.

        This method's signature intentionally matches
        network.packet_capture.PacketCallback, so it can be passed
        directly to PacketCapture.register_callback() without any
        adapter code:

            capture.register_callback(threat_detector.process_packet)

        Args:
            packet: The captured packet to analyze.

        Returns:
            The completed DetectionResult for this packet.
        """
        feature_vector = self._feature_extractor.extract(packet)
        prediction = self._predictor.predict(feature_vector)
        risk_assessment = self._risk_analyzer.analyze(prediction)

        result = DetectionResult(
            timestamp=datetime.now(timezone.utc),
            packet=packet,
            prediction=prediction,
            risk_assessment=risk_assessment,
            is_threat=prediction.is_threat,
        )

        with self._lock:
            self.stats.total_processed += 1
            if result.is_threat:
                self.stats.total_threats_detected += 1

        if result.is_threat:
            self._persist_threat(result)

        with self._lock:
            self._recent_detections.append(result)

        self._dispatch(result)
        return result

    def get_recent_detections(self, limit: int = 50) -> List[DetectionResult]:
        """
        Returns the most recent detections held in memory, newest
        first, without requiring a database query. Intended for
        instant-loading UI panels (e.g., the dashboard's Recent
        Alerts section) that don't need full historical persistence.

        Args:
            limit: Maximum number of results to return.

        Returns:
            A list of DetectionResult instances, newest first.
        """
        with self._lock:
            items = list(self._recent_detections)
        return list(reversed(items))[:limit]

    def get_summary(self) -> dict:
        """
        Returns a single combined snapshot merging this detector's own
        pipeline counters, the AI predictor's rolling confidence
        stats, and the risk analyzer's current smoothed risk level.

        This is the one call app/core/engine.py needs on each refresh
        tick to update the dashboard's top stat-card row (Packets,
        Threats, Risk Score) and AI Status panel.

        Returns:
            A dictionary combining detector, predictor, and risk stats.
        """
        with self._lock:
            own_stats = self.stats.to_dict()

        return {
            **own_stats,
            "predictor": self._predictor.stats.to_dict(),
            "active_model": self._predictor.active_model_name,
            "current_risk_score": self._risk_analyzer.get_current_risk_score(),
            "current_severity": self._risk_analyzer.get_current_severity().value,
            "active_flows": self._feature_extractor.active_flow_count,
        }

    def reset(self) -> None:
        """
        Clears this detector's own pipeline counters, recent-detections
        buffer, and flow-tracking state (via FeatureExtractor.reset()).

        Note: this does NOT reset the shared ThreatPredictor's stats or
        the shared RiskAnalyzer's history/smoothed score, since those
        are application-wide singletons that may be read independently
        elsewhere. Callers wanting a fully clean session (e.g.,
        app/core/engine.py on "Stop Monitoring" -> "Start Monitoring")
        should reset those explicitly alongside this call.
        """
        with self._lock:
            self.stats.reset()
            self._recent_detections.clear()

        self._feature_extractor.reset()
        logger.info("ThreatDetector state reset.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist_threat(self, result: DetectionResult) -> None:
        """
        Persists a threat-worthy DetectionResult as both a threat
        record and a security log entry via DatabaseManager. Updates
        the result in place with persistence outcome (persisted flag
        and record ID) so callbacks/UI can reflect whether it was
        successfully saved.

        Args:
            result: The DetectionResult to persist (result.is_threat
                is assumed True).
        """
        threat_document = {
            **result.prediction.to_dict(),
            **result.risk_assessment.to_dict(),
        }

        record_id = self._db_manager.insert_threat_record(threat_document)
        result.persisted = record_id is not None
        result.threat_record_id = record_id

        if result.persisted:
            with self._lock:
                self.stats.total_threats_persisted += 1

        log_level = _SEVERITY_TO_LOG_LEVEL.get(result.risk_assessment.severity, "WARNING")
        log_message = (
            f"{result.prediction.predicted_class} detected: "
            f"{result.packet.src_ip}:{result.packet.src_port} -> "
            f"{result.packet.dst_ip}:{result.packet.dst_port} "
            f"(risk {result.risk_assessment.risk_score}/100, "
            f"confidence {result.prediction.confidence * 100:.1f}%)"
        )

        self._db_manager.insert_log_record(
            {
                "level": log_level,
                "source": "ThreatDetector",
                "message": log_message,
            }
        )

        if not result.persisted:
            logger.warning(
                "Threat detected but not persisted (database unavailable): %s",
                log_message,
            )

    def _dispatch(self, result: DetectionResult) -> None:
        """
        Invokes every registered callback with the completed
        DetectionResult, isolating callback failures so one broken
        subscriber (e.g., a UI handler) cannot interrupt the capture
        pipeline or prevent other subscribers from receiving the result.

        Args:
            result: The completed DetectionResult to dispatch.
        """
        for callback in self._callbacks:
            try:
                callback(result)
            except Exception as error:
                logger.error(
                    "Detection callback '%s' raised an error: %s",
                    getattr(callback, "__qualname__", callback),
                    error,
                )


# ==========================================================================
# Module-level singleton accessor
# ==========================================================================

_threat_detector_instance: Optional[ThreatDetector] = None


def get_threat_detector() -> ThreatDetector:
    """
    Returns the shared, application-wide ThreatDetector instance,
    creating it on first call.

    Mirrors the singleton pattern used by utils.logger.get_logger(),
    database.db_manager.get_database_manager(), ai.model_loader.get_model_bundle(),
    ai.predictor.get_predictor(), and security.risk_analyzer.get_risk_analyzer().
    app/core/engine.py should call this rather than instantiating
    ThreatDetector directly.

    Returns:
        The singleton ThreatDetector instance.
    """
    global _threat_detector_instance
    if _threat_detector_instance is None:
        _threat_detector_instance = ThreatDetector()
    return _threat_detector_instance