"""
security/alert_manager.py

User-facing alert / notification management for CyberSentinel AI.

Responsibilities:
    - Consume DetectionResult objects from security/threat_detector.py
      via handle_detection(), whose signature matches
      security.threat_detector.DetectionCallback exactly, so it can be
      registered directly:
          threat_detector.register_callback(alert_manager.handle_detection)
    - Decide WHEN a threat detection should become a user-facing Alert:
        * Only detections at or above a configured severity threshold
          are alert-worthy (low-severity noise doesn't reach the
          Notification Center).
        * Repeat detections from the same (source IP, attack type)
          pair are suppressed within a cooldown window, so a sustained
          attack produces one alert, not hundreds.
    - Provide create_system_alert() as a general entry point for
      non-threat notifications (monitoring lifecycle events, database
      connectivity changes, etc.), so this module is the single owner
      of everything that appears in the Notification Center.
    - Persist every alert via database.db_manager.DatabaseManager and
      maintain an in-memory ring buffer for instant UI access.
    - Dispatch every created Alert to registered callbacks, giving the
      UI's notification bell a live, decoupled update source.

This module contains no AI/scoring logic of its own (that belongs to
ai/predictor.py and security/risk_analyzer.py) and no PySide6 imports,
per the project's strict UI separation rule.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Deque, Dict, List, Optional, Tuple

from database.db_manager import DatabaseManager, get_database_manager
from security.risk_analyzer import SeverityLevel
from security.threat_detector import DetectionResult
from utils.logger import get_logger, get_threat_logger

logger = get_logger(__name__)
threat_logger = get_threat_logger()

# Minimum severity a threat detection must reach to generate a
# user-facing Alert. Detections below this are still logged/persisted
# as threats by ThreatDetector, but do not reach the Notification
# Center, keeping it meaningful rather than noisy.
ALERT_SEVERITY_THRESHOLD: SeverityLevel = SeverityLevel.MEDIUM

# Minimum time between alerts generated for the same (source IP,
# attack type) pair, in seconds. Prevents a sustained attack from
# flooding the Notification Center with near-duplicate alerts.
ALERT_COOLDOWN_SECONDS: float = 30.0

# Default size of the in-memory recent-alerts ring buffer, used for
# instant Notification Center display without a database round-trip.
_DEFAULT_RECENT_ALERTS_LIMIT: int = 50

# SeverityLevel is a plain str Enum (no intrinsic ordering), so a
# local rank table is used to compare "at least MEDIUM" style
# thresholds. Kept local to this module since it's an internal
# alert-filtering detail, not a cross-cutting app setting.
_SEVERITY_RANK: Dict[SeverityLevel, int] = {
    SeverityLevel.SAFE: 0,
    SeverityLevel.LOW: 1,
    SeverityLevel.MEDIUM: 2,
    SeverityLevel.HIGH: 3,
    SeverityLevel.CRITICAL: 4,
}


@dataclass
class Alert:
    """
    A single user-facing notification, either generated from a threat
    detection (category="threat") or raised directly by another
    subsystem via create_system_alert() (category="system").

    This is the data contract persisted via
    DatabaseManager.insert_alert_record() and dispatched to every
    registered AlertCallback for live UI updates.
    """
    timestamp: datetime
    title: str
    message: str
    severity: SeverityLevel
    category: str = "system"           # "system" or "threat"
    source_ip: Optional[str] = None
    target_ip: Optional[str] = None
    attack_type: Optional[str] = None
    risk_score: Optional[int] = None
    is_read: bool = False
    alert_id: Optional[str] = None     # populated after successful persistence

    def to_dict(self) -> dict:
        """
        Converts this alert into a plain dictionary suitable for
        MongoDB insertion via DatabaseManager.insert_alert_record().
        Excludes alert_id, since that field is only populated AFTER
        persistence assigns it.
        """
        return {
            "timestamp": self.timestamp,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category,
            "source_ip": self.source_ip,
            "target_ip": self.target_ip,
            "attack_type": self.attack_type,
            "risk_score": self.risk_score,
            "is_read": self.is_read,
        }


@dataclass
class AlertStats:
    """
    Rolling counters tracking alert pipeline activity, for the future
    Settings/Reports pages and for debugging alert-noise tuning.
    """
    total_alerts_created: int = 0
    threat_alerts_created: int = 0
    system_alerts_created: int = 0
    suppressed_by_threshold: int = 0
    suppressed_by_cooldown: int = 0

    def to_dict(self) -> dict:
        """Returns a plain-dict snapshot for display or logging."""
        return {
            "total_alerts_created": self.total_alerts_created,
            "threat_alerts_created": self.threat_alerts_created,
            "system_alerts_created": self.system_alerts_created,
            "suppressed_by_threshold": self.suppressed_by_threshold,
            "suppressed_by_cooldown": self.suppressed_by_cooldown,
        }

    def reset(self) -> None:
        """Clears all counters, e.g. when a monitoring session restarts."""
        self.total_alerts_created = 0
        self.threat_alerts_created = 0
        self.system_alerts_created = 0
        self.suppressed_by_threshold = 0
        self.suppressed_by_cooldown = 0


# Type alias for functions that consume a newly created Alert.
AlertCallback = Callable[[Alert], None]

# Deduplication key: (source_ip, attack_type).
_CooldownKey = Tuple[str, str]


class AlertManager:
    """
    Filters threat detections into user-facing Alerts, persists them,
    and dispatches them to registered callbacks for live UI updates
    (notification bell, Notification Center panel).

    Usage:
        alert_manager = AlertManager()
        alert_manager.register_callback(my_notification_handler)
        threat_detector.register_callback(alert_manager.handle_detection)
        ...
        alert_manager.create_system_alert("Monitoring Started", "...")
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        severity_threshold: SeverityLevel = ALERT_SEVERITY_THRESHOLD,
        cooldown_seconds: float = ALERT_COOLDOWN_SECONDS,
        recent_alerts_limit: int = _DEFAULT_RECENT_ALERTS_LIMIT,
    ) -> None:
        """
        Args:
            db_manager: Optional explicit DatabaseManager. Defaults to
                the shared application singleton via
                database.db_manager.get_database_manager().
            severity_threshold: Minimum SeverityLevel a threat
                detection must reach to generate an Alert.
            cooldown_seconds: Minimum time between alerts for the same
                (source IP, attack type) pair.
            recent_alerts_limit: Maximum number of recent Alerts to
                retain in memory for instant UI access.
        """
        self._db_manager = db_manager or get_database_manager()
        self._severity_threshold = severity_threshold
        self._cooldown_seconds = cooldown_seconds

        self._lock = threading.Lock()
        self._callbacks: List[AlertCallback] = []
        self._recent_alerts: Deque[Alert] = deque(maxlen=recent_alerts_limit)
        self._last_alert_time: Dict[_CooldownKey, float] = {}

        self.stats = AlertStats()

        logger.info(
            "AlertManager initialized (severity_threshold=%s, cooldown=%.0fs).",
            severity_threshold.value,
            cooldown_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_callback(self, callback: AlertCallback) -> None:
        """
        Registers a function to be invoked with every newly created
        Alert (threat-origin or system-origin). Multiple callbacks may
        be registered (e.g., the notification bell badge and a future
        toast/sound notification handler).

        Args:
            callback: A callable accepting a single Alert argument.
        """
        self._callbacks.append(callback)
        logger.debug(
            "Registered alert callback: %s",
            getattr(callback, "__qualname__", callback),
        )

    def handle_detection(self, result: DetectionResult) -> Optional[Alert]:
        """
        Evaluates a completed DetectionResult and creates a user-facing
        Alert if it clears both the severity threshold and the
        per-source cooldown window.

        This method's signature matches
        security.threat_detector.DetectionCallback exactly, so it can
        be passed directly to ThreatDetector.register_callback():

            threat_detector.register_callback(alert_manager.handle_detection)

        Args:
            result: The completed DetectionResult to evaluate.

        Returns:
            The newly created Alert, or None if the detection was not
            a threat, did not meet the severity threshold, or was
            suppressed by the cooldown window.
        """
        if not result.is_threat:
            return None

        severity = result.risk_assessment.severity
        if not self._meets_severity_threshold(severity):
            with self._lock:
                self.stats.suppressed_by_threshold += 1
            return None

        cooldown_key: _CooldownKey = (result.packet.src_ip, result.prediction.predicted_class)
        if self._is_within_cooldown(cooldown_key):
            with self._lock:
                self.stats.suppressed_by_cooldown += 1
            return None

        title, message = self._build_threat_alert_content(result)
        alert = Alert(
            timestamp=datetime.now(timezone.utc),
            title=title,
            message=message,
            severity=severity,
            category="threat",
            source_ip=result.packet.src_ip,
            target_ip=result.packet.dst_ip,
            attack_type=result.prediction.predicted_class,
            risk_score=result.risk_assessment.risk_score,
        )

        created = self._finalize_alert(alert, is_threat_alert=True)

        threat_logger.warning(
            "Alert raised [%s]: %s (cooldown key=%s)",
            severity.value,
            title,
            cooldown_key,
        )
        return created

    def create_system_alert(
        self,
        title: str,
        message: str,
        severity: SeverityLevel = SeverityLevel.LOW,
    ) -> Alert:
        """
        Creates a non-threat, system-originated Alert (e.g.,
        "Monitoring Started", "Database Disconnected", "AI Model
        Reloaded"). Bypasses the severity threshold and cooldown logic
        used for threat alerts, since callers here are explicitly
        requesting a single, deliberate notification.

        Args:
            title: Short alert title shown in the Notification Center.
            message: Full alert description.
            severity: Severity level to associate with this alert,
                affecting its display styling in the UI.

        Returns:
            The newly created Alert.
        """
        alert = Alert(
            timestamp=datetime.now(timezone.utc),
            title=title,
            message=message,
            severity=severity,
            category="system",
        )
        return self._finalize_alert(alert, is_threat_alert=False)

    def get_recent_alerts(self, limit: int = 20) -> List[Alert]:
        """
        Returns the most recent alerts held in memory, newest first,
        without requiring a database query. Intended for instant-
        loading UI panels (the notification bell dropdown, Notification
        Center page).

        Args:
            limit: Maximum number of alerts to return.

        Returns:
            A list of Alert instances, newest first.
        """
        with self._lock:
            items = list(self._recent_alerts)
        return list(reversed(items))[:limit]

    def get_unread_count(self) -> int:
        """
        Returns the number of unread alerts, for the notification
        bell's badge counter. Prefers an authoritative database count
        (since unread alerts may exist beyond the in-memory buffer's
        capacity); falls back to counting the in-memory buffer if the
        database is unavailable.

        Returns:
            The number of unread alerts.
        """
        if self._db_manager.is_connected():
            return self._db_manager.get_unread_alert_count()

        with self._lock:
            return sum(1 for alert in self._recent_alerts if not alert.is_read)

    def mark_as_read(self, alert_id: str) -> bool:
        """
        Marks a specific alert as read, both in the database and in
        the in-memory buffer if it's still present there.

        Args:
            alert_id: The string ID of the alert to mark as read.

        Returns:
            True if the database update succeeded, False otherwise.
        """
        success = self._db_manager.mark_alert_as_read(alert_id)

        if success:
            with self._lock:
                for alert in self._recent_alerts:
                    if alert.alert_id == alert_id:
                        alert.is_read = True
                        break

        return success

    def mark_all_as_read(self) -> int:
        """
        Marks every currently unread alert in the in-memory buffer as
        read, both in the database and locally. Intended for a future
        "Mark All Read" button in the Notification Center.

        Returns:
            The number of alerts successfully marked as read.
        """
        with self._lock:
            unread_alerts = [alert for alert in self._recent_alerts if not alert.is_read]

        marked_count = 0
        for alert in unread_alerts:
            if alert.alert_id and self._db_manager.mark_alert_as_read(alert.alert_id):
                alert.is_read = True
                marked_count += 1

        return marked_count

    def get_summary(self) -> dict:
        """
        Returns a combined snapshot of alert pipeline stats and the
        current unread count, for the dashboard's notification bell
        and a future Notification Center summary header.

        Returns:
            A dictionary combining AlertStats and the unread count.
        """
        with self._lock:
            own_stats = self.stats.to_dict()

        return {
            **own_stats,
            "unread_count": self.get_unread_count(),
        }
        
        
  
    def set_severity_threshold(self, severity: SeverityLevel) -> None:
        """
        Updates the minimum severity a threat detection must reach to
        generate a user-facing Alert. Intended to be called from
        app/ui/pages/settings_page.py.

        Args:
            severity: The new minimum severity threshold.
        """
        with self._lock:
            self._severity_threshold = severity
        logger.info("Alert severity threshold updated to: %s", severity.value)

    def get_severity_threshold(self) -> SeverityLevel:
        """Returns the currently configured minimum alert severity threshold."""
        return self._severity_threshold    

    def reset(self) -> None:
        """
        Clears this manager's in-memory recent-alerts buffer, cooldown
        tracker, and stats counters. Does NOT delete previously
        persisted alerts from the database — only in-memory session
        state is cleared, matching the reset() scope convention used
        by security/threat_detector.py.
        """
        with self._lock:
            self._recent_alerts.clear()
            self._last_alert_time.clear()
            self.stats.reset()
        logger.info("AlertManager state reset.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _meets_severity_threshold(self, severity: SeverityLevel) -> bool:
        """
        Checks whether a severity level is at or above this manager's
        configured alert threshold.

        Args:
            severity: The severity level to check.

        Returns:
            True if the severity qualifies for alert creation.
        """
        return _SEVERITY_RANK[severity] >= _SEVERITY_RANK[self._severity_threshold]

    def _is_within_cooldown(self, key: _CooldownKey) -> bool:
        """
        Checks whether an alert was already raised for this
        (source_ip, attack_type) pair within the cooldown window, and
        if not, records the current time against the key.

        Args:
            key: The (source_ip, attack_type) deduplication key.

        Returns:
            True if a matching alert is still within its cooldown
            window (i.e., a new alert should be suppressed).
        """
        with self._lock:
            now = time.monotonic()
            last_time = self._last_alert_time.get(key)

            if last_time is not None and (now - last_time) < self._cooldown_seconds:
                return True

            self._last_alert_time[key] = now
            return False

    @staticmethod
    def _build_threat_alert_content(result: DetectionResult) -> Tuple[str, str]:
        """
        Builds a human-readable title and message for a threat-origin
        alert from a completed DetectionResult.

        Args:
            result: The DetectionResult to summarize.

        Returns:
            A tuple of (title, message) strings.
        """
        severity = result.risk_assessment.severity
        predicted_class = result.prediction.predicted_class
        packet = result.packet

        title = f"{severity.value} Alert: {predicted_class} Detected"
        message = (
            f"{predicted_class} traffic detected from {packet.src_ip}:{packet.src_port} "
            f"to {packet.dst_ip}:{packet.dst_port} over {packet.protocol}. "
            f"Risk score {result.risk_assessment.risk_score}/100 "
            f"({result.prediction.confidence * 100:.1f}% model confidence)."
        )
        return title, message

    def _finalize_alert(self, alert: Alert, is_threat_alert: bool) -> Alert:
        """
        Shared finalization logic for both threat and system alerts:
        persists the alert, updates stats, stores it in the in-memory
        buffer, and dispatches it to registered callbacks.

        Args:
            alert: The Alert to finalize (not yet persisted).
            is_threat_alert: Whether this alert originated from a
                threat detection (affects which stats counter increments).

        Returns:
            The same Alert instance, with alert_id populated if
            persistence succeeded.
        """
        record_id = self._db_manager.insert_alert_record(alert.to_dict())
        alert.alert_id = record_id

        if record_id is None:
            logger.warning("Alert created but not persisted (database unavailable): %s", alert.title)

        with self._lock:
            self.stats.total_alerts_created += 1
            if is_threat_alert:
                self.stats.threat_alerts_created += 1
            else:
                self.stats.system_alerts_created += 1
            self._recent_alerts.append(alert)

        self._dispatch(alert)
        return alert

    def _dispatch(self, alert: Alert) -> None:
        """
        Invokes every registered callback with the newly created
        Alert, isolating callback failures so one broken subscriber
        cannot prevent other subscribers from receiving the alert or
        interrupt the detection pipeline upstream.

        Args:
            alert: The newly created Alert to dispatch.
        """
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as error:
                logger.error(
                    "Alert callback '%s' raised an error: %s",
                    getattr(callback, "__qualname__", callback),
                    error,
                )


# ==========================================================================
# Module-level singleton accessor
# ==========================================================================

_alert_manager_instance: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """
    Returns the shared, application-wide AlertManager instance,
    creating it on first call.

    Mirrors the singleton pattern used throughout the project
    (utils.logger.get_logger(), database.db_manager.get_database_manager(),
    ai.model_loader.get_model_bundle(), ai.predictor.get_predictor(),
    security.risk_analyzer.get_risk_analyzer(),
    security.threat_detector.get_threat_detector()). app/core/engine.py
    should call this rather than instantiating AlertManager directly.

    Returns:
        The singleton AlertManager instance.
    """
    global _alert_manager_instance
    if _alert_manager_instance is None:
        _alert_manager_instance = AlertManager()
    return _alert_manager_instance