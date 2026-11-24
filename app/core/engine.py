"""
app/core/engine.py

Central orchestration bridge between CyberSentinel AI's threading/
callback-based backend subsystems and the PySide6 UI layer.

Responsibilities:
    - Own the single live PacketCapture and NetworkMonitor instances
      for the application.
    - Own the file-scanning subsystem (FileScanner + QuarantineManager),
      which runs independently of network packet capture - real-time
      file protection starts as soon as the engine is constructed,
      the same way system-resource monitoring does, rather than being
      gated behind the Start/Stop Monitoring toggle (which controls
      packet capture specifically).
    - Wire the full detection pipeline together.
    - Re-emit backend events as Qt Signals.
    - Provide a small, UI-friendly control surface for the dashboard.
    - Manage clean startup and shutdown of monitoring resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QObject, Signal

from ai.predictor import ThreatPredictor, get_predictor
from network.network_monitor import DeviceInfo, NetworkMonitor, SystemStats
from network.packet_capture import PacketCapture
from security.alert_manager import Alert, AlertManager, get_alert_manager
from security.file_scanner import FileRiskSeverity, FileScanResult, get_file_scanner
from security.quarantine_manager import get_quarantine_manager
from security.risk_analyzer import RiskAnalyzer, SeverityLevel, get_risk_analyzer
from security.threat_detector import (
    DetectionResult,
    ThreatDetector,
    get_threat_detector,
)
from utils.logger import get_logger


logger = get_logger(__name__)

# Maps FileScanner's own severity band to the shared SeverityLevel enum
# used everywhere else in the alert pipeline, so a malicious file scan
# result can raise a system alert through the exact same path a
# network-based threat does. FileRiskSeverity was deliberately named
# to mirror SeverityLevel's members (SAFE/LOW/MEDIUM/HIGH/CRITICAL);
# if risk_analyzer.py's actual SeverityLevel doesn't have MEDIUM or
# CRITICAL members, this is the one place to fix that mismatch.
_FILE_SEVERITY_TO_SYSTEM_SEVERITY = {
    FileRiskSeverity.CRITICAL: SeverityLevel.CRITICAL,
    FileRiskSeverity.HIGH: SeverityLevel.HIGH,
    FileRiskSeverity.MEDIUM: SeverityLevel.MEDIUM,
    FileRiskSeverity.LOW: SeverityLevel.LOW,
    FileRiskSeverity.SAFE: SeverityLevel.SAFE,
}


@dataclass
class DashboardSnapshot:
    """
    A single combined bundle of everything the dashboard needs
    to refresh on one tick.
    """

    timestamp: datetime
    system_stats: SystemStats
    device_info: DeviceInfo
    packets_captured: int
    packets_per_second: int
    bytes_per_second: int
    is_monitoring: bool
    threat_summary: dict
    alert_summary: dict

    def to_dict(self) -> dict:
        """Returns a plain-dict representation."""
        return {
            "timestamp": self.timestamp,
            "system_stats": self.system_stats.to_dict(),
            "device_info": self.device_info.to_dict(),
            "packets_captured": self.packets_captured,
            "packets_per_second": self.packets_per_second,
            "bytes_per_second": self.bytes_per_second,
            "is_monitoring": self.is_monitoring,
            "threat_summary": self.threat_summary,
            "alert_summary": self.alert_summary,
        }


class MonitoringEngine(QObject):
    """
    Owns and coordinates every backend subsystem and translates
    their callback-based events into Qt signals for the UI.
    """

    dashboard_updated = Signal(object)
    threat_detected = Signal(object)
    alert_created = Signal(object)

    monitoring_started = Signal()
    monitoring_stopped = Signal()

    capture_error = Signal(str)

    # Emitted for every completed file scan (not just malicious ones),
    # for a future dedicated "File Activity" page. Not consumed by any
    # existing widget yet - malicious verdicts are already surfaced
    # through alert_created via the existing alert pipeline, so no UI
    # changes are required for this feature to be visible right now.
    file_scan_completed = Signal(object)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        """
        Constructs all backend subsystems, wires their callbacks,
        and starts system-resource monitoring and file scanning.
        """
        super().__init__(parent)

        # Live OS-resource subsystems.
        self._packet_capture = PacketCapture()
        self._network_monitor = NetworkMonitor()

        # Shared AI/security singletons.
        self._threat_detector: ThreatDetector = get_threat_detector()
        self._alert_manager: AlertManager = get_alert_manager()
        self._risk_analyzer: RiskAnalyzer = get_risk_analyzer()
        self._predictor: ThreatPredictor = get_predictor()

        # File-scanning subsystem (Phase A): watches Downloads/Desktop,
        # scans new files, quarantines malicious ones. Independent of
        # packet capture - see class docstring.
        self._file_scanner = get_file_scanner()
        self._quarantine_manager = get_quarantine_manager()

        # Cache static device information.
        self._device_info: DeviceInfo = (
            self._network_monitor.get_device_info()
        )

        # Preferred capture interface.
        # None means auto-detection.
        self._preferred_interface: Optional[str] = None

        self._wire_callbacks()

        # System monitoring starts immediately.
        self._network_monitor.start()

        # File scanning starts immediately too - real-time file
        # protection shouldn't require the user to separately opt in
        # the way packet capture does (which needs Npcap/admin
        # privileges and an explicit interface choice).
        if not self._file_scanner.start():
            logger.warning(
                "FileScanner did not start - no valid folders to "
                "watch (Downloads/Desktop not found). File-scanning "
                "protection is inactive this session."
            )

        logger.info(
            "MonitoringEngine initialized; "
            "system resource monitoring and file scanning active."
        )

    # ------------------------------------------------------------------
    # Callback wiring
    # ------------------------------------------------------------------

    def _wire_callbacks(self) -> None:
        """
        Connects backend callbacks into the complete detection pipeline.
        """

        self._threat_detector.register_callback(
            self._alert_manager.handle_detection
        )

        self._threat_detector.register_callback(
            self._on_detection
        )

        self._packet_capture.register_callback(
            self._threat_detector.process_packet
        )

        self._alert_manager.register_callback(
            self._on_alert
        )

        self._network_monitor.register_callback(
            self._on_system_tick
        )

        # File-scanning pipeline: QuarantineManager and this engine
        # both subscribe independently to FileScanner's scan-result
        # callback (multi-callback, same pattern as ThreatDetector
        # above) - one acts on malicious verdicts, the other bridges
        # every result into a Qt signal for the UI.
        self._quarantine_manager.connect_to_scanner()
        self._file_scanner.register_scan_result_callback(
            self._on_file_scan_result
        )

        logger.debug("MonitoringEngine callback pipeline wired.")

    # ------------------------------------------------------------------
    # Public control API
    # ------------------------------------------------------------------

    def start_monitoring(
        self,
        interface: Optional[str] = None,
    ) -> bool:
        """
        Starts live packet capture and AI threat detection.

        Args:
            interface:
                Explicit interface to capture on. If None, the
                preferred interface is used. If neither is set,
                PacketCapture performs its normal auto-detection.
        """

        if self._packet_capture.is_running:
            logger.warning(
                "start_monitoring() called while already running; "
                "ignoring."
            )
            return False

        # Reset session-scoped state.
        self._threat_detector.reset()
        self._risk_analyzer.reset()
        self._predictor.stats.reset()

        # Explicit interface takes priority over the preferred interface.
        resolved_interface = interface or self._preferred_interface

        started = self._packet_capture.start(
            interface=resolved_interface
        )

        if not started:
            message = (
                "Failed to start packet capture. Ensure Npcap is "
                "installed (with WinPcap API-compatible mode enabled) "
                "and that CyberSentinel AI is running with "
                "Administrator privileges."
            )

            logger.error(message)
            self.capture_error.emit(message)

            self._alert_manager.create_system_alert(
                title="Monitoring Failed to Start",
                message=message,
                severity=SeverityLevel.HIGH,
            )

            return False

        self._alert_manager.create_system_alert(
            title="Monitoring Started",
            message=(
                "Live packet capture and AI threat detection "
                "started on interface "
                f"'{self._packet_capture.active_interface}'."
            ),
            severity=SeverityLevel.LOW,
        )

        self.monitoring_started.emit()

        logger.info(
            "Monitoring started on interface: %s",
            self._packet_capture.active_interface,
        )

        return True

    def stop_monitoring(self) -> None:
        """
        Stops packet capture and AI threat detection.
        """

        if not self._packet_capture.is_running:
            return

        self._packet_capture.stop()

        self._alert_manager.create_system_alert(
            title="Monitoring Stopped",
            message=(
                "Live packet capture and AI threat detection "
                "have been stopped."
            ),
            severity=SeverityLevel.LOW,
        )

        self.monitoring_stopped.emit()

        logger.info("Monitoring stopped.")

    @property
    def is_monitoring(self) -> bool:
        """Returns True if packet capture is currently active."""

        return self._packet_capture.is_running

    def get_device_info(self) -> DeviceInfo:
        """
        Returns the cached device information.
        """

        return self._device_info

    def get_available_interfaces(self) -> List[str]:
        """
        Returns a list of network interfaces available for capture.
        """

        return PacketCapture.list_available_interfaces()

    def set_preferred_interface(
        self,
        interface: Optional[str],
    ) -> None:
        """
        Sets the network interface to use on the next
        start_monitoring() call that doesn't explicitly override it.

        Intended to be called from
        app/ui/pages/settings_page.py.

        Args:
            interface:
                The interface name to prefer, or None to restore
                auto-detection.
        """

        self._preferred_interface = interface

        logger.info(
            "Preferred capture interface set to: %s",
            interface or "Auto-detect",
        )

    @property
    def preferred_interface(self) -> Optional[str]:
        """Returns the currently configured preferred interface, if any."""

        return self._preferred_interface

    @property
    def active_interface(self) -> Optional[str]:
        """
        Returns:
            The name of the network interface currently being
            captured on, or None if monitoring is not currently active.
        """

        return self._packet_capture.active_interface

    def get_current_snapshot(self) -> DashboardSnapshot:
        """
        Builds and returns a current DashboardSnapshot immediately.
        """

        stats = (
            self._network_monitor.latest_stats
            or self._network_monitor.get_current_stats()
        )

        return self._build_snapshot(stats)

    def shutdown(self) -> None:
        """
        Stops every background subsystem owned by this engine.
        """

        logger.info("MonitoringEngine shutting down...")

        self.stop_monitoring()
        self._network_monitor.stop()
        self._file_scanner.stop()

        logger.info("MonitoringEngine shutdown complete.")

    # ------------------------------------------------------------------
    # Internal Qt-bridging callbacks
    # ------------------------------------------------------------------

    def _on_system_tick(
        self,
        stats: SystemStats,
    ) -> None:
        """
        NetworkMonitor periodic callback.
        """

        snapshot = self._build_snapshot(stats)

        self.dashboard_updated.emit(snapshot)

    def _build_snapshot(
        self,
        stats: SystemStats,
    ) -> DashboardSnapshot:
        """
        Builds a DashboardSnapshot from current subsystem state.
        """

        packets_per_second, bytes_per_second = (
            self._packet_capture.rate_counter.get_rates()
        )

        return DashboardSnapshot(
            timestamp=datetime.now(timezone.utc),
            system_stats=stats,
            device_info=self._device_info,
            packets_captured=(
                self._packet_capture.total_packets_captured
            ),
            packets_per_second=packets_per_second,
            bytes_per_second=bytes_per_second,
            is_monitoring=self._packet_capture.is_running,
            threat_summary=self._threat_detector.get_summary(),
            alert_summary=self._alert_manager.get_summary(),
        )

    def _on_detection(
        self,
        result: DetectionResult,
    ) -> None:
        """
        Forwards threat-classified detections to the UI.
        """

        if result.is_threat:
            self.threat_detected.emit(result)

    def _on_alert(
        self,
        alert: Alert,
    ) -> None:
        """
        Forwards newly created alerts to the UI.
        """

        self.alert_created.emit(alert)

    def _on_file_scan_result(self, result: FileScanResult) -> None:
        """
        FileScanner's scan-result callback (registered alongside
        QuarantineManager - both subscribe independently, see
        _wire_callbacks). Forwards every scan to the UI via
        file_scan_completed for future use, and raises a system alert
        through the existing alert pipeline for malicious verdicts, so
        a quarantined file surfaces through Recent Alerts, the
        header's notification badge, and the desktop toast exactly
        like a network-based threat does - without needing any new UI
        plumbing right now.

        Args:
            result: The completed FileScanResult from FileScanner.
        """
        self.file_scan_completed.emit(result)

        if result.scan_error or not result.is_malicious:
            return

        severity = _FILE_SEVERITY_TO_SYSTEM_SEVERITY.get(
            result.severity, SeverityLevel.HIGH
        )

        self._alert_manager.create_system_alert(
            title="Malicious File Quarantined",
            message=(
                f"'{Path(result.file_path).name}' was flagged "
                f"(risk score {result.risk_score}/100) and moved to "
                f"quarantine. Reasons: "
                f"{', '.join(result.reasons) or 'signature match'}."
            ),
            severity=severity,
        )


# ==========================================================================
# Module-level singleton accessor
# ==========================================================================

_engine_instance: Optional[MonitoringEngine] = None


def get_engine() -> MonitoringEngine:
    """
    Returns the shared application-wide MonitoringEngine instance.

    The engine must be created after QApplication exists.
    """

    global _engine_instance

    if _engine_instance is None:
        _engine_instance = MonitoringEngine()

    return _engine_instance