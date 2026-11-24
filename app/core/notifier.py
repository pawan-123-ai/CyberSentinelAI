"""
app/core/notifier.py

Native OS desktop notification bridge for CyberSentinel AI.

Responsibilities:
    - Own a QSystemTrayIcon (PySide6's built-in cross-platform tray/
      notification API — no additional dependency required) that
      stays present even when the main window is minimized or closed
      to the tray.
    - Show a native desktop toast notification for HIGH and CRITICAL
      severity alerts only, filtering out MEDIUM-and-below alerts to
      avoid notification fatigue during a sustained attack (those
      remain fully visible in-app via app/ui/widgets/recent_alerts_widget.py).
    - Provide a right-click tray context menu (Show Dashboard,
      Start/Stop Monitoring, Quit), a standard pattern for
      security-monitoring desktop applications that need to keep
      running in the background.

This module fulfills the "Show an immediate desktop alert/
notification" requirement independently of platform (Windows/Linux/
macOS), since QSystemTrayIcon.showMessage() maps to each OS's native
notification system automatically.
"""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import QObject
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QMenu, QSystemTrayIcon

from app.core.engine import MonitoringEngine
from config.settings import AppInfo, PathConfig
from security.alert_manager import Alert
from security.risk_analyzer import SeverityLevel
from utils.logger import get_logger

logger = get_logger(__name__)

# Minimum severity required for a real OS toast notification to fire.
# Kept local to this module (same pattern already used for severity-
# related constants in security/alert_manager.py and
# app/ui/widgets/threat_timeline_widget.py) since it's a notification-
# specific display policy, not a cross-cutting app setting.
_TOAST_SEVERITY_THRESHOLD: SeverityLevel = SeverityLevel.HIGH

# Local severity rank table, mirroring the private one in
# security/alert_manager.py. Not imported directly since that table
# is an internal implementation detail of AlertManager's own cooldown
# logic — each display-layer module defining its own copy (as
# app/ui/widgets/threat_timeline_widget.py and
# app/ui/widgets/recent_alerts_widget.py already do for severity
# colors) keeps this module independently understandable.
_SEVERITY_RANK: Dict[SeverityLevel, int] = {
    SeverityLevel.SAFE: 0,
    SeverityLevel.LOW: 1,
    SeverityLevel.MEDIUM: 2,
    SeverityLevel.HIGH: 3,
    SeverityLevel.CRITICAL: 4,
}

# Maps SeverityLevel -> the native OS notification icon style Qt
# should render the toast with (a colored info/warning/critical
# glyph, drawn by the OS itself).
_SEVERITY_TO_QT_ICON: Dict[SeverityLevel, QSystemTrayIcon.MessageIcon] = {
    SeverityLevel.HIGH: QSystemTrayIcon.MessageIcon.Warning,
    SeverityLevel.CRITICAL: QSystemTrayIcon.MessageIcon.Critical,
}

# How long (milliseconds) a toast notification stays visible before
# the OS auto-dismisses it. Actual on-screen duration is ultimately
# controlled by the OS notification system, not guaranteed by Qt.
_TOAST_DURATION_MS: int = 8000


class DesktopNotifier(QObject):
    """
    Bridges MonitoringEngine's alert_created signal to native OS
    desktop notifications via QSystemTrayIcon, and hosts a persistent
    tray icon with a right-click control menu.

    Usage (from MainWindow, after self.engine is constructed):
        self.notifier = DesktopNotifier(main_window=self, engine=self.engine)
        self.engine.alert_created.connect(self.notifier.handle_alert)
    """

    def __init__(
        self,
        main_window: QMainWindow,
        engine: MonitoringEngine,
        parent: Optional[QObject] = None,
    ) -> None:
        """
        Args:
            main_window: The application's MainWindow, used to
                show/restore/hide from the tray menu.
            engine: The shared MonitoringEngine instance, used to
                wire the tray menu's Start/Stop Monitoring action.
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self._main_window = main_window
        self._engine = engine
        
        self._notifications_enabled: bool = True

        self._tray_icon: Optional[QSystemTrayIcon] = None
        self._toggle_monitoring_action = None  # set in _build_tray_menu()

        if not QSystemTrayIcon.isSystemTrayAvailable():
            logger.warning(
                "System tray is not available on this platform/session; "
                "desktop toast notifications will be disabled. In-app "
                "alerts (Recent Alerts panel, notification bell) are "
                "unaffected."
            )
            return

        self._build_tray_icon()
        logger.info("DesktopNotifier initialized; system tray icon active.")

    # ------------------------------------------------------------------
    # Tray icon construction
    # ------------------------------------------------------------------

    def _build_tray_icon(self) -> None:
        """
        Constructs the QSystemTrayIcon, assigns it an icon (the app
        icon if available, otherwise the main window's current
        icon as a fallback), attaches the right-click context menu,
        and makes it visible.
        """
        icon = self._resolve_icon()
        self._tray_icon = QSystemTrayIcon(icon, self)
        self._tray_icon.setToolTip(AppInfo.NAME)
        self._tray_icon.setContextMenu(self._build_tray_menu())
        self._tray_icon.activated.connect(self._on_tray_activated)
        self._tray_icon.show()

    def _resolve_icon(self) -> QIcon:
        """
        Resolves the icon used for both the tray and toast
        notifications. Prefers assets/icon.ico (per PathConfig),
        falling back to the main window's current window icon so a
        missing icon file never prevents the tray icon itself from
        appearing.

        Returns:
            A QIcon instance (possibly null/empty if no icon is
            available anywhere; Qt renders a default in that case).
        """
        icon_path = PathConfig.ASSETS_DIR / "icon.ico"
        if icon_path.exists():
            return QIcon(str(icon_path))

        logger.debug(
            "No assets/icon.ico found for tray icon; falling back to "
            "MainWindow's current window icon."
        )
        return self._main_window.windowIcon()

    def _build_tray_menu(self) -> QMenu:
        """
        Builds the right-click context menu attached to the tray icon:
        Show Dashboard, a Start/Stop Monitoring toggle kept in sync
        with the engine's actual state, and Quit.

        Returns:
            The constructed QMenu.
        """
        menu = QMenu()

        show_action = menu.addAction("Show Dashboard")
        show_action.triggered.connect(self._show_main_window)

        menu.addSeparator()

        self._toggle_monitoring_action = menu.addAction("Start Monitoring")
        self._toggle_monitoring_action.triggered.connect(self._on_toggle_monitoring)
        self._refresh_toggle_action_text()

        menu.addSeparator()

        quit_action = menu.addAction("Quit CyberSentinel AI")
        quit_action.triggered.connect(self._on_quit_requested)

        # Keep the toggle action's label in sync with the engine's
        # actual monitoring state, regardless of whether it was
        # started/stopped from the tray menu or the header button.
        self._engine.monitoring_started.connect(self._refresh_toggle_action_text)
        self._engine.monitoring_stopped.connect(self._refresh_toggle_action_text)

        return menu

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def handle_alert(self, alert: Alert) -> None:
        """
        Handles a newly created Alert: shows a native desktop toast
        notification if the alert's severity meets
        _TOAST_SEVERITY_THRESHOLD, and silently does nothing otherwise
        (the alert is still fully visible in-app regardless — this
        method only controls the OS-level popup).

        Intended to be connected directly to
        app.core.engine.MonitoringEngine.alert_created:
            engine.alert_created.connect(notifier.handle_alert)

        Args:
            alert: The newly created Alert to potentially notify about.
        """
        if self._tray_icon is None:
             return  # system tray unavailable; already logged at startup

        if not self._notifications_enabled:
            return

        if _SEVERITY_RANK[alert.severity] < _SEVERITY_RANK[_TOAST_SEVERITY_THRESHOLD]:
            return

        icon_style = _SEVERITY_TO_QT_ICON.get(
    alert.severity, QSystemTrayIcon.MessageIcon.Warning
)

        self._tray_icon.showMessage(
            alert.title,
            alert.message,
            icon_style,
            _TOAST_DURATION_MS,
        )
        logger.debug("Desktop toast shown for alert: %s", alert.title)
        
        
    # Add these two new methods anywhere in the public API section:
    def set_notifications_enabled(self, enabled: bool) -> None:
        """
        Enables or disables desktop toast notifications without
        removing the tray icon itself. Intended to be called from
        app/ui/pages/settings_page.py.

        Args:
            enabled: True to allow toasts for HIGH/CRITICAL alerts,
                False to suppress them (in-app alerts are unaffected).
        """
        self._notifications_enabled = enabled
        logger.info("Desktop toast notifications %s.", "enabled" if enabled else "disabled")

    @property
    def notifications_enabled(self) -> bool:
        """Returns True if desktop toast notifications are currently enabled."""
        return self._notifications_enabled    

    # ------------------------------------------------------------------
    # Internal slots
    # ------------------------------------------------------------------

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """
        Handles clicks on the tray icon itself (not the context menu):
        a left-click or double-click restores and focuses the main
        window, matching standard desktop tray-icon behavior.

        Args:
            reason: The Qt activation reason (click type).
        """
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self._show_main_window()

    def _show_main_window(self) -> None:
        """Restores, raises, and focuses the main window."""
        self._main_window.showNormal()
        self._main_window.raise_()
        self._main_window.activateWindow()

    def _on_toggle_monitoring(self) -> None:
        """
        Handles the tray menu's Start/Stop Monitoring action: mirrors
        the exact toggle logic used by the header button in
        app/ui/dashboard.py's MainWindow, so monitoring can be
        controlled without restoring the window first.
        """
        if self._engine.is_monitoring:
            self._engine.stop_monitoring()
        else:
            self._engine.start_monitoring()

    def _refresh_toggle_action_text(self) -> None:
        """
        Updates the tray menu's Start/Stop Monitoring action label to
        reflect the engine's current state. Connected to both
        monitoring_started and monitoring_stopped so it stays correct
        regardless of what triggered the state change.
        """
        if self._toggle_monitoring_action is None:
            return
        self._toggle_monitoring_action.setText(
            "Stop Monitoring" if self._engine.is_monitoring else "Start Monitoring"
        )

    def _on_quit_requested(self) -> None:
        """
        Handles the tray menu's Quit action: closes the main window
        (triggering its normal closeEvent -> MonitoringEngine.shutdown()
        cleanup path), then quits the QApplication event loop.
        """
        logger.info("Quit requested from system tray menu.")
        self._main_window.close()
        QApplication.instance().quit()