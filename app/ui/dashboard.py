"""
app/ui/dashboard.py

Main application window (shell) for CyberSentinel AI, fully wired to
live backend data via app.core.engine.MonitoringEngine.

Responsibilities:
    - Build the overall window layout: sidebar + header + content area.
    - Apply the dark cybersecurity theme (QSS) using ThemeConfig.
    - Provide navigation between pages via a QStackedWidget.
    - Own the single MonitoringEngine instance and pass it into every
      page that needs it (DashboardPage, ThreatDetectionPage), which
      each independently compose their own widgets and connect to the
      engine's Qt signals.
    - Own the single DesktopNotifier instance (system tray icon +
      native OS toast notifications for HIGH/CRITICAL alerts).
    - Host the AI status indicator, monitoring toggle, and
      notification bell in the header, all wired to live state.
    - Handle graceful shutdown: MainWindow.closeEvent() calls
      MonitoringEngine.shutdown(), stopping all background threads
      before main.py's own DatabaseManager.disconnect() runs.

Network Monitor / Logs / Reports / Settings use their dedicated
fully-wired page implementations; About remains a lightweight
PagePlaceholder.
"""

from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QFont, QCloseEvent, QAction
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QFrame,
    QGridLayout,
    QMessageBox,
    QSizePolicy,
    QSpacerItem,
    QScrollArea,
    QMenu,
    QDialog,
)

from ai.predictor import get_predictor
from app.ui.pages.network_monitor_page import NetworkMonitorPage
from app.ui.login_panel import LoginPanel
from web.auth_manager import get_auth_manager
from app.core.engine import DashboardSnapshot, MonitoringEngine, get_engine
from app.core.notifier import DesktopNotifier
from app.ui.pages.threat_detection_page import ThreatDetectionPage
from app.ui.pages.logs_page import LogsPage
from app.ui.pages.reports_page import ReportsPage
from app.ui.pages.settings_page import SettingsPage
from app.ui.widgets.network_graph_widget import NetworkGraphWidget
from app.ui.widgets.network_topology_widget import NetworkTopologyWidget, TopologyConnection
from app.ui.widgets.recent_alerts_widget import RecentAlertsWidget
from app.ui.widgets.system_info_widget import SystemInfoWidget
from app.ui.widgets.threat_timeline_widget import ThreatTimelineWidget
from config.settings import AppInfo, ThemeConfig, UIConfig
from security.alert_manager import Alert, get_alert_manager
from security.risk_analyzer import SeverityLevel
from security.threat_detector import DetectionResult, get_threat_detector
from utils.logger import get_logger

logger = get_logger(__name__)

# Number of most-recent detections pulled each tick to feed the
# Network Topology widget. Kept small and local to this module since
# it's a pure UI-refresh-rate concern, not an application setting.
_TOPOLOGY_SAMPLE_LIMIT: int = 20


def _format_bandwidth(bytes_per_second: float) -> str:
    """
    Formats a bytes/sec value as a compact, human-readable string,
    switching from KB/s to MB/s once the value is large enough that
    KB/s would be unwieldy. Shared by the dashboard's Bandwidth stat
    card so its formatting matches what a real monitoring tool shows.

    Args:
        bytes_per_second: The raw bandwidth reading in bytes/sec.

    Returns:
        A formatted string, e.g. "128 KB/s" or "4.2 MB/s".
    """
    kb_per_second = bytes_per_second / 1024.0
    if kb_per_second >= 1024.0:
        return f"{kb_per_second / 1024.0:.1f} MB/s"
    return f"{kb_per_second:.0f} KB/s"


class SidebarButton(QPushButton):
    """
    A single navigation entry in the sidebar.

    Behaves like a toggle button (checkable) so the currently active
    page can be visually highlighted. Styling is applied globally via
    the MainWindow's stylesheet using the 'sidebarButton' objectName
    and dynamic 'active' property.
    """

    def __init__(self, label: str, icon_text: str, parent: Optional[QWidget] = None) -> None:
        """
        Args:
            label: The visible navigation label (e.g., "Dashboard").
            icon_text: A short glyph/emoji-style marker shown before
                the label (keeps the UI recognizable without bundling
                icon image assets for every entry).
            parent: Optional parent widget.
        """
        super().__init__(f"  {icon_text}   {label}", parent)
        self.setObjectName("sidebarButton")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(46)
        self.setFont(QFont(ThemeConfig.FONT_FAMILY, 10))
        self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)


class StatusPill(QLabel):
    """
    A small rounded label used to display live status text (e.g.,
    "AI: ACTIVE", "MONITORING: OFF") with a colored border/text that
    reflects state (safe/neon, warning, alert).
    """

    def __init__(self, text: str, color: str, parent: Optional[QWidget] = None) -> None:
        """
        Args:
            text: The status text to display.
            color: Hex color string used for the border/text accent.
            parent: Optional parent widget.
        """
        super().__init__(text, parent)
        self._color = color
        self.setObjectName("statusPill")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFont(QFont(ThemeConfig.FONT_FAMILY, 9, QFont.Weight.DemiBold))
        self.set_state(text, color)

    def set_state(self, text: str, color: str) -> None:
        """
        Updates both the label text and its accent color at runtime.

        Args:
            text: New status text.
            color: New accent hex color.
        """
        self._color = color
        self.setText(text)
        self.setStyleSheet(
            f"""
            QLabel#statusPill {{
                color: {color};
                border: 1px solid {color};
                border-radius: 10px;
                padding: 4px 12px;
                background-color: {ThemeConfig.BACKGROUND_CARD};
            }}
            """
        )


class StatCard(QFrame):
    """
    A rounded metric card used in the top row of the dashboard page,
    displaying a title, a large value, and a small subtitle/unit.
    """

    def __init__(
        self,
        title: str,
        initial_value: str = "--",
        accent_color: str = ThemeConfig.NEON_BLUE,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Args:
            title: The metric's label (e.g., "Packets Captured").
            initial_value: The initially displayed value string.
            accent_color: Hex color used for the value text and left border.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setObjectName("statCard")
        self._accent_color = accent_color
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(6)

        self._title_label = QLabel(title.upper())
        self._title_label.setObjectName("statCardTitle")
        self._title_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 9, QFont.Weight.Medium))

        self._value_label = QLabel(initial_value)
        self._value_label.setObjectName("statCardValue")
        self._value_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 24, QFont.Weight.Bold))
        self._value_label.setStyleSheet(f"color: {accent_color};")

        layout.addWidget(self._title_label)
        layout.addWidget(self._value_label)
        layout.addStretch()

        self.setStyleSheet(
            f"""
            QFrame#statCard {{
                background-color: {ThemeConfig.BACKGROUND_CARD};
                border: 1px solid {ThemeConfig.BORDER_COLOR};
                border-left: 3px solid {accent_color};
                border-radius: 10px;
            }}
            QLabel#statCardTitle {{
                color: {ThemeConfig.TEXT_SECONDARY};
                letter-spacing: 1px;
            }}
            """
        )

    def update_value(self, value: str) -> None:
        """
        Updates the displayed metric value.

        Args:
            value: The new value string to display (e.g., "1,204").
        """
        self._value_label.setText(value)


class PagePlaceholder(QWidget):
    """
    Temporary placeholder content for a sidebar page that has not yet
    been implemented with its full widget set.
    """

    def __init__(self, page_title: str, description: str, parent: Optional[QWidget] = None) -> None:
        """
        Args:
            page_title: The page's display name (e.g., "Network Monitor").
            description: A short explanatory line shown beneath the title.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel(page_title)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont(ThemeConfig.FONT_FAMILY, 20, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY};")

        subtitle = QLabel(description)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setFont(QFont(ThemeConfig.FONT_FAMILY, 11))
        subtitle.setStyleSheet(f"color: {ThemeConfig.TEXT_SECONDARY};")

        layout.addWidget(title)
        layout.addWidget(subtitle)


class DashboardPage(QWidget):
    """
    The primary Dashboard page.

    Layout order:
        1. Live Security Overview
        2. Six security/system statistic cards
        3. Live Network Traffic + System Information
        4. Threat Timeline + Network Topology
        5. Recent Alerts

    The complete dashboard content is hosted inside a vertical
    QScrollArea. This prevents the Recent Alerts panel from being
    squeezed over the graph widgets on smaller screens or when the
    application window is resized.
    """

    def __init__(self, engine: MonitoringEngine, parent: Optional[QWidget] = None) -> None:
        """
        Args:
            engine: The application's shared MonitoringEngine instance.
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._engine = engine

        # --------------------------------------------------------------
        # Root layout
        # --------------------------------------------------------------
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # --------------------------------------------------------------
        # Scrollable dashboard container
        # --------------------------------------------------------------
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("dashboardScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        self.content_widget = QWidget()
        self.content_widget.setObjectName("dashboardContent")

        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(24, 20, 24, 24)
        self.content_layout.setSpacing(16)

        # --------------------------------------------------------------
        # Heading
        # --------------------------------------------------------------
        heading = QLabel("Live Security Overview")
        heading.setFont(
            QFont(ThemeConfig.FONT_FAMILY, 16, QFont.Weight.Bold)
        )
        heading.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY};")
        self.content_layout.addWidget(heading)

        # --------------------------------------------------------------
        # Six statistic cards
        # --------------------------------------------------------------
        self.content_layout.addLayout(self._build_stat_card_grid())

        # --------------------------------------------------------------
        # Row 1: Live Network Traffic + System Information
        # --------------------------------------------------------------
        middle_row_one = QHBoxLayout()
        middle_row_one.setSpacing(16)

        self.network_graph = NetworkGraphWidget()
        self.system_info = SystemInfoWidget()

        # Fixed minimum heights keep the graph and system panel readable
        # instead of allowing the layout to collapse vertically.
        self.network_graph.setMinimumHeight(300)
        self.system_info.setMinimumHeight(300)

        self.network_graph.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.system_info.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        middle_row_one.addWidget(self.network_graph, stretch=2)
        middle_row_one.addWidget(self.system_info, stretch=1)
        self.content_layout.addLayout(middle_row_one)

        # --------------------------------------------------------------
        # Row 2: Threat Timeline + Network Topology
        # --------------------------------------------------------------
        middle_row_two = QHBoxLayout()
        middle_row_two.setSpacing(16)

        self.threat_timeline = ThreatTimelineWidget()
        self.network_topology = NetworkTopologyWidget()

        self.threat_timeline.setMinimumHeight(280)
        self.network_topology.setMinimumHeight(280)

        self.threat_timeline.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.network_topology.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        middle_row_two.addWidget(self.threat_timeline, stretch=2)
        middle_row_two.addWidget(self.network_topology, stretch=1)
        self.content_layout.addLayout(middle_row_two)

        # --------------------------------------------------------------
        # Recent Alerts
        # --------------------------------------------------------------
        # IMPORTANT: Recent Alerts is deliberately added AFTER both
        # graph rows. It is now part of the normal vertical flow and
        # cannot overlap either graph.
        self.recent_alerts_section = QFrame()
        self.recent_alerts_section.setObjectName("recentAlertsSection")

        recent_alerts_section_layout = QVBoxLayout(self.recent_alerts_section)
        recent_alerts_section_layout.setContentsMargins(8, 8, 8, 8)
        recent_alerts_section_layout.setSpacing(8)

        recent_alerts_header = QHBoxLayout()
        recent_alerts_header.setContentsMargins(0, 0, 0, 0)

        recent_alerts_title = QLabel("Recent Alerts")
        recent_alerts_title.setFont(
            QFont(ThemeConfig.FONT_FAMILY, 14, QFont.Weight.DemiBold)
        )
        recent_alerts_title.setStyleSheet(
            f"color: {ThemeConfig.TEXT_PRIMARY};"
        )

        recent_alerts_header.addWidget(recent_alerts_title)
        recent_alerts_header.addStretch()

        self.recent_alerts_toggle = QPushButton("−")
        self.recent_alerts_toggle.setObjectName("recentAlertsToggle")
        self.recent_alerts_toggle.setFixedSize(34, 30)
        self.recent_alerts_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.recent_alerts_toggle.setToolTip("Minimize Recent Alerts")
        self.recent_alerts_toggle.clicked.connect(self._toggle_recent_alerts)
        recent_alerts_header.addWidget(self.recent_alerts_toggle)
        recent_alerts_section_layout.addLayout(recent_alerts_header)

        self.recent_alerts = RecentAlertsWidget()
        self.recent_alerts.setMinimumHeight(240)
        self.recent_alerts.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        recent_alerts_section_layout.addWidget(self.recent_alerts)

        self.content_layout.addWidget(self.recent_alerts_section)
        self.content_layout.addSpacing(10)
        self._recent_alerts_expanded = True

        # Install content in the scroll area only after the complete
        # vertical layout has been constructed.
        self.scroll_area.setWidget(self.content_widget)
        root_layout.addWidget(self.scroll_area)

        # --------------------------------------------------------------
        # Engine wiring + initial data
        # --------------------------------------------------------------
        self._connect_engine_signals()
        self._load_initial_data()

    # ------------------------------------------------------------------
    # Recent Alerts controls
    # ------------------------------------------------------------------

    def _toggle_recent_alerts(self) -> None:
        """Expand or minimize the Recent Alerts list without overlaying other widgets."""
        self._recent_alerts_expanded = not self._recent_alerts_expanded
        self.recent_alerts.setVisible(self._recent_alerts_expanded)
        if self._recent_alerts_expanded:
            self.recent_alerts_toggle.setText("−")
            self.recent_alerts_toggle.setToolTip("Minimize Recent Alerts")
        else:
            self.recent_alerts_toggle.setText("+")
            self.recent_alerts_toggle.setToolTip("Expand Recent Alerts")
        self.recent_alerts_section.adjustSize()
        self.content_widget.adjustSize()

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_stat_card_grid(self) -> QGridLayout:
        """Builds the six dashboard statistic cards."""
        self.cards: dict[str, StatCard] = {
            "packets": StatCard(
                "Packets Captured", "0", ThemeConfig.NEON_BLUE
            ),
            "threats": StatCard(
                "Threats Detected", "0", ThemeConfig.ALERT_RED
            ),
            "risk_score": StatCard(
                "Risk Score", "0 / 100", ThemeConfig.WARNING_ORANGE
            ),
            "cpu": StatCard(
                "CPU Usage", "0%", ThemeConfig.NEON_CYAN
            ),
            "ram": StatCard(
                "RAM Usage", "0%", ThemeConfig.NEON_CYAN
            ),
            "bandwidth": StatCard(
                "Bandwidth", "0 KB/s", ThemeConfig.ACCENT_PURPLE
            ),
        }

        card_grid = QGridLayout()
        card_grid.setSpacing(16)

        positions = [
            (0, 0), (0, 1), (0, 2),
            (1, 0), (1, 1), (1, 2),
        ]

        for (row, col), card in zip(positions, self.cards.values()):
            card_grid.addWidget(card, row, col)

        return card_grid

    # ------------------------------------------------------------------
    # Engine wiring
    # ------------------------------------------------------------------

    def _connect_engine_signals(self) -> None:
        """Connect MonitoringEngine signals to dashboard widgets."""
        self._engine.dashboard_updated.connect(self._on_dashboard_updated)
        self._engine.threat_detected.connect(self._on_threat_detected)
        self._engine.alert_created.connect(self._on_alert_created)
        self.recent_alerts.alert_clicked.connect(self._on_alert_clicked)

    def _load_initial_data(self) -> None:
        """Populate the dashboard immediately with the current engine state."""
        self.system_info.set_device_info(self._engine.get_device_info())
        self.recent_alerts.set_alerts(
            get_alert_manager().get_recent_alerts(limit=20)
        )
        self._on_dashboard_updated(self._engine.get_current_snapshot())

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_dashboard_updated(self, snapshot: DashboardSnapshot) -> None:
        """Refresh all live dashboard widgets from the latest snapshot."""
        threat_summary = snapshot.threat_summary

        self.cards["packets"].update_value(
            f"{snapshot.packets_captured:,}"
        )
        self.cards["threats"].update_value(
            f"{threat_summary.get('total_threats_detected', 0):,}"
        )
        self.cards["risk_score"].update_value(
            f"{threat_summary.get('current_risk_score', 0)} / 100"
        )
        self.cards["cpu"].update_value(
            f"{snapshot.system_stats.cpu_percent:.0f}%"
        )
        self.cards["ram"].update_value(
            f"{snapshot.system_stats.ram_percent:.0f}%"
        )
        self.cards["bandwidth"].update_value(
            _format_bandwidth(snapshot.bytes_per_second)
        )

        self.network_graph.update_data(
            snapshot.packets_per_second,
            snapshot.bytes_per_second,
        )

        self.threat_timeline.update_risk_score(
            threat_summary.get("current_risk_score", 0),
            threat_summary.get(
                "current_severity",
                SeverityLevel.SAFE.value,
            ),
        )

        self.system_info.update_system_stats(
            snapshot.system_stats
        )

        self._refresh_topology(snapshot)

    def _refresh_topology(self, snapshot: DashboardSnapshot) -> None:
        """Refresh Network Topology from the latest threat detections."""
        recent_detections: List[DetectionResult] = (
            get_threat_detector().get_recent_detections(
                limit=_TOPOLOGY_SAMPLE_LIMIT
            )
        )

        connections = [
            TopologyConnection(
                remote_ip=result.packet.dst_ip,
                is_threat=result.is_threat,
                protocol=result.packet.protocol,
                port=result.packet.dst_port,
            )
            for result in recent_detections
        ]

        self.network_topology.update_connections(
            local_ip=snapshot.device_info.ip_address,
            connections=connections,
            local_label=snapshot.device_info.hostname,
        )

    def _on_threat_detected(self, result: DetectionResult) -> None:
        """Add an immediate threat event to the threat timeline."""
        self.threat_timeline.add_threat_event(
            result.risk_assessment.risk_score,
            result.risk_assessment.severity.value,
        )

    def _on_alert_created(self, alert: Alert) -> None:
        """Prepend a newly created alert to Recent Alerts."""
        self.recent_alerts.add_alert(alert)

    def _on_alert_clicked(self, alert_id: str) -> None:
        """Mark a clicked alert as read through AlertManager."""
        get_alert_manager().mark_as_read(alert_id)

    def reset_live_widgets(self) -> None:
        """Clear rolling graphs and topology when monitoring restarts."""
        self.network_graph.clear()
        self.threat_timeline.clear()
        self.network_topology.clear()


class Sidebar(QFrame):
    """
    The left-hand navigation panel. Displays the application brand,
    a list of SidebarButton entries, and emits page-change requests
    to the MainWindow via a callback.
    """

    NAV_ITEMS: list[tuple[str, str]] = [
        ("Dashboard", "🛡"),
        ("Threat Detection", "⚠"),
        ("Network Monitor", "🌐"),
        ("Logs", "📜"),
        ("Reports", "📊"),
        ("Settings", "⚙"),
        ("About", "ℹ"),
    ]

    def __init__(self, on_navigate: Callable[[str], None], parent: Optional[QWidget] = None) -> None:
        """
        Args:
            on_navigate: Callback invoked with the nav item label
                whenever the user selects a different page.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(UIConfig.SIDEBAR_WIDTH_EXPANDED)

        self._on_navigate = on_navigate
        self._buttons: dict[str, SidebarButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        layout.addWidget(self._build_brand_section())
        layout.addWidget(self._build_separator())

        for label, icon in self.NAV_ITEMS:
            button = SidebarButton(label, icon)
            button.clicked.connect(lambda checked, name=label: self._handle_click(name))
            layout.addWidget(button)
            self._buttons[label] = button

        layout.addStretch()

        self._buttons["Dashboard"].setChecked(True)

    def _build_brand_section(self) -> QWidget:
        """
        Returns:
            A QWidget containing the top brand/logo section layout.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(2)

        name_label = QLabel(f"🛡  {AppInfo.NAME}")
        name_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 13, QFont.Weight.Bold))
        name_label.setStyleSheet(f"color: {ThemeConfig.NEON_BLUE};")

        tagline_label = QLabel("AI Network Defense")
        tagline_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 8))
        tagline_label.setStyleSheet(f"color: {ThemeConfig.TEXT_SECONDARY};")

        layout.addWidget(name_label)
        layout.addWidget(tagline_label)
        return container

    def _build_separator(self) -> QFrame:
        """Returns a thin horizontal divider line matching the theme border color."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background-color: {ThemeConfig.BORDER_COLOR}; max-height: 1px;")
        return line

    def _handle_click(self, label: str) -> None:
        """
        Handles a sidebar button click: ensures only the selected
        button remains checked, then forwards the navigation request.

        Args:
            label: The clicked navigation item's label.
        """
        for name, button in self._buttons.items():
            button.setChecked(name == label)
        self._on_navigate(label)


class Header(QFrame):
    """
    The top header bar. Shows the current page title on the left, and
    the AI status pill, monitoring status pill, a Start/Stop
    Monitoring toggle button, and the notification bell (with an
    unread badge) on the right — all wired to live engine state by
    MainWindow.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("header")
        self.setFixedHeight(64)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 0, 24, 0)
        layout.setSpacing(14)

        self.page_title_label = QLabel("Dashboard")
        self.page_title_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 15, QFont.Weight.DemiBold))
        self.page_title_label.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY};")

        layout.addWidget(self.page_title_label)
        layout.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))

        self.ai_status_pill = StatusPill("AI: LOADING", ThemeConfig.TEXT_SECONDARY)
        self.monitor_status_pill = StatusPill("MONITORING: OFF", ThemeConfig.TEXT_SECONDARY)

        self.monitor_toggle_button = QPushButton("▶  Start Monitoring")
        self.monitor_toggle_button.setObjectName("monitorToggleButton")
        self.monitor_toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.monitor_toggle_button.setFont(QFont(ThemeConfig.FONT_FAMILY, 9, QFont.Weight.DemiBold))
        self._style_toggle_button(is_monitoring=False)

        self.notification_button = QPushButton("🔔")
        self.notification_button.setObjectName("notificationButton")
        self.notification_button.setFixedSize(QSize(36, 36))
        self.notification_button.setCursor(Qt.CursorShape.PointingHandCursor)

        self.notification_badge = QLabel("0")
        self.notification_badge.setObjectName("notificationBadge")
        self.notification_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.notification_badge.setFixedSize(QSize(16, 16))
        self._style_notification_badge(count=0)

        # Account button: clicking the user identity opens the account/action menu.
        # There is intentionally no separate Logout button in the header.
        self.user_button = QPushButton("👤  Guest  ▾")
        self.user_button.setObjectName("headerUserButton")
        self.user_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.user_button.setFont(
            QFont(ThemeConfig.FONT_FAMILY, 9, QFont.Weight.DemiBold)
        )
        self.user_button.setToolTip("Account and application options")
        self.user_button.setMinimumWidth(120)

        bell_container = QWidget()
        bell_layout = QHBoxLayout(bell_container)
        bell_layout.setContentsMargins(0, 0, 0, 0)
        bell_layout.setSpacing(0)
        bell_layout.addWidget(self.notification_button)

        layout.addWidget(self.ai_status_pill)
        layout.addWidget(self.monitor_status_pill)
        layout.addWidget(self.monitor_toggle_button)
        layout.addWidget(bell_container)
        layout.addWidget(self.notification_badge)
        layout.addWidget(self.user_button)

    def set_authenticated_user(self, username: str, role: str) -> None:
        """Display the authenticated desktop user in the account button."""
        role_text = role.replace("_", " ").title() if role else "User"
        self.user_button.setText(f"👤  {username}  ▾")
        self.user_button.setToolTip(f"Signed in as {username} ({role_text})")

    def clear_authenticated_user(self) -> None:
        """Clear the header identity after logout/switch-account."""
        self.user_button.setText("👤  Guest  ▾")
        self.user_button.setToolTip("Account and application options")

    def set_page_title(self, title: str) -> None:
        """
        Updates the header's page title label when navigation occurs.

        Args:
            title: The new page title text.
        """
        self.page_title_label.setText(title)

    def set_ai_status(self, model_name: str) -> None:
        """
        Updates the AI status pill once the active model is known
        (called once, shortly after the engine is constructed).

        Args:
            model_name: The display name of the currently active AI model.
        """
        self.ai_status_pill.set_state(f"AI: {model_name.upper()}", ThemeConfig.NEON_CYAN)

    def set_monitoring_active(self, is_active: bool) -> None:
        """
        Updates the monitoring status pill and the toggle button's
        text/styling to reflect whether packet capture is running.

        Args:
            is_active: True if monitoring is currently active.
        """
        if is_active:
            self.monitor_status_pill.set_state("MONITORING: ON", ThemeConfig.SAFE_GREEN)
            self.monitor_toggle_button.setText("■  Stop Monitoring")
        else:
            self.monitor_status_pill.set_state("MONITORING: OFF", ThemeConfig.TEXT_SECONDARY)
            self.monitor_toggle_button.setText("▶  Start Monitoring")

        self._style_toggle_button(is_monitoring=is_active)

    def set_unread_count(self, count: int) -> None:
        """
        Updates the notification bell's unread badge.

        Args:
            count: The current number of unread alerts.
        """
        self._style_notification_badge(count)

    def _style_toggle_button(self, is_monitoring: bool) -> None:
        """
        Applies color styling to the monitor toggle button: neon blue
        when idle (inviting a "Start"), alert red when active
        (signaling a live, ongoing capture session).

        Args:
            is_monitoring: True if monitoring is currently active.
        """
        color = ThemeConfig.ALERT_RED if is_monitoring else ThemeConfig.NEON_BLUE
        self.monitor_toggle_button.setStyleSheet(
            f"""
            QPushButton#monitorToggleButton {{
                color: {color};
                border: 1px solid {color};
                border-radius: 6px;
                padding: 6px 14px;
                background-color: {ThemeConfig.BACKGROUND_CARD};
            }}
            QPushButton#monitorToggleButton:hover {{
                background-color: {color};
                color: {ThemeConfig.BACKGROUND_PRIMARY};
            }}
            """
        )

    def _style_notification_badge(self, count: int) -> None:
        """
        Updates the small circular unread-count badge overlapping the
        notification bell. Hides itself entirely at zero unread, so
        the bell reads as "clear" rather than showing a "0" badge.

        Args:
            count: The current unread alert count.
        """
        self.notification_badge.setVisible(count > 0)
        self.notification_badge.setText(str(min(count, 99)))
        self.notification_badge.setStyleSheet(
            f"""
            QLabel#notificationBadge {{
                background-color: {ThemeConfig.ALERT_RED};
                color: white;
                border-radius: 8px;
                font-size: 9px;
                font-weight: 600;
            }}
            """
        )


class MainWindow(QMainWindow):
    """
    The top-level application window for CyberSentinel AI.

    Owns the single MonitoringEngine and DesktopNotifier instances,
    assembles the Sidebar, Header, and QStackedWidget of pages, wires
    header controls (monitoring toggle, AI status, notification
    badge) to live engine signals, and handles graceful shutdown of
    all background subsystems when the window is closed.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(UIConfig.WINDOW_TITLE)
        self.setMinimumSize(UIConfig.WINDOW_MIN_WIDTH, UIConfig.WINDOW_MIN_HEIGHT)

        # MonitoringEngine is a QObject and must be constructed after
        # QApplication exists (guaranteed here, since MainWindow itself
        # is only ever constructed from main.py after the QApplication
        # is created) and BEFORE _build_layout(), since DashboardPage
        # and ThreatDetectionPage both require it at construction time.
        self.engine: MonitoringEngine = get_engine()

        self._pages: dict[str, QWidget] = {}
        self._stacked_widget = QStackedWidget()

        self._auth = get_auth_manager()
        self._authenticated_username: Optional[str] = None
        self._authenticated_role: Optional[str] = None
        self._session_token: Optional[str] = None
        self._authenticated = False
        self._dark_mode = True

        # SettingsPage receives the notifier during page construction, so
        # DesktopNotifier must exist BEFORE _build_layout() calls
        # _register_pages(). self and self.engine already exist here.
        self.notifier = DesktopNotifier(main_window=self, engine=self.engine)
        self.engine.alert_created.connect(self.notifier.handle_alert)

        self._build_layout()
        self._apply_theme()
        self._connect_header_signals()
        self._initialize_header_state()
        self.header.user_button.clicked.connect(self._show_account_menu)
        self._show_login_page()

        logger.info("MainWindow initialized.")

    def _build_layout(self) -> None:
        """
        Constructs the root layout: sidebar on the left, and a
        vertical stack of (header, page content) on the right.
        """
        central_widget = QWidget()
        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.sidebar = Sidebar(on_navigate=self._navigate_to)
        root_layout.addWidget(self.sidebar)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self.header = Header()
        right_layout.addWidget(self.header)
        self.header_separator = self._build_separator()
        right_layout.addWidget(self.header_separator)
        right_layout.addWidget(self._stacked_widget)

        self.sidebar.hide()
        self.header.hide()
        self.header_separator.hide()

        root_layout.addWidget(right_container)

        self.setCentralWidget(central_widget)
        self._register_pages()

    def _build_separator(self) -> QFrame:
        """Returns a thin horizontal divider placed beneath the header."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"background-color: {ThemeConfig.BORDER_COLOR}; max-height: 1px;")
        return line

    def _register_pages(self) -> None:
        """
        Creates each sidebar-navigable page and adds it to the
        QStackedWidget. All implemented application pages are wired
        here so navigation opens the real UI instead of placeholders.
        """
        self.login_page = LoginPanel()
        self.login_page.authenticated.connect(self._on_authenticated)
        self._stacked_widget.addWidget(self.login_page)

        self.dashboard_page = DashboardPage(engine=self.engine)
        self._pages["Dashboard"] = self.dashboard_page

        self.threat_detection_page = ThreatDetectionPage(engine=self.engine)
        self._pages["Threat Detection"] = self.threat_detection_page

        self.network_monitor_page = NetworkMonitorPage(engine=self.engine)
        self._pages["Network Monitor"] = self.network_monitor_page
        self.logs_page = LogsPage(engine=self.engine)
        self._pages["Logs"] = self.logs_page

        self.reports_page = ReportsPage(engine=self.engine)
        self._pages["Reports"] = self.reports_page

        self.settings_page = SettingsPage(
            engine=self.engine,
            notifier=self.notifier,
        )
        self._pages["Settings"] = self.settings_page
        self._pages["About"] = PagePlaceholder(
            "About",
            f"{AppInfo.NAME} v{AppInfo.VERSION} — {AppInfo.DESCRIPTION}",
        )

        for page_widget in self._pages.values():
            self._stacked_widget.addWidget(page_widget)

    def _show_login_page(self) -> None:
        """Show the embedded login/signup page and hide authenticated UI chrome."""
        self._authenticated = False
        self._stacked_widget.setCurrentWidget(self.login_page)
        self.header.set_page_title("Sign In")
        self.header.clear_authenticated_user()
        self.sidebar.hide()
        self.header.hide()
        self.header_separator.hide()

    def _on_authenticated(self, username: str, role: str, token: str) -> None:
        """Validate the emitted session and reveal the authenticated dashboard."""
        if not token or not self._auth.validate_session(token):
            logger.warning("LoginPanel emitted an invalid session token; access denied.")
            return

        self._authenticated = True
        self._authenticated_username = username
        self._authenticated_role = role
        self._session_token = token

        self.header.set_authenticated_user(username, role)
        self.sidebar.show()
        self.header.show()
        self.header_separator.show()
        self._navigate_to("Dashboard")
        logger.info("Desktop session established for '%s' (role=%s).", username, role)

    def _logout(self) -> None:
        """Invalidate the current session and return to the login page."""
        username = self._authenticated_username or "unknown"

        if self.engine.is_monitoring:
            self.engine.stop_monitoring()

        if self._session_token:
            try:
                self._auth.logout(self._session_token)
            except Exception as exc:
                logger.warning("Session logout failed: %s", exc)

        self._session_token = None
        self._authenticated_username = None
        self._authenticated_role = None
        self.dashboard_page.reset_live_widgets()
        self.login_page.reset_for_next_login()
        self._show_login_page()
        logger.info("Desktop session ended for '%s'.", username)

    def _show_account_menu(self) -> None:
        """Show the account menu from the header user button."""
        if not self._authenticated:
            return

        menu = QMenu(self)
        menu.setObjectName("accountMenu")

        username = self._authenticated_username or "User"
        email = self._auth.get_email_for_session(self._session_token or "") or ""
        role = (self._authenticated_role or "user").replace("_", " ").title()

        account_action = QAction(f"👤  {username}", menu)
        account_action.setEnabled(False)
        menu.addAction(account_action)
        if email:
            email_action = QAction(f"✉  {email}", menu)
            email_action.setEnabled(False)
            menu.addAction(email_action)
        role_action = QAction(f"🔐  Role: {role}", menu)
        role_action.setEnabled(False)
        menu.addAction(role_action)
        menu.addSeparator()

        dark_action = QAction("🌙  Dark Mode", menu)
        dark_action.setCheckable(True)
        dark_action.setChecked(self._dark_mode)
        dark_action.triggered.connect(self._toggle_dark_mode)
        menu.addAction(dark_action)

        settings_action = QAction("⚙  Settings", menu)
        settings_action.triggered.connect(lambda: self._navigate_to("Settings"))
        menu.addAction(settings_action)

        about_action = QAction("ℹ  About CyberSentinel AI", menu)
        about_action.triggered.connect(lambda: self._navigate_to("About"))
        menu.addAction(about_action)

        menu.addSeparator()

        switch_action = QAction("⇄  Switch Account", menu)
        switch_action.triggered.connect(self._switch_account)
        menu.addAction(switch_action)

        logout_action = QAction("⏻  Logout", menu)
        logout_action.triggered.connect(self._logout)
        menu.addAction(logout_action)

        menu.exec(self.header.user_button.mapToGlobal(self.header.user_button.rect().bottomLeft()))

    def _toggle_dark_mode(self, enabled: bool) -> None:
        """Switch the application shell between dark and light appearance."""
        self._dark_mode = enabled
        self._apply_theme()

    def _switch_account(self) -> None:
        """End the current session and return to the account chooser/sign-in page."""
        self._logout()

    def _navigate_to(self, page_label: str) -> None:
        """
        Switches the visible page in the QStackedWidget and updates
        the header title to match.

        Args:
            page_label: The sidebar navigation label that was clicked.
        """
        if not self._authenticated:
            return

        page_widget = self._pages.get(page_label)
        if page_widget is None:
            logger.warning("Navigation requested for unknown page: %s", page_label)
            return

        self._stacked_widget.setCurrentWidget(page_widget)
        self.header.set_page_title(page_label)
        logger.debug("Navigated to page: %s", page_label)

    # ------------------------------------------------------------------
    # Engine <-> Header wiring
    # ------------------------------------------------------------------

    def _connect_header_signals(self) -> None:
        """
        Wires the header's monitoring toggle button to
        MonitoringEngine.start_monitoring()/stop_monitoring(), and
        connects the engine's lifecycle/error signals back to header
        state updates and user-facing error dialogs.
        """
        self.header.monitor_toggle_button.clicked.connect(self._on_monitor_toggle_clicked)
        self.header.notification_button.clicked.connect(self._on_notification_clicked)

        self.engine.monitoring_started.connect(self._on_monitoring_started)
        self.engine.monitoring_stopped.connect(self._on_monitoring_stopped)
        self.engine.capture_error.connect(self._on_capture_error)
        self.engine.alert_created.connect(self._on_alert_created_for_badge)

    def _initialize_header_state(self) -> None:
        """
        Populates the header's AI status pill and notification badge
        with their initial values immediately on startup, rather than
        waiting for the first relevant signal to fire.
        """
        predictor = get_predictor()
        self.header.set_ai_status(predictor.active_model_name)
        self.header.set_monitoring_active(self.engine.is_monitoring)
        self.header.set_unread_count(get_alert_manager().get_unread_count())

    def _on_monitor_toggle_clicked(self) -> None:
        """
        Handles a click on the header's Start/Stop Monitoring button:
        starts or stops the engine's packet capture pipeline based on
        its current state.
        """
        if self.engine.is_monitoring:
            self.engine.stop_monitoring()
        else:
            self.engine.start_monitoring()

    def _on_monitoring_started(self) -> None:
        """Handles MonitoringEngine.monitoring_started: updates the header state."""
        self.header.set_monitoring_active(True)

    def _on_monitoring_stopped(self) -> None:
        """
        Handles MonitoringEngine.monitoring_stopped: updates the
        header state and clears the dashboard's rolling live widgets
        so a subsequent restart begins from a visually clean trend.
        """
        self.header.set_monitoring_active(False)
        self.dashboard_page.reset_live_widgets()

    def _on_capture_error(self, message: str) -> None:
        """
        Handles MonitoringEngine.capture_error: shows a warning dialog
        explaining why packet capture failed to start (e.g., missing
        Npcap or insufficient privileges), rather than failing silently.

        Args:
            message: The human-readable error message from the engine.
        """
        QMessageBox.warning(self, f"{AppInfo.NAME} - Monitoring Error", message)

    def _on_notification_clicked(self) -> None:
        """
        Opens the notification center when the user clicks the bell icon.
        The dialog shows recent alerts, allows per-alert read-state updates,
        and provides a quick "Mark all as read" action for the unread count.
        """
        alert_manager = get_alert_manager()
        alerts = alert_manager.get_recent_alerts(limit=20)

        if not alerts:
            QMessageBox.information(
                self,
                f"{AppInfo.NAME} - Notifications",
                "No alerts yet.",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"{AppInfo.NAME} - Notifications")
        dialog.resize(560, 420)
        dialog.setModal(False)

        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(12, 12, 12, 12)
        dialog_layout.setSpacing(10)

        panel = RecentAlertsWidget(dialog)
        panel.set_alerts(alerts)

        def refresh_panel() -> None:
            panel.set_alerts(alert_manager.get_recent_alerts(limit=20))
            self.header.set_unread_count(alert_manager.get_unread_count())

        def handle_alert_click(alert_id: str) -> None:
            if alert_manager.mark_as_read(alert_id):
                refresh_panel()

        def mark_all_as_read() -> None:
            alert_manager.mark_all_as_read()
            refresh_panel()

        panel.alert_clicked.connect(handle_alert_click)
        dialog_layout.addWidget(panel)

        mark_all_button = QPushButton("Mark all as read")
        mark_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        mark_all_button.clicked.connect(mark_all_as_read)
        dialog_layout.addWidget(mark_all_button)

        dialog.exec()

    def _on_alert_created_for_badge(self, alert: Alert) -> None:
        """
        Handles MonitoringEngine.alert_created: refreshes the header's
        notification badge count. Kept separate from DashboardPage's
        own alert_created handler (which updates the Recent Alerts
        list) and from DesktopNotifier's own handler (which shows the
        OS toast), since the header badge is visible across every
        page, not just the Dashboard.

        Args:
            alert: The newly created Alert (unused directly; the
                badge reflects the authoritative unread count from
                AlertManager rather than a local increment, since
                mark_as_read() can reduce it independently of this signal).
        """
        self.header.set_unread_count(get_alert_manager().get_unread_count())

    def _apply_theme(self) -> None:
        """
        Applies the global QSS stylesheet to the entire window,
        derived from ThemeConfig color constants.
        """
        if self._dark_mode:
            bg_primary = ThemeConfig.BACKGROUND_PRIMARY
            bg_secondary = ThemeConfig.BACKGROUND_SECONDARY
            bg_card = ThemeConfig.BACKGROUND_CARD
            bg_sidebar = ThemeConfig.BACKGROUND_SIDEBAR
            border = ThemeConfig.BORDER_COLOR
            text_primary = ThemeConfig.TEXT_PRIMARY
            text_secondary = ThemeConfig.TEXT_SECONDARY
        else:
            bg_primary = "#f4f7fb"
            bg_secondary = "#ffffff"
            bg_card = "#ffffff"
            bg_sidebar = "#eaf0f7"
            border = "#cbd5e1"
            text_primary = "#0f172a"
            text_secondary = "#475569"

        self.setStyleSheet(
            f"""
            QMainWindow {{
                background-color: {bg_primary};
            }}

            QWidget {{
                background-color: {bg_primary};
                color: {text_primary};
                font-family: '{ThemeConfig.FONT_FAMILY}';
            }}

            QFrame#sidebar {{
                background-color: {bg_sidebar};
                border-right: 1px solid {border};
            }}

            QFrame#header {{
                background-color: {bg_secondary};
            }}

            QPushButton#sidebarButton {{
                background-color: transparent;
                color: {text_secondary};
                text-align: left;
                border: none;
                border-left: 3px solid transparent;
                padding-left: 6px;
            }}

            QPushButton#sidebarButton:hover {{
                background-color: {bg_card};
                color: {text_primary};
            }}

            QPushButton#sidebarButton:checked {{
                background-color: {bg_card};
                color: {ThemeConfig.NEON_BLUE};
                border-left: 3px solid {ThemeConfig.NEON_BLUE};
                font-weight: 600;
            }}

            QPushButton#notificationButton {{
                background-color: {bg_card};
                border: 1px solid {border};
                border-radius: 18px;
                font-size: 14px;
            }}

            QPushButton#notificationButton:hover {{
                border: 1px solid {ThemeConfig.NEON_BLUE};
            }}

            QFrame#recentAlertsSection {{
                background-color: {bg_card};
                border: 1px solid {border};
                border-radius: 10px;
            }}

            QPushButton#recentAlertsToggle {{
                background-color: {bg_secondary};
                color: {text_secondary};
                border: 1px solid {border};
                border-radius: 6px;
                font-size: 18px;
                font-weight: 600;
            }}

            QPushButton#recentAlertsToggle:hover {{
                color: {ThemeConfig.NEON_BLUE};
                border: 1px solid {ThemeConfig.NEON_BLUE};
                background-color: {bg_card};
            }}

            QPushButton#headerUserButton {{
                color: {text_secondary};
                background-color: {bg_card};
                border: 1px solid {border};
                border-radius: 7px;
                padding: 7px 10px;
            }}

            QPushButton#headerUserButton:hover {{
                color: {text_primary};
                border: 1px solid {ThemeConfig.NEON_BLUE};
                background-color: {bg_secondary};
            }}

            QMenu#accountMenu {{
                background-color: {bg_secondary};
                color: {text_primary};
                border: 1px solid {border};
                padding: 6px;
            }}

            QMenu#accountMenu::item {{
                padding: 8px 22px 8px 12px;
                border-radius: 5px;
            }}

            QMenu#accountMenu::item:selected {{
                background-color: {bg_card};
                color: {ThemeConfig.NEON_BLUE};
            }}

            QMenu#accountMenu::separator {{
                height: 1px;
                background: {border};
                margin: 5px 8px;
            }}

            QScrollBar:vertical {{
                background: {bg_primary};
                width: 10px;
            }}

            QScrollBar::handle:vertical {{
                background: {border};
                border-radius: 5px;
                min-height: 20px;
            }}

            QScrollBar::handle:vertical:hover {{
                background: {ThemeConfig.NEON_BLUE};
            }}
            """
        )

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Handles the window close request: stops every background
        subsystem owned by MonitoringEngine (packet capture thread,
        system monitor poll thread) before the window closes. The
        actual MongoDB disconnect happens afterward in main.py's own
        shutdown sequence.

        Args:
            event: The Qt close event to accept once cleanup is done.
        """
        logger.info("MainWindow closing. Shutting down MonitoringEngine...")
        self.engine.shutdown()
        event.accept()