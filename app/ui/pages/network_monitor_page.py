"""
app/ui/pages/network_monitor_page.py

Network Monitor page for CyberSentinel AI's sidebar navigation.

Responsibilities:
    - Render a dedicated, larger live network traffic graph (a second
      independent NetworkGraphWidget instance, separate from the one
      on the Dashboard page).
    - Show interface/session info: active capture interface, total
      packets captured, packets/sec, and active tracked flows.
    - Break down observed traffic by protocol (TCP/UDP/ICMP/Other).
    - Render a live-scrolling table of ALL recent traffic (not just
      threats, unlike app/ui/pages/threat_detection_page.py), newest
      first, color-coded Normal vs. Threat per row.

Unlike ThreatDetectionPage (which only ever receives threat-classified
results via MonitoringEngine.threat_detected), this page needs to see
EVERY processed packet. Rather than adding a new high-frequency
per-packet Qt signal to MonitoringEngine (which would flood the event
queue at real capture rates), this page samples
security.threat_detector.ThreatDetector.get_recent_detections() once
per dashboard tick — the same in-memory, all-traffic buffer already
used by app/ui/dashboard.py's DashboardPage for its Network Topology
widget — and tracks the last-seen timestamp to insert only genuinely
new rows each tick, avoiding duplicate display of already-shown packets.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.engine import DashboardSnapshot, MonitoringEngine
from app.ui.widgets.network_graph_widget import NetworkGraphWidget
from config.settings import ThemeConfig
from security.threat_detector import DetectionResult, get_threat_detector
from utils.logger import get_logger

logger = get_logger(__name__)

# Maximum number of rows kept in the live traffic table at once,
# matching the eviction pattern already used by
# app/ui/pages/threat_detection_page.py's MAX_DISPLAYED_ROWS.
MAX_DISPLAYED_ROWS: int = 200

# How many recent detections to sample from ThreatDetector's buffer
# each tick. Must be large enough that, at expected demo traffic
# rates, no packet is evicted from the buffer before this page has a
# chance to see it (the buffer itself defaults to 100 — see
# security/threat_detector.py's _DEFAULT_RECENT_DETECTIONS_LIMIT).
_SAMPLE_LIMIT: int = 100

_COLUMNS: List[str] = [
    "Time",
    "Source",
    "Destination",
    "Protocol",
    "Size (B)",
    "Classification",
    "Risk Score",
]

_NORMAL_ROW_COLOR: str = ThemeConfig.TEXT_PRIMARY
_THREAT_ROW_COLOR: str = ThemeConfig.ALERT_RED


class _InfoStat(QFrame):
    """
    A small stat card used in the Network Monitor page's summary row.
    Deliberately a local, lightweight equivalent of
    app/ui/pages/threat_detection_page.py's _SummaryStat rather than a
    shared import, since that class is a private (underscore-prefixed)
    implementation detail of its own module — each page owning its
    own small variant keeps pages independently self-contained.
    """

    def __init__(self, title: str, accent_color: str, parent: Optional[QWidget] = None) -> None:
        """
        Args:
            title: The stat's label (e.g., "Active Interface").
            accent_color: Hex color used for the value text and left border.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setObjectName("infoStat")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        title_label = QLabel(title.upper())
        title_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 8, QFont.Weight.Medium))
        title_label.setStyleSheet(f"color: {ThemeConfig.TEXT_SECONDARY}; background: transparent;")

        self._value_label = QLabel("--")
        self._value_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 16, QFont.Weight.Bold))
        self._value_label.setStyleSheet(f"color: {accent_color}; background: transparent;")
        self._value_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(self._value_label)

        self.setStyleSheet(
            f"""
            QFrame#infoStat {{
                background-color: {ThemeConfig.BACKGROUND_CARD};
                border: 1px solid {ThemeConfig.BORDER_COLOR};
                border-left: 3px solid {accent_color};
                border-radius: 8px;
            }}
            """
        )

    def update_value(self, value: str) -> None:
        """
        Updates the displayed stat value.

        Args:
            value: The new value string to display.
        """
        self._value_label.setText(value)


class NetworkMonitorPage(QWidget):
    """
    Sidebar page displaying detailed live network traffic: a dedicated
    traffic graph, interface/session info, protocol breakdown, and a
    live table of every recent connection (Normal and Threat alike).

    Like DashboardPage and ThreatDetectionPage, this page receives its
    MonitoringEngine via constructor injection from MainWindow.

    Usage:
        page = NetworkMonitorPage(engine=self.engine)
    """

    def __init__(self, engine: MonitoringEngine, parent: Optional[QWidget] = None) -> None:
        """
        Args:
            engine: The application's shared MonitoringEngine instance.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._engine = engine

        self._protocol_counts: Counter[str] = Counter()
        self._last_seen_timestamp: Optional[datetime] = None

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(24, 20, 24, 20)
        outer_layout.setSpacing(16)

        heading = QLabel("Network Traffic Monitor")
        heading.setFont(QFont(ThemeConfig.FONT_FAMILY, 16, QFont.Weight.Bold))
        heading.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY};")
        outer_layout.addWidget(heading)

        outer_layout.addLayout(self._build_info_row())
        outer_layout.addLayout(self._build_protocol_row())

        self.network_graph = NetworkGraphWidget()
        self.network_graph.setMinimumHeight(220)
        outer_layout.addWidget(self.network_graph)

        outer_layout.addWidget(self._build_table(), stretch=1)

        self._connect_engine_signals()
        self._load_initial_data()

        logger.debug("NetworkMonitorPage initialized.")

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_info_row(self) -> QHBoxLayout:
        """
        Builds the four-card interface/session info row.

        Returns:
            The populated QHBoxLayout, ready to add to the page.
        """
        self._info_cards: Dict[str, _InfoStat] = {
            "interface": _InfoStat("Active Interface", ThemeConfig.NEON_BLUE),
            "total_packets": _InfoStat("Total Packets", ThemeConfig.NEON_CYAN),
            "packets_per_second": _InfoStat("Packets/s", ThemeConfig.ACCENT_PURPLE),
            "active_flows": _InfoStat("Active Flows", ThemeConfig.WARNING_ORANGE),
        }

        row = QHBoxLayout()
        row.setSpacing(14)
        for card in self._info_cards.values():
            row.addWidget(card)

        return row

    def _build_protocol_row(self) -> QHBoxLayout:
        """
        Builds the four-card protocol breakdown row.

        Returns:
            The populated QHBoxLayout, ready to add to the page.
        """
        self._protocol_cards: Dict[str, _InfoStat] = {
            "TCP": _InfoStat("TCP Packets", ThemeConfig.NEON_BLUE),
            "UDP": _InfoStat("UDP Packets", ThemeConfig.NEON_CYAN),
            "ICMP": _InfoStat("ICMP Packets", ThemeConfig.ACCENT_PURPLE),
            "OTHER": _InfoStat("Other", ThemeConfig.TEXT_SECONDARY),
        }
        for card in self._protocol_cards.values():
            card.update_value("0")

        row = QHBoxLayout()
        row.setSpacing(14)
        for card in self._protocol_cards.values():
            row.addWidget(card)

        return row

    def _build_table(self) -> QTableWidget:
        """
        Builds and styles the live traffic table.

        Returns:
            The configured QTableWidget, ready to add to the page.
        """
        table = QTableWidget()
        table.setColumnCount(len(_COLUMNS))
        table.setHorizontalHeaderLabels(_COLUMNS)
        table.setObjectName("trafficTable")

        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setWordWrap(False)

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header.setHighlightSections(False)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        table.setStyleSheet(
            f"""
            QTableWidget#trafficTable {{
                background-color: {ThemeConfig.BACKGROUND_CARD};
                border: 1px solid {ThemeConfig.BORDER_COLOR};
                border-radius: 10px;
                gridline-color: {ThemeConfig.BORDER_COLOR};
            }}
            QTableWidget#trafficTable::item {{
                padding: 6px 10px;
                border-bottom: 1px solid {ThemeConfig.BORDER_COLOR};
            }}
            QTableWidget#trafficTable::item:selected {{
                background-color: {ThemeConfig.BACKGROUND_SECONDARY};
            }}
            QHeaderView::section {{
                background-color: {ThemeConfig.BACKGROUND_SECONDARY};
                color: {ThemeConfig.TEXT_SECONDARY};
                padding: 8px 10px;
                border: none;
                border-bottom: 1px solid {ThemeConfig.BORDER_COLOR};
                font-weight: 600;
            }}
            """
        )

        self._table = table
        return table

    # ------------------------------------------------------------------
    # Engine wiring
    # ------------------------------------------------------------------

    def _connect_engine_signals(self) -> None:
        """
        Connects MonitoringEngine.dashboard_updated to this page's
        per-tick refresh slot. Unlike ThreatDetectionPage (which
        listens to threat_detected, a per-event signal), this page
        deliberately uses the periodic aggregate signal, since it
        needs to sample and diff against the traffic buffer as a
        batch each tick rather than reacting to individual packets.
        """
        self._engine.dashboard_updated.connect(self._on_dashboard_updated)

    def _load_initial_data(self) -> None:
        """
        Populates the info cards and table immediately from the
        engine's current state and ThreatDetector's in-memory buffer,
        so the page shows real data right away rather than waiting
        for the first tick.
        """
        self._on_dashboard_updated(self._engine.get_current_snapshot())

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_dashboard_updated(self, snapshot: DashboardSnapshot) -> None:
        """
        Handles MonitoringEngine.dashboard_updated: refreshes the info
        cards, live traffic graph, and appends any genuinely new rows
        to the traffic table.

        Args:
            snapshot: The latest DashboardSnapshot from the engine.
        """
        threat_summary = snapshot.threat_summary

        self._info_cards["interface"].update_value(
            self._engine.active_interface or "Not Active"
        )
        self._info_cards["total_packets"].update_value(f"{snapshot.packets_captured:,}")
        self._info_cards["packets_per_second"].update_value(str(snapshot.packets_per_second))
        self._info_cards["active_flows"].update_value(
            str(threat_summary.get("active_flows", 0))
        )

        self.network_graph.update_data(snapshot.packets_per_second, snapshot.bytes_per_second)

        self._append_new_traffic_rows()

    def _append_new_traffic_rows(self) -> None:
        """
        Samples ThreatDetector's in-memory recent-detections buffer
        (which holds ALL processed traffic, not just threats — see
        security/threat_detector.py's process_packet()), filters to
        only results newer than the last tick's newest timestamp, and
        inserts them into the table newest-first. Also folds each new
        result's protocol into the running protocol breakdown counters.
        """
        recent_detections: List[DetectionResult] = get_threat_detector().get_recent_detections(
            limit=_SAMPLE_LIMIT
        )
        if not recent_detections:
            return

        new_results: List[DetectionResult] = []
        for result in recent_detections:  # newest first
            if self._last_seen_timestamp is not None and result.timestamp <= self._last_seen_timestamp:
                break
            new_results.append(result)

        if not new_results:
            return

        self._last_seen_timestamp = recent_detections[0].timestamp

        # new_results is newest-first; insert oldest-of-new first at
        # the top so the final table ordering ends up correctly
        # newest-overall-at-top, matching ThreatDetectionPage's
        # equivalent insertion logic.
        for result in reversed(new_results):
            self._insert_row(result)
            self._protocol_counts[self._normalize_protocol(result.packet.protocol)] += 1

        self._refresh_protocol_cards()

    def _insert_row(self, result: DetectionResult) -> None:
        """
        Inserts a single DetectionResult as a new row at the top of
        the table, color-coded by classification, and evicts the
        oldest row if the table exceeds MAX_DISPLAYED_ROWS.

        Args:
            result: The DetectionResult to display.
        """
        self._table.insertRow(0)

        color = QColor(_THREAT_ROW_COLOR if result.is_threat else _NORMAL_ROW_COLOR)
        classification = (
            f"⚠ {result.prediction.predicted_class}" if result.is_threat else "Normal"
        )

        values = [
            result.timestamp.strftime("%H:%M:%S"),
            f"{result.packet.src_ip}:{result.packet.src_port}",
            f"{result.packet.dst_ip}:{result.packet.dst_port}",
            result.packet.protocol,
            f"{result.packet.size:,}",
            classification,
            f"{result.risk_assessment.risk_score}/100",
        ]

        for column_index, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFont(QFont(ThemeConfig.FONT_FAMILY_MONO, 9))
            item.setForeground(color)
            if column_index == 5 and result.is_threat:
                item.setFont(QFont(ThemeConfig.FONT_FAMILY, 9, QFont.Weight.DemiBold))
            self._table.setItem(0, column_index, item)

        while self._table.rowCount() > MAX_DISPLAYED_ROWS:
            self._table.removeRow(self._table.rowCount() - 1)

    @staticmethod
    def _normalize_protocol(protocol: str) -> str:
        """
        Maps a raw protocol string to one of the four protocol
        breakdown buckets, defaulting unrecognized values to "OTHER".

        Args:
            protocol: The raw protocol string from CapturedPacket
                (e.g., "TCP", "UDP", "ICMP").

        Returns:
            One of "TCP", "UDP", "ICMP", or "OTHER".
        """
        if protocol in ("TCP", "UDP", "ICMP"):
            return protocol
        return "OTHER"

    def _refresh_protocol_cards(self) -> None:
        """Updates all four protocol breakdown cards from the running counters."""
        for protocol_key, card in self._protocol_cards.items():
            card.update_value(f"{self._protocol_counts.get(protocol_key, 0):,}")

    def reset(self) -> None:
        """
        Clears the table, protocol counters, and traffic graph.
        Intended to be called by MainWindow when monitoring restarts,
        matching app/ui/dashboard.py's DashboardPage.reset_live_widgets()
        pattern — a fresh monitoring session should show a fresh
        traffic view rather than continuing to display the previous
        session's packets.
        """
        self._table.setRowCount(0)
        self._protocol_counts.clear()
        self._last_seen_timestamp = None
        self._refresh_protocol_cards()
        self.network_graph.clear()