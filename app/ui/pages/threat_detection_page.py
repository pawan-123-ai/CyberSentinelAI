"""
app/ui/pages/threat_detection_page.py

Threat Detection page for CyberSentinel AI's sidebar navigation.

Responsibilities:
    - Render a live-scrolling, newest-first table of every threat
      classification produced by the AI detection pipeline: source/
      destination IP, protocol, predicted attack type, model
      confidence, risk score, severity, and timestamp.
    - Show a summary stat row (Total Threats, Critical Count, High
      Count, Most Common Attack Type) above the table.
    - Load initial data instantly from
      security.threat_detector.ThreatDetector.get_recent_detections()
      (in-memory, no database round-trip), then append live rows via
      app.core.engine.MonitoringEngine.threat_detected as new threats
      are classified.

This page reuses security.threat_detector.DetectionResult directly as
its row data source rather than defining a new DTO, since
DetectionResult is already the natural "one row of the threat table"
shape and this page (unlike app/ui/widgets/network_topology_widget.py)
has no need to decouple from its internal structure — it displays
exactly what DetectionResult already contains.
"""

from __future__ import annotations

from collections import Counter
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

from app.core.engine import MonitoringEngine
from config.settings import ThemeConfig
from security.risk_analyzer import SeverityLevel
from security.threat_detector import DetectionResult, get_threat_detector
from utils.logger import get_logger

logger = get_logger(__name__)

# Maximum number of rows kept in the table at once. Beyond this, the
# oldest (bottom-most) row is dropped whenever a new one is added,
# mirroring the eviction pattern already used by
# app/ui/widgets/recent_alerts_widget.py's MAX_DISPLAYED_ALERTS.
MAX_DISPLAYED_ROWS: int = 200

# Column layout for the threat table, defined once so header setup
# and row population can never drift out of sync with each other.
_COLUMNS: List[str] = [
    "Time",
    "Source IP",
    "Destination IP",
    "Protocol",
    "Attack Type",
    "Confidence",
    "Risk Score",
    "Severity",
]

# Maps SeverityLevel -> row accent color, reusing the exact same
# mapping already established in
# app/ui/widgets/threat_timeline_widget.py and
# app/ui/widgets/recent_alerts_widget.py, so severity reads identically
# everywhere in the dashboard rather than each page inventing its own scheme.
_SEVERITY_COLORS: Dict[SeverityLevel, str] = {
    SeverityLevel.SAFE: ThemeConfig.SAFE_GREEN,
    SeverityLevel.LOW: ThemeConfig.SAFE_GREEN,
    SeverityLevel.MEDIUM: ThemeConfig.WARNING_ORANGE,
    SeverityLevel.HIGH: ThemeConfig.ALERT_RED,
    SeverityLevel.CRITICAL: ThemeConfig.ALERT_RED,
}
_DEFAULT_SEVERITY_COLOR: str = ThemeConfig.TEXT_SECONDARY


class _SummaryStat(QFrame):
    """
    A small stat card used in the Threat Detection page's summary
    row. Deliberately more compact than
    app/ui/dashboard.py's StatCard, since this page's own table is
    the main content — the summary row is secondary context, not the
    page's headline metrics.
    """

    def __init__(self, title: str, accent_color: str, parent: Optional[QWidget] = None) -> None:
        """
        Args:
            title: The stat's label (e.g., "Total Threats").
            accent_color: Hex color used for the value text and left border.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setObjectName("summaryStat")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(4)

        title_label = QLabel(title.upper())
        title_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 8, QFont.Weight.Medium))
        title_label.setStyleSheet(f"color: {ThemeConfig.TEXT_SECONDARY}; background: transparent;")

        self._value_label = QLabel("--")
        self._value_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 18, QFont.Weight.Bold))
        self._value_label.setStyleSheet(f"color: {accent_color}; background: transparent;")

        layout.addWidget(title_label)
        layout.addWidget(self._value_label)

        self.setStyleSheet(
            f"""
            QFrame#summaryStat {{
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


class ThreatDetectionPage(QWidget):
    """
    Sidebar page displaying a live-scrolling table of every AI threat
    classification, with a summary stat row above it.

    Like app/ui/dashboard.py's DashboardPage, this page receives its
    MonitoringEngine via constructor injection from MainWindow rather
    than calling get_engine() itself.

    Usage:
        page = ThreatDetectionPage(engine=self.engine)
    """

    def __init__(self, engine: MonitoringEngine, parent: Optional[QWidget] = None) -> None:
        """
        Args:
            engine: The application's shared MonitoringEngine instance.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._engine = engine

        self._severity_counts: Counter[str] = Counter()
        self._attack_type_counts: Counter[str] = Counter()
        self._total_threats: int = 0

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(24, 20, 24, 20)
        outer_layout.setSpacing(16)

        heading = QLabel("AI Threat Classifications")
        heading.setFont(QFont(ThemeConfig.FONT_FAMILY, 16, QFont.Weight.Bold))
        heading.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY};")
        outer_layout.addWidget(heading)

        outer_layout.addLayout(self._build_summary_row())
        outer_layout.addWidget(self._build_table(), stretch=1)

        self._connect_engine_signals()
        self._load_initial_data()

        logger.debug("ThreatDetectionPage initialized.")

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_summary_row(self) -> QHBoxLayout:
        """
        Builds the four-card summary row above the table.

        Returns:
            The populated QHBoxLayout, ready to add to the page.
        """
        self._summary_cards: Dict[str, _SummaryStat] = {
            "total": _SummaryStat("Total Threats", ThemeConfig.NEON_BLUE),
            "critical": _SummaryStat("Critical", ThemeConfig.ALERT_RED),
            "high": _SummaryStat("High", ThemeConfig.WARNING_ORANGE),
            "top_attack": _SummaryStat("Most Common", ThemeConfig.ACCENT_PURPLE),
        }

        row = QHBoxLayout()
        row.setSpacing(14)
        for card in self._summary_cards.values():
            row.addWidget(card)

        return row

    def _build_table(self) -> QTableWidget:
        """
        Builds and styles the main threat classification table.

        Returns:
            The configured QTableWidget, ready to add to the page.
        """
        table = QTableWidget()
        table.setColumnCount(len(_COLUMNS))
        table.setHorizontalHeaderLabels(_COLUMNS)
        table.setObjectName("threatTable")

        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setAlternatingRowColors(False)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setWordWrap(False)

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.ResizeMode.ResizeToContents)
        header.setHighlightSections(False)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        table.setStyleSheet(
            f"""
            QTableWidget#threatTable {{
                background-color: {ThemeConfig.BACKGROUND_CARD};
                border: 1px solid {ThemeConfig.BORDER_COLOR};
                border-radius: 10px;
                gridline-color: {ThemeConfig.BORDER_COLOR};
            }}
            QTableWidget#threatTable::item {{
                padding: 6px 10px;
                border-bottom: 1px solid {ThemeConfig.BORDER_COLOR};
            }}
            QTableWidget#threatTable::item:selected {{
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
        """Connects MonitoringEngine.threat_detected to this page's row-insertion slot."""
        self._engine.threat_detected.connect(self._on_threat_detected)

    def _load_initial_data(self) -> None:
        """
        Populates the table immediately from ThreatDetector's
        in-memory recent-detections buffer, so the page shows data
        right away rather than waiting for the next live threat.
        """
        recent_detections = get_threat_detector().get_recent_detections(limit=MAX_DISPLAYED_ROWS)
        for result in reversed(recent_detections):  # oldest first, so _insert_row's "top" ordering ends up newest-first
            self._insert_row(result, at_top=True)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_threat_detected(self, result: DetectionResult) -> None:
        """
        Handles MonitoringEngine.threat_detected: inserts a new row at
        the top of the table and refreshes the summary stat cards.

        Args:
            result: The completed DetectionResult for a threat-classified packet.
        """
        self._insert_row(result, at_top=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert_row(self, result: DetectionResult, at_top: bool) -> None:
        """
        Inserts a single DetectionResult as a new table row, evicts
        the oldest row if the table exceeds MAX_DISPLAYED_ROWS, and
        updates the running summary counters.

        Args:
            result: The DetectionResult to display.
            at_top: If True, inserts at row 0 (newest-first ordering).
                Always True in current usage; kept as an explicit
                parameter for clarity at call sites.
        """
        row_index = 0 if at_top else self._table.rowCount()
        self._table.insertRow(row_index)

        severity = result.risk_assessment.severity
        color = QColor(_SEVERITY_COLORS.get(severity, _DEFAULT_SEVERITY_COLOR))

        values = [
            result.timestamp.strftime("%H:%M:%S"),
            f"{result.packet.src_ip}:{result.packet.src_port}",
            f"{result.packet.dst_ip}:{result.packet.dst_port}",
            result.packet.protocol,
            result.prediction.predicted_class,
            f"{result.prediction.confidence * 100:.1f}%",
            f"{result.risk_assessment.risk_score}/100",
            severity.value,
        ]

        for column_index, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFont(QFont(ThemeConfig.FONT_FAMILY_MONO, 9))
            item.setForeground(color if column_index == 7 else QColor(ThemeConfig.TEXT_PRIMARY))
            if column_index == 7:
                item.setFont(QFont(ThemeConfig.FONT_FAMILY, 9, QFont.Weight.DemiBold))
            self._table.setItem(row_index, column_index, item)

        while self._table.rowCount() > MAX_DISPLAYED_ROWS:
            self._table.removeRow(self._table.rowCount() - 1)

        self._update_summary(severity, result.prediction.predicted_class)

    def _update_summary(self, severity: SeverityLevel, attack_type: str) -> None:
        """
        Folds a new detection into the running summary counters and
        refreshes the four summary stat cards.

        Args:
            severity: The severity of the newly added detection.
            attack_type: The predicted attack class of the newly
                added detection.
        """
        self._total_threats += 1
        self._severity_counts[severity.value] += 1
        self._attack_type_counts[attack_type] += 1

        top_attack_type = (
            self._attack_type_counts.most_common(1)[0][0] if self._attack_type_counts else "--"
        )

        self._summary_cards["total"].update_value(f"{self._total_threats:,}")
        self._summary_cards["critical"].update_value(
            f"{self._severity_counts.get(SeverityLevel.CRITICAL.value, 0):,}"
        )
        self._summary_cards["high"].update_value(
            f"{self._severity_counts.get(SeverityLevel.HIGH.value, 0):,}"
        )
        self._summary_cards["top_attack"].update_value(top_attack_type)

    def reset(self) -> None:
        """
        Clears the table and all summary counters. Intended to be
        called by MainWindow if/when a "Clear Threat Log" action is
        added later; not currently wired to monitoring start/stop,
        since (like security/alert_manager.py's alert history) a
        threat log arguably should persist across sessions rather
        than disappearing on restart.
        """
        self._table.setRowCount(0)
        self._severity_counts.clear()
        self._attack_type_counts.clear()
        self._total_threats = 0
        self._summary_cards["total"].update_value("0")
        self._summary_cards["critical"].update_value("0")
        self._summary_cards["high"].update_value("0")
        self._summary_cards["top_attack"].update_value("--")