"""
app/ui/pages/reports_page.py

Reports page for CyberSentinel AI's sidebar navigation.

Responsibilities:
    - Query database.db_manager.DatabaseManager.get_threat_statistics()
      for a time-range-scoped summary of historical threat data: total
      count, average risk score, average confidence, severity
      breakdown, and attack-type breakdown.
    - Provide a time-range filter (Today / 7 Days / 30 Days / All Time)
      via a segmented button control, matching the visual pattern
      already established by app/ui/pages/logs_page.py's level filter.
    - Render a severity breakdown bar chart via matplotlib (same
      FigureCanvasQTAgg embedding technique already used by
      app/ui/widgets/network_topology_widget.py) and an attack-type
      breakdown table.
    - Fulfil the project spec's "Generated security reports and
      summaries" feature: a "Generate PDF Report" action that renders
      an HTML summary document and exports it to a real PDF file via
      QTextDocument + QPrinter (both part of PySide6's QtPrintSupport
      module — no new dependency required).

Like app/ui/pages/logs_page.py, this page queries the database
directly rather than any in-memory buffer, since a report must
reflect the full historical record (including prior sessions), not
just data collected since this application launch.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QTextDocument
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.engine import MonitoringEngine
from config.settings import AppInfo, PathConfig, ThemeConfig
from database.db_manager import get_database_manager
from utils.logger import get_logger

logger = get_logger(__name__)

_TIME_RANGE_OPTIONS: Dict[str, Optional[int]] = {
    "Today": 0,
    "7 Days": 7,
    "30 Days": 30,
    "All Time": None,
}

_SEVERITY_ORDER: list[str] = ["Critical", "High", "Medium", "Low", "Safe"]

# Maps severity -> bar color, reusing the same scheme already
# established in app/ui/widgets/threat_timeline_widget.py,
# app/ui/widgets/recent_alerts_widget.py, app/core/notifier.py, and
# app/ui/pages/logs_page.py, so severity colors read identically
# across the entire application.
_SEVERITY_COLORS: Dict[str, str] = {
    "Critical": ThemeConfig.ALERT_RED,
    "High": ThemeConfig.ALERT_RED,
    "Medium": ThemeConfig.WARNING_ORANGE,
    "Low": ThemeConfig.SAFE_GREEN,
    "Safe": ThemeConfig.SAFE_GREEN,
}


class _SummaryStat(QFrame):
    """
    A small stat card used in the Reports page's summary row. A local
    equivalent of the analogous private stat-card classes already
    defined independently in app/ui/pages/threat_detection_page.py
    and app/ui/pages/network_monitor_page.py — each page owns its own
    small variant rather than sharing one, keeping every page
    self-contained.
    """

    def __init__(self, title: str, accent_color: str, parent: Optional[QWidget] = None) -> None:
        """
        Args:
            title: The stat's label (e.g., "Total Threats").
            accent_color: Hex color used for the value text and left border.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setObjectName("reportStat")
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
        self._value_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(self._value_label)

        self.setStyleSheet(
            f"""
            QFrame#reportStat {{
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


class _RangeFilterButton(QPushButton):
    """
    A single entry in the time-range segmented control. Visually and
    behaviorally identical to app/ui/pages/logs_page.py's
    _FilterButton — duplicated locally rather than imported, since
    that class is a private implementation detail of its own module.
    """

    def __init__(self, label: str, parent: Optional[QWidget] = None) -> None:
        """
        Args:
            label: The range option's display text (e.g., "7 Days").
            parent: Optional parent widget.
        """
        super().__init__(label, parent)
        self.setObjectName("rangeFilterButton")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont(ThemeConfig.FONT_FAMILY, 9, QFont.Weight.DemiBold))
        self.setStyleSheet(
            f"""
            QPushButton#rangeFilterButton {{
                color: {ThemeConfig.TEXT_SECONDARY};
                background-color: {ThemeConfig.BACKGROUND_CARD};
                border: 1px solid {ThemeConfig.BORDER_COLOR};
                border-radius: 6px;
                padding: 6px 14px;
            }}
            QPushButton#rangeFilterButton:checked {{
                color: {ThemeConfig.NEON_BLUE};
                border: 1px solid {ThemeConfig.NEON_BLUE};
                background-color: {ThemeConfig.BACKGROUND_SECONDARY};
                font-weight: 700;
            }}
            QPushButton#rangeFilterButton:hover {{
                color: {ThemeConfig.TEXT_PRIMARY};
            }}
            """
        )


class ReportsPage(QWidget):
    """
    Sidebar page displaying historical threat report summaries, with
    time-range filtering, a severity breakdown chart, an attack-type
    breakdown table, and PDF export.

    Like the other fully-implemented pages, this page receives its
    MonitoringEngine via constructor injection from MainWindow — used
    here only to know when new threats have occurred (via
    threat_detected) so the report can be silently marked "stale"
    without forcing an expensive re-aggregation on every single event.

    Usage:
        page = ReportsPage(engine=self.engine)
    """

    def __init__(self, engine: MonitoringEngine, parent: Optional[QWidget] = None) -> None:
        """
        Args:
            engine: The application's shared MonitoringEngine instance.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._engine = engine
        self._db_manager = get_database_manager()

        self._active_range: str = "All Time"
        self._current_stats: Dict = {}

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(24, 20, 24, 20)
        outer_layout.setSpacing(16)

        outer_layout.addLayout(self._build_header_row())
        outer_layout.addLayout(self._build_range_row())
        outer_layout.addLayout(self._build_summary_row())

        middle_row = QHBoxLayout()
        middle_row.setSpacing(16)
        middle_row.addWidget(self._build_chart_card(), stretch=3)
        middle_row.addWidget(self._build_attack_table_card(), stretch=2)
        outer_layout.addLayout(middle_row, stretch=1)

        self._empty_state_label = QLabel(
            "No threat data available for this time range yet."
        )
        self._empty_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_state_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 10))
        self._empty_state_label.setStyleSheet(f"color: {ThemeConfig.TEXT_DISABLED};")
        self._empty_state_label.setVisible(False)
        outer_layout.addWidget(self._empty_state_label)

        self._connect_engine_signals()
        self._refresh_report()

        logger.debug("ReportsPage initialized.")

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_header_row(self) -> QHBoxLayout:
        """
        Builds the page heading plus the Refresh and Generate PDF
        Report action buttons.

        Returns:
            The populated QHBoxLayout, ready to add to the page.
        """
        heading = QLabel("Security Reports")
        heading.setFont(QFont(ThemeConfig.FONT_FAMILY, 16, QFont.Weight.Bold))
        heading.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY};")

        self._refresh_button = QPushButton("⟳  Refresh")
        self._refresh_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_button.setFont(QFont(ThemeConfig.FONT_FAMILY, 9, QFont.Weight.DemiBold))
        self._refresh_button.clicked.connect(self._refresh_report)
        self._style_action_button(self._refresh_button, ThemeConfig.NEON_CYAN)

        self._generate_button = QPushButton("📄  Generate PDF Report")
        self._generate_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._generate_button.setFont(QFont(ThemeConfig.FONT_FAMILY, 9, QFont.Weight.DemiBold))
        self._generate_button.clicked.connect(self._on_generate_pdf_clicked)
        self._style_action_button(self._generate_button, ThemeConfig.NEON_BLUE)

        row = QHBoxLayout()
        row.addWidget(heading)
        row.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        row.addWidget(self._refresh_button)
        row.addWidget(self._generate_button)
        return row

    @staticmethod
    def _style_action_button(button: QPushButton, color: str) -> None:
        """
        Applies consistent outlined-pill styling to a header action
        button, matching the pattern used in
        app/ui/pages/logs_page.py's _style_action_button().

        Args:
            button: The button to style.
            color: Hex accent color for the border/text/hover fill.
        """
        button.setStyleSheet(
            f"""
            QPushButton {{
                color: {color};
                border: 1px solid {color};
                border-radius: 6px;
                padding: 6px 14px;
                background-color: {ThemeConfig.BACKGROUND_CARD};
            }}
            QPushButton:hover {{
                background-color: {color};
                color: {ThemeConfig.BACKGROUND_PRIMARY};
            }}
            """
        )

    def _build_range_row(self) -> QHBoxLayout:
        """
        Builds the time-range segmented control (Today / 7 Days /
        30 Days / All Time).

        Returns:
            The populated QHBoxLayout, ready to add to the page.
        """
        self._range_buttons: Dict[str, _RangeFilterButton] = {}

        row = QHBoxLayout()
        row.setSpacing(8)
        for range_label in _TIME_RANGE_OPTIONS:
            button = _RangeFilterButton(range_label)
            button.clicked.connect(lambda checked, lbl=range_label: self._on_range_selected(lbl))
            row.addWidget(button)
            self._range_buttons[range_label] = button

        row.addStretch()
        self._range_buttons["All Time"].setChecked(True)
        return row

    def _build_summary_row(self) -> QHBoxLayout:
        """
        Builds the four-card summary row (Total Threats, Avg Risk
        Score, Avg Confidence, Most Common Attack).

        Returns:
            The populated QHBoxLayout, ready to add to the page.
        """
        self._summary_cards: Dict[str, _SummaryStat] = {
            "total": _SummaryStat("Total Threats", ThemeConfig.NEON_BLUE),
            "avg_risk": _SummaryStat("Avg Risk Score", ThemeConfig.WARNING_ORANGE),
            "avg_confidence": _SummaryStat("Avg Confidence", ThemeConfig.NEON_CYAN),
            "top_attack": _SummaryStat("Most Common Attack", ThemeConfig.ACCENT_PURPLE),
        }

        row = QHBoxLayout()
        row.setSpacing(14)
        for card in self._summary_cards.values():
            row.addWidget(card)
        return row

    def _build_chart_card(self) -> QWidget:
        """
        Builds the rounded card hosting the severity breakdown bar
        chart, using the same FigureCanvasQTAgg embedding technique
        as app/ui/widgets/network_topology_widget.py.

        Returns:
            The constructed card widget.
        """
        card = QFrame()
        card.setObjectName("chartCard")
        card.setStyleSheet(
            f"""
            QFrame#chartCard {{
                background-color: {ThemeConfig.BACKGROUND_CARD};
                border: 1px solid {ThemeConfig.BORDER_COLOR};
                border-radius: 10px;
            }}
            """
        )

        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        title_label = QLabel("Threats by Severity")
        title_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 12, QFont.Weight.DemiBold))
        title_label.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(title_label)

        self._figure = Figure(figsize=(5, 3))
        self._axes = self._figure.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(240)
        layout.addWidget(self._canvas)

        return card

    def _build_attack_table_card(self) -> QWidget:
        """
        Builds the rounded card hosting the attack-type breakdown
        table.

        Returns:
            The constructed card widget.
        """
        card = QFrame()
        card.setObjectName("attackTableCard")
        card.setStyleSheet(
            f"""
            QFrame#attackTableCard {{
                background-color: {ThemeConfig.BACKGROUND_CARD};
                border: 1px solid {ThemeConfig.BORDER_COLOR};
                border-radius: 10px;
            }}
            """
        )

        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(10)

        title_label = QLabel("Attack Type Breakdown")
        title_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 12, QFont.Weight.DemiBold))
        title_label.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY}; background: transparent;")
        layout.addWidget(title_label)

        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Attack Type", "Count"])
        table.setObjectName("attackTable")
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)

        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setHighlightSections(False)

        table.setStyleSheet(
            f"""
            QTableWidget#attackTable {{
                background-color: transparent;
                border: none;
                gridline-color: {ThemeConfig.BORDER_COLOR};
            }}
            QTableWidget#attackTable::item {{
                padding: 6px 10px;
                border-bottom: 1px solid {ThemeConfig.BORDER_COLOR};
            }}
            QHeaderView::section {{
                background-color: {ThemeConfig.BACKGROUND_SECONDARY};
                color: {ThemeConfig.TEXT_SECONDARY};
                padding: 6px 10px;
                border: none;
                font-weight: 600;
            }}
            """
        )

        layout.addWidget(table)
        self._attack_table = table
        return card

    # ------------------------------------------------------------------
    # Engine wiring
    # ------------------------------------------------------------------

    def _connect_engine_signals(self) -> None:
        """
        Connects MonitoringEngine.threat_detected to a lightweight
        staleness indicator on the Refresh button, rather than
        triggering a full re-aggregation on every single threat.
        Unlike app/ui/pages/logs_page.py (where a single new log entry
        is cheap to merge), a report re-aggregation touches the entire
        matched threat set via MongoDB aggregation pipelines, so
        re-running it on every event would be wasteful — the user
        explicitly refreshes when ready to see updated numbers.
        """
        self._engine.threat_detected.connect(self._mark_stale)

    def _mark_stale(self) -> None:
        """
        Visually flags the Refresh button to indicate new threat data
        may be available since the report was last generated, without
        forcing an immediate re-aggregation.
        """
        self._refresh_button.setText("⟳  Refresh (new data)")

    def _on_range_selected(self, range_label: str) -> None:
        """
        Handles a click on a time-range filter button: updates the
        active range, ensures only the clicked button stays checked,
        and re-runs the report aggregation for the new range.

        Args:
            range_label: The selected range option (e.g., "7 Days").
        """
        for name, button in self._range_buttons.items():
            button.setChecked(name == range_label)
        self._active_range = range_label
        self._refresh_report()

    def _refresh_report(self) -> None:
        """
        Re-queries DatabaseManager.get_threat_statistics() for the
        currently selected time range and redraws every section of
        the page (summary cards, chart, attack table) from the result.
        """
        since = self._resolve_since_timestamp()
        self._current_stats = self._db_manager.get_threat_statistics(since=since)
        self._refresh_button.setText("⟳  Refresh")

        self._render_summary_cards()
        self._render_chart()
        self._render_attack_table()

        has_data = self._current_stats.get("total", 0) > 0
        self._empty_state_label.setVisible(not has_data)

    def _resolve_since_timestamp(self) -> Optional[datetime]:
        """
        Converts the currently selected range label into a concrete
        "since" timestamp for the database query.

        Returns:
            A UTC datetime for the start of the selected range, or
            None for "All Time" (no lower bound).
        """
        days = _TIME_RANGE_OPTIONS[self._active_range]
        if days is None:
            return None
        if days == 0:  # "Today"
            now = datetime.now(timezone.utc)
            return now.replace(hour=0, minute=0, second=0, microsecond=0)
        return datetime.now(timezone.utc) - timedelta(days=days)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_summary_cards(self) -> None:
        """Updates the four summary stat cards from self._current_stats."""
        total = self._current_stats.get("total", 0)
        avg_risk = self._current_stats.get("avg_risk_score", 0.0)
        avg_confidence = self._current_stats.get("avg_confidence", 0.0)
        attack_counts = self._current_stats.get("attack_type_counts", {})

        top_attack = (
            max(attack_counts, key=attack_counts.get) if attack_counts else "--"
        )

        self._summary_cards["total"].update_value(f"{total:,}")
        self._summary_cards["avg_risk"].update_value(f"{avg_risk:.0f} / 100")
        self._summary_cards["avg_confidence"].update_value(f"{avg_confidence * 100:.1f}%")
        self._summary_cards["top_attack"].update_value(top_attack)

    def _render_chart(self) -> None:
        """Redraws the severity breakdown bar chart from self._current_stats."""
        self._axes.clear()
        self._figure.patch.set_facecolor(ThemeConfig.BACKGROUND_CARD)
        self._axes.set_facecolor(ThemeConfig.BACKGROUND_CARD)

        severity_counts = self._current_stats.get("severity_counts", {})
        labels = [level for level in _SEVERITY_ORDER if severity_counts.get(level, 0) > 0]
        values = [severity_counts[level] for level in labels]
        colors = [_SEVERITY_COLORS.get(level, ThemeConfig.TEXT_SECONDARY) for level in labels]

        if labels:
            bars = self._axes.bar(labels, values, color=colors, width=0.55)
            self._axes.bar_label(
                bars, padding=3, color=ThemeConfig.TEXT_PRIMARY, fontsize=8
            )

        self._axes.tick_params(colors=ThemeConfig.TEXT_SECONDARY, labelsize=8)
        for spine in self._axes.spines.values():
            spine.set_color(ThemeConfig.BORDER_COLOR)
        self._axes.set_ylabel("Count", color=ThemeConfig.TEXT_SECONDARY, fontsize=8)
        self._figure.subplots_adjust(left=0.12, right=0.97, top=0.95, bottom=0.12)

        self._canvas.draw_idle()

    def _render_attack_table(self) -> None:
        """Redraws the attack-type breakdown table from self._current_stats, sorted descending by count."""
        attack_counts = self._current_stats.get("attack_type_counts", {})
        sorted_items = sorted(attack_counts.items(), key=lambda item: item[1], reverse=True)

        self._attack_table.setRowCount(0)
        for row_index, (attack_type, count) in enumerate(sorted_items):
            self._attack_table.insertRow(row_index)

            type_item = QTableWidgetItem(attack_type)
            type_item.setFont(QFont(ThemeConfig.FONT_FAMILY, 9))
            type_item.setForeground(QColor(ThemeConfig.TEXT_PRIMARY))

            count_item = QTableWidgetItem(f"{count:,}")
            count_item.setFont(QFont(ThemeConfig.FONT_FAMILY_MONO, 9))
            count_item.setForeground(QColor(ThemeConfig.NEON_BLUE))

            self._attack_table.setItem(row_index, 0, type_item)
            self._attack_table.setItem(row_index, 1, count_item)

    # ------------------------------------------------------------------
    # PDF export
    # ------------------------------------------------------------------

    def _on_generate_pdf_clicked(self) -> None:
        """
        Handles a click on Generate PDF Report: builds an HTML summary
        document from self._current_stats and exports it to a real PDF
        file via QTextDocument + QPrinter (PySide6's built-in
        QtPrintSupport module — no new third-party dependency
        required), after prompting for a save location.
        """
        if self._current_stats.get("total", 0) == 0:
            QMessageBox.information(
                self, "Generate PDF Report",
                "There is no threat data for the selected time range to report on.",
            )
            return

        default_filename = (
            f"cybersentinel_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        )
        default_path = str(PathConfig.EXPORT_DIR / default_filename)

        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self, "Generate PDF Report", default_path, "PDF Files (*.pdf)"
        )
        if not file_path:
            return  # user cancelled the dialog

        html_report = self._build_report_html()

        document = QTextDocument()
        document.setHtml(html_report)

        printer = QPrinter(QPrinter.PrinterMode.HighResolution)
        printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
        printer.setOutputFileName(file_path)

        try:
            document.print_(printer)
            logger.info("Generated PDF report at %s", file_path)
            QMessageBox.information(
                self, "Generate PDF Report", f"Report saved to:\n{file_path}"
            )
        except Exception as error:  # broad: Qt's print pipeline can raise varied platform errors
            logger.error("Failed to generate PDF report at %s: %s", file_path, error)
            QMessageBox.warning(
                self, "Generate PDF Report", f"Failed to generate the report:\n{error}"
            )

    def _build_report_html(self) -> str:
        """
        Builds a styled HTML document summarizing self._current_stats,
        suitable for rendering to PDF via QTextDocument. Kept as plain
        HTML/CSS (Qt's QTextDocument supports a subset of HTML4/CSS2)
        rather than a templating library, since the layout is simple
        and this avoids adding another dependency.

        Returns:
            A complete HTML document as a string.
        """
        total = self._current_stats.get("total", 0)
        avg_risk = self._current_stats.get("avg_risk_score", 0.0)
        avg_confidence = self._current_stats.get("avg_confidence", 0.0)
        severity_counts = self._current_stats.get("severity_counts", {})
        attack_counts = self._current_stats.get("attack_type_counts", {})

        severity_rows = "".join(
            f"<tr><td>{level}</td><td>{severity_counts.get(level, 0)}</td></tr>"
            for level in _SEVERITY_ORDER
            if severity_counts.get(level, 0) > 0
        )
        attack_rows = "".join(
            f"<tr><td>{attack_type}</td><td>{count}</td></tr>"
            for attack_type, count in sorted(
                attack_counts.items(), key=lambda item: item[1], reverse=True
            )
        )

        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return f"""
        <html>
        <body style="font-family: Segoe UI, sans-serif; color: #111;">
            <h1 style="color: #1a1a2e;">{AppInfo.NAME} — Security Report</h1>
            <p><b>Time Range:</b> {self._active_range}<br/>
               <b>Generated:</b> {generated_at}</p>
            <hr/>
            <h2>Summary</h2>
            <table border="1" cellspacing="0" cellpadding="6" width="100%">
                <tr><td><b>Total Threats Detected</b></td><td>{total}</td></tr>
                <tr><td><b>Average Risk Score</b></td><td>{avg_risk:.1f} / 100</td></tr>
                <tr><td><b>Average Model Confidence</b></td><td>{avg_confidence * 100:.1f}%</td></tr>
            </table>
            <h2>Threats by Severity</h2>
            <table border="1" cellspacing="0" cellpadding="6" width="100%">
                <tr><th>Severity</th><th>Count</th></tr>
                {severity_rows}
            </table>
            <h2>Threats by Attack Type</h2>
            <table border="1" cellspacing="0" cellpadding="6" width="100%">
                <tr><th>Attack Type</th><th>Count</th></tr>
                {attack_rows}
            </table>
            <hr/>
            <p style="color: #666; font-size: 10px;">
                Generated by {AppInfo.NAME} v{AppInfo.VERSION} — {AppInfo.DESCRIPTION}
            </p>
        </body>
        </html>
        """