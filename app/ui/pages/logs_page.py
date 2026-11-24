"""
app/ui/pages/logs_page.py

Logs page for CyberSentinel AI's sidebar navigation.

Responsibilities:
    - Render a live-updating table of the "security_logs" MongoDB
      collection (see database.db_manager.DatabaseManager.get_recent_logs()),
      the audit trail written by security.threat_detector.ThreatDetector
      whenever a threat is confirmed.
    - Provide level filtering (All / INFO / WARNING / ERROR / CRITICAL)
      via a segmented button control, applied client-side against a
      cached copy of loaded logs.
    - Provide a manual Refresh action, and automatically pull in new
      entries whenever MonitoringEngine.threat_detected fires (since
      that is exactly when ThreatDetector persists a new log entry).
    - Fulfil the project spec's "Export Logs" feature: writes the
      currently filtered set of logs to a real CSV file via a native
      save dialog.

Unlike app/ui/pages/threat_detection_page.py and
app/ui/pages/network_monitor_page.py (which both read from
ThreatDetector's in-memory, session-only recent-detections buffer),
this page queries database.db_manager.DatabaseManager directly, since
logs should reflect the full persisted audit history — including
entries written during previous application sessions — not just
what's happened since this launch.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from typing import Dict, List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
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
from config.settings import PathConfig, ThemeConfig
from database.db_manager import get_database_manager
from security.threat_detector import DetectionResult
from utils.logger import get_logger

logger = get_logger(__name__)

# Maximum number of logs cached/displayed at once. Matches the
# eviction cap pattern used by
# app/ui/pages/threat_detection_page.py's MAX_DISPLAYED_ROWS and
# app/ui/pages/network_monitor_page.py's MAX_DISPLAYED_ROWS.
MAX_DISPLAYED_LOGS: int = 500

# Number of most-recent logs re-queried from the database each time
# MonitoringEngine.threat_detected fires, to pick up the newly
# persisted log entry (and any others written since the last check).
_REFRESH_SAMPLE_LIMIT: int = 50

_COLUMNS: List[str] = ["Time", "Level", "Source", "Message"]

# Maps a log "level" string (as written by
# security/threat_detector.py's _SEVERITY_TO_LOG_LEVEL) to a display
# color. Kept local to this module, same rationale as the local
# severity-color tables already defined in
# app/ui/widgets/threat_timeline_widget.py,
# app/ui/widgets/recent_alerts_widget.py, and app/core/notifier.py.
_LEVEL_COLORS: Dict[str, str] = {
    "INFO": ThemeConfig.TEXT_SECONDARY,
    "WARNING": ThemeConfig.WARNING_ORANGE,
    "ERROR": ThemeConfig.ALERT_RED,
    "CRITICAL": ThemeConfig.ALERT_RED,
}
_DEFAULT_LEVEL_COLOR: str = ThemeConfig.TEXT_PRIMARY

_FILTER_LEVELS: List[str] = ["All", "INFO", "WARNING", "ERROR", "CRITICAL"]


class _FilterButton(QPushButton):
    """
    A single entry in the level-filter segmented control. Behaves
    like a checkable toggle button, styled to look like a compact
    pill rather than a standard push button, matching the visual
    language of app/ui/dashboard.py's StatusPill without reusing that
    class directly (StatusPill is a QLabel, not clickable).
    """

    def __init__(self, label: str, parent: Optional[QWidget] = None) -> None:
        """
        Args:
            label: The filter option's display text (e.g., "WARNING").
            parent: Optional parent widget.
        """
        super().__init__(label, parent)
        self.setObjectName("filterButton")
        self.setCheckable(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont(ThemeConfig.FONT_FAMILY, 9, QFont.Weight.DemiBold))
        self.setStyleSheet(
            f"""
            QPushButton#filterButton {{
                color: {ThemeConfig.TEXT_SECONDARY};
                background-color: {ThemeConfig.BACKGROUND_CARD};
                border: 1px solid {ThemeConfig.BORDER_COLOR};
                border-radius: 6px;
                padding: 6px 14px;
            }}
            QPushButton#filterButton:checked {{
                color: {ThemeConfig.NEON_BLUE};
                border: 1px solid {ThemeConfig.NEON_BLUE};
                background-color: {ThemeConfig.BACKGROUND_SECONDARY};
                font-weight: 700;
            }}
            QPushButton#filterButton:hover {{
                color: {ThemeConfig.TEXT_PRIMARY};
            }}
            """
        )


class LogsPage(QWidget):
    """
    Sidebar page displaying the persisted security log audit trail,
    with level filtering, manual refresh, and CSV export.

    Like the other fully-implemented pages, this page receives its
    MonitoringEngine via constructor injection from MainWindow — used
    here only to know when to re-query the database (via
    threat_detected), not for any packet-level data directly.

    Usage:
        page = LogsPage(engine=self.engine)
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

        self._all_logs: List[dict] = []       # newest first, capped at MAX_DISPLAYED_LOGS
        self._active_filter: str = "All"
        self._last_seen_timestamp: Optional[datetime] = None

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(24, 20, 24, 20)
        outer_layout.setSpacing(16)

        outer_layout.addLayout(self._build_header_row())
        outer_layout.addLayout(self._build_filter_row())
        outer_layout.addWidget(self._build_table(), stretch=1)

        self._empty_state_label = QLabel(
            "No security events logged yet. Logs appear here once threats are detected."
        )
        self._empty_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_state_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 10))
        self._empty_state_label.setStyleSheet(f"color: {ThemeConfig.TEXT_DISABLED};")
        outer_layout.addWidget(self._empty_state_label)

        self._connect_engine_signals()
        self._load_initial_data()

        logger.debug("LogsPage initialized.")

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_header_row(self) -> QHBoxLayout:
        """
        Builds the page heading plus the Refresh and Export Logs
        action buttons.

        Returns:
            The populated QHBoxLayout, ready to add to the page.
        """
        heading = QLabel("Security Event Logs")
        heading.setFont(QFont(ThemeConfig.FONT_FAMILY, 16, QFont.Weight.Bold))
        heading.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY};")

        self._refresh_button = QPushButton("⟳  Refresh")
        self._refresh_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_button.setFont(QFont(ThemeConfig.FONT_FAMILY, 9, QFont.Weight.DemiBold))
        self._refresh_button.clicked.connect(self._on_refresh_clicked)
        self._style_action_button(self._refresh_button, ThemeConfig.NEON_CYAN)

        self._export_button = QPushButton("⬇  Export Logs")
        self._export_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_button.setFont(QFont(ThemeConfig.FONT_FAMILY, 9, QFont.Weight.DemiBold))
        self._export_button.clicked.connect(self._on_export_clicked)
        self._style_action_button(self._export_button, ThemeConfig.NEON_BLUE)

        row = QHBoxLayout()
        row.addWidget(heading)
        row.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        row.addWidget(self._refresh_button)
        row.addWidget(self._export_button)
        return row

    @staticmethod
    def _style_action_button(button: QPushButton, color: str) -> None:
        """
        Applies consistent outlined-pill styling to a header action
        button, matching the visual language of
        app/ui/dashboard.py's Header.monitor_toggle_button.

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

    def _build_filter_row(self) -> QHBoxLayout:
        """
        Builds the level-filter segmented control (All / INFO /
        WARNING / ERROR / CRITICAL).

        Returns:
            The populated QHBoxLayout, ready to add to the page.
        """
        self._filter_buttons: Dict[str, _FilterButton] = {}

        row = QHBoxLayout()
        row.setSpacing(8)
        for level in _FILTER_LEVELS:
            button = _FilterButton(level)
            button.clicked.connect(lambda checked, lvl=level: self._on_filter_selected(lvl))
            row.addWidget(button)
            self._filter_buttons[level] = button

        row.addStretch()
        self._filter_buttons["All"].setChecked(True)
        return row

    def _build_table(self) -> QTableWidget:
        """
        Builds and styles the logs table.

        Returns:
            The configured QTableWidget, ready to add to the page.
        """
        table = QTableWidget()
        table.setColumnCount(len(_COLUMNS))
        table.setHorizontalHeaderLabels(_COLUMNS)
        table.setObjectName("logsTable")

        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.verticalHeader().setVisible(False)
        table.setShowGrid(False)
        table.setWordWrap(False)

        header = table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setHighlightSections(False)
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        table.setStyleSheet(
            f"""
            QTableWidget#logsTable {{
                background-color: {ThemeConfig.BACKGROUND_CARD};
                border: 1px solid {ThemeConfig.BORDER_COLOR};
                border-radius: 10px;
                gridline-color: {ThemeConfig.BORDER_COLOR};
            }}
            QTableWidget#logsTable::item {{
                padding: 6px 10px;
                border-bottom: 1px solid {ThemeConfig.BORDER_COLOR};
            }}
            QTableWidget#logsTable::item:selected {{
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
        Connects MonitoringEngine.threat_detected to this page's
        refresh slot. This is the correct trigger (rather than the
        higher-frequency dashboard_updated) since a new "security_logs"
        entry is only ever written by
        security.threat_detector.ThreatDetector._persist_threat(),
        which runs exactly once per confirmed threat — the same event
        that fires threat_detected.
        """
        self._engine.threat_detected.connect(self._on_threat_detected)

    def _load_initial_data(self) -> None:
        """
        Loads the most recent persisted logs from the database
        immediately on construction, so the page shows the full
        historical audit trail (including prior sessions) right away.
        """
        self._refresh_from_database(limit=MAX_DISPLAYED_LOGS)

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_threat_detected(self, result: DetectionResult) -> None:
        """
        Handles MonitoringEngine.threat_detected: re-queries the
        database for recently written logs and merges in anything new.

        Args:
            result: The completed DetectionResult (unused directly —
                this handler exists purely as a trigger; the actual
                log content is read back from MongoDB as the
                authoritative source, not reconstructed locally).
        """
        self._refresh_from_database(limit=_REFRESH_SAMPLE_LIMIT, merge_only=True)

    def _on_refresh_clicked(self) -> None:
        """Handles a manual click on the Refresh button: full reload from the database."""
        self._refresh_from_database(limit=MAX_DISPLAYED_LOGS)

    def _on_filter_selected(self, level: str) -> None:
        """
        Handles a click on a level-filter button: updates the active
        filter, ensures only the clicked button stays checked, and
        redraws the table from the cached log list.

        Args:
            level: The selected filter level ("All", "INFO", etc.).
        """
        for name, button in self._filter_buttons.items():
            button.setChecked(name == level)
        self._active_filter = level
        self._render_table()

    def _on_export_clicked(self) -> None:
        """
        Handles a click on the Export Logs button: opens a native save
        dialog and writes the currently filtered set of logs to a CSV
        file. Fulfils the project spec's "Export Logs" feature.
        """
        if not self._all_logs:
            QMessageBox.information(
                self, "Export Logs", "There are no logs to export yet."
            )
            return

        default_filename = (
            f"cybersentinel_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        default_path = str(PathConfig.EXPORT_DIR / default_filename)

        file_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Logs",
            default_path,
            "CSV Files (*.csv)",
        )
        if not file_path:
            return  # user cancelled the dialog

        rows_to_export = self._filtered_logs()
        try:
            with open(file_path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Time", "Level", "Source", "Message"])
                for log in rows_to_export:
                    writer.writerow(
                        [
                            self._format_timestamp(log.get("timestamp")),
                            log.get("level", ""),
                            log.get("source", ""),
                            log.get("message", ""),
                        ]
                    )
            logger.info("Exported %d log(s) to %s", len(rows_to_export), file_path)
            QMessageBox.information(
                self, "Export Logs", f"Exported {len(rows_to_export)} log(s) to:\n{file_path}"
            )
        except OSError as error:
            logger.error("Failed to export logs to %s: %s", file_path, error)
            QMessageBox.warning(
                self, "Export Logs", f"Failed to write the export file:\n{error}"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_from_database(self, limit: int, merge_only: bool = False) -> None:
        """
        Queries DatabaseManager.get_recent_logs() and either fully
        replaces the cached log list (merge_only=False, used for
        initial load and manual refresh) or merges in only genuinely
        new entries newer than the last-seen timestamp (merge_only=True,
        used for the lightweight per-threat refresh), mirroring the
        diffing approach already used by
        app/ui/pages/network_monitor_page.py's _append_new_traffic_rows().

        Args:
            limit: Maximum number of logs to request from the database.
            merge_only: If True, only prepends logs newer than
                self._last_seen_timestamp rather than replacing the
                whole cached list.
        """
        fetched_logs = self._db_manager.get_recent_logs(limit=limit)  # newest first
        if not fetched_logs:
            if not merge_only:
                self._all_logs = []
                self._render_table()
            return

        if merge_only and self._last_seen_timestamp is not None:
            new_logs = [
                log for log in fetched_logs
                if log.get("timestamp") and log["timestamp"] > self._last_seen_timestamp
            ]
            if not new_logs:
                return
            self._all_logs = (new_logs + self._all_logs)[:MAX_DISPLAYED_LOGS]
        else:
            self._all_logs = fetched_logs[:MAX_DISPLAYED_LOGS]

        if fetched_logs[0].get("timestamp"):
            self._last_seen_timestamp = fetched_logs[0]["timestamp"]

        self._render_table()

    def _filtered_logs(self) -> List[dict]:
        """
        Returns:
            The cached log list filtered by the currently active
            level filter ("All" returns everything unmodified).
        """
        if self._active_filter == "All":
            return self._all_logs
        return [log for log in self._all_logs if log.get("level") == self._active_filter]

    def _render_table(self) -> None:
        """
        Fully rebuilds the table from the filtered log list. A full
        rebuild (rather than incremental row insertion) is used here
        since level filtering requires re-evaluating the whole visible
        set on every filter change anyway, and log volume is low
        (threats only, not every packet), so the redraw cost is
        negligible compared to app/ui/pages/network_monitor_page.py's
        much higher-frequency traffic table.
        """
        filtered = self._filtered_logs()

        self._table.setRowCount(0)
        self._empty_state_label.setVisible(len(filtered) == 0)
        self._table.setVisible(len(filtered) > 0)

        for log in filtered:
            self._insert_row(log)

    def _insert_row(self, log: dict) -> None:
        """
        Appends a single log document as a new row at the bottom of
        the table (the cached list is already newest-first, so
        sequential appends preserve that order visually).

        Args:
            log: A single log document, as returned by
                DatabaseManager.get_recent_logs() (with "_id" already
                converted to a string).
        """
        row_index = self._table.rowCount()
        self._table.insertRow(row_index)

        level = log.get("level", "INFO")
        color = QColor(_LEVEL_COLORS.get(level, _DEFAULT_LEVEL_COLOR))

        values = [
            self._format_timestamp(log.get("timestamp")),
            level,
            log.get("source", "--"),
            log.get("message", ""),
        ]

        for column_index, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setFont(QFont(ThemeConfig.FONT_FAMILY_MONO, 9))
            item.setForeground(color)
            if column_index == 1:
                item.setFont(QFont(ThemeConfig.FONT_FAMILY, 9, QFont.Weight.DemiBold))
            self._table.setItem(row_index, column_index, item)

    @staticmethod
    def _format_timestamp(timestamp: Optional[datetime]) -> str:
        """
        Formats a log document's timestamp for table/CSV display.

        Args:
            timestamp: The UTC-aware timestamp, or None if missing.

        Returns:
            A formatted "YYYY-MM-DD HH:MM:SS" string, or "--" if the
            timestamp is missing.
        """
        if timestamp is None:
            return "--"
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")