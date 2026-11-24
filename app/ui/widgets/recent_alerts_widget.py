"""
app/ui/widgets/recent_alerts_widget.py

Recent alerts / notification panel widget for CyberSentinel AI's
dashboard.

Responsibilities:
    - Render a scrollable, newest-first list of recent Alert records
      (security.alert_manager.Alert), each shown as a severity-colored
      card with title, message, relative timestamp, and a read/unread
      indicator.
    - Fulfil the project spec's "Recent Alerts" (dashboard) and
      "Notification Center" (bell dropdown / dedicated panel) features
      from a single reusable widget.
    - Expose set_alerts() (bulk refresh, e.g. on initial load) and
      add_alert() (incremental prepend, intended to be connected to
      app.core.engine.MonitoringEngine.alert_created for live updates).
    - Emit alert_clicked(alert_id) rather than calling
      security.alert_manager.AlertManager.mark_as_read() directly, so
      this widget stays a pure display component and the composition
      layer (app/ui/dashboard.py) owns the decision of what a click
      does.

This widget reuses security.alert_manager.Alert directly rather than
defining a local DTO, since Alert is a plain, side-effect-free
dataclass (the same reasoning already applied to DeviceInfo/SystemStats
in app/ui/widgets/system_info_widget.py) — not a business-logic object
carrying threading or persistence concerns.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from config.settings import ThemeConfig
from security.alert_manager import Alert
from security.risk_analyzer import SeverityLevel
from utils.logger import get_logger

logger = get_logger(__name__)

# Maximum number of alert cards kept in the scrollable list at once.
# Beyond this, the oldest (bottom-most) card is dropped whenever a new
# one is added, keeping the panel responsive during a sustained attack
# that generates many alerts.
MAX_DISPLAYED_ALERTS: int = 40

# Maps SeverityLevel -> accent color, mirroring the color choices
# already used for event markers in
# app/ui/widgets/threat_timeline_widget.py, so a "High" alert reads
# the same color everywhere in the dashboard.
_SEVERITY_COLORS: Dict[SeverityLevel, str] = {
    SeverityLevel.SAFE: ThemeConfig.SAFE_GREEN,
    SeverityLevel.LOW: ThemeConfig.SAFE_GREEN,
    SeverityLevel.MEDIUM: ThemeConfig.WARNING_ORANGE,
    SeverityLevel.HIGH: ThemeConfig.ALERT_RED,
    SeverityLevel.CRITICAL: ThemeConfig.ALERT_RED,
}
_DEFAULT_SEVERITY_COLOR: str = ThemeConfig.TEXT_SECONDARY


def _format_relative_time(timestamp: datetime) -> str:
    """
    Formats a UTC timestamp as a short, human-readable relative time
    string (e.g., "just now", "5s ago", "12m ago", "3h ago", "2d ago"),
    for compact display inside an alert card.

    Args:
        timestamp: The UTC-aware timestamp to format.

    Returns:
        A short relative time string.
    """
    now = datetime.now(timezone.utc)
    delta_seconds = max((now - timestamp).total_seconds(), 0)

    if delta_seconds < 5:
        return "just now"
    if delta_seconds < 60:
        return f"{int(delta_seconds)}s ago"
    if delta_seconds < 3600:
        return f"{int(delta_seconds // 60)}m ago"
    if delta_seconds < 86400:
        return f"{int(delta_seconds // 3600)}h ago"
    return f"{int(delta_seconds // 86400)}d ago"


class _AlertCard(QFrame):
    """
    A single alert entry within the scrollable list: a severity-colored
    left border, title, message, relative timestamp, and an unread dot
    indicator. Clicking anywhere on the card invokes a callback with
    the alert's ID (internal implementation detail — the public signal
    lives on RecentAlertsWidget itself).
    """

    def __init__(self, alert: Alert, on_click, parent: Optional[QWidget] = None) -> None:
        """
        Args:
            alert: The Alert this card displays.
            on_click: A callable invoked with alert.alert_id when the
                card is clicked (typically RecentAlertsWidget's own
                signal-emitting method).
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._alert = alert
        self._on_click = on_click
        self.setObjectName("alertCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        self._build_layout()
        self._apply_style()

    def _build_layout(self) -> None:
        """Builds the card's internal layout: unread dot, title/message column, and timestamp."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(10)

        self._unread_dot = QLabel("●")
        self._unread_dot.setFixedWidth(14)
        self._unread_dot.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)

        text_column = QVBoxLayout()
        text_column.setSpacing(2)

        title_row = QHBoxLayout()
        title_label = QLabel(self._alert.title)
        title_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 9, QFont.Weight.DemiBold))
        title_label.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY}; background: transparent;")
        title_label.setWordWrap(True)

        time_label = QLabel(_format_relative_time(self._alert.timestamp))
        time_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 8))
        time_label.setStyleSheet(f"color: {ThemeConfig.TEXT_DISABLED}; background: transparent;")

        title_row.addWidget(title_label, stretch=1)
        title_row.addWidget(time_label)

        message_label = QLabel(self._alert.message)
        message_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 8))
        message_label.setStyleSheet(f"color: {ThemeConfig.TEXT_SECONDARY}; background: transparent;")
        message_label.setWordWrap(True)

        text_column.addLayout(title_row)
        text_column.addWidget(message_label)

        layout.addWidget(self._unread_dot)
        layout.addLayout(text_column, stretch=1)

    def _apply_style(self) -> None:
        """
        Applies severity-colored left border styling and dims the
        unread dot for already-read alerts, so read/unread state and
        severity are both visible at a glance without extra chrome.
        """
        color = _SEVERITY_COLORS.get(self._alert.severity, _DEFAULT_SEVERITY_COLOR)

        self.setStyleSheet(
            f"""
            QFrame#alertCard {{
                background-color: {ThemeConfig.BACKGROUND_SECONDARY};
                border: 1px solid {ThemeConfig.BORDER_COLOR};
                border-left: 3px solid {color};
                border-radius: 6px;
            }}
            QFrame#alertCard:hover {{
                border: 1px solid {color};
                border-left: 3px solid {color};
            }}
            """
        )

        dot_color = color if not self._alert.is_read else ThemeConfig.TEXT_DISABLED
        self._unread_dot.setStyleSheet(f"color: {dot_color}; background: transparent; font-size: 8px;")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """
        Handles a click on the card: invokes the on_click callback
        with this alert's ID (if it was successfully persisted and
        thus has one), then defers to the base implementation.

        Args:
            event: The Qt mouse press event.
        """
        if self._alert.alert_id:
            self._on_click(self._alert.alert_id)
        super().mousePressEvent(event)


class RecentAlertsWidget(QWidget):
    """
    A rounded-card dashboard panel displaying a scrollable, newest-
    first list of recent alerts, fulfilling the project spec's
    "Recent Alerts" and "Notification Center" features.

    Usage:
        panel = RecentAlertsWidget()
        layout.addWidget(panel)
        panel.alert_clicked.connect(
            lambda alert_id: alert_manager.mark_as_read(alert_id)
        )
        ...
        panel.set_alerts(alert_manager.get_recent_alerts())   # initial load
        engine.alert_created.connect(panel.add_alert)          # live updates
    """

    # Emitted with an alert's ID when its card is clicked. The widget
    # itself does not call AlertManager.mark_as_read() directly — it
    # only reports the interaction, keeping this a pure display
    # component per the project's strict UI separation rule.
    alert_clicked = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("recentAlertsWidget")

        self._cards: List[_AlertCard] = []

        self._build_layout()
        logger.debug("RecentAlertsWidget initialized (max_displayed=%d).", MAX_DISPLAYED_ALERTS)

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Builds the card container, header row (title + unread badge), and scrollable list area."""
        self.setStyleSheet(
            f"""
            QWidget#recentAlertsWidget {{
                background-color: {ThemeConfig.BACKGROUND_CARD};
                border: 1px solid {ThemeConfig.BORDER_COLOR};
                border-radius: 10px;
            }}
            """
        )

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(18, 14, 18, 14)
        outer_layout.setSpacing(10)

        header_row = QHBoxLayout()

        title_label = QLabel("Recent Alerts")
        title_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 12, QFont.Weight.DemiBold))
        title_label.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY}; background: transparent;")

        self._unread_badge = QLabel("0 unread")
        self._unread_badge.setFont(QFont(ThemeConfig.FONT_FAMILY, 8, QFont.Weight.DemiBold))
        self._unread_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_unread_badge_style(0)

        header_row.addWidget(title_label)
        header_row.addStretch()
        header_row.addWidget(self._unread_badge)
        outer_layout.addLayout(header_row)

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setStyleSheet("background: transparent;")

        self._list_container = QWidget()
        self._list_container.setStyleSheet("background: transparent;")
        self._list_layout = QVBoxLayout(self._list_container)
        self._list_layout.setContentsMargins(0, 0, 4, 0)
        self._list_layout.setSpacing(6)
        self._list_layout.addStretch()  # keeps cards top-aligned as the list grows

        self._scroll_area.setWidget(self._list_container)
        outer_layout.addWidget(self._scroll_area, stretch=1)

        self._empty_state_label = QLabel("No alerts yet. Start monitoring to begin detection.")
        self._empty_state_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 9))
        self._empty_state_label.setStyleSheet(f"color: {ThemeConfig.TEXT_DISABLED}; background: transparent;")
        self._empty_state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_state_label.setWordWrap(True)
        outer_layout.addWidget(self._empty_state_label)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_alerts(self, alerts: List[Alert]) -> None:
        """
        Replaces the entire displayed list with the given alerts,
        newest first. Intended for initial population (e.g., loading
        AlertManager.get_recent_alerts() when the dashboard first
        appears) rather than per-tick updates — use add_alert() for
        incremental live updates instead, which is cheaper than a
        full rebuild.

        Args:
            alerts: The alerts to display, in any order (this method
                sorts them newest-first internally).
        """
        self._clear_cards()

        ordered = sorted(alerts, key=lambda alert: alert.timestamp, reverse=True)
        for alert in ordered[:MAX_DISPLAYED_ALERTS]:
            self._insert_card(alert)

        self._refresh_empty_state()
        self._refresh_unread_badge(alerts)

    def add_alert(self, alert: Alert) -> None:
        """
        Prepends a single new alert to the top of the list. Intended
        to be connected directly to
        app.core.engine.MonitoringEngine.alert_created for live
        updates as new alerts are created, without rebuilding the
        entire list on every call.

        Args:
            alert: The newly created alert to display.
        """
        self._insert_card(alert, at_top=True)

        while len(self._cards) > MAX_DISPLAYED_ALERTS:
            oldest_card = self._cards.pop()
            self._list_layout.removeWidget(oldest_card)
            oldest_card.deleteLater()

        self._refresh_empty_state()
        self._bump_unread_badge()

    def clear(self) -> None:
        """
        Removes all displayed alert cards. Note: unlike most other
        widgets' clear() methods in this project, this does NOT
        correspond to a "monitoring restarted" event — per
        app.core.engine.MonitoringEngine's design, AlertManager's
        history is intentionally NOT reset on start_monitoring(), so
        callers should generally prefer refreshing via set_alerts()
        rather than calling this directly during normal operation.
        """
        self._clear_cards()
        self._refresh_empty_state()
        self._refresh_unread_badge([])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert_card(self, alert: Alert, at_top: bool = False) -> None:
        """
        Creates an _AlertCard for the given alert and inserts it into
        the list layout, keeping self._cards and the visual layout
        order in sync (index 0 == newest == top of the list).

        Args:
            alert: The alert to create a card for.
            at_top: If True, inserts above all existing cards
                (used by add_alert()); otherwise inserts just before
                the trailing stretch, preserving the caller's
                iteration order (used by set_alerts(), which already
                iterates newest-first).
        """
        card = _AlertCard(alert, on_click=self._emit_alert_clicked)
        insert_index = 0 if at_top else max(self._list_layout.count() - 1, 0)
        self._list_layout.insertWidget(insert_index, card)
        self._cards.insert(0 if at_top else len(self._cards), card)

    def _clear_cards(self) -> None:
        """Removes and destroys every currently displayed alert card."""
        for card in self._cards:
            self._list_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

    def _refresh_empty_state(self) -> None:
        """Shows the empty-state message only when no cards are currently displayed."""
        has_cards = len(self._cards) > 0
        self._empty_state_label.setVisible(not has_cards)
        self._scroll_area.setVisible(has_cards)

    def _refresh_unread_badge(self, alerts: List[Alert]) -> None:
        """
        Recomputes the unread badge count from a full alert list
        (used after set_alerts()).

        Args:
            alerts: The full list of alerts currently considered loaded.
        """
        unread_count = sum(1 for alert in alerts if not alert.is_read)
        self._update_unread_badge_style(unread_count)

    def _bump_unread_badge(self) -> None:
        """
        Increments the unread badge count by one (used after
        add_alert(), where re-scanning the full alert list would be
        unnecessary work for a single new — and always unread — alert).
        """
        current_text = self._unread_badge.text().split()[0]
        current_count = int(current_text) if current_text.isdigit() else 0
        self._update_unread_badge_style(current_count + 1)

    def _update_unread_badge_style(self, unread_count: int) -> None:
        """
        Updates the unread badge's text and styling: a neutral style
        with "0 unread" when nothing is pending, and a highlighted
        alert-red pill when one or more alerts are unread — reinforcing
        the "Red Alerts" visual language from the project's UI spec.

        Args:
            unread_count: The number of unread alerts to display.
        """
        self._unread_badge.setText(f"{unread_count} unread")
        color = ThemeConfig.ALERT_RED if unread_count > 0 else ThemeConfig.TEXT_SECONDARY

        self._unread_badge.setStyleSheet(
            f"""
            QLabel {{
                color: {color};
                border: 1px solid {color};
                border-radius: 8px;
                padding: 2px 10px;
                background-color: {ThemeConfig.BACKGROUND_SECONDARY};
            }}
            """
        )

    def _emit_alert_clicked(self, alert_id: str) -> None:
        """
        Emits the alert_clicked signal. Kept as a small dedicated
        method (rather than passing self.alert_clicked.emit directly
        as _AlertCard's on_click callback) so a log line or breakpoint
        can be added here later without touching _AlertCard.

        Args:
            alert_id: The clicked alert's persisted ID.
        """
        self.alert_clicked.emit(alert_id)