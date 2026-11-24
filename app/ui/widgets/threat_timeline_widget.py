"""
app/ui/widgets/threat_timeline_widget.py

Live risk score / threat timeline widget for CyberSentinel AI's dashboard.

Responsibilities:
    - Render a rolling, real-time line chart of the smoothed risk
      score (0-100), with a filled area under the curve and dashed
      horizontal reference lines at the Medium/High/Critical severity
      thresholds.
    - Overlay scatter markers for individual threat detection events,
      color-coded by severity, so bursts of activity are visually
      distinguishable from the smoothed trend line beneath them.
    - Maintain fixed-length rolling buffers for both the trend line
      and the event markers, fed via update_risk_score() (intended to
      be called once per tick from app.core.engine.MonitoringEngine.
      dashboard_updated) and add_threat_event() (intended to be called
      immediately from MonitoringEngine.threat_detected).

This widget is purely presentational: it holds no business logic and
knows nothing about AI prediction, risk analysis, threading, or Qt
signals beyond its own public update methods. It only knows how to
plot the numbers it's given. The one exception is importing the
RISK_SCORE_CRITICAL constant from security.risk_analyzer, so the
displayed threshold line can never drift out of sync with the actual
scoring logic that constant governs.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from config.settings import AIConfig, ThemeConfig, UIConfig
from security.risk_analyzer import RISK_SCORE_CRITICAL
from utils.logger import get_logger

logger = get_logger(__name__)

# Maps severity display strings (as produced by
# security.risk_analyzer.SeverityLevel.value, and echoed through
# ai/predictor.py and security/threat_detector.py's to_dict() methods)
# to marker colors. Kept local to the widget since it's a pure display
# concern, not a scoring rule.
_SEVERITY_MARKER_COLORS: dict[str, str] = {
    "Safe": ThemeConfig.SAFE_GREEN,
    "Low": ThemeConfig.SAFE_GREEN,
    "Medium": ThemeConfig.WARNING_ORANGE,
    "High": ThemeConfig.ALERT_RED,
    "Critical": ThemeConfig.ALERT_RED,
}
_DEFAULT_MARKER_COLOR: str = ThemeConfig.TEXT_SECONDARY

# Event marker buffer holds more entries than the line's tick window,
# since several threat events can land within the same ~1s tick
# during a burst (e.g., a DDoS). Excess/off-screen markers are
# filtered out at render time based on the current visible tick range.
_EVENT_MARKER_BUFFER_MULTIPLIER: int = 5


class ThreatTimelineWidget(QWidget):
    """
    A rounded-card widget hosting a real-time risk score trend line
    (with severity threshold reference lines) and threat event
    markers, styled to match the dark cybersecurity dashboard theme.

    Usage:
        timeline = ThreatTimelineWidget()
        layout.addWidget(timeline)
        ...
        # Typically wired via Qt signals, e.g.:
        # engine.dashboard_updated.connect(
        #     lambda snap: timeline.update_risk_score(
        #         snap.threat_summary["current_risk_score"],
        #         snap.threat_summary["current_severity"],
        #     )
        # )
        # engine.threat_detected.connect(
        #     lambda result: timeline.add_threat_event(
        #         result.risk_assessment.risk_score,
        #         result.risk_assessment.severity.value,
        #     )
        # )
        timeline.update_risk_score(risk_score=42, severity="Medium")
        timeline.add_threat_event(risk_score=78, severity="High")
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("threatTimelineWidget")

        self._max_points = UIConfig.MAX_GRAPH_DATA_POINTS
        self._risk_history: Deque[float] = deque(maxlen=self._max_points)
        self._x_axis: Deque[int] = deque(maxlen=self._max_points)
        self._tick_counter: int = 0

        # Each entry: (tick, risk_score, severity_string)
        self._event_markers: Deque[Tuple[int, int, str]] = deque(
            maxlen=self._max_points * _EVENT_MARKER_BUFFER_MULTIPLIER
        )

        self._build_layout()
        self._configure_plot()

        logger.debug("ThreatTimelineWidget initialized (max_points=%d).", self._max_points)

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Builds the card container, header row (title + legend), and embedded plot widget."""
        self.setStyleSheet(
            f"""
            QWidget#threatTimelineWidget {{
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

        title_label = QLabel("Threat Timeline")
        title_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 12, QFont.Weight.DemiBold))
        title_label.setStyleSheet(
            f"color: {ThemeConfig.TEXT_PRIMARY}; background: transparent;"
        )

        legend_label = QLabel(
            f"<span style='color:{ThemeConfig.ACCENT_PURPLE};'>&#9679;</span> Risk Score"
            f"&nbsp;&nbsp;&nbsp;"
            f"<span style='color:{ThemeConfig.ALERT_RED};'>&#9679;</span> Threat Event"
        )
        legend_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 9))
        legend_label.setStyleSheet(
            f"color: {ThemeConfig.TEXT_SECONDARY}; background: transparent;"
        )

        header_row.addWidget(title_label)
        header_row.addStretch()
        header_row.addWidget(legend_label)
        outer_layout.addLayout(header_row)

        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setMinimumHeight(240)
        outer_layout.addWidget(self._plot_widget)

    def _configure_plot(self) -> None:
        """
        Configures the PyQtGraph plot: dark theme colors, a fixed
        0-100 Y range (risk scores are always on this scale, unlike
        the network graph's variable-magnitude series), dashed
        severity threshold reference lines, the filled risk score
        trend curve, and an empty scatter layer for threat event
        markers. Mouse interaction is disabled, matching
        NetworkGraphWidget's "glanceable live display, not an
        interactive chart" design.
        """
        self._plot_widget.setBackground(QColor(ThemeConfig.BACKGROUND_CARD))

        plot_item = self._plot_widget.getPlotItem()
        plot_item.showGrid(x=True, y=True, alpha=0.15)
        plot_item.setMenuEnabled(False)
        plot_item.hideButtons()
        plot_item.setYRange(0, 100, padding=0.02)

        axis_pen = pg.mkPen(color=ThemeConfig.BORDER_COLOR, width=1)

        left_axis = plot_item.getAxis("left")
        left_axis.setPen(axis_pen)
        left_axis.setTextPen(ThemeConfig.TEXT_SECONDARY)
        left_axis.setLabel("Risk Score", color=ThemeConfig.ACCENT_PURPLE)

        bottom_axis = plot_item.getAxis("bottom")
        bottom_axis.setPen(axis_pen)
        bottom_axis.setTextPen(ThemeConfig.TEXT_SECONDARY)
        bottom_axis.setLabel("Time (ticks ago)")

        self._add_threshold_line(plot_item, AIConfig.RISK_SCORE_MEDIUM, ThemeConfig.WARNING_ORANGE)
        self._add_threshold_line(plot_item, AIConfig.RISK_SCORE_HIGH, ThemeConfig.ALERT_RED)
        self._add_threshold_line(plot_item, RISK_SCORE_CRITICAL, ThemeConfig.ALERT_RED)

        # Filled trend curve: a translucent brush under the line gives
        # the "security dashboard" area-chart look without obscuring
        # the gridlines or threshold markers beneath it.
        line_pen = pg.mkPen(color=ThemeConfig.ACCENT_PURPLE, width=2)
        fill_color = QColor(ThemeConfig.ACCENT_PURPLE)
        fill_color.setAlpha(40)
        self._risk_curve = plot_item.plot(
            pen=line_pen,
            fillLevel=0,
            brush=pg.mkBrush(fill_color),
            name="Risk Score",
        )

        # Scatter layer for discrete threat event markers, added on
        # top of the trend curve so bursts of activity stand out.
        self._event_scatter = pg.ScatterPlotItem(size=9, pen=pg.mkPen(None))
        plot_item.addItem(self._event_scatter)

        self._plot_widget.setMouseEnabled(x=False, y=False)

    @staticmethod
    def _add_threshold_line(plot_item: pg.PlotItem, value: int, color: str) -> None:
        """
        Adds a dashed horizontal reference line at the given risk
        score value, used to visually mark the Medium/High/Critical
        severity boundaries on the chart.

        Args:
            plot_item: The PyQtGraph PlotItem to add the line to.
            value: The Y-axis (risk score) value to draw the line at.
            color: Hex color string for the line.
        """
        pen = pg.mkPen(color=color, width=1, style=Qt.PenStyle.DashLine)
        threshold_line = pg.InfiniteLine(pos=value, angle=0, pen=pen)
        plot_item.addItem(threshold_line)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_risk_score(self, risk_score: int, severity: str) -> None:
        """
        Appends a new risk score data point to the rolling trend line
        and redraws it. Intended to be called once per dashboard tick
        with the smoothed, current risk level (not a raw per-packet
        score).

        Args:
            risk_score: The current smoothed risk score, 0-100.
            severity: The current severity label (unused directly by
                the trend line itself, accepted for API symmetry with
                add_threat_event() and potential future use, e.g.
                dynamically recoloring the curve).
        """
        self._tick_counter += 1
        self._x_axis.append(self._tick_counter)
        self._risk_history.append(float(max(0, min(risk_score, 100))))

        x_values = list(self._x_axis)
        self._risk_curve.setData(x_values, list(self._risk_history))

        self._redraw_event_markers()

    def add_threat_event(self, risk_score: int, severity: str) -> None:
        """
        Records a discrete threat detection event as a marker at the
        current tick position. Intended to be called immediately when
        a threat is detected (e.g., from MonitoringEngine.threat_detected),
        independently of the once-per-tick trend line updates, so
        rapid bursts of threats are all captured even between ticks.

        Args:
            risk_score: The individual event's risk score, 0-100,
                used as the marker's Y position.
            severity: The severity label for this event (e.g., "High",
                "Critical"), used to color the marker.
        """
        current_tick = self._tick_counter if self._x_axis else 0
        clamped_score = max(0, min(risk_score, 100))
        self._event_markers.append((current_tick, clamped_score, severity))
        self._redraw_event_markers()

    def clear(self) -> None:
        """
        Clears all plotted history and event markers. Intended to be
        called when monitoring restarts, so the timeline shows a
        fresh trend rather than continuing to plot alongside a stale
        previous session.
        """
        self._risk_history.clear()
        self._x_axis.clear()
        self._event_markers.clear()
        self._tick_counter = 0
        self._risk_curve.setData([], [])
        self._event_scatter.setData([])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _redraw_event_markers(self) -> None:
        """
        Rebuilds the scatter plot from the event marker buffer,
        filtering out any markers that have scrolled outside the
        currently visible tick window (matching the trend line's
        rolling behavior, so old markers don't linger off to the left
        of the visible chart area indefinitely).
        """
        if not self._x_axis:
            self._event_scatter.setData([])
            return

        visible_minimum_tick = self._x_axis[0]
        visible_markers = [
            marker for marker in self._event_markers if marker[0] >= visible_minimum_tick
        ]

        spots = [
            {
                "pos": (tick, score),
                "brush": pg.mkBrush(
                    _SEVERITY_MARKER_COLORS.get(severity, _DEFAULT_MARKER_COLOR)
                ),
            }
            for tick, score, severity in visible_markers
        ]
        self._event_scatter.setData(spots)