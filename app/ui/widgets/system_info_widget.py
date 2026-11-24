"""
app/ui/widgets/system_info_widget.py

System and device information panel widget for CyberSentinel AI's
dashboard.

Responsibilities:
    - Render live CPU/RAM/Disk usage as custom-painted circular
      gauges (CircularGauge), each shifting color from its base
      accent through warning-orange to alert-red as usage crosses
      configured thresholds.
    - Display static device information (hostname, IP address, OS
      name/version, architecture) in a simple label-value list.
    - Expose set_device_info() (called once, since this data rarely
      changes during a session) and update_system_stats() (called
      once per dashboard tick, intended to be connected to
      app.core.engine.MonitoringEngine.dashboard_updated).

Both CircularGauge and SystemInfoWidget are purely presentational:
neither imports psutil, threading, or app.core.engine directly. They
only know how to render the DeviceInfo/SystemStats objects they're
given, which are defined in network/network_monitor.py.

LAYOUT FIX (see project changelog): this widget previously relied on
the layout squeezing itself to fit whatever height its sibling
(NetworkGraphWidget) in DashboardPage's row happened to need, which
was not always tall enough for three gauges + four device-info rows,
causing visual overlap. This version declares an explicit minimum
height on the widget itself, forcing Qt to allocate enough vertical
space for both sections regardless of sibling widgets.
"""

from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPaintEvent, QPen
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from config.settings import ThemeConfig
from network.network_monitor import DeviceInfo, SystemStats
from utils.logger import get_logger

logger = get_logger(__name__)

# Minimum total height for the whole SystemInfoWidget card. This is
# the key fix: without an explicit floor here, DashboardPage's
# middle_row_one QHBoxLayout sizes this widget to match
# NetworkGraphWidget's height, which is not always tall enough for
# the gauge row + four device-info rows, causing visual overlap.
_WIDGET_MIN_HEIGHT: int = 360

# Gauge sizing, tightened from the original (130, 170) so three
# gauges comfortably fit side-by-side within a card this width while
# still leaving clean room for the device-info rows beneath them.
_GAUGE_MIN_WIDTH: int = 100
_GAUGE_MIN_HEIGHT: int = 132
_GAUGE_BOTTOM_TEXT_RESERVE: int = 36  # reserved space below the arc for title+detail text


class CircularGauge(QWidget):
    """
    A custom-painted circular progress gauge displaying a 0-100
    percentage value with an optional detail line beneath it (e.g.
    "12.4 / 16.0 GB" for a RAM gauge).

    The arc color escalates from the gauge's assigned base accent
    color to WARNING_ORANGE and then ALERT_RED as the value crosses
    configured thresholds, giving an at-a-glance danger cue consistent
    with the dashboard's "Red Alerts" theme requirement, without
    requiring the caller to manage coloring logic itself.
    """

    _WARNING_THRESHOLD: float = 70.0
    _CRITICAL_THRESHOLD: float = 90.0

    _ARC_START_ANGLE_SIXTEENTHS: int = 225 * 16   # Qt angles are in 1/16th-degree units
    _ARC_FULL_SPAN_SIXTEENTHS: int = -270 * 16    # negative = clockwise sweep
    _ARC_LINE_WIDTH: int = 9

    def __init__(
        self,
        title: str,
        accent_color: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Args:
            title: The gauge's label, shown beneath the arc (e.g., "CPU").
            accent_color: The base hex color used while the value is
                below the warning threshold.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._title = title
        self._accent_color = accent_color
        self._value: float = 0.0
        self._detail_text: str = ""

        self.setMinimumSize(_GAUGE_MIN_WIDTH, _GAUGE_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_value(self, value: float, detail_text: str = "") -> None:
        """
        Updates the gauge's displayed value and optional detail text,
        then schedules a repaint.

        Args:
            value: The percentage value to display; clamped to 0-100.
            detail_text: Optional secondary text shown below the
                percentage and title (e.g., used/total for RAM/Disk).
                Left as an empty string for gauges with no natural
                detail line (e.g., CPU).
        """
        self._value = max(0.0, min(value, 100.0))
        self._detail_text = detail_text
        self.update()  # schedules paintEvent()

    def _resolve_arc_color(self) -> QColor:
        """
        Returns:
            The arc's current color: the gauge's base accent color
            under the warning threshold, WARNING_ORANGE between the
            warning and critical thresholds, and ALERT_RED at or
            above the critical threshold.
        """
        if self._value >= self._CRITICAL_THRESHOLD:
            return QColor(ThemeConfig.ALERT_RED)
        if self._value >= self._WARNING_THRESHOLD:
            return QColor(ThemeConfig.WARNING_ORANGE)
        return QColor(self._accent_color)

    def paintEvent(self, event: QPaintEvent) -> None:
        """
        Custom-paints the gauge: a dim 270-degree background track, a
        colored foreground arc proportional to the current value, the
        percentage centered inside the arc, and the title/detail text
        beneath it — all sized relative to this widget's ACTUAL
        allocated size (self.width()/self.height()) rather than a
        fixed assumption, so the gauge degrades gracefully even if
        given less space than its minimum hint requests.

        Args:
            event: The Qt paint event (unused directly; required by
                the QWidget override signature).
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Reserve space below the arc for the title + detail text so
        # the arc itself stays perfectly circular regardless of widget height.
        available_height_for_arc = max(self.height() - _GAUGE_BOTTOM_TEXT_RESERVE, 10)
        side = max(min(self.width(), available_height_for_arc), 10)
        arc_rect = QRectF(
            (self.width() - side) / 2 + self._ARC_LINE_WIDTH / 2,
            self._ARC_LINE_WIDTH / 2,
            side - self._ARC_LINE_WIDTH,
            side - self._ARC_LINE_WIDTH,
        )

        # Background track: full 270-degree arc in a dim border color.
        track_pen = QPen(QColor(ThemeConfig.BORDER_COLOR))
        track_pen.setWidth(self._ARC_LINE_WIDTH)
        track_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track_pen)
        painter.drawArc(arc_rect, self._ARC_START_ANGLE_SIXTEENTHS, self._ARC_FULL_SPAN_SIXTEENTHS)

        # Foreground value arc: spans a fraction of the full sweep
        # proportional to the current value.
        value_pen = QPen(self._resolve_arc_color())
        value_pen.setWidth(self._ARC_LINE_WIDTH)
        value_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(value_pen)
        value_span = int(self._ARC_FULL_SPAN_SIXTEENTHS * (self._value / 100.0))
        painter.drawArc(arc_rect, self._ARC_START_ANGLE_SIXTEENTHS, value_span)

        # Percentage value, centered inside the arc.
        painter.setPen(QColor(ThemeConfig.TEXT_PRIMARY))
        painter.setFont(QFont(ThemeConfig.FONT_FAMILY, 13, QFont.Weight.Bold))
        painter.drawText(arc_rect, Qt.AlignmentFlag.AlignCenter, f"{self._value:.0f}%")

        # Title, directly beneath the arc.
        painter.setFont(QFont(ThemeConfig.FONT_FAMILY, 8, QFont.Weight.DemiBold))
        painter.setPen(QColor(ThemeConfig.TEXT_SECONDARY))
        title_rect = QRectF(0, arc_rect.bottom() + 4, self.width(), 14)
        painter.drawText(title_rect, Qt.AlignmentFlag.AlignCenter, self._title.upper())

        # Optional detail text (e.g., "12.4 / 16.0 GB"), beneath the title.
        if self._detail_text:
            painter.setFont(QFont(ThemeConfig.FONT_FAMILY, 7))
            painter.setPen(QColor(ThemeConfig.TEXT_DISABLED))
            detail_rect = QRectF(0, arc_rect.bottom() + 18, self.width(), 12)
            painter.drawText(detail_rect, Qt.AlignmentFlag.AlignCenter, self._detail_text)

        painter.end()


class SystemInfoWidget(QWidget):
    """
    A rounded-card dashboard panel combining live CPU/RAM/Disk usage
    gauges with static device information (hostname, IP address, OS
    details), fulfilling the project spec's "System Information" and
    "Device Information" dashboard features in a single panel.

    Declares an explicit minimum height (_WIDGET_MIN_HEIGHT) so its
    layout always has enough room for both the gauge row and the
    device-info rows, regardless of what height a sibling widget in
    the same dashboard row would otherwise impose.

    Usage:
        panel = SystemInfoWidget()
        layout.addWidget(panel)
        ...
        panel.set_device_info(engine.get_device_info())  # once, static
        engine.dashboard_updated.connect(
            lambda snap: panel.update_system_stats(snap.system_stats)
        )
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("systemInfoWidget")
        self.setMinimumHeight(_WIDGET_MIN_HEIGHT)

        self._device_value_labels: dict[str, QLabel] = {}

        self._build_layout()
        logger.debug("SystemInfoWidget initialized (min_height=%d).", _WIDGET_MIN_HEIGHT)

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Builds the card container, gauge row, separator, and device info rows."""
        self.setStyleSheet(
            f"""
            QWidget#systemInfoWidget {{
                background-color: {ThemeConfig.BACKGROUND_CARD};
                border: 1px solid {ThemeConfig.BORDER_COLOR};
                border-radius: 10px;
            }}
            """
        )

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(18, 14, 18, 16)
        outer_layout.setSpacing(10)

        title_label = QLabel("System Information")
        title_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 12, QFont.Weight.DemiBold))
        title_label.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY}; background: transparent;")
        outer_layout.addWidget(title_label)

        gauge_row = QHBoxLayout()
        gauge_row.setSpacing(4)

        self._cpu_gauge = CircularGauge("CPU", ThemeConfig.NEON_CYAN)
        self._ram_gauge = CircularGauge("RAM", ThemeConfig.NEON_BLUE)
        self._disk_gauge = CircularGauge("Disk", ThemeConfig.ACCENT_PURPLE)

        gauge_row.addWidget(self._cpu_gauge)
        gauge_row.addWidget(self._ram_gauge)
        gauge_row.addWidget(self._disk_gauge)
        outer_layout.addLayout(gauge_row)

        outer_layout.addWidget(self._build_separator())

        device_info_fields: Tuple[Tuple[str, str], ...] = (
            ("hostname", "Hostname"),
            ("ip_address", "IP Address"),
            ("os", "Operating System"),
            ("architecture", "Architecture"),
        )

        info_rows_container = QVBoxLayout()
        info_rows_container.setSpacing(6)
        for field_key, display_label in device_info_fields:
            row_layout, value_label = self._build_info_row(display_label)
            self._device_value_labels[field_key] = value_label
            info_rows_container.addLayout(row_layout)

        outer_layout.addLayout(info_rows_container)
        outer_layout.addStretch()

    def _build_separator(self) -> QFrame:
        """Returns a thin horizontal divider between the gauge row and device info rows."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedHeight(1)
        line.setStyleSheet(f"background-color: {ThemeConfig.BORDER_COLOR}; max-height: 1px;")
        return line

    def _build_info_row(self, label_text: str) -> Tuple[QHBoxLayout, QLabel]:
        """
        Builds a single "label: value" row for the device info section.

        Args:
            label_text: The static field label (e.g., "Hostname").

        Returns:
            A tuple of (row_layout, value_label). The caller retains
            a reference to value_label so its text can be updated
            later via set_device_info().
        """
        row_layout = QHBoxLayout()
        row_layout.setSpacing(8)

        key_label = QLabel(label_text)
        key_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 8))
        key_label.setStyleSheet(f"color: {ThemeConfig.TEXT_SECONDARY}; background: transparent;")
        key_label.setFixedWidth(120)

        value_label = QLabel("—")
        value_label.setFont(QFont(ThemeConfig.FONT_FAMILY_MONO, 8))
        value_label.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY}; background: transparent;")
        value_label.setWordWrap(True)

        row_layout.addWidget(key_label)
        row_layout.addWidget(value_label, stretch=1)
        return row_layout, value_label

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_device_info(self, device_info: DeviceInfo) -> None:
        """
        Populates the static device information rows. Intended to be
        called once, shortly after construction (device details don't
        change during a normal application session).

        Args:
            device_info: The DeviceInfo instance to display, typically
                obtained via app.core.engine.MonitoringEngine.get_device_info().
        """
        self._device_value_labels["hostname"].setText(device_info.hostname)
        self._device_value_labels["ip_address"].setText(device_info.ip_address)
        self._device_value_labels["os"].setText(
            f"{device_info.os_name} {device_info.os_version}".strip()
        )
        self._device_value_labels["architecture"].setText(device_info.architecture)

    def update_system_stats(self, stats: SystemStats) -> None:
        """
        Updates all three gauges from a fresh SystemStats snapshot.
        Intended to be called once per dashboard tick.

        Args:
            stats: The latest system resource snapshot, typically
                obtained from an app.core.engine.DashboardSnapshot's
                system_stats field.
        """
        self._cpu_gauge.set_value(stats.cpu_percent)
        self._ram_gauge.set_value(
            stats.ram_percent,
            detail_text=f"{stats.ram_used_gb:.1f} / {stats.ram_total_gb:.1f} GB",
        )
        self._disk_gauge.set_value(
            stats.disk_percent,
            detail_text=f"{stats.disk_used_gb:.1f} / {stats.disk_total_gb:.1f} GB",
        )