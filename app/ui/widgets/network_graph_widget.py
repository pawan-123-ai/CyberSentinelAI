"""
app/ui/widgets/network_graph_widget.py

Live network traffic graph widget for CyberSentinel AI's dashboard.

Responsibilities:
    - Render a rolling, real-time dual-axis line chart of network
      throughput: packets/second (left axis) and bandwidth in KB/s
      (right axis — a separate scale is required since the two
      series differ by orders of magnitude).
    - Maintain a fixed-length rolling data buffer
      (UIConfig.MAX_GRAPH_DATA_POINTS) fed via update_data(), intended
      to be called once per tick from a slot connected to
      app.core.engine.MonitoringEngine.dashboard_updated.
    - Apply the application's dark cybersecurity theme (ThemeConfig)
      to PyQtGraph's plot styling.

This widget is purely presentational: it holds no business logic and
knows nothing about packet capture, AI, threading, or Qt signals
beyond its own update_data() method. It only knows how to plot
numbers it's given.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import pyqtgraph as pg
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from config.settings import ThemeConfig, UIConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class NetworkGraphWidget(QWidget):
    """
    A rounded-card widget hosting a dual-axis, real-time line chart of
    packets/second and bandwidth (KB/s), styled to match the dark
    cybersecurity dashboard theme.

    Usage:
        graph = NetworkGraphWidget()
        layout.addWidget(graph)
        ...
        # Typically wired via a Qt signal, e.g.:
        # engine.dashboard_updated.connect(
        #     lambda snap: graph.update_data(
        #         snap.packets_per_second, snap.bytes_per_second
        #     )
        # )
        graph.update_data(packets_per_second=42, bytes_per_second=51200.0)
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("networkGraphWidget")

        self._max_points = UIConfig.MAX_GRAPH_DATA_POINTS
        self._packet_history: Deque[float] = deque(maxlen=self._max_points)
        self._bandwidth_history: Deque[float] = deque(maxlen=self._max_points)
        self._x_axis: Deque[int] = deque(maxlen=self._max_points)
        self._tick_counter: int = 0

        self._build_layout()
        self._configure_plot()

        logger.debug("NetworkGraphWidget initialized (max_points=%d).", self._max_points)

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Builds the card container, header row (title + legend), and embedded plot widget."""
        self.setStyleSheet(
            f"""
            QWidget#networkGraphWidget {{
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

        title_label = QLabel("Live Network Traffic")
        title_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 12, QFont.Weight.DemiBold))
        title_label.setStyleSheet(
            f"color: {ThemeConfig.TEXT_PRIMARY}; background: transparent;"
        )

        legend_label = QLabel(
            f"<span style='color:{ThemeConfig.NEON_BLUE};'>&#9679;</span> Packets/s"
            f"&nbsp;&nbsp;&nbsp;"
            f"<span style='color:{ThemeConfig.NEON_CYAN};'>&#9679;</span> Bandwidth (KB/s)"
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
        Configures the PyQtGraph plot: dark theme colors, gridlines,
        a primary curve (packets/sec) on the left axis, and a
        secondary curve (bandwidth) on an independently-scaled right
        axis via a linked ViewBox — PyQtGraph's standard dual-axis
        technique. Mouse interaction is disabled so the dashboard
        remains a pure live display rather than an interactive chart.
        """
        self._plot_widget.setBackground(QColor(ThemeConfig.BACKGROUND_CARD))

        plot_item = self._plot_widget.getPlotItem()
        plot_item.showGrid(x=True, y=True, alpha=0.15)
        plot_item.setMenuEnabled(False)
        plot_item.hideButtons()

        axis_pen = pg.mkPen(color=ThemeConfig.BORDER_COLOR, width=1)

        left_axis = plot_item.getAxis("left")
        left_axis.setPen(axis_pen)
        left_axis.setTextPen(ThemeConfig.TEXT_SECONDARY)
        left_axis.setLabel("Packets/s", color=ThemeConfig.NEON_BLUE)

        bottom_axis = plot_item.getAxis("bottom")
        bottom_axis.setPen(axis_pen)
        bottom_axis.setTextPen(ThemeConfig.TEXT_SECONDARY)
        bottom_axis.setLabel("Time (ticks ago)")

        # Primary curve: packets/sec, drawn on the plot's default (left) ViewBox.
        packet_pen = pg.mkPen(color=ThemeConfig.NEON_BLUE, width=2)
        self._packet_curve = plot_item.plot(pen=packet_pen, name="Packets/s")

        # Secondary curve: bandwidth (KB/s), drawn on a second ViewBox
        # sharing the same X axis but with its own independent Y scale.
        # Packet counts and byte throughput differ by orders of
        # magnitude, so sharing one Y axis would flatten one series.
        self._bandwidth_view_box = pg.ViewBox()
        plot_item.showAxis("right")
        plot_item.scene().addItem(self._bandwidth_view_box)

        right_axis = plot_item.getAxis("right")
        right_axis.linkToView(self._bandwidth_view_box)
        right_axis.setPen(axis_pen)
        right_axis.setTextPen(ThemeConfig.TEXT_SECONDARY)
        right_axis.setLabel("Bandwidth (KB/s)", color=ThemeConfig.NEON_CYAN)

        self._bandwidth_view_box.setXLink(plot_item.getViewBox())

        bandwidth_pen = pg.mkPen(color=ThemeConfig.NEON_CYAN, width=2)
        self._bandwidth_curve = pg.PlotCurveItem(pen=bandwidth_pen)
        self._bandwidth_view_box.addItem(self._bandwidth_curve)

        # PyQtGraph does not automatically keep a manually-added
        # secondary ViewBox's geometry in sync with the primary plot
        # area — this must be reconnected on every resize.
        plot_item.getViewBox().sigResized.connect(self._sync_bandwidth_view_box)
        self._sync_bandwidth_view_box()  # run once immediately at construction

        self._plot_widget.setMouseEnabled(x=False, y=False)

    def _sync_bandwidth_view_box(self) -> None:
        """
        Resizes and re-links the secondary (bandwidth) ViewBox to
        match the primary plot's current geometry. Required by
        PyQtGraph's dual-axis pattern whenever the widget is resized.
        """
        plot_item = self._plot_widget.getPlotItem()
        primary_view_box = plot_item.getViewBox()

        self._bandwidth_view_box.setGeometry(primary_view_box.sceneBoundingRect())
        self._bandwidth_view_box.linkedViewChanged(
            primary_view_box, self._bandwidth_view_box.XAxis
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_data(self, packets_per_second: int, bytes_per_second: float) -> None:
        """
        Appends a new data point to the rolling history and redraws
        both curves. Intended to be called once per dashboard tick.

        Args:
            packets_per_second: The latest packets/sec reading.
            bytes_per_second: The latest bandwidth reading in bytes/sec
                (converted to KB/s for display).
        """
        self._tick_counter += 1
        self._x_axis.append(self._tick_counter)
        self._packet_history.append(float(packets_per_second))
        self._bandwidth_history.append(bytes_per_second / 1024.0)

        x_values = list(self._x_axis)
        self._packet_curve.setData(x_values, list(self._packet_history))
        self._bandwidth_curve.setData(x_values, list(self._bandwidth_history))

    def clear(self) -> None:
        """
        Clears all plotted history. Intended to be called when
        monitoring restarts, so the graph shows a fresh trend rather
        than continuing to plot alongside a stale previous session.
        """
        self._packet_history.clear()
        self._bandwidth_history.clear()
        self._x_axis.clear()
        self._tick_counter = 0
        self._packet_curve.setData([], [])
        self._bandwidth_curve.setData([], [])