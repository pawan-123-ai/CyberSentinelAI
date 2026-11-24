"""
app/ui/widgets/network_topology_widget.py

Live network topology visualization widget for CyberSentinel AI's
dashboard.

Responsibilities:
    - Render a hub-and-spoke graph: the local device at the center,
      with recently-seen remote IPs arranged radially around it,
      connected by edges colored by traffic status (normal vs threat).
    - Use networkx purely for graph structure (nodes/edges) and a
      deterministic radial layout computation. A force-directed layout
      (e.g. spring_layout) was deliberately avoided, since recomputing
      it every tick causes nodes to jitter/rearrange on each redraw,
      which reads as "broken" on a continuously live-updating panel.
    - Render via matplotlib's FigureCanvasQTAgg embedded directly into
      the Qt layout, styled to match the dark cybersecurity theme.
    - Track a rolling set of remote endpoints with per-node "last
      seen" ticks, evicting idle/excess nodes automatically so the
      graph stays readable rather than growing unbounded — mirroring
      the same idle-cleanup pattern used by ai/feature_extractor.py's
      flow table and security/alert_manager.py's cooldown tracker.

This widget is purely presentational: it accepts a lightweight local
TopologyConnection DTO rather than importing CapturedPacket,
DetectionResult, or any AI/security-layer type, keeping it fully
decoupled and independently reusable. app/ui/dashboard.py is
responsible for translating live engine data into TopologyConnection
instances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QHBoxLayout, QLabel, QVBoxLayout, QWidget

from config.settings import ThemeConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# Maximum number of remote endpoint nodes shown at once. Beyond this,
# the least-recently-seen nodes are evicted first, keeping the graph
# readable even under a high-volume simulated attack with many
# distinct source IPs.
MAX_TOPOLOGY_NODES: int = 10

# A remote node not seen for this many ticks (~seconds, matching the
# dashboard's ~1s tick cadence) is evicted even if under the node cap,
# so the topology reflects genuinely "recent" activity rather than
# accumulating every IP seen since the app launched.
NODE_IDLE_TIMEOUT_TICKS: int = 45

# Node visual sizing (matplotlib scatter-style units via networkx).
_LOCAL_NODE_SIZE: int = 900
_REMOTE_NODE_SIZE_NORMAL: int = 380
_REMOTE_NODE_SIZE_THREAT: int = 550

# Radius of the circle remote nodes are arranged on; local device
# sits at the origin (0, 0).
_LAYOUT_RADIUS: float = 1.0


@dataclass
class TopologyConnection:
    """
    A lightweight description of one observed connection to a remote
    IP, used as input to NetworkTopologyWidget.update_connections().

    This DTO intentionally mirrors only the fields the widget actually
    needs for display, rather than accepting a full CapturedPacket or
    DetectionResult — keeping this widget decoupled from the
    network/AI/security layers' internal data models.
    """
    remote_ip: str
    is_threat: bool = False
    protocol: str = ""
    port: int = 0


@dataclass
class _RemoteNodeState:
    """
    Internal per-node tracking state, keyed by remote IP in
    NetworkTopologyWidget._remote_nodes. Not exposed outside this module.
    """
    is_threat: bool
    protocol: str
    port: int
    last_seen_tick: int


class NetworkTopologyWidget(QWidget):
    """
    A rounded-card widget rendering a live hub-and-spoke network
    topology graph via networkx (structure/layout) and matplotlib
    (rendering), styled to match the dark cybersecurity dashboard theme.

    Usage:
        topology = NetworkTopologyWidget()
        layout.addWidget(topology)
        ...
        # Typically wired from a slot that combines the local device's
        # IP with recent detection data, e.g.:
        # engine.dashboard_updated.connect(
        #     lambda snap: topology.update_connections(
        #         local_ip=snap.device_info.ip_address,
        #         connections=[
        #             TopologyConnection(d.packet.dst_ip, d.is_threat, d.packet.protocol, d.packet.dst_port)
        #             for d in threat_detector.get_recent_detections(limit=20)
        #         ],
        #     )
        # )
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("networkTopologyWidget")

        self._remote_nodes: Dict[str, _RemoteNodeState] = {}
        self._tick_counter: int = 0
        self._local_ip: str = ""
        self._local_label: str = "This Device"

        self._build_layout()
        self._style_axes()
        self._canvas.draw_idle()

        logger.debug("NetworkTopologyWidget initialized (max_nodes=%d).", MAX_TOPOLOGY_NODES)

    # ------------------------------------------------------------------
    # Layout construction
    # ------------------------------------------------------------------

    def _build_layout(self) -> None:
        """Builds the card container, header row (title + legend), and embedded matplotlib canvas."""
        self.setStyleSheet(
            f"""
            QWidget#networkTopologyWidget {{
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

        title_label = QLabel("Network Topology")
        title_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 12, QFont.Weight.DemiBold))
        title_label.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY}; background: transparent;")

        legend_label = QLabel(
            f"<span style='color:{ThemeConfig.NEON_BLUE};'>&#9679;</span> This Device"
            f"&nbsp;&nbsp;&nbsp;"
            f"<span style='color:{ThemeConfig.NEON_CYAN};'>&#9679;</span> Normal"
            f"&nbsp;&nbsp;&nbsp;"
            f"<span style='color:{ThemeConfig.ALERT_RED};'>&#9679;</span> Threat"
        )
        legend_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 9))
        legend_label.setStyleSheet(f"color: {ThemeConfig.TEXT_SECONDARY}; background: transparent;")

        header_row.addWidget(title_label)
        header_row.addStretch()
        header_row.addWidget(legend_label)
        outer_layout.addLayout(header_row)

        self._figure = Figure(figsize=(5, 4))
        self._axes = self._figure.add_subplot(111)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setMinimumHeight(260)
        self._canvas.setStyleSheet("background-color: transparent;")
        outer_layout.addWidget(self._canvas)

    def _style_axes(self) -> None:
        """
        Resets the matplotlib figure/axes to the dashboard's dark
        theme and a fixed square viewport. Called after every
        self._axes.clear(), since clear() wipes prior styling along
        with the plotted content.
        """
        self._figure.patch.set_facecolor(ThemeConfig.BACKGROUND_CARD)
        self._axes.set_facecolor(ThemeConfig.BACKGROUND_CARD)
        self._axes.set_xlim(-1.3, 1.3)
        self._axes.set_ylim(-1.3, 1.3)
        self._axes.set_aspect("equal")
        self._axes.axis("off")
        self._figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_connections(
        self,
        local_ip: str,
        connections: List[TopologyConnection],
        local_label: Optional[str] = None,
    ) -> None:
        """
        Advances the widget by one tick: folds newly observed
        connections into the tracked remote-node set, evicts idle/
        excess nodes, and redraws the graph.

        Args:
            local_ip: The local device's IP address, used as the
                center hub node and to skip self-referential connections.
            connections: Connections observed since the last call.
                Duplicate remote IPs within a single call are merged
                (is_threat is OR'd together, so a threat connection to
                an IP isn't masked by a later benign one in the same tick).
            local_label: Optional display label for the hub node
                (e.g., the device hostname). Defaults to "This Device"
                if never provided.
        """
        self._tick_counter += 1
        self._local_ip = local_ip
        if local_label:
            self._local_label = local_label

        merged_this_tick: Dict[str, TopologyConnection] = {}
        for connection in connections:
            if connection.remote_ip == local_ip or not connection.remote_ip:
                continue
            existing = merged_this_tick.get(connection.remote_ip)
            if existing is None:
                merged_this_tick[connection.remote_ip] = connection
            else:
                existing.is_threat = existing.is_threat or connection.is_threat

        for remote_ip, connection in merged_this_tick.items():
            self._remote_nodes[remote_ip] = _RemoteNodeState(
                is_threat=connection.is_threat,
                protocol=connection.protocol,
                port=connection.port,
                last_seen_tick=self._tick_counter,
            )

        self._evict_stale_and_excess_nodes()
        self._redraw()

    def clear(self) -> None:
        """
        Clears all tracked remote nodes and redraws an empty topology
        (just the local hub, if known). Intended to be called when
        monitoring restarts, so the topology doesn't keep displaying
        endpoints from a stale previous session.
        """
        self._remote_nodes.clear()
        self._tick_counter = 0
        self._redraw()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_stale_and_excess_nodes(self) -> None:
        """
        Removes remote nodes that have been idle longer than
        NODE_IDLE_TIMEOUT_TICKS, then — if still over MAX_TOPOLOGY_NODES —
        removes the least-recently-seen nodes until back at the cap.
        """
        stale_ips = [
            ip
            for ip, state in self._remote_nodes.items()
            if (self._tick_counter - state.last_seen_tick) > NODE_IDLE_TIMEOUT_TICKS
        ]
        for ip in stale_ips:
            del self._remote_nodes[ip]

        if len(self._remote_nodes) > MAX_TOPOLOGY_NODES:
            ranked_by_recency = sorted(
                self._remote_nodes.items(), key=lambda item: item[1].last_seen_tick
            )
            excess_count = len(self._remote_nodes) - MAX_TOPOLOGY_NODES
            for ip, _state in ranked_by_recency[:excess_count]:
                del self._remote_nodes[ip]

    def _compute_radial_layout(self, remote_ips: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Computes a deterministic hub-and-spoke layout: the local
        device fixed at the origin, and remote nodes evenly spaced
        around a circle of radius _LAYOUT_RADIUS. Deterministic (no
        randomness, no iterative force simulation) so nodes don't
        jump around between redraws — only entering/leaving smoothly
        as they're added or evicted.

        Args:
            remote_ips: The remote IPs to position, in a stable order.

        Returns:
            A dict mapping every node key (local IP + each remote IP)
            to its (x, y) position.
        """
        positions: Dict[str, Tuple[float, float]] = {self._local_ip: (0.0, 0.0)}

        node_count = len(remote_ips)
        if node_count == 0:
            return positions

        for index, remote_ip in enumerate(remote_ips):
            angle = (2 * math.pi * index) / node_count
            positions[remote_ip] = (
                _LAYOUT_RADIUS * math.cos(angle),
                _LAYOUT_RADIUS * math.sin(angle),
            )

        return positions

    def _redraw(self) -> None:
        """
        Rebuilds a networkx Graph from the current local IP + tracked
        remote nodes, computes the radial layout, and renders it onto
        the matplotlib canvas with theme-consistent node/edge coloring.
        """
        self._axes.clear()
        self._style_axes()

        if not self._local_ip:
            self._canvas.draw_idle()
            return

        remote_ips = list(self._remote_nodes.keys())

        graph = nx.Graph()
        graph.add_node(self._local_ip)
        for remote_ip in remote_ips:
            graph.add_node(remote_ip)
            graph.add_edge(self._local_ip, remote_ip)

        positions = self._compute_radial_layout(remote_ips)

        node_order = [self._local_ip] + remote_ips
        node_colors = [ThemeConfig.NEON_BLUE] + [
            ThemeConfig.ALERT_RED if self._remote_nodes[ip].is_threat else ThemeConfig.NEON_CYAN
            for ip in remote_ips
        ]
        node_sizes = [_LOCAL_NODE_SIZE] + [
            _REMOTE_NODE_SIZE_THREAT if self._remote_nodes[ip].is_threat else _REMOTE_NODE_SIZE_NORMAL
            for ip in remote_ips
        ]

        edge_colors = [
            ThemeConfig.ALERT_RED if self._remote_nodes[ip].is_threat else ThemeConfig.BORDER_COLOR
            for ip in remote_ips
        ]

        nx.draw_networkx_edges(
            graph,
            positions,
            ax=self._axes,
            edgelist=[(self._local_ip, ip) for ip in remote_ips],
            edge_color=edge_colors,
            width=1.5,
        )
        nx.draw_networkx_nodes(
            graph,
            positions,
            ax=self._axes,
            nodelist=node_order,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors=ThemeConfig.BACKGROUND_CARD,
            linewidths=1.5,
        )

        labels = {self._local_ip: self._local_label}
        labels.update({ip: ip for ip in remote_ips})
        nx.draw_networkx_labels(
            graph,
            positions,
            ax=self._axes,
            labels=labels,
            font_size=7,
            font_color=ThemeConfig.TEXT_PRIMARY,
            font_family=ThemeConfig.FONT_FAMILY,
        )

        self._canvas.draw_idle()