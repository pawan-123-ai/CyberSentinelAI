"""
network/traffic_analyzer.py

Traffic intelligence layer for CyberSentinel AI.

Architecture position (per the project's target architecture):
    Network Capture -> Packet Analyzer -> Traffic Analyzer ->
        { AI Detection, Behavioral Analysis, Protocol Classification }
        -> Threat Correlation

This module owns ALL flow-level and source-IP-level state tracking —
packet/byte counts, flow duration, per-source destination port/IP
fan-out, connection rate, and SYN ratio — as the single source of
truth. ai/feature_extractor.py and the upcoming
security/behavior_analyzer.py both read from this same tracked state
via TrafficAnalyzer, rather than each maintaining their own duplicate
tracker that could silently drift out of sync with the other.

This module has no GUI, AI, or database dependency and knows nothing
about Predictions, DetectionResults, or Alerts — it only tracks and
reports raw traffic behavior.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple

from network.packet_capture import CapturedPacket
from utils.logger import get_logger

logger = get_logger(__name__)

# Flow identity key: a direction-independent tuple identifying the
# connection a packet belongs to.
FlowKey = Tuple[str, int, str, int, str]  # (ip_a, port_a, ip_b, port_b, protocol)

# How long a flow may sit idle before it's considered ended and purged.
FLOW_TIMEOUT_SECONDS: float = 120.0

# Rolling window for per-source-IP behavioral tracking (destination
# port/IP fan-out, connection rate, SYN ratio). Resets every window so
# these stats reflect RECENT behavior, not the whole session — a
# source IP that scanned aggressively an hour ago shouldn't still
# look suspicious now.
SOURCE_IP_WINDOW_SECONDS: float = 30.0

# Minimum time between stale-state cleanup sweeps, throttled to avoid
# scanning the full flow/source-IP tables on every single packet.
CLEANUP_INTERVAL_SECONDS: float = 30.0

# Source-IP states are purged entirely after this many idle windows
# (rather than just rotating), bounding memory for source IPs that
# appeared once and never returned.
SOURCE_IP_IDLE_MULTIPLIER: float = 4.0


@dataclass
class _FlowState:
    """Internal mutable per-flow tracking state. Not exposed outside this module."""
    start_time: float
    last_seen: float
    packet_count: int = 0
    byte_count: int = 0
    flag_count: int = 0


@dataclass
class _SourceIPState:
    """
    Internal mutable per-source-IP tracking state, rolling over every
    SOURCE_IP_WINDOW_SECONDS. Not exposed outside this module —
    TrafficSnapshot is the public read-only view.
    """
    window_start: float
    dst_ports: Set[int] = field(default_factory=set)
    dst_ips: Set[str] = field(default_factory=set)
    connection_count: int = 0
    syn_count: int = 0
    last_seen: float = field(default_factory=lambda: time.monotonic())


@dataclass
class TrafficSnapshot:
    """
    A read-only, point-in-time view of everything TrafficAnalyzer
    knows about the flow and source IP a single packet belongs to,
    computed at the moment analyze() is called.

    This is the data contract handed to downstream consumers:
    ai/feature_extractor.py builds its numeric FeatureVector from
    this, and the upcoming security/behavior_analyzer.py will read
    the same fields to run its independent rule-based checks —
    guaranteeing both layers agree on the underlying traffic facts.
    """
    # Flow-level (this specific src<->dst pair)
    flow_duration: float
    flow_packet_count: int
    flow_byte_count: int
    flow_packet_rate: float
    flow_byte_rate: float
    flow_flag_count: int

    # Source-IP-level (this src IP's recent behavior across ALL destinations)
    unique_dst_ports: int
    unique_dst_ips: int
    connection_count: int
    connection_rate: float
    syn_count: int
    syn_ratio: float  # syn_count / connection_count, 0.0 if no connections yet

    def to_dict(self) -> dict:
        """Returns a plain-dict representation, for logging/debugging."""
        return {
            "flow_duration": round(self.flow_duration, 4),
            "flow_packet_count": self.flow_packet_count,
            "flow_byte_count": self.flow_byte_count,
            "flow_packet_rate": round(self.flow_packet_rate, 2),
            "flow_byte_rate": round(self.flow_byte_rate, 2),
            "flow_flag_count": self.flow_flag_count,
            "unique_dst_ports": self.unique_dst_ports,
            "unique_dst_ips": self.unique_dst_ips,
            "connection_count": self.connection_count,
            "connection_rate": round(self.connection_rate, 2),
            "syn_count": self.syn_count,
            "syn_ratio": round(self.syn_ratio, 4),
        }


class TrafficAnalyzer:
    """
    Tracks live flow-level and source-IP-level traffic state and
    produces TrafficSnapshot objects on demand for each captured
    packet.

    Thread-safe: intended to be called from the same synchronous
    per-packet pipeline as ai/feature_extractor.py (itself invoked
    from security/threat_detector.py's process_packet(), which runs
    on network/packet_capture.py's background capture thread) — the
    internal lock protects against any future concurrent callers
    (e.g., a UI page polling current state directly).

    Usage:
        analyzer = TrafficAnalyzer()
        snapshot = analyzer.analyze(captured_packet)
    """

    def __init__(
        self,
        flow_timeout_seconds: float = FLOW_TIMEOUT_SECONDS,
        source_ip_window_seconds: float = SOURCE_IP_WINDOW_SECONDS,
        cleanup_interval_seconds: float = CLEANUP_INTERVAL_SECONDS,
    ) -> None:
        """
        Args:
            flow_timeout_seconds: Idle duration after which a flow is purged.
            source_ip_window_seconds: Rolling window length for
                per-source-IP behavioral tracking.
            cleanup_interval_seconds: Minimum time between stale-state
                cleanup sweeps.
        """
        self._flows: Dict[FlowKey, _FlowState] = {}
        self._source_ip_states: Dict[str, _SourceIPState] = {}
        self._lock = threading.Lock()

        self._flow_timeout = flow_timeout_seconds
        self._source_ip_window = source_ip_window_seconds
        self._cleanup_interval = cleanup_interval_seconds
        self._last_cleanup_time = time.monotonic()

        logger.info(
            "TrafficAnalyzer initialized (flow_timeout=%.0fs, source_ip_window=%.0fs).",
            flow_timeout_seconds,
            source_ip_window_seconds,
        )

    def analyze(self, packet: CapturedPacket) -> TrafficSnapshot:
        """
        Folds a single captured packet into both its flow's state and
        its source IP's rolling state, then returns a TrafficSnapshot
        reflecting the updated totals.

        Args:
            packet: The parsed packet to analyze.

        Returns:
            A TrafficSnapshot computed from the current flow and
            source-IP state, immediately after incorporating this packet.
        """
        with self._lock:
            flow_state = self._update_flow_state(packet)
            source_state = self._update_source_ip_state(packet)

            flow_duration = max(flow_state.last_seen - flow_state.start_time, 1e-6)
            flow_packet_rate = flow_state.packet_count / flow_duration
            flow_byte_rate = flow_state.byte_count / flow_duration

            source_elapsed = max(time.monotonic() - source_state.window_start, 1e-6)
            connection_rate = source_state.connection_count / source_elapsed
            syn_ratio = (
                source_state.syn_count / source_state.connection_count
                if source_state.connection_count > 0
                else 0.0
            )

            snapshot = TrafficSnapshot(
                flow_duration=flow_duration,
                flow_packet_count=flow_state.packet_count,
                flow_byte_count=flow_state.byte_count,
                flow_packet_rate=flow_packet_rate,
                flow_byte_rate=flow_byte_rate,
                flow_flag_count=flow_state.flag_count,
                unique_dst_ports=len(source_state.dst_ports),
                unique_dst_ips=len(source_state.dst_ips),
                connection_count=source_state.connection_count,
                connection_rate=connection_rate,
                syn_count=source_state.syn_count,
                syn_ratio=syn_ratio,
            )

            self._maybe_cleanup_stale_state()

        return snapshot

    def get_source_ip_snapshot(self, src_ip: str) -> Optional[TrafficSnapshot]:
        """
        Returns the current source-IP-level snapshot for a given IP
        without requiring a new packet, for read-only queries from
        other modules (e.g., the upcoming security/behavior_analyzer.py
        checking "is this IP currently mid-scan" independent of a
        specific flow). Flow-level fields are zeroed since no
        specific flow is implied by this query.

        Args:
            src_ip: The source IP address to look up.

        Returns:
            A TrafficSnapshot with only source-IP fields populated, or
            None if this IP has no currently tracked state.
        """
        with self._lock:
            source_state = self._source_ip_states.get(src_ip)
            if source_state is None:
                return None

            source_elapsed = max(time.monotonic() - source_state.window_start, 1e-6)
            connection_rate = source_state.connection_count / source_elapsed
            syn_ratio = (
                source_state.syn_count / source_state.connection_count
                if source_state.connection_count > 0
                else 0.0
            )

            return TrafficSnapshot(
                flow_duration=0.0,
                flow_packet_count=0,
                flow_byte_count=0,
                flow_packet_rate=0.0,
                flow_byte_rate=0.0,
                flow_flag_count=0,
                unique_dst_ports=len(source_state.dst_ports),
                unique_dst_ips=len(source_state.dst_ips),
                connection_count=source_state.connection_count,
                connection_rate=connection_rate,
                syn_count=source_state.syn_count,
                syn_ratio=syn_ratio,
            )

    def reset(self) -> None:
        """
        Clears all tracked flow and source-IP state. Should be called
        when monitoring is stopped and later restarted, so stale
        state from a previous session doesn't influence new analysis
        (mirrors security/threat_detector.py's reset() convention).
        """
        with self._lock:
            flow_count = len(self._flows)
            source_count = len(self._source_ip_states)
            self._flows.clear()
            self._source_ip_states.clear()
            self._last_cleanup_time = time.monotonic()
        logger.info(
            "TrafficAnalyzer state reset (%d flow(s), %d source IP(s) cleared).",
            flow_count,
            source_count,
        )

    @property
    def active_flow_count(self) -> int:
        """Returns the number of flows currently tracked in memory."""
        with self._lock:
            return len(self._flows)

    @property
    def active_source_ip_count(self) -> int:
        """Returns the number of source IPs currently tracked in memory."""
        with self._lock:
            return len(self._source_ip_states)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_flow_key(packet: CapturedPacket) -> FlowKey:
        """
        Builds a direction-independent flow identity key so traffic
        from A->B and B->A is treated as the same flow.

        Args:
            packet: The packet to derive a flow key from.

        Returns:
            A normalized FlowKey tuple.
        """
        endpoint_a = (packet.src_ip, packet.src_port)
        endpoint_b = (packet.dst_ip, packet.dst_port)
        first, second = sorted([endpoint_a, endpoint_b])
        return (first[0], first[1], second[0], second[1], packet.protocol)

    def _update_flow_state(self, packet: CapturedPacket) -> _FlowState:
        """
        Retrieves (creating if needed) the flow state for this
        packet's flow and folds the packet into it. Must be called
        while holding self._lock.

        Args:
            packet: The packet to fold into its flow's running stats.

        Returns:
            The updated _FlowState for this packet's flow.
        """
        now = time.monotonic()
        flow_key = self._build_flow_key(packet)
        flow_state = self._flows.get(flow_key)

        if flow_state is None:
            flow_state = _FlowState(start_time=now, last_seen=now)
            self._flows[flow_key] = flow_state

        flow_state.last_seen = now
        flow_state.packet_count += 1
        flow_state.byte_count += packet.size

        if packet.protocol == "TCP" and packet.flags:
            flow_state.flag_count += len(packet.flags)

        return flow_state

    def _update_source_ip_state(self, packet: CapturedPacket) -> _SourceIPState:
        """
        Retrieves (creating/rotating if needed) the source-IP tracking
        state for this packet's source, folding in its destination
        port/IP and connection/SYN counters. The window rotates
        (resets) every source_ip_window_seconds so these stats reflect
        recent behavior only. Must be called while holding self._lock.

        Args:
            packet: The packet to fold into its source IP's state.

        Returns:
            The updated _SourceIPState for this packet's source IP.
        """
        now = time.monotonic()
        state = self._source_ip_states.get(packet.src_ip)

        if state is None or (now - state.window_start) > self._source_ip_window:
            state = _SourceIPState(window_start=now)
            self._source_ip_states[packet.src_ip] = state

        state.last_seen = now
        state.dst_ports.add(packet.dst_port)
        state.dst_ips.add(packet.dst_ip)
        state.connection_count += 1

        flags = packet.flags or ""
        if packet.protocol == "TCP" and "S" in flags and "A" not in flags:
            # SYN-without-ACK: a genuinely NEW connection attempt, as
            # opposed to a mid-conversation packet on an already-
            # established session. This is what makes connection_rate
            # and syn_ratio meaningful indicators of scanning/flooding
            # behavior rather than just counting every packet of one
            # legitimate, long-lived connection.
            state.syn_count += 1

        return state

    def _maybe_cleanup_stale_state(self) -> None:
        """
        Purges flows idle longer than flow_timeout_seconds and source-
        IP states idle longer than source_ip_window_seconds *
        SOURCE_IP_IDLE_MULTIPLIER, throttled to run at most once per
        cleanup_interval_seconds. Must be called while holding self._lock.
        """
        now = time.monotonic()
        if (now - self._last_cleanup_time) < self._cleanup_interval:
            return

        stale_flow_keys = [
            key for key, state in self._flows.items()
            if (now - state.last_seen) > self._flow_timeout
        ]
        for key in stale_flow_keys:
            del self._flows[key]

        source_idle_timeout = self._source_ip_window * SOURCE_IP_IDLE_MULTIPLIER
        stale_source_ips = [
            ip for ip, state in self._source_ip_states.items()
            if (now - state.last_seen) > source_idle_timeout
        ]
        for ip in stale_source_ips:
            del self._source_ip_states[ip]

        self._last_cleanup_time = now

        if stale_flow_keys or stale_source_ips:
            logger.debug(
                "Cleanup: purged %d stale flow(s), %d stale source IP(s). "
                "Active: %d flow(s), %d source IP(s).",
                len(stale_flow_keys),
                len(stale_source_ips),
                len(self._flows),
                len(self._source_ip_states),
            )


# ==========================================================================
# Module-level singleton accessor
# ==========================================================================

_traffic_analyzer_instance: Optional[TrafficAnalyzer] = None


def get_traffic_analyzer() -> TrafficAnalyzer:
    """
    Returns the shared, application-wide TrafficAnalyzer instance,
    creating it on first call.

    Mirrors the singleton pattern used throughout the project. Both
    ai/feature_extractor.py and the upcoming
    security/behavior_analyzer.py should call this rather than
    instantiating TrafficAnalyzer directly, so both layers observe
    the exact same tracked flow/source-IP state.

    Returns:
        The singleton TrafficAnalyzer instance.
    """
    global _traffic_analyzer_instance
    if _traffic_analyzer_instance is None:
        _traffic_analyzer_instance = TrafficAnalyzer()
    return _traffic_analyzer_instance