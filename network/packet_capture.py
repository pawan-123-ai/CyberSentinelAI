"""
Live packet capture engine for CyberSentinel AI, built on Scapy.

Responsibilities:
    - Sniff live network traffic on a background thread (non-blocking).
    - Extract lightweight, structured metadata from each captured packet
      (IPs, ports, protocol, size, TCP flags, payload preview).
    - Forward each parsed packet to one or more registered callbacks
      for downstream processing (feature extraction, AI prediction,
      persistence, UI updates).
    - Track live throughput statistics (packets/sec, bytes/sec).

This module is intentionally free of any GUI framework imports (no
PySide6) so it can run headlessly and be unit-tested in isolation.
app/core/engine.py is responsible for bridging its callbacks to Qt
signals consumed by the dashboard.

IMPORTANT (Windows):
    Live packet capture requires Npcap
    (https://npcap.com/) installed with "WinPcap API-compatible mode"
    enabled, AND the application must be run with Administrator
    privileges. Without both, sniff() will raise a PermissionError/OSError,
    which this module catches and logs rather than crashing the app.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, List, Optional, Tuple

from scapy.all import sniff, get_if_list, conf
from scapy.packet import Packet
from scapy.layers.inet import IP, TCP, UDP, ICMP

from config.settings import NetworkConfig
from utils.logger import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Payload configuration
# ---------------------------------------------------------------------------

# Number of payload bytes retained in memory for lightweight protocol
# signature detection.
#
# Examples:
#   - HTTP request methods
#   - SSH banners
#   - TLS ClientHello
#   - Other lightweight protocol signatures
#
# Raw payload is intentionally NOT persisted to MongoDB.
_PAYLOAD_PREVIEW_BYTES: int = 64


# ---------------------------------------------------------------------------
# Captured packet data structure
# ---------------------------------------------------------------------------

@dataclass
class CapturedPacket:
    """
    Structured representation of a single captured network packet,
    extracted from raw Scapy packet data.

    This is the common data contract passed to all downstream consumers:
        - feature extractor
        - AI predictor
        - database manager
        - UI
        - traffic analyzer
        - security/behavior analyzer

    payload_preview contains only the first few bytes of a TCP/UDP
    payload and is intended for lightweight protocol signature detection.

    IMPORTANT:
        payload_preview is intentionally excluded from to_dict()
        because raw payload bytes should not be persisted to MongoDB.
    """

    timestamp: datetime
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    size: int
    flags: str = ""
    interface: str = ""

    # NEW:
    # First N bytes of TCP/UDP payload.
    payload_preview: bytes = b""

    def to_dict(self) -> dict:
        """
        Converts this packet record into a plain dictionary, suitable
        for MongoDB insertion via DatabaseManager.insert_packet_record().

        NOTE:
            payload_preview is intentionally NOT included here.
            Raw payload bytes should not be stored in MongoDB because:

                1. They are unnecessary for existing packet queries.
                2. They can significantly increase document size.
                3. They may contain sensitive application data.
                4. Only lightweight protocol-signature detection needs
                   the in-memory preview.

        Returns:
            A dictionary representation of the packet.
        """

        return {
            "timestamp": self.timestamp,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "size": self.size,
            "flags": self.flags,
            "interface": self.interface,
        }


# ---------------------------------------------------------------------------
# Packet callback type
# ---------------------------------------------------------------------------

PacketCallback = Callable[[CapturedPacket], None]


# ---------------------------------------------------------------------------
# Traffic rate counter
# ---------------------------------------------------------------------------

class TrafficRateCounter:
    """
    Tracks live packets-per-second and bytes-per-second throughput
    using a rolling one-second window.

    Thread-safe:
        Packets are recorded from the background capture thread,
        but read from the Qt UI thread via get_rates().
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

        self._window_start = time.monotonic()

        self._packet_count_in_window = 0
        self._byte_count_in_window = 0

        self._last_packets_per_second = 0
        self._last_bytes_per_second = 0

    def record_packet(self, size: int) -> None:
        """
        Registers a newly captured packet's size in the current
        rolling window, rotating the window once a full second has
        elapsed since it began.

        Args:
            size:
                Size of the captured packet in bytes.
        """

        with self._lock:
            now = time.monotonic()
            elapsed = now - self._window_start

            if elapsed >= NetworkConfig.PACKETS_PER_SECOND_WINDOW:
                self._last_packets_per_second = self._packet_count_in_window
                self._last_bytes_per_second = self._byte_count_in_window

                self._packet_count_in_window = 0
                self._byte_count_in_window = 0

                self._window_start = now

            self._packet_count_in_window += 1
            self._byte_count_in_window += size

    def get_rates(self) -> Tuple[int, int]:
        """
        Returns:

            (
                packets_per_second,
                bytes_per_second
            )

        reflecting the most recently completed one-second window.
        """

        with self._lock:
            return (
                self._last_packets_per_second,
                self._last_bytes_per_second,
            )


# ---------------------------------------------------------------------------
# Packet capture engine
# ---------------------------------------------------------------------------

class PacketCapture:
    """
    Captures live network traffic using Scapy on a dedicated
    background thread and dispatches parsed packets to registered
    callbacks.

    Usage:

        capture = PacketCapture()

        capture.register_callback(my_handler)

        capture.start(interface=None)

        capture.stop()
    """

    def __init__(self) -> None:
        """
        Initializes the packet capture engine in a stopped state.

        Call start() to begin sniffing.
        """

        self._thread: Optional[threading.Thread] = None

        self._stop_event = threading.Event()

        self._callbacks: List[PacketCallback] = []

        self._interface: Optional[str] = None

        self._is_running: bool = False

        self._total_packets_captured: int = 0

        self.rate_counter = TrafficRateCounter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_callback(self, callback: PacketCallback) -> None:
        """
        Registers a function to be invoked with each successfully
        parsed CapturedPacket.

        Multiple callbacks may be registered.

        Examples:

            - AI prediction
            - database persistence
            - UI updates
            - traffic analysis
            - logging

        Args:
            callback:
                A callable accepting a single CapturedPacket argument.
        """

        self._callbacks.append(callback)

        logger.debug(
            "Registered packet callback: %s",
            getattr(callback, "__qualname__", callback),
        )

    def start(self, interface: Optional[str] = None) -> bool:
        """
        Starts live packet capture on a background thread.

        Args:
            interface:
                Network interface name to sniff on.

                If None or "auto", the first active interface is
                auto-detected.

        Returns:
            True if the capture thread was started successfully.

            False if capture was already running or no interface
            could be resolved.
        """

        if self._is_running:
            logger.warning(
                "Packet capture already running; ignoring start() call."
            )
            return False

        resolved_interface = self._resolve_interface(interface)

        if resolved_interface is None:
            logger.error(
                "No usable network interface found. Capture not started."
            )
            return False

        self._interface = resolved_interface

        self._stop_event.clear()

        self._total_packets_captured = 0

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="PacketCaptureThread",
            daemon=True,
        )

        self._thread.start()

        self._is_running = True

        logger.info(
            "Packet capture started on interface: %s",
            resolved_interface,
        )

        return True

    def stop(self) -> None:
        """
        Signals the capture thread to stop and waits briefly for it
        to exit cleanly.

        Safe to call even if capture is not currently running.
        """

        if not self._is_running:
            return

        logger.info("Stopping packet capture...")

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=3.0)

        self._is_running = False

        logger.info(
            "Packet capture stopped. "
            "Total packets captured this session: %d",
            self._total_packets_captured,
        )

    @property
    def is_running(self) -> bool:
        """
        Returns True if the capture thread is currently active.
        """

        return self._is_running

    @property
    def total_packets_captured(self) -> int:
        """
        Returns the cumulative number of packets captured
        during this session.
        """

        return self._total_packets_captured

    @property
    def active_interface(self) -> Optional[str]:
        """
        Returns the name of the interface currently being sniffed,
        if any.
        """

        return self._interface

    @staticmethod
    def list_available_interfaces() -> List[str]:
        """
        Lists all network interfaces visible to Scapy.

        Used by the Settings page to populate the interface selector.

        Returns:
            A list of interface name strings.

            Empty list on failure.
        """

        try:
            return list(get_if_list())

        except Exception as error:
            logger.error(
                "Failed to list network interfaces: %s",
                error,
            )

            return []

    # ------------------------------------------------------------------
    # Internal capture logic
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_interface(
        interface: Optional[str],
    ) -> Optional[str]:
        """
        Resolves the interface to sniff on.

        If "auto"/None is requested, falls back to:

            1. Scapy's configured default interface.
            2. First interface Scapy can see.

        Args:
            interface:
                Requested interface name, "auto", or None.

        Returns:
            A concrete interface name, or None if none could be resolved.
        """

        if interface and interface.lower() != "auto":
            return interface

        try:
            default_iface = conf.iface

            if default_iface:
                return str(default_iface)

        except Exception as error:
            logger.error(
                "Failed to auto-detect default interface: %s",
                error,
            )

        available = PacketCapture.list_available_interfaces()

        return available[0] if available else None

    def _capture_loop(self) -> None:
        """
        Background thread's main function.

        Runs Scapy's sniff() repeatedly with a short timeout so that
        the stop event can be checked between iterations.

        This allows clean shutdown instead of blocking indefinitely
        inside sniff().
        """

        try:
            while not self._stop_event.is_set():

                sniff(
                    iface=self._interface,
                    prn=self._on_packet_captured,
                    store=False,
                    timeout=NetworkConfig.PACKET_CAPTURE_TIMEOUT,
                    promisc=NetworkConfig.PROMISCUOUS_MODE,
                    stop_filter=lambda _pkt: self._stop_event.is_set(),
                )

        except PermissionError:
            logger.error(
                "Permission denied while capturing packets. "
                "Run CyberSentinel AI as Administrator and ensure "
                "Npcap is installed with WinPcap API-compatible mode "
                "enabled."
            )

            self._is_running = False

        except OSError as error:
            logger.error(
                "Network capture OS error on interface '%s': %s",
                self._interface,
                error,
            )

            self._is_running = False

        except Exception as error:
            logger.error(
                "Unexpected error in packet capture loop: %s",
                error,
            )

            self._is_running = False

    def _on_packet_captured(
        self,
        raw_packet: Packet,
    ) -> None:
        """
        Scapy callback invoked synchronously for every sniffed packet.

        The packet is:

            1. Parsed.
            2. Throughput statistics are updated.
            3. Total packet count is incremented.
            4. Registered callbacks are notified.

        Args:
            raw_packet:
                Raw Scapy packet object.
        """

        parsed = self._parse_packet(raw_packet)

        if parsed is None:
            return

        self._total_packets_captured += 1

        self.rate_counter.record_packet(parsed.size)

        for callback in self._callbacks:

            try:
                callback(parsed)

            except Exception as error:
                logger.error(
                    "Packet callback '%s' raised an error: %s",
                    getattr(callback, "__qualname__", callback),
                    error,
                )

    def _parse_packet(
        self,
        raw_packet: Packet,
    ) -> Optional[CapturedPacket]:
        """
        Extracts structured metadata from a raw Scapy packet.

        Extracted information includes:

            - source IP
            - destination IP
            - source port
            - destination port
            - protocol
            - packet size
            - TCP flags
            - payload preview

        The payload preview contains only the first
        _PAYLOAD_PREVIEW_BYTES bytes of TCP/UDP payload.

        Args:
            raw_packet:
                Raw Scapy packet.

        Returns:
            CapturedPacket instance.

            None if the packet does not contain an IP layer,
            such as ARP packets.
        """

        # --------------------------------------------------------------
        # Ignore packets without an IP layer
        # --------------------------------------------------------------

        if not raw_packet.haslayer(IP):
            return None

        # --------------------------------------------------------------
        # Basic IP metadata
        # --------------------------------------------------------------

        ip_layer = raw_packet[IP]

        src_ip = ip_layer.src
        dst_ip = ip_layer.dst

        size = len(raw_packet)

        # --------------------------------------------------------------
        # Default transport metadata
        # --------------------------------------------------------------

        src_port = 0
        dst_port = 0

        protocol = "OTHER"

        flags = ""

        # --------------------------------------------------------------
        # TCP
        # --------------------------------------------------------------

        if raw_packet.haslayer(TCP):

            tcp_layer = raw_packet[TCP]

            protocol = "TCP"

            src_port = int(tcp_layer.sport)

            dst_port = int(tcp_layer.dport)

            flags = str(tcp_layer.flags)

        # --------------------------------------------------------------
        # UDP
        # --------------------------------------------------------------

        elif raw_packet.haslayer(UDP):

            udp_layer = raw_packet[UDP]

            protocol = "UDP"

            src_port = int(udp_layer.sport)

            dst_port = int(udp_layer.dport)

        # --------------------------------------------------------------
        # ICMP
        # --------------------------------------------------------------

        elif raw_packet.haslayer(ICMP):

            protocol = "ICMP"

        # --------------------------------------------------------------
        # Payload preview
        # --------------------------------------------------------------
        #
        # Only TCP/UDP payload is inspected.
        #
        # We retain at most 64 bytes in memory.
        #
        # This is intentionally NOT persisted in CapturedPacket.to_dict().
        # --------------------------------------------------------------

        payload_preview = b""

        if raw_packet.haslayer(TCP) or raw_packet.haslayer(UDP):

            payload_layer = (
                raw_packet[TCP].payload
                if raw_packet.haslayer(TCP)
                else raw_packet[UDP].payload
            )

            try:
                payload_preview = bytes(payload_layer)[
                    :_PAYLOAD_PREVIEW_BYTES
                ]

            except Exception:
                # Malformed, truncated, or otherwise unusual packets
                # must never crash the capture engine.
                payload_preview = b""

        # --------------------------------------------------------------
        # Build structured packet
        # --------------------------------------------------------------

        return CapturedPacket(
            timestamp=datetime.now(timezone.utc),

            src_ip=src_ip,

            dst_ip=dst_ip,

            src_port=src_port,

            dst_port=dst_port,

            protocol=protocol,

            size=size,

            flags=flags,

            interface=self._interface or "",

            # NEW:
            # Lightweight in-memory payload preview.
            payload_preview=payload_preview,
        )