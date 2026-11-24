"""
network/network_monitor.py

System and network resource monitoring for CyberSentinel AI, built on psutil.

Responsibilities:
    - Collect live system resource statistics: CPU, RAM, Disk, and
      network bandwidth (upload/download rates).
    - Collect static device/host information: hostname, IP address,
      OS details, architecture.
    - Run periodic polling on a dedicated background thread and
      dispatch snapshots to registered callbacks, mirroring the
      threading + callback pattern established in
      network/packet_capture.py, so app/core/engine.py can bridge
      both subsystems into Qt signals the same way.

This module is intentionally free of any GUI framework imports.
"""

from __future__ import annotations

import platform
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, List, Optional

import psutil

from config.settings import UIConfig
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DeviceInfo:
    """
    Static (rarely-changing) information about the host machine,
    displayed in the dashboard's "Device Information" panel.
    """
    hostname: str
    ip_address: str
    os_name: str
    os_version: str
    architecture: str
    processor: str

    def to_dict(self) -> dict:
        """Returns a plain-dict representation for logging/export."""
        return {
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "architecture": self.architecture,
            "processor": self.processor,
        }


@dataclass
class SystemStats:
    """
    A single point-in-time snapshot of system resource usage,
    matching the fields expected by
    DatabaseManager.insert_system_stat() (see database/db_manager.py).
    """
    timestamp: datetime
    cpu_percent: float
    ram_percent: float
    ram_used_gb: float
    ram_total_gb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    bytes_sent_rate: float   # bytes/sec, upload
    bytes_recv_rate: float   # bytes/sec, download

    def to_dict(self) -> dict:
        """
        Converts this snapshot into a plain dictionary suitable for
        MongoDB insertion via DatabaseManager.insert_system_stat().
        """
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "ram_percent": self.ram_percent,
            "ram_used_gb": self.ram_used_gb,
            "ram_total_gb": self.ram_total_gb,
            "disk_percent": self.disk_percent,
            "disk_used_gb": self.disk_used_gb,
            "disk_total_gb": self.disk_total_gb,
            "bytes_sent_rate": self.bytes_sent_rate,
            "bytes_recv_rate": self.bytes_recv_rate,
        }


# Type alias for functions that consume a system stats snapshot.
SystemStatsCallback = Callable[[SystemStats], None]


class BandwidthRateCalculator:
    """
    Computes live network upload/download rates (bytes/sec) from
    psutil's cumulative counters by tracking the delta between
    consecutive polls.

    Intended to be updated exclusively from the NetworkMonitor's
    single polling thread, so no internal locking is required.
    """

    def __init__(self) -> None:
        self._last_sent: Optional[int] = None
        self._last_recv: Optional[int] = None
        self._last_time: Optional[float] = None

    def compute(self) -> tuple[float, float]:
        """
        Reads the current cumulative network counters and computes
        the upload/download rate since the previous call.

        Returns:
            A tuple of (bytes_sent_rate, bytes_recv_rate) in bytes/sec.
            Returns (0.0, 0.0) on the first call, since no delta exists yet.
        """
        counters = psutil.net_io_counters()
        now = time.monotonic()

        if self._last_time is None:
            self._last_sent = counters.bytes_sent
            self._last_recv = counters.bytes_recv
            self._last_time = now
            return 0.0, 0.0

        elapsed = max(now - self._last_time, 1e-6)  # avoid division by zero
        sent_rate = (counters.bytes_sent - self._last_sent) / elapsed
        recv_rate = (counters.bytes_recv - self._last_recv) / elapsed

        self._last_sent = counters.bytes_sent
        self._last_recv = counters.bytes_recv
        self._last_time = now

        return max(sent_rate, 0.0), max(recv_rate, 0.0)


class NetworkMonitor:
    """
    Polls system resource and network usage statistics on a dedicated
    background thread and dispatches snapshots to registered callbacks.

    Usage:
        monitor = NetworkMonitor()
        monitor.register_callback(my_handler)
        monitor.start()
        ...
        monitor.stop()
    """

    def __init__(
        self,
        poll_interval_seconds: float = UIConfig.DASHBOARD_REFRESH_INTERVAL_MS / 1000,
    ) -> None:
        """
        Args:
            poll_interval_seconds: How often to poll system stats, in
                seconds. Defaults to UIConfig.DASHBOARD_REFRESH_INTERVAL_MS
                so the backend polling cadence matches the UI's refresh rate.
        """
        self._poll_interval = poll_interval_seconds
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._callbacks: List[SystemStatsCallback] = []
        self._is_running: bool = False
        self._bandwidth_calculator = BandwidthRateCalculator()
        self._device_info: Optional[DeviceInfo] = None
        self._latest_stats: Optional[SystemStats] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_callback(self, callback: SystemStatsCallback) -> None:
        """
        Registers a function to be invoked with each new SystemStats
        snapshot as it's polled. Multiple callbacks may be registered
        (e.g., dashboard stat cards, database persistence, live graphs).

        Args:
            callback: A callable accepting a single SystemStats argument.
        """
        self._callbacks.append(callback)
        logger.debug(
            "Registered system stats callback: %s",
            getattr(callback, "__qualname__", callback),
        )

    def start(self) -> None:
        """
        Starts the background polling thread. Safe to call once;
        subsequent calls while already running are ignored.
        """
        if self._is_running:
            logger.warning("NetworkMonitor already running; ignoring start() call.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop,
            name="NetworkMonitorThread",
            daemon=True,
        )
        self._thread.start()
        self._is_running = True
        logger.info("NetworkMonitor started (poll interval: %.1fs).", self._poll_interval)

    def stop(self) -> None:
        """
        Signals the polling thread to stop and waits briefly for it
        to exit cleanly. Safe to call even if not currently running.
        """
        if not self._is_running:
            return

        logger.info("Stopping NetworkMonitor...")
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=3.0)

        self._is_running = False
        logger.info("NetworkMonitor stopped.")

    @property
    def is_running(self) -> bool:
        """Returns True if the polling thread is currently active."""
        return self._is_running

    @property
    def latest_stats(self) -> Optional[SystemStats]:
        """Returns the most recently polled SystemStats snapshot, if any."""
        return self._latest_stats

    def get_device_info(self) -> DeviceInfo:
        """
        Returns static device/host information (hostname, IP, OS
        details). Computed once and cached, since this data does not
        change during a normal application session.

        Returns:
            A DeviceInfo instance describing the local machine.
        """
        if self._device_info is None:
            self._device_info = self._collect_device_info()
        return self._device_info

    def get_current_stats(self) -> SystemStats:
        """
        Synchronously collects and returns a single SystemStats
        snapshot on demand, without requiring the background polling
        thread to be running. Useful for one-off reads (e.g.,
        populating the dashboard immediately on startup before the
        first poll cycle completes).

        Returns:
            A freshly collected SystemStats snapshot.
        """
        return self._collect_system_stats()

    # ------------------------------------------------------------------
    # Internal polling logic
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        """
        The background thread's main function. Repeatedly collects a
        SystemStats snapshot, stores it, and dispatches it to all
        registered callbacks, sleeping between iterations while
        remaining responsive to stop() requests.
        """
        try:
            while not self._stop_event.is_set():
                stats = self._collect_system_stats()
                self._latest_stats = stats

                for callback in self._callbacks:
                    try:
                        callback(stats)
                    except Exception as error:
                        logger.error(
                            "System stats callback '%s' raised an error: %s",
                            getattr(callback, "__qualname__", callback),
                            error,
                        )

                self._stop_event.wait(timeout=self._poll_interval)
        except Exception as error:  # broad: psutil can raise varied platform errors
            logger.error("Unexpected error in NetworkMonitor poll loop: %s", error)
            self._is_running = False

    def _collect_system_stats(self) -> SystemStats:
        """
        Gathers a fresh snapshot of CPU, RAM, Disk, and bandwidth usage
        using psutil.

        Returns:
            A populated SystemStats instance.
        """
        cpu_percent = psutil.cpu_percent(interval=None)

        virtual_memory = psutil.virtual_memory()
        ram_percent = virtual_memory.percent
        ram_used_gb = virtual_memory.used / (1024 ** 3)
        ram_total_gb = virtual_memory.total / (1024 ** 3)

        disk_usage = psutil.disk_usage(self._get_system_drive())
        disk_percent = disk_usage.percent
        disk_used_gb = disk_usage.used / (1024 ** 3)
        disk_total_gb = disk_usage.total / (1024 ** 3)

        bytes_sent_rate, bytes_recv_rate = self._bandwidth_calculator.compute()

        return SystemStats(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=round(cpu_percent, 1),
            ram_percent=round(ram_percent, 1),
            ram_used_gb=round(ram_used_gb, 2),
            ram_total_gb=round(ram_total_gb, 2),
            disk_percent=round(disk_percent, 1),
            disk_used_gb=round(disk_used_gb, 2),
            disk_total_gb=round(disk_total_gb, 2),
            bytes_sent_rate=round(bytes_sent_rate, 2),
            bytes_recv_rate=round(bytes_recv_rate, 2),
        )

    @staticmethod
    def _get_system_drive() -> str:
        """
        Returns the appropriate root path for psutil.disk_usage() on
        the current platform (e.g., "C:\\" on Windows, "/" elsewhere).
        This is primarily a Windows desktop application, but this
        guard prevents a hard crash if run on another OS during
        development.

        Returns:
            The root path string to pass to psutil.disk_usage().
        """
        return "C:\\" if platform.system() == "Windows" else "/"

    @staticmethod
    def _collect_device_info() -> DeviceInfo:
        """
        Gathers static host/device information using the socket and
        platform standard library modules.

        Returns:
            A populated DeviceInfo instance.
        """
        hostname = socket.gethostname()
        ip_address = NetworkMonitor._resolve_local_ip()

        return DeviceInfo(
            hostname=hostname,
            ip_address=ip_address,
            os_name=platform.system(),
            os_version=platform.version(),
            architecture=platform.machine(),
            processor=platform.processor() or "Unknown",
        )

    @staticmethod
    def _resolve_local_ip() -> str:
        """
        Determines the machine's primary local IP address by opening
        a throwaway UDP socket to a public address (no data is
        actually sent). This is more reliable than
        socket.gethostbyname(hostname), which often incorrectly
        returns "127.0.0.1" on Windows.

        Returns:
            The resolved local IP address, or "127.0.0.1" as a
            last-resort fallback if resolution fails entirely
            (e.g., no network adapters are active).
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as temp_socket:
                temp_socket.connect(("8.8.8.8", 80))
                return temp_socket.getsockname()[0]
        except OSError as error:
            logger.warning("Could not resolve local IP address: %s", error)
            return "127.0.0.1"