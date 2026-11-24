"""
security/file_watcher.py

Real-time file-system watcher for CyberSentinel AI's file-scanning
feature (Phase A: catch a file the moment it lands on disk, before any
content scanning happens).

Responsibility:
    - Monitor a configurable set of folders (defaults: the user's
      Downloads and Desktop) for newly created or modified files, using
      the `watchdog` library.
    - Wait for a file to finish being written before reporting it -
      a file mid-download has a growing size and a locked handle;
      scanning it too early would scan a truncated file or fail to
      open it at all.
    - Report ready-to-scan files via a plain callback, NOT a Qt signal,
      keeping this module PySide6-free like the rest of security/.
      Whatever orchestrates the full scan (file_scanner.py, wired up
      later into MonitoringEngine) subscribes to this callback and is
      responsible for bridging results into Qt signals for the UI.

Design notes:
    - Singleton pattern, consistent with the rest of the codebase.
    - No PySide6 imports.
    - Self-contained: does NOT depend on config/settings.py, so it
      works immediately without needing that file pasted first.
      Default watched folders are resolved directly via pathlib from
      the current user's home directory. add_watch_path() lets a
      later Settings page add more folders (e.g. a USB drive mount)
      without changing this file.
    - "File finished writing" is detected by polling the file's size
      twice, STABILITY_CHECK_INTERVAL_SECONDS apart, and only firing
      the callback once two consecutive reads match. This is a
      pragmatic approach that works across browsers/download managers
      without needing OS-specific "download complete" hooks.
    - Runs watchdog's Observer on its own background thread (owned by
      watchdog itself); the stability-check delay is handled via a
      short-lived helper thread per pending file so the watcher's own
      event-handling thread is never blocked.
"""

import threading
import time
from pathlib import Path
from typing import Callable, List, Optional, Set

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from utils.logger import get_logger

logger = get_logger(__name__)

# How long to wait between size checks when confirming a file has
# finished being written. Two consecutive matching reads this far
# apart are treated as "done".
_STABILITY_CHECK_INTERVAL_SECONDS: float = 1.0

# Maximum time to keep polling a single file's size before giving up
# and reporting it anyway (handles the rare case of a file that never
# stabilizes, e.g. a continuously-appended log file).
_STABILITY_CHECK_TIMEOUT_SECONDS: float = 30.0

# File extensions ignored outright - temp/partial download markers
# that browsers rename once the real file lands, so scanning them is
# pointless and they'd otherwise trigger duplicate scans.
_IGNORED_EXTENSIONS: Set[str] = {".crdownload", ".part", ".tmp", ".download"}


class _WatchedFolderEventHandler(FileSystemEventHandler):
    """
    Internal watchdog event handler. Forwards every created/modified
    file path to FileWatcher for stability-checking, filtering out
    directories and known partial-download extensions.
    """

    def __init__(self, on_candidate: Callable[[Path], None]) -> None:
        """
        Args:
            on_candidate: Callback invoked with a candidate file Path
                whenever a relevant filesystem event occurs.
        """
        super().__init__()
        self._on_candidate = on_candidate

    def on_created(self, event: FileSystemEvent) -> None:
        """Handles a new file/directory appearing in a watched folder."""
        self._forward_if_file(event)

    def on_modified(self, event: FileSystemEvent) -> None:
        """
        Handles a file being modified. Needed because many browsers
        create the destination file immediately (on_created) and only
        finish writing bytes into it afterward - the modification
        events are what let us detect "still being written".
        """
        self._forward_if_file(event)

    def _forward_if_file(self, event: FileSystemEvent) -> None:
        """Filters an event down to real, non-ignored files before forwarding."""
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() in _IGNORED_EXTENSIONS:
            return
        self._on_candidate(path)


class FileWatcher:
    """
    Singleton real-time file-system watcher.

    Monitors a set of folders for new files, waits for each one to
    finish being written, then reports it as ready-to-scan via a
    registered callback. Does not scan file content itself - that is
    file_scanner.py's job, kept separate so this class has exactly one
    responsibility (noticing files), matching the codebase's one-file-
    one-responsibility convention.
    """

    _instance: Optional["FileWatcher"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "FileWatcher":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._watch_paths: Set[Path] = self._default_watch_paths()
        self._observer: Optional[Observer] = None
        self._is_running: bool = False

        self._on_file_ready: Optional[Callable[[Path], None]] = None
        self._pending_files: Set[Path] = set()
        self._pending_lock = threading.Lock()

        logger.info(f"FileWatcher initialized with default paths: {self._watch_paths}")

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _default_watch_paths() -> Set[Path]:
        """
        Resolves the default set of folders to watch: the current
        user's Downloads and Desktop directories. Silently skips any
        that don't exist (e.g. a minimal/non-standard Windows setup)
        rather than failing.

        Returns:
            A set of existing, resolved directory paths to watch.
        """
        home = Path.home()
        candidates = [home / "Downloads", home / "Desktop"]
        existing = {path for path in candidates if path.is_dir()}
        if not existing:
            logger.warning("Neither Downloads nor Desktop found under the user's home directory.")
        return existing

    def add_watch_path(self, path: str) -> bool:
        """
        Adds an additional folder to watch (e.g. a USB drive mount
        point, or a custom folder from the Settings page). Takes
        effect immediately if the watcher is already running.

        Args:
            path: Filesystem path to a directory.

        Returns:
            True if the path exists and was added (or already present),
            False if the path does not exist or is not a directory.
        """
        folder = Path(path)
        if not folder.is_dir():
            logger.warning(f"Cannot watch '{path}' - not a valid directory.")
            return False

        if folder in self._watch_paths:
            return True

        self._watch_paths.add(folder)
        if self._is_running and self._observer is not None:
            self._observer.schedule(
                _WatchedFolderEventHandler(self._on_event_candidate), str(folder), recursive=False
            )
            logger.info(f"Now watching additional folder: {folder}")
        return True

    def get_watch_paths(self) -> List[str]:
        """Returns the currently configured watch folders as strings, for display in Settings."""
        return [str(path) for path in self._watch_paths]

    def set_file_ready_callback(self, callback: Callable[[Path], None]) -> None:
        """
        Registers the callback invoked whenever a file has finished
        being written and is ready for content scanning.

        Args:
            callback: Function accepting the completed file's Path.
        """
        self._on_file_ready = callback

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        Starts watching all configured folders in a background thread
        owned by watchdog's Observer.

        Returns:
            True if watching started (or was already running), False
            if there are no valid folders to watch.
        """
        if self._is_running:
            logger.info("FileWatcher already running.")
            return True

        if not self._watch_paths:
            logger.error("No valid folders to watch - FileWatcher not started.")
            return False

        self._observer = Observer()
        handler = _WatchedFolderEventHandler(self._on_event_candidate)
        for folder in self._watch_paths:
            self._observer.schedule(handler, str(folder), recursive=False)

        self._observer.start()
        self._is_running = True
        logger.info(f"FileWatcher started, monitoring: {[str(p) for p in self._watch_paths]}")
        return True

    def stop(self) -> None:
        """Stops the observer and waits for its thread to exit cleanly."""
        if not self._is_running or self._observer is None:
            return

        self._observer.stop()
        self._observer.join(timeout=5.0)
        self._is_running = False
        logger.info("FileWatcher stopped.")

    def is_running(self) -> bool:
        """Returns True if the watcher is currently active."""
        return self._is_running

    # ------------------------------------------------------------------
    # Stability checking
    # ------------------------------------------------------------------

    def _on_event_candidate(self, path: Path) -> None:
        """
        Handles a raw filesystem event candidate: deduplicates against
        files already being tracked, then spawns a short-lived thread
        to confirm the file has finished being written before firing
        the ready callback.

        Args:
            path: The candidate file path from a watchdog event.
        """
        with self._pending_lock:
            if path in self._pending_files:
                return  # already being stability-checked
            self._pending_files.add(path)

        threading.Thread(
            target=self._wait_for_stable_and_report,
            args=(path,),
            name=f"FileStabilityCheck-{path.name}",
            daemon=True,
        ).start()

    def _wait_for_stable_and_report(self, path: Path) -> None:
        """
        Polls a file's size until two consecutive reads match (the
        file has stopped growing) or the timeout is reached, then
        invokes the ready callback exactly once.

        Args:
            path: The file being watched for write-completion.
        """
        deadline = time.monotonic() + _STABILITY_CHECK_TIMEOUT_SECONDS
        last_size: Optional[int] = None

        try:
            while time.monotonic() < deadline:
                if not path.exists():
                    logger.debug(f"'{path}' disappeared before stabilizing - abandoning.")
                    return

                try:
                    current_size = path.stat().st_size
                except OSError:
                    # File briefly locked/inaccessible mid-write; retry on the next tick.
                    time.sleep(_STABILITY_CHECK_INTERVAL_SECONDS)
                    continue

                if last_size is not None and current_size == last_size:
                    self._report_ready(path)
                    return

                last_size = current_size
                time.sleep(_STABILITY_CHECK_INTERVAL_SECONDS)

            logger.warning(f"'{path}' did not stabilize within timeout - reporting anyway.")
            self._report_ready(path)

        finally:
            with self._pending_lock:
                self._pending_files.discard(path)

    def _report_ready(self, path: Path) -> None:
        """Invokes the registered ready-callback, guarding against a missing callback or its errors."""
        if self._on_file_ready is None:
            logger.warning(f"'{path}' is ready to scan, but no callback is registered.")
            return
        try:
            self._on_file_ready(path)
        except Exception as exc:
            logger.error(f"file_ready callback failed for '{path}': {exc}")


def get_file_watcher() -> FileWatcher:
    """Module-level accessor returning the singleton FileWatcher instance."""
    return FileWatcher()


if __name__ == "__main__":
    # Standalone smoke test - starts watching the real Downloads/Desktop
    # folders and prints every file reported as ready. Copy or download
    # a file into either folder while this is running to see it fire.
    import sys

    watcher = get_file_watcher()
    print(f"Watching: {watcher.get_watch_paths()}")

    def handle_ready(path: Path) -> None:
        print(f"READY TO SCAN: {path} ({path.stat().st_size} bytes)")

    watcher.set_file_ready_callback(handle_ready)

    if not watcher.start():
        print("Failed to start - no valid folders found.")
        sys.exit(1)

    print("Watching for new files. Drop a file into Downloads or Desktop. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
        print("Stopped.")