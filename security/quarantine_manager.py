"""
security/quarantine_manager.py

Acts on FileScanner's malicious verdicts by isolating dangerous files:
moves them into data/quarantine/, renames them to disable accidental
execution, and records enough metadata to restore them later if a
verdict turns out to be a false positive.

Responsibility:
    - Subscribe to FileScanner's scan-result callback and quarantine
      any file whose verdict is_malicious is True.
    - Store quarantined files under data/quarantine/, renamed with a
      neutral extension (.quarantined) so double-clicking one by
      accident does nothing.
    - Maintain a JSON index of every quarantined file's original
      location, hash, and the scan result that triggered isolation.
    - Provide restore_file() and delete_permanently(), so quarantine
      is a reversible holding area, not a silent deletion - matches
      the project's stated "non-destructive by design" requirement.

This file does NOT do:
    - Automatic permanent deletion. Ever. A human decision is always
      required to permanently remove a quarantined file.
    - Its own risk scoring - it only acts on FileScanner's verdict.

Design notes:
    - Singleton pattern, consistent with the rest of the codebase.
    - No PySide6 imports.
    - Self-contained: uses a local JSON index (data/quarantine/index.json),
      not MongoDB, consistent with how file_watcher.py/signature_manager.py
      avoided a database dependency for this feature so far.
    - Quarantine IDs are UUIDs, not filenames, so two files with the
      same name from different folders never collide.
"""

import json
import shutil
import threading
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from security.file_scanner import FileScanResult, get_file_scanner
from utils.logger import get_logger

logger = get_logger(__name__)

_QUARANTINE_DIR = Path("data/quarantine")
_QUARANTINE_INDEX_FILE = _QUARANTINE_DIR / "index.json"
_QUARANTINE_SUFFIX = ".quarantined"


@dataclass
class QuarantineEntry:
    """
    Metadata for one quarantined file, persisted to the JSON index.

    Attributes:
        quarantine_id: Unique identifier (UUID4) for this entry.
        original_path: The file's absolute path before quarantine.
        quarantined_path: Where the isolated file currently lives.
        sha256: The file's hash at time of quarantine.
        risk_score: The FileScanner risk score that triggered this.
        severity: The FileRiskSeverity value (as a string) at time of quarantine.
        reasons: The scan reasons that triggered quarantine.
        quarantined_at: ISO timestamp of when this happened.
        restored: True if this file has since been restored to its
            original location (kept in the index as history rather
            than deleted, so the record isn't lost).
    """

    quarantine_id: str
    original_path: str
    quarantined_path: str
    sha256: str
    risk_score: int
    severity: str
    reasons: List[str]
    quarantined_at: str
    restored: bool = False


class QuarantineManager:
    """
    Singleton manager that isolates malicious files flagged by
    FileScanner and keeps a restorable record of every quarantine
    action.
    """

    _instance: Optional["QuarantineManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "QuarantineManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._state_lock = threading.RLock()
        self._entries: Dict[str, QuarantineEntry] = {}

        _QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
        self._load_index()

        logger.info(f"QuarantineManager initialized ({len(self._entries)} existing entries).")

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def connect_to_scanner(self) -> None:
        """
        Registers this manager as FileScanner's scan-result callback,
        so every malicious verdict is quarantined automatically. Call
        this once during application startup, alongside FileScanner.start().
        """
        get_file_scanner().register_scan_result_callback(self.handle_scan_result)
        logger.info("QuarantineManager connected to FileScanner.")

    def handle_scan_result(self, result: FileScanResult) -> None:
        """
        FileScanner's scan-result callback. Quarantines the file if
        its verdict is malicious (HIGH or CRITICAL severity); ignores
        everything else, including scan errors (a file that could not
        be read has nothing here to quarantine).

        Args:
            result: The completed FileScanResult from FileScanner.
        """
        if result.scan_error or not result.is_malicious:
            return
        self.quarantine_file(result)

    # ------------------------------------------------------------------
    # Quarantine actions
    # ------------------------------------------------------------------

    def quarantine_file(self, result: FileScanResult) -> Optional[QuarantineEntry]:
        """
        Moves a malicious file into data/quarantine/, renaming it with
        a neutral .quarantined extension, and records an index entry.

        Args:
            result: The FileScanResult that triggered quarantine.

        Returns:
            The created QuarantineEntry, or None if the move failed
            (e.g. the file was already removed by another process).
        """
        original_path = Path(result.file_path)
        if not original_path.exists():
            logger.warning(f"Cannot quarantine '{original_path}' - it no longer exists.")
            return None

        quarantine_id = str(uuid.uuid4())
        destination = _QUARANTINE_DIR / f"{quarantine_id}{_QUARANTINE_SUFFIX}"

        try:
            shutil.move(str(original_path), str(destination))
        except OSError as move_error:
            logger.error(f"Failed to quarantine '{original_path}': {move_error}")
            return None

        entry = QuarantineEntry(
            quarantine_id=quarantine_id,
            original_path=str(original_path),
            quarantined_path=str(destination),
            sha256=result.sha256,
            risk_score=result.risk_score,
            severity=result.severity.value,
            reasons=result.reasons,
            quarantined_at=datetime.now(timezone.utc).isoformat(),
        )

        with self._state_lock:
            self._entries[quarantine_id] = entry
            self._persist_index()

        logger.error(
            f"QUARANTINED: '{original_path}' -> '{destination}' "
            f"(severity={entry.severity}, score={entry.risk_score})"
        )
        return entry

    def restore_file(self, quarantine_id: str) -> bool:
        """
        Moves a quarantined file back to its original location and
        marks the index entry as restored. Fails safely if the
        original location's parent folder no longer exists, or a file
        already occupies that path.

        Args:
            quarantine_id: The entry to restore.

        Returns:
            True if the file was restored, False otherwise.
        """
        with self._state_lock:
            entry = self._entries.get(quarantine_id)
            if entry is None:
                logger.warning(f"No quarantine entry found for id '{quarantine_id}'.")
                return False
            if entry.restored:
                logger.warning(f"Entry '{quarantine_id}' was already restored.")
                return False

            quarantined_path = Path(entry.quarantined_path)
            original_path = Path(entry.original_path)

            if not quarantined_path.exists():
                logger.error(f"Quarantined file missing on disk: '{quarantined_path}'.")
                return False
            if original_path.exists():
                logger.error(f"Cannot restore - a file already exists at '{original_path}'.")
                return False
            if not original_path.parent.exists():
                logger.error(f"Cannot restore - original folder no longer exists: '{original_path.parent}'.")
                return False

            try:
                shutil.move(str(quarantined_path), str(original_path))
            except OSError as move_error:
                logger.error(f"Failed to restore '{quarantine_id}': {move_error}")
                return False

            entry.restored = True
            self._persist_index()

        logger.info(f"Restored '{original_path}' from quarantine.")
        return True

    def delete_permanently(self, quarantine_id: str) -> bool:
        """
        Permanently deletes a quarantined file from disk and removes
        its index entry. This is the ONLY path that actually destroys
        data in this module - it always requires an explicit call,
        never happens automatically.

        Args:
            quarantine_id: The entry to delete.

        Returns:
            True if deleted, False if the entry/file was not found.
        """
        with self._state_lock:
            entry = self._entries.get(quarantine_id)
            if entry is None:
                logger.warning(f"No quarantine entry found for id '{quarantine_id}'.")
                return False

            quarantined_path = Path(entry.quarantined_path)
            if quarantined_path.exists():
                try:
                    quarantined_path.unlink()
                except OSError as delete_error:
                    logger.error(f"Failed to delete '{quarantine_id}': {delete_error}")
                    return False

            del self._entries[quarantine_id]
            self._persist_index()

        logger.info(f"Permanently deleted quarantine entry '{quarantine_id}'.")
        return True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_entries(self, include_restored: bool = False) -> List[QuarantineEntry]:
        """
        Returns all quarantine entries, most recent first.

        Args:
            include_restored: If False (default), already-restored
                entries are excluded - a future UI's default view is
                "what's currently isolated", not full history.

        Returns:
            A list of QuarantineEntry objects.
        """
        with self._state_lock:
            entries = list(self._entries.values())

        if not include_restored:
            entries = [entry for entry in entries if not entry.restored]

        return sorted(entries, key=lambda entry: entry.quarantined_at, reverse=True)

    def get_entry(self, quarantine_id: str) -> Optional[QuarantineEntry]:
        """Returns a single entry by id, or None if not found."""
        with self._state_lock:
            return self._entries.get(quarantine_id)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_index(self) -> None:
        """Loads the quarantine index from disk, tolerating a missing or corrupt file."""
        if not _QUARANTINE_INDEX_FILE.exists():
            return
        try:
            raw = json.loads(_QUARANTINE_INDEX_FILE.read_text())
            self._entries = {
                quarantine_id: QuarantineEntry(**data) for quarantine_id, data in raw.items()
            }
        except (json.JSONDecodeError, TypeError) as parse_error:
            logger.error(f"Corrupt quarantine index ({parse_error}) - starting empty.")
            self._entries = {}

    def _persist_index(self) -> None:
        """Writes the current index to disk as JSON. Caller must hold _state_lock."""
        payload = {qid: asdict(entry) for qid, entry in self._entries.items()}
        _QUARANTINE_INDEX_FILE.write_text(json.dumps(payload, indent=2))


def get_quarantine_manager() -> QuarantineManager:
    """Module-level accessor returning the singleton QuarantineManager instance."""
    return QuarantineManager()


if __name__ == "__main__":
    # Standalone smoke test - manufactures a fake "malicious" file,
    # quarantines it, restores it, then quarantines and permanently
    # deletes it. Doesn't touch the real Downloads/Desktop watcher.
    import tempfile

    from security.file_scanner import FileRiskSeverity

    manager = get_quarantine_manager()

    with tempfile.TemporaryDirectory() as tmp_dir:
        fake_malicious = Path(tmp_dir) / "definitely_not_a_virus.exe"
        fake_malicious.write_bytes(b"pretend malicious content")

        fake_result = FileScanResult(
            file_path=str(fake_malicious),
            sha256="deadbeef" * 8,
            size_bytes=fake_malicious.stat().st_size,
            risk_score=95,
            severity=FileRiskSeverity.CRITICAL,
            reasons=["Simulated detection for smoke test"],
        )

        print(f"Quarantining {fake_malicious.name}...")
        entry = manager.quarantine_file(fake_result)
        if entry is None:
            print("  FAILED to quarantine.")
        else:
            print(f"  Quarantine ID: {entry.quarantine_id}")
            print(f"  Original file still on disk: {fake_malicious.exists()} (expect False)")
            print(f"  Quarantined file exists: {Path(entry.quarantined_path).exists()} (expect True)")

            print("\nActive quarantine entries:")
            for active in manager.list_entries():
                print(f"  {active.quarantine_id[:8]}... - {active.original_path} ({active.severity})")

            print(f"\nRestoring {entry.quarantine_id[:8]}...")
            restored = manager.restore_file(entry.quarantine_id)
            print(f"  Restored: {restored}")
            print(f"  File back at original location: {fake_malicious.exists()} (expect True)")

            print(f"\nActive entries after restore: {len(manager.list_entries())} (expect 0)")

            print("\nQuarantining again, then permanently deleting...")
            entry2 = manager.quarantine_file(fake_result)
            if entry2:
                deleted = manager.delete_permanently(entry2.quarantine_id)
                print(f"  Permanently deleted: {deleted}")
                print(f"  Entry still in index: {manager.get_entry(entry2.quarantine_id) is not None} (expect False)")