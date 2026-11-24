"""
security/file_scanner.py

Orchestrator for CyberSentinel AI's file-scanning feature (Phase A):
combines FileWatcher's "a file is ready" notice with SignatureManager's
hash/YARA results and a local extension/entropy heuristic layer into
one risk verdict per file.

Responsibility:
    - Register as FileWatcher's ready-callback, so every file that
      finishes writing in a watched folder gets scanned automatically.
    - Run SignatureManager.scan() (known-hash lookup + YARA matching).
    - Add heuristic checks this file owns directly: suspicious file
      extensions (.exe, .ps1, etc.), the "double extension" disguise
      trick (invoice.pdf.exe), and Shannon entropy (packed/obfuscated
      content tends to look statistically random).
    - Combine every signal into a single 0-100 risk score and severity
      band, and report the result via a registered callback.
    - Maintain a small in-memory history of recent scans for later UI
      use (mirrors ThreatDetector.get_recent_detections()' pattern).

Responsibility this file does NOT have:
    - Acting on a malicious verdict (deleting/moving the file) - that
      is quarantine_manager.py's job, kept separate on purpose.
    - ML-based classification - that's Phase B (file_predictor.py),
      not built yet.

Design notes:
    - Singleton pattern, consistent with the rest of the codebase.
    - No PySide6 imports.
    - Deliberately does NOT import security.risk_analyzer's severity
      enum, since that file hasn't been shared in this conversation
      yet and guessing at its exact member names risks another
      import mismatch (as happened with signature_manager.py). This
      module defines its own small FileRiskSeverity enum instead;
      unifying the two is a trivial follow-up once risk_analyzer.py
      is available.
    - Entropy is only checked for file types where high entropy is
      unusual. Already-compressed formats (zip, jpg, mp4, docx, etc.
      - docx/xlsx/pptx are themselves zip containers) are naturally
      high-entropy without being suspicious, so they're excluded to
      avoid false positives.
"""

import math
import threading
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Deque, List, Optional

from security.file_watcher import get_file_watcher
from security.signature_manager import get_signature_manager
from utils.logger import get_logger

logger = get_logger(__name__)

# Extensions treated as executable/script-capable - the final segment
# of a filename matching one of these is inherently higher risk.
_SUSPICIOUS_EXTENSIONS = {
    ".exe", ".scr", ".bat", ".cmd", ".ps1", ".vbs", ".js",
    ".jar", ".msi", ".com", ".pif", ".hta", ".wsf",
}

# Extensions commonly used as the "disguise" in a double-extension
# trick, e.g. "invoice.pdf.exe" - the .pdf here is meant to look safe.
_TRUSTED_LOOKING_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".txt", ".jpg", ".jpeg", ".png", ".gif", ".mp3", ".mp4", ".zip",
}

# Formats that are naturally high-entropy (already compressed) - entropy
# checks are skipped for these to avoid flagging ordinary photos/zips.
_KNOWN_COMPRESSED_EXTENSIONS = {
    ".zip", ".7z", ".rar", ".gz", ".jpg", ".jpeg", ".png", ".gif",
    ".mp3", ".mp4", ".docx", ".xlsx", ".pptx", ".pdf",
}

_HIGH_ENTROPY_THRESHOLD: float = 7.2  # out of a max of 8.0 bits/byte
_MAX_ENTROPY_READ_BYTES: int = 2_000_000  # cap the read for large files

_YARA_SCORE_PER_MATCH: int = 40
_YARA_SCORE_CAP: int = 80
_DOUBLE_EXTENSION_SCORE: int = 30
_SUSPICIOUS_EXTENSION_SCORE: int = 20
_HIGH_ENTROPY_SCORE: int = 15

_SCAN_HISTORY_SIZE: int = 200


class FileRiskSeverity(Enum):
    """Severity band for a file scan verdict, ordered from safest to most dangerous."""

    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class FileScanResult:
    """
    The complete verdict for one scanned file.

    Attributes:
        file_path: Absolute path to the scanned file.
        sha256: The file's hash, empty if it could not be read.
        size_bytes: File size at scan time, 0 if unreadable.
        risk_score: Combined 0-100 risk score from every signal.
        severity: The FileRiskSeverity band risk_score falls into.
        reasons: Human-readable explanations for each signal that
            contributed to the score, in the order they were checked.
        scan_error: Set if the file could not be read at all (e.g.
            removed or quarantined by another security product between
            being noticed and being scanned). When set, sha256/hash
            and YARA/entropy checks were skipped; risk_score reflects
            extension-only heuristics.
    """

    file_path: str
    sha256: str
    size_bytes: int
    risk_score: int
    severity: FileRiskSeverity
    reasons: List[str] = field(default_factory=list)
    scan_error: Optional[str] = None

    @property
    def is_malicious(self) -> bool:
        """True if the verdict is severe enough to warrant quarantine (HIGH or CRITICAL)."""
        return self.severity in (FileRiskSeverity.HIGH, FileRiskSeverity.CRITICAL)


class FileScanner:
    """
    Singleton orchestrator combining FileWatcher, SignatureManager, and
    local extension/entropy heuristics into one risk verdict per file.
    """

    _instance: Optional["FileScanner"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "FileScanner":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True

        self._file_watcher = get_file_watcher()
        self._signature_manager = get_signature_manager()

        self._history_lock = threading.Lock()
        self._recent_scans: Deque[FileScanResult] = deque(maxlen=_SCAN_HISTORY_SIZE)

        self._scan_result_callbacks: List[Callable[[FileScanResult], None]] = []

        logger.info("FileScanner initialized.")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        Registers this scanner as FileWatcher's ready-callback and
        starts watching. Every file that finishes writing in a watched
        folder will be scanned automatically from this point on.

        Returns:
            True if watching started successfully, False otherwise.
        """
        self._file_watcher.set_file_ready_callback(self._handle_file_ready)
        return self._file_watcher.start()

    def stop(self) -> None:
        """Stops the underlying FileWatcher."""
        self._file_watcher.stop()

    def register_scan_result_callback(self, callback: Callable[[FileScanResult], None]) -> None:
        """
        Registers a callback invoked after every completed scan
        (whether triggered by the watcher or a manual scan_file()
        call). Multiple callbacks can be registered — e.g.
        quarantine_manager.py (acts on malicious verdicts) and
        MonitoringEngine (bridges every result into a Qt signal for
        the UI) both subscribe independently, matching the multi-
        callback pattern ThreatDetector.register_callback() already
        uses elsewhere in the pipeline.

        Args:
            callback: Function accepting the completed FileScanResult.
        """
        self._scan_result_callbacks.append(callback)

    def _handle_file_ready(self, file_path: Path) -> None:
        """Handles FileWatcher's ready-callback by scanning the file."""
        self.scan_file(file_path)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan_file(self, file_path: Path) -> FileScanResult:
        """
        Runs the full scan pipeline on a single file and returns a
        verdict: extension heuristics, signature (hash + YARA) checks,
        and entropy analysis, combined into one risk score.

        Safe to call directly (not just via the watcher) - a future
        "scan now" UI action or quarantine_manager can use this too.

        Args:
            file_path: Path to the file to scan.

        Returns:
            The completed FileScanResult.
        """
        reasons: List[str] = []
        score = 0

        ext_score, ext_reasons = self._check_extension(file_path)
        score += ext_score
        reasons.extend(ext_reasons)

        sig_result = self._signature_manager.scan(file_path)

        if sig_result.error:
            result = FileScanResult(
                file_path=str(file_path),
                sha256="",
                size_bytes=self._safe_size(file_path),
                risk_score=min(score, 100),
                severity=self._severity_for_score(min(score, 100)),
                reasons=reasons,
                scan_error=sig_result.error,
            )
            self._finalize(result)
            return result

        if sig_result.hash_match_label:
            score = 100
            reasons.append(f"Known malicious hash: {sig_result.hash_match_label}")
        else:
            if sig_result.yara_matches:
                score += min(_YARA_SCORE_CAP, _YARA_SCORE_PER_MATCH * len(sig_result.yara_matches))
                reasons.extend(f"Matched YARA rule: {name}" for name in sig_result.yara_matches)

            entropy_score, entropy_reason = self._check_entropy(file_path)
            score += entropy_score
            if entropy_reason:
                reasons.append(entropy_reason)

        score = min(score, 100)
        result = FileScanResult(
            file_path=str(file_path),
            sha256=sig_result.sha256,
            size_bytes=self._safe_size(file_path),
            risk_score=score,
            severity=self._severity_for_score(score),
            reasons=reasons,
        )
        self._finalize(result)
        return result

    # ------------------------------------------------------------------
    # Heuristic checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_extension(file_path: Path) -> tuple[int, List[str]]:
        """
        Checks a file's name for suspicious or disguised extensions,
        without needing to read its content.

        Args:
            file_path: Path to check.

        Returns:
            A (score_contribution, reasons) tuple.
        """
        suffixes = [s.lower() for s in file_path.suffixes]
        final_ext = file_path.suffix.lower()

        if len(suffixes) >= 2 and final_ext in _SUSPICIOUS_EXTENSIONS and suffixes[-2] in _TRUSTED_LOOKING_EXTENSIONS:
            return _DOUBLE_EXTENSION_SCORE, [
                f"Double extension disguise: '{file_path.name}' looks like "
                f"a {suffixes[-2]} file but actually runs as {final_ext}"
            ]

        if final_ext in _SUSPICIOUS_EXTENSIONS:
            return _SUSPICIOUS_EXTENSION_SCORE, [f"Executable/script extension: {final_ext}"]

        return 0, []

    @staticmethod
    def _check_entropy(file_path: Path) -> tuple[int, Optional[str]]:
        """
        Computes Shannon entropy for file types where high entropy is
        unusual (skipped entirely for already-compressed formats).

        Args:
            file_path: Path to check.

        Returns:
            A (score_contribution, reason_or_None) tuple.
        """
        if file_path.suffix.lower() in _KNOWN_COMPRESSED_EXTENSIONS:
            return 0, None

        try:
            entropy = FileScanner.compute_shannon_entropy(file_path)
        except OSError as read_error:
            logger.warning(f"Could not compute entropy for '{file_path}': {read_error}")
            return 0, None

        if entropy > _HIGH_ENTROPY_THRESHOLD:
            return _HIGH_ENTROPY_SCORE, (
                f"High entropy ({entropy:.2f}/8.0) suggests packed or encrypted content"
            )
        return 0, None

    @staticmethod
    def compute_shannon_entropy(file_path: Path, max_bytes: int = _MAX_ENTROPY_READ_BYTES) -> float:
        """
        Computes the Shannon entropy of a file's content, in bits per
        byte (0.0 = completely uniform, 8.0 = maximally random). Reads
        at most max_bytes so very large files don't need to be fully
        loaded into memory - a sample from the start is sufficient to
        characterize whether the content looks packed/encrypted.

        Args:
            file_path: Path to the file to analyze.
            max_bytes: Maximum number of bytes to read for the sample.

        Returns:
            The computed entropy value.
        """
        data = file_path.read_bytes()[:max_bytes]
        if not data:
            return 0.0

        byte_counts = Counter(data)
        length = len(data)
        return -sum(
            (count / length) * math.log2(count / length)
            for count in byte_counts.values()
        )

    @staticmethod
    def _severity_for_score(score: int) -> FileRiskSeverity:
        """Maps a 0-100 risk score to a severity band."""
        if score == 0:
            return FileRiskSeverity.SAFE
        if score < 30:
            return FileRiskSeverity.LOW
        if score < 60:
            return FileRiskSeverity.MEDIUM
        if score < 90:
            return FileRiskSeverity.HIGH
        return FileRiskSeverity.CRITICAL

    @staticmethod
    def _safe_size(file_path: Path) -> int:
        """Returns a file's size in bytes, or 0 if it can't be stat'd."""
        try:
            return file_path.stat().st_size
        except OSError:
            return 0

    # ------------------------------------------------------------------
    # History + reporting
    # ------------------------------------------------------------------

    def _finalize(self, result: FileScanResult) -> None:
        """Records a completed scan in history, logs it, and notifies the registered callback."""
        with self._history_lock:
            self._recent_scans.append(result)

        self._log_result(result)

        for callback in self._scan_result_callbacks:
            try:
                callback(result)
            except Exception as callback_error:
                logger.error(f"scan_result callback failed for '{result.file_path}': {callback_error}")

    @staticmethod
    def _log_result(result: FileScanResult) -> None:
        """Logs a scan result at a level matching its severity."""
        if result.scan_error:
            logger.warning(f"Could not scan '{result.file_path}': {result.scan_error}")
        elif result.severity in (FileRiskSeverity.HIGH, FileRiskSeverity.CRITICAL):
            logger.error(
                f"MALICIOUS FILE DETECTED: '{result.file_path}' "
                f"(score={result.risk_score}, severity={result.severity.value}) - {result.reasons}"
            )
        elif result.severity == FileRiskSeverity.MEDIUM:
            logger.warning(
                f"Suspicious file: '{result.file_path}' (score={result.risk_score}) - {result.reasons}"
            )
        else:
            logger.info(f"File scanned clean: '{result.file_path}' (score={result.risk_score})")

    def get_recent_scans(self, limit: int = 20) -> List[FileScanResult]:
        """
        Returns the most recent scan results, newest first.

        Args:
            limit: Maximum number of results to return.

        Returns:
            A list of up to `limit` FileScanResult objects.
        """
        with self._history_lock:
            return list(self._recent_scans)[-limit:][::-1]


def get_file_scanner() -> FileScanner:
    """Module-level accessor returning the singleton FileScanner instance."""
    return FileScanner()


if __name__ == "__main__":
    # Standalone smoke test - manually scans a set of manufactured
    # temp files covering every heuristic path, without touching the
    # real Downloads/Desktop folders or starting the live watcher.
    import os
    import tempfile

    scanner = get_file_scanner()

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        test_files = {
            "Clean text file": (tmp_path / "notes.txt", b"Just a normal set of project notes."),
            "Suspicious extension": (tmp_path / "tool.exe", b"MZ fake header, arbitrary content here."),
            "Double extension disguise": (tmp_path / "invoice.pdf.exe", b"Looks like a PDF, isn't one."),
            "High entropy (random bytes)": (tmp_path / "mystery.bin", os.urandom(4096)),
        }

        for label, (path, content) in test_files.items():
            path.write_bytes(content)
            result = scanner.scan_file(path)
            print(f"\n{label}: {path.name}")
            print(f"  Risk score: {result.risk_score}  Severity: {result.severity.value}")
            print(f"  Reasons: {result.reasons or ['none']}")
            print(f"  Malicious verdict: {result.is_malicious}")

    print(f"\nScan history recorded: {len(scanner.get_recent_scans(limit=10))} entries")