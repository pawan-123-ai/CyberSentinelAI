"""
security/signature_manager.py

Signature-based detection layer for CyberSentinel AI's file-scanning
feature: known-malicious-hash lookup and YARA pattern matching.

Responsibility:
    - Maintain a local database of known-malicious SHA256 hashes
      (data/known_malicious_hashes.json), checked via exact match.
    - Load and compile YARA rules from data/yara_rules/*.yar, matched
      against file content for pattern-based detection that doesn't
      depend on an exact hash (catches variants, not just exact copies).
    - Expose a single scan() entry point that file_scanner.py calls,
      returning both the file's computed hash and any signature
      matches, without file_scanner.py needing to know how either
      detection mechanism works internally.

Design notes:
    - Singleton pattern, consistent with the rest of the codebase.
    - No PySide6 imports.
    - Self-contained: does not depend on config/settings.py or
      database/db_manager.py.
    - Ships with exactly one starter YARA rule (EICAR test-string
      detection) auto-created on first run if the rules folder is
      empty, and one built-in known hash (the EICAR test file's
      SHA256) - both safe, standard, non-malicious test signatures
      used industry-wide to verify a scanner is actually working.
    - Gracefully degrades if yara-python is not installed: hash-based
      detection still works, YARA matching is simply skipped with a
      logged warning, mirroring the pattern used for fastapi/uvicorn
      in web/server.py.
"""

import hashlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

try:
    import yara

    _YARA_AVAILABLE = True
except ImportError:
    _YARA_AVAILABLE = False


# Paths are relative to the project root (matches the data/ convention
# already used elsewhere, e.g. data/raw, data/processed).
_YARA_RULES_DIR = Path("data/yara_rules")
_KNOWN_HASHES_FILE = Path("data/known_malicious_hashes.json")

# The EICAR Standard Anti-Malware Test File - a harmless string every
# real antivirus/EDR engine is built to flag. Used here purely to give
# both detection layers (hash + YARA) something safe to demonstrate
# against. Never a real malicious sample.
_EICAR_STRING = r'X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*'
_EICAR_SHA256 = "275a021bbfb6489e54d471899f7db9d1663fc695ec2fe2a2c4538aabf651fd0"

_STARTER_YARA_RULE = f'''rule EICAR_Test_File
{{
    meta:
        description = "Detects the EICAR antivirus test string"
        reference = "https://www.eicar.org/download-anti-malware-testfile/"
    strings:
        $eicar = "{_EICAR_STRING}"
    condition:
        $eicar
}}
'''


@dataclass
class SignatureScanResult:
    """
    The combined outcome of checking one file against both detection
    layers this module owns.

    Attributes:
        sha256: The file's computed SHA256 hash.
        hash_match_label: Description of the known-hash match, or None.
        yara_matches: Names of every YARA rule that matched, empty if none.
        error: Set if the file could not be read/scanned (e.g. removed
            or quarantined by another security product between being
            noticed and being scanned). When set, sha256 is empty and
            hash_match_label/yara_matches are meaningless.
    """

    sha256: str
    hash_match_label: Optional[str] = None
    yara_matches: List[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def is_flagged(self) -> bool:
        """True if either detection layer found something."""
        return self.hash_match_label is not None or bool(self.yara_matches)


class SignatureManager:
    """
    Singleton manager for signature-based file detection: known-hash
    lookup and YARA pattern matching.
    """

    _instance: Optional["SignatureManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "SignatureManager":
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
        self._known_hashes: Dict[str, str] = {}
        self._compiled_rules: Optional["yara.Rules"] = None

        self._load_known_hashes()
        self._load_yara_rules()

    # ------------------------------------------------------------------
    # Known-hash detection
    # ------------------------------------------------------------------

    def _load_known_hashes(self) -> None:
        """
        Loads the known-malicious-hash database from disk, creating it
        with the built-in EICAR hash on first run if it doesn't exist.
        """
        if not _KNOWN_HASHES_FILE.exists():
            logger.info("No known-hash database found - creating one with the EICAR test hash.")
            self._known_hashes = {_EICAR_SHA256: "EICAR test file (safe, standard AV test signature)"}
            self._persist_known_hashes()
            return

        try:
            self._known_hashes = json.loads(_KNOWN_HASHES_FILE.read_text())
            logger.info(f"Loaded {len(self._known_hashes)} known malicious hash(es).")
        except json.JSONDecodeError as parse_error:
            logger.error(f"Corrupt known-hash database ({parse_error}) - starting empty.")
            self._known_hashes = {}

    def _persist_known_hashes(self) -> None:
        """Writes the current known-hash dict to disk as JSON."""
        _KNOWN_HASHES_FILE.parent.mkdir(parents=True, exist_ok=True)
        _KNOWN_HASHES_FILE.write_text(json.dumps(self._known_hashes, indent=2))

    def add_known_hash(self, sha256: str, label: str) -> None:
        """
        Adds a new hash to the known-malicious database. Intended for
        a future "mark as malicious" action once a file has been
        manually confirmed bad (e.g. from a quarantine review UI).

        Args:
            sha256: The SHA256 hash to flag on future scans.
            label: A human-readable description shown when it matches.
        """
        with self._state_lock:
            self._known_hashes[sha256.lower()] = label
            self._persist_known_hashes()
        logger.info(f"Added known malicious hash: {sha256[:16]}...")

    def check_hash(self, sha256: str) -> Optional[str]:
        """
        Checks a SHA256 hash against the known-malicious database.

        Args:
            sha256: The hash to check.

        Returns:
            The match's description label if found, otherwise None.
        """
        with self._state_lock:
            return self._known_hashes.get(sha256.lower())

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """
        Computes the SHA256 hash of a file, reading it in chunks so
        large files don't need to be loaded into memory all at once.

        Args:
            file_path: Path to the file to hash.

        Returns:
            The hex-encoded SHA256 hash.
        """
        hasher = hashlib.sha256()
        with open(file_path, "rb") as file_handle:
            for chunk in iter(lambda: file_handle.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    # ------------------------------------------------------------------
    # YARA pattern detection
    # ------------------------------------------------------------------

    def _load_yara_rules(self) -> None:
        """
        Compiles every .yar/.yara file in the rules directory into a
        single ruleset. Creates the directory with a starter EICAR
        rule on first run if it's missing or empty. No-ops with a
        logged warning if yara-python isn't installed.
        """
        if not _YARA_AVAILABLE:
            logger.warning(
                "yara-python not installed - YARA matching disabled. "
                "Install with: pip install yara-python"
            )
            return

        _YARA_RULES_DIR.mkdir(parents=True, exist_ok=True)
        rule_files = list(_YARA_RULES_DIR.glob("*.yar")) + list(_YARA_RULES_DIR.glob("*.yara"))

        if not rule_files:
            logger.info("No YARA rules found - creating starter rule (EICAR detection).")
            starter_path = _YARA_RULES_DIR / "eicar_test.yar"
            starter_path.write_text(_STARTER_YARA_RULE)
            rule_files = [starter_path]

        try:
            filepaths = {path.stem: str(path) for path in rule_files}
            self._compiled_rules = yara.compile(filepaths=filepaths)
            logger.info(f"Compiled {len(rule_files)} YARA rule file(s).")
        except yara.Error as compile_error:
            logger.error(f"Failed to compile YARA rules: {compile_error}")
            self._compiled_rules = None

    def match_yara(self, file_path: Path) -> List[str]:
        """
        Matches a file's content against every compiled YARA rule.

        Args:
            file_path: Path to the file to scan.

        Returns:
            Names of every rule that matched. Empty if YARA is
            unavailable, no rules are compiled, or nothing matched.
        """
        if self._compiled_rules is None:
            return []

        try:
            matches = self._compiled_rules.match(str(file_path))
            return [match.rule for match in matches]
        except yara.Error as match_error:
            logger.error(f"YARA match failed for '{file_path}': {match_error}")
            return []

    def reload_rules(self) -> None:
        """
        Re-compiles YARA rules from disk. Intended to be called after
        a user adds/edits a .yar file via a future rules-management UI,
        without needing to restart the application.
        """
        self._load_yara_rules()

    # ------------------------------------------------------------------
    # Combined entry point
    # ------------------------------------------------------------------

    def scan(self, file_path: Path) -> SignatureScanResult:
        """
        Runs both detection layers against a single file: computes its
        hash and checks it against the known-malicious database, and
        matches its content against compiled YARA rules.

        Args:
            file_path: Path to the file to scan.

        Returns:
            A SignatureScanResult combining both layers' findings, or
            one with `error` set if the file could not be read (e.g.
            removed/quarantined by another security product between
            being noticed and being scanned).
        """
        try:
            sha256 = self.compute_sha256(file_path)
        except (OSError, FileNotFoundError, PermissionError) as read_error:
            logger.warning(
                f"Could not read '{file_path}' for scanning ({read_error}). "
                "It may have been removed or quarantined by another "
                "security product (e.g. Windows Defender reacting to "
                "the file's own content) between being noticed and "
                "being scanned."
            )
            return SignatureScanResult(sha256="", error=str(read_error))

        return SignatureScanResult(
            sha256=sha256,
            hash_match_label=self.check_hash(sha256),
            yara_matches=self.match_yara(file_path),
        )


def get_signature_manager() -> SignatureManager:
    """Module-level accessor returning the singleton SignatureManager instance."""
    return SignatureManager()


if __name__ == "__main__":
    # Standalone smoke test - creates a real EICAR test file in a temp
    # location, scans it, and confirms both detection layers fire.
    import tempfile

    manager = get_signature_manager()

    print(f"YARA available: {_YARA_AVAILABLE}")
    print(f"Known hashes loaded: {len(manager._known_hashes)}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        eicar_path = Path(tmp_dir) / "eicar_test.txt"
        eicar_path.write_text(_EICAR_STRING)

        print(f"\nScanning EICAR test file at {eicar_path}...")
        result = manager.scan(eicar_path)

        if result.error:
            print(f"  Could not scan file: {result.error}")
            print("  (Likely Windows Defender quarantined/removed it first - expected on a protected machine.)")
        else:
            print(f"  SHA256: {result.sha256}")
            print(f"  Hash match: {result.hash_match_label}")
            print(f"  YARA matches: {result.yara_matches}")
            print(f"  Flagged: {result.is_flagged}")

    print("\nScanning a harmless file (expect no matches)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        clean_path = Path(tmp_dir) / "clean.txt"
        clean_path.write_text("This is just a normal text file, nothing suspicious here.")

        result = manager.scan(clean_path)
        if result.error:
            print(f"  Could not scan file: {result.error}")
        else:
            print(f"  Hash match: {result.hash_match_label}")
            print(f"  YARA matches: {result.yara_matches}")
            print(f"  Flagged: {result.is_flagged}")