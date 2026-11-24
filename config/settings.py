"""
config/settings.py

Central configuration file for CyberSentinel AI.

Holds:
    - Application metadata
    - Filesystem paths
    - UI theme
    - Network monitoring configuration
    - AI/ML model configuration
    - MongoDB configuration
    - UI configuration
    - Logging configuration

This module must NOT import from any other project module
to avoid circular imports. It is the root of the dependency tree.
"""

import os
from pathlib import Path


# ==========================================================================
# Application Information
# ==========================================================================


class AppInfo:
    """
    Static application metadata used throughout CyberSentinel AI.
    """

    NAME: str = "CyberSentinel AI"
    VERSION: str = "1.0.0"
    ORGANIZATION: str = "CyberSentinel Labs"
    DESCRIPTION: str = (
        "AI-Powered Real-Time Network Intrusion Detection System"
    )
    AUTHOR: str = "MCA Final Year Project"


# ==========================================================================
# Filesystem Paths
# ==========================================================================


class PathConfig:
    """
    Centralized filesystem paths.

    All paths are resolved relative to the project root so that the
    application works regardless of the current working directory.
    """

    ROOT_DIR: Path = Path(__file__).resolve().parent.parent

    LOGS_DIR: Path = ROOT_DIR / "logs"

    MODELS_DIR: Path = ROOT_DIR / "models"

    DATA_DIR: Path = ROOT_DIR / "data"

    RAW_DATA_DIR: Path = DATA_DIR / "raw"

    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    ASSETS_DIR: Path = ROOT_DIR / "assets"

    APP_LOG_FILE: Path = LOGS_DIR / "app.log"

    THREAT_LOG_FILE: Path = LOGS_DIR / "threats.log"

    EXPORT_DIR: Path = LOGS_DIR / "exports"

    # ------------------------------------------------------------------
    # Legacy/default model paths
    #
    # Kept for compatibility with any older modules that may still
    # reference these constants.
    # ------------------------------------------------------------------

    DEFAULT_MODEL_FILE: Path = MODELS_DIR / "random_forest_model.pkl"

    SCALER_FILE: Path = MODELS_DIR / "random_forest_scaler.pkl"

    @classmethod
    def ensure_directories_exist(cls) -> None:
        """
        Creates all required directories.
        """

        required_dirs = [
            cls.LOGS_DIR,
            cls.MODELS_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.ASSETS_DIR,
            cls.EXPORT_DIR,
        ]

        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)


# ==========================================================================
# UI Theme
# ==========================================================================


class ThemeConfig:
    """
    Dark cybersecurity theme used by the PySide6 UI.
    """

    # Core backgrounds
    BACKGROUND_PRIMARY: str = "#0B0E14"
    BACKGROUND_SECONDARY: str = "#121722"
    BACKGROUND_CARD: str = "#161B26"
    BACKGROUND_SIDEBAR: str = "#0D1117"

    # Accent colors
    NEON_BLUE: str = "#00D9FF"
    NEON_CYAN: str = "#00FFC6"
    ACCENT_PURPLE: str = "#7B61FF"

    # Status colors
    ALERT_RED: str = "#FF3B5C"
    WARNING_ORANGE: str = "#FFA500"
    SAFE_GREEN: str = "#00E676"

    # Text colors
    TEXT_PRIMARY: str = "#E6EDF3"
    TEXT_SECONDARY: str = "#8B949E"
    TEXT_DISABLED: str = "#4A5568"

    # Border
    BORDER_COLOR: str = "#2A2F3A"

    # Fonts
    FONT_FAMILY: str = "Segoe UI"
    FONT_FAMILY_MONO: str = "Consolas"


# ==========================================================================
# Network Configuration
# ==========================================================================


class NetworkConfig:
    """
    Configuration for network monitoring and packet capture.
    """

    DEFAULT_INTERFACE: str = "auto"

    PACKET_CAPTURE_TIMEOUT: int = 1

    MAX_PACKETS_IN_MEMORY: int = 5000

    PACKETS_PER_SECOND_WINDOW: int = 1

    PROMISCUOUS_MODE: bool = True


# ==========================================================================
# AI / ML Configuration
# ==========================================================================


class AIConfig:
    """
    Configuration for the CyberSentinel AI/ML threat detection system.

    Supported algorithms:

        1. Random Forest
        2. XGBoost
        3. LightGBM
        4. Deep Learning - MLP

    Change ACTIVE_MODEL to switch the active algorithm.
    """

    # ------------------------------------------------------------------
    # Active model
    # ------------------------------------------------------------------

    ACTIVE_MODEL: str = "random_forest"

    # Supported model identifiers
    AVAILABLE_MODELS: list[str] = [
        "random_forest",
        "xgboost",
        "lightgbm",
        "deep_learning",
    ]

    # ------------------------------------------------------------------
    # Threat detection configuration
    # ------------------------------------------------------------------

    CONFIDENCE_THRESHOLD: float = 0.75

    RISK_SCORE_HIGH: int = 75

    RISK_SCORE_MEDIUM: int = 40

    # ------------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------------

    RANDOM_FOREST_PARAMS: dict = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": 42,
        "n_jobs": -1,
    }

    # ------------------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------------------

    XGBOOST_PARAMS: dict = {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
    }

    # ------------------------------------------------------------------
    # LightGBM
    # ------------------------------------------------------------------

    LIGHTGBM_PARAMS: dict = {
        "n_estimators": 200,
        "learning_rate": 0.08,
        "max_depth": -1,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "multiclass",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    # ------------------------------------------------------------------
    # Deep Learning / MLP
    # ------------------------------------------------------------------

    DEEP_LEARNING_PARAMS: dict = {
        "hidden_layer_sizes": (128, 64, 32),
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": 64,
        "learning_rate_init": 0.001,
        "max_iter": 300,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 15,
        "random_state": 42,
    }

    # ------------------------------------------------------------------
    # Feature columns
    # ------------------------------------------------------------------

    FEATURE_COLUMNS: list[str] = [
        "packet_size",
        "protocol_type",
        "src_port",
        "dst_port",
        "duration",
        "packet_rate",
        "byte_rate",
        "flag_count",
    ]

    # ------------------------------------------------------------------
    # Attack classes
    # ------------------------------------------------------------------

    ATTACK_CLASSES: list[str] = [
        "Normal",
        "DoS",
        "Probe",
        "R2L",
        "U2R",
        "Port Scan",
        "DDoS",
    ]


# ==========================================================================
# Database Configuration
# ==========================================================================


class DatabaseConfig:
    """
    MongoDB configuration.
    """

    MONGO_URI: str = "mongodb://localhost:27017/"

    MONGO_DB_NAME: str = "cybersentinel_db"

    SERVER_SELECTION_TIMEOUT_MS: int = 5000

    MAX_LOG_RETENTION_DAYS: int = 30

    COLLECTION_PACKETS: str = "packets"

    COLLECTION_THREATS: str = "threats"

    COLLECTION_ALERTS: str = "alerts"

    COLLECTION_LOGS: str = "security_logs"

    COLLECTION_SYSTEM_STATS: str = "system_stats"

    COLLECTION_AI_PREDICTIONS: str = "ai_predictions"


# ==========================================================================
# UI Configuration
# ==========================================================================


class UIConfig:
    """
    Main application window and dashboard refresh configuration.
    """

    WINDOW_TITLE: str = (
        f"{AppInfo.NAME} - {AppInfo.DESCRIPTION}"
    )

    WINDOW_MIN_WIDTH: int = 1366

    WINDOW_MIN_HEIGHT: int = 768

    DASHBOARD_REFRESH_INTERVAL_MS: int = 1000

    GRAPH_REFRESH_INTERVAL_MS: int = 1000

    MAX_GRAPH_DATA_POINTS: int = 60

    SIDEBAR_WIDTH_EXPANDED: int = 240

    SIDEBAR_WIDTH_COLLAPSED: int = 70


# ==========================================================================
# Logging Configuration
# ==========================================================================


class LoggingConfig:
    """
    Python logging configuration.
    """

    LOG_LEVEL: str = "INFO"

    LOG_FORMAT: str = (
        "%(asctime)s | %(levelname)-8s | "
        "%(name)s | %(message)s"
    )

    DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"

    MAX_LOG_FILE_SIZE_MB: int = 10

    BACKUP_COUNT: int = 5