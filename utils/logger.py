"""
utils/logger.py

Centralized logging utility for CyberSentinel AI.

This module configures and provides logger instances used across
the entire application (UI, network, AI, security, database layers).
It uses rotating file handlers to prevent unbounded log growth and
mirrors output to the console for live debugging during development.

Usage:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import PathConfig, LoggingConfig


class LoggerFactory:
    """
    Factory class responsible for creating and configuring logger
    instances with consistent formatting, rotation policy, and
    output handlers (file + console).

    This class ensures handlers are only attached once per logger
    name, preventing duplicate log lines when get_logger() is called
    multiple times for the same module (e.g., on re-import).
    """

    _configured_loggers: set[str] = set()
    _initialized: bool = False

    @classmethod
    def _initialize_directories(cls) -> None:
        """
        Ensures the logs directory exists before any file handler
        attempts to write to it. Safe to call multiple times.
        """
        if not cls._initialized:
            PathConfig.ensure_directories_exist()
            cls._initialized = True

    @classmethod
    def create_logger(
        cls,
        name: str,
        log_file: Path = PathConfig.APP_LOG_FILE,
        level: str = LoggingConfig.LOG_LEVEL,
    ) -> logging.Logger:
        """
        Creates (or retrieves) a configured logger instance.

        Args:
            name: The logger name, typically __name__ of the calling module.
            log_file: The target log file path for the RotatingFileHandler.
            level: The minimum logging level as a string (e.g., "INFO", "DEBUG").

        Returns:
            A fully configured logging.Logger instance with both file
            and console handlers attached.
        """
        cls._initialize_directories()

        logger = logging.getLogger(name)

        # Prevent attaching duplicate handlers if this logger was
        # already configured earlier in the application lifecycle.
        handler_key = f"{name}::{log_file}"
        if handler_key in cls._configured_loggers:
            return logger

        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        logger.propagate = False  # avoid duplicate logs via root logger

        formatter = logging.Formatter(
            fmt=LoggingConfig.LOG_FORMAT,
            datefmt=LoggingConfig.DATE_FORMAT,
        )

        # Rotating file handler: caps log file size and keeps backups
        file_handler = RotatingFileHandler(
            filename=str(log_file),
            maxBytes=LoggingConfig.MAX_LOG_FILE_SIZE_MB * 1024 * 1024,
            backupCount=LoggingConfig.BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

        # Console handler: useful during development and debugging
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        cls._configured_loggers.add(handler_key)
        return logger


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function for retrieving a standard application logger.

    This is the primary entry point that other modules should use.

    Args:
        name: The name of the module requesting a logger (use __name__).

    Returns:
        A configured logging.Logger instance writing to the main
        application log file (logs/app.log) and the console.
    """
    return LoggerFactory.create_logger(
        name=name,
        log_file=PathConfig.APP_LOG_FILE,
        level=LoggingConfig.LOG_LEVEL,
    )


def get_threat_logger() -> logging.Logger:
    """
    Convenience function for retrieving the dedicated threat/security
    event logger.

    This logger writes to a separate file (logs/threats.log) so that
    security-relevant events (detected intrusions, blocked packets,
    high risk scores) can be audited independently from general
    application logs.

    Returns:
        A configured logging.Logger instance writing to the threat
        log file (logs/threats.log) and the console.
    """
    return LoggerFactory.create_logger(
        name="cybersentinel.threats",
        log_file=PathConfig.THREAT_LOG_FILE,
        level=LoggingConfig.LOG_LEVEL,
    )