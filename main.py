"""
main.py

Application entry point for CyberSentinel AI.

Responsibilities:
    1. Ensure required runtime directories exist.
    2. Initialize application-wide logging.
    3. Establish the MongoDB connection (non-blocking failure).
    4. Configure the Qt Application (metadata, icon, dark theme).
    5. Install a global exception hook so uncaught errors are logged
       rather than crashing the app silently.
    6. Launch the MainWindow and start the Qt event loop. MainWindow
       itself now owns the login/signup gate as its first page, so
       this file no longer shows any pre-window dialog.
    7. Guarantee clean resource shutdown on exit.

Run with:
    python main.py

This is also the target script for PyInstaller packaging into a
standalone Windows .exe (see README.md for build instructions).
"""


import sys
import traceback
from types import TracebackType
from typing import Optional, Type

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from config.settings import AppInfo, PathConfig, ThemeConfig
from utils.logger import get_logger
from database.db_manager import get_database_manager

logger = get_logger(__name__)


class CyberSentinelApplication:
    """
    Bootstraps and owns the lifecycle of the CyberSentinel AI desktop
    application.

    This class is intentionally the only place in the codebase that
    touches QApplication construction and top-level startup/shutdown
    sequencing. Everything else (UI, network, AI, security, and now
    authentication) is wired together inside MainWindow, which this
    class simply constructs and shows.
    """

    def __init__(self, argv: list[str]) -> None:
        """
        Initializes the application bootstrapper.

        Args:
            argv: Command-line arguments, typically sys.argv.
        """
        self._argv = argv
        self._qt_app: Optional[QApplication] = None
        self._main_window = None  # type: ignore[assignment]  # set in _launch_main_window

    def run(self) -> int:
        """
        Executes the full startup sequence and enters the Qt event loop.

        Returns:
            The process exit code returned by the Qt event loop.
        """
        self._prepare_environment()
        self._install_global_exception_hook()

        self._qt_app = self._create_qt_application()
        self._connect_database()
        self._launch_main_window()

        logger.info("%s v%s started successfully.", AppInfo.NAME, AppInfo.VERSION)
        exit_code = self._qt_app.exec()

        self._shutdown()
        return exit_code

    # ------------------------------------------------------------------
    # Startup steps
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_environment() -> None:
        """
        Creates all required directories (logs, models, data, exports)
        before any other subsystem attempts to read or write files.
        """
        PathConfig.ensure_directories_exist()
        logger.info("Runtime directories verified.")

    def _install_global_exception_hook(self) -> None:
        """
        Installs a global handler for uncaught exceptions so that any
        error anywhere in the application (UI callbacks, background
        threads excluded — those need their own guards) is logged with
        a full traceback instead of silently killing the process, and
        the user is shown a graceful error dialog rather than a raw
        Python traceback on screen.
        """

        def handle_exception(
            exc_type: Type[BaseException],
            exc_value: BaseException,
            exc_traceback: Optional[TracebackType],
        ) -> None:
            if issubclass(exc_type, KeyboardInterrupt):
                sys.__excepthook__(exc_type, exc_value, exc_traceback)
                return

            formatted = "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)
            )
            logger.critical("Uncaught exception:\n%s", formatted)

            if self._qt_app is not None:
                QMessageBox.critical(
                    None,
                    f"{AppInfo.NAME} - Unexpected Error",
                    "An unexpected error occurred and has been logged.\n"
                    "CyberSentinel AI will attempt to continue running.\n\n"
                    f"Details: {exc_value}",
                )

        sys.excepthook = handle_exception
        logger.info("Global exception hook installed.")

    def _create_qt_application(self) -> QApplication:
        """
        Constructs and configures the QApplication instance, including
        organization metadata (used for QSettings later) and the
        application icon.

        Returns:
            The configured QApplication instance.
        """
        qt_app = QApplication(self._argv)
        qt_app.setApplicationName(AppInfo.NAME)
        qt_app.setApplicationVersion(AppInfo.VERSION)
        qt_app.setOrganizationName(AppInfo.ORGANIZATION)
        qt_app.setQuitOnLastWindowClosed(True)

        icon_path = PathConfig.ASSETS_DIR / "icon.ico"
        if icon_path.exists():
            qt_app.setWindowIcon(QIcon(str(icon_path)))
        else:
            logger.warning(
                "Application icon not found at %s (will use default Qt icon).",
                icon_path,
            )

        qt_app.setStyle("Fusion")  # neutral base style; dashboard.py applies the full dark theme
        logger.info("QApplication configured.")
        return qt_app

    @staticmethod
    def _connect_database() -> None:
        """
        Establishes the MongoDB connection via the shared DatabaseManager
        singleton. Connection failure is logged but is NOT fatal — the
        application continues to run in a degraded (no-persistence) mode,
        per DatabaseManager's design.
        """
        db_manager = get_database_manager()
        if db_manager.is_connected():
            logger.info("Database layer ready.")
        else:
            logger.warning(
                "Database layer unavailable. The application will run "
                "without data persistence until MongoDB is reachable."
            )

    def _launch_main_window(self) -> None:
        """
        Imports and instantiates the main dashboard window.

        The import is deliberately deferred to this method (rather than
        placed at module level) so that logging and directory setup are
        always guaranteed to run first, even though dashboard.py itself
        will end up importing many other subsystems.

        MainWindow opens showing its embedded login page first; the
        dashboard shell only becomes visible after successful
        authentication, handled entirely inside MainWindow.
        """
        from app.ui.dashboard import MainWindow  # deferred import, see docstring

        self._main_window = MainWindow()
        self._main_window.setWindowState(Qt.WindowState.WindowMaximized)
        self._main_window.show()
        logger.info("Main window launched (showing login page).")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    @staticmethod
    def _shutdown() -> None:
        """
        Performs graceful cleanup when the Qt event loop exits:
        closes the MongoDB connection and flushes logs. UI-owned
        resources (capture threads, timers) are expected to be
        stopped by MainWindow's own closeEvent handler before this
        point is reached.
        """
        logger.info("Shutting down %s...", AppInfo.NAME)
        get_database_manager().disconnect()
        logger.info("Shutdown complete.")


def main() -> int:
    """
    Module-level entry point. Creates and runs the CyberSentinelApplication.

    Returns:
        The process exit code.
    """
    app = CyberSentinelApplication(sys.argv)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())