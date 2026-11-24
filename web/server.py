"""
CyberSentinel AI - Web Server Manager

Owns the FastAPI application instance and manages the lifecycle of its
Uvicorn server inside a background thread, allowing the web/remote-access
layer to run alongside the PySide6 desktop UI within the same main.py process.

Responsibilities:
    - Construct and configure the FastAPI application.
    - Configure CORS.
    - Serve the web/static directory.
    - Register authentication/API/WebSocket routers.
    - Start and stop Uvicorn in a background thread.
    - Provide server status information.
    - Detect the local LAN IP address.

This module contains no business logic and no PySide6 imports.
"""

# ============================================================================
# STANDARD LIBRARY IMPORTS
# ============================================================================

import socket
import threading
from pathlib import Path
from typing import Callable, Optional


# ============================================================================
# PROJECT IMPORTS
# ============================================================================

from utils.logger import get_logger

# Authentication router
from web.auth_manager import auth_router


# ============================================================================
# LOGGER
# ============================================================================

logger = get_logger(__name__)


# ============================================================================
# OPTIONAL WEB DEPENDENCIES
# ============================================================================

try:
    import uvicorn

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles

    _WEB_DEPENDENCIES_AVAILABLE = True

except ImportError:
    _WEB_DEPENDENCIES_AVAILABLE = False

    uvicorn = None
    FastAPI = None
    CORSMiddleware = None
    StaticFiles = None


# ============================================================================
# WEB SERVER MANAGER
# ============================================================================

class WebServerManager:
    """
    Singleton manager for CyberSentinel AI's web/remote-access layer.

    The web server runs Uvicorn in a dedicated background thread so that
    it does not block the PySide6 desktop application's main event loop.

    Default:
        Host: 0.0.0.0
        Port: 8443
    """

    # ------------------------------------------------------------------------
    # Singleton state
    # ------------------------------------------------------------------------

    _instance: Optional["WebServerManager"] = None
    _lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------------
    # Singleton constructor
    # ------------------------------------------------------------------------

    def __new__(cls) -> "WebServerManager":
        with cls._lock:

            if cls._instance is None:
                cls._instance = super().__new__(cls)

                cls._instance._initialized = False

        return cls._instance

    # ------------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------------

    def __init__(self) -> None:

        if self._initialized:
            return

        self._initialized = True

        # FastAPI application
        self._app: Optional["FastAPI"] = None

        # Uvicorn server
        self._server: Optional["uvicorn.Server"] = None

        # Background server thread
        self._server_thread: Optional[threading.Thread] = None

        # Runtime state
        self._is_running: bool = False

        # Network configuration
        self._host: str = "0.0.0.0"
        self._port: int = 8443

        # Optional callback for PySide6 UI
        self._status_callback: Optional[
            Callable[[bool, str], None]
        ] = None

        # --------------------------------------------------------------------
        # Build FastAPI application
        # --------------------------------------------------------------------

        if _WEB_DEPENDENCIES_AVAILABLE:

            self._app = self._build_app()

            # ---------------------------------------------------------------
            # Register authentication router
            # ---------------------------------------------------------------

            if auth_router is not None:

                self.register_router(
                    auth_router,
                    prefix="/api/auth"
                )

                logger.info(
                    "Authentication router registered at /api/auth"
                )

        else:

            logger.warning(
                "FastAPI/Uvicorn not installed - web layer disabled. "
                "Install with: pip install fastapi uvicorn[standard]"
            )

    # =========================================================================
    # FASTAPI APPLICATION
    # =========================================================================

    def _build_app(self) -> "FastAPI":
        """
        Construct and configure the FastAPI application.

        Includes:

            - Application metadata
            - CORS
            - Static file serving
            - Health-check endpoint
        """

        app = FastAPI(
            title="CyberSentinel AI - Remote Dashboard",
            description=(
                "LAN-based remote monitoring interface "
                "for CyberSentinel AI"
            ),
            version="1.0.0",
        )

        # --------------------------------------------------------------------
        # CORS
        # --------------------------------------------------------------------

        # LAN dashboard access.

        app.add_middleware(
            CORSMiddleware,

            allow_origins=["*"],

            allow_credentials=True,

            allow_methods=["*"],

            allow_headers=["*"],
        )

        # --------------------------------------------------------------------
        # Static files
        # --------------------------------------------------------------------

        static_dir = Path(__file__).parent / "static"

        if static_dir.exists():

            app.mount(
                "/static",
                StaticFiles(
                    directory=str(static_dir)
                ),
                name="static",
            )

            logger.info(
                f"Static web directory mounted: {static_dir}"
            )

        else:

            logger.warning(
                f"Static directory not found at {static_dir}"
            )

        # --------------------------------------------------------------------
        # Root endpoint
        # --------------------------------------------------------------------

        @app.get("/")
        async def root() -> dict:
            """
            Basic root endpoint.

            The actual login page is available at:

                /static/login.html
            """

            return {
                "service": "CyberSentinel AI",
                "status": "online",
                "message": "CyberSentinel AI Web Layer",
                "login": "/static/login.html",
                "health": "/api/health",
            }

        # --------------------------------------------------------------------
        # Health check
        # --------------------------------------------------------------------

        @app.get("/api/health")
        def health_check() -> dict:
            """
            Lightweight liveness endpoint.

            Used to confirm that the web server is responsive.
            """

            return {
                "status": "online",
                "service": "CyberSentinel AI Web Layer",
            }

        return app

    # =========================================================================
    # ROUTER REGISTRATION
    # =========================================================================

    def register_router(
        self,
        router: object,
        prefix: str = ""
    ) -> None:
        """
        Register a FastAPI APIRouter.

        This method is intended for:

            - auth_manager.py
            - api_routes.py
            - websocket_manager.py

        Example:

            manager.register_router(
                auth_router,
                prefix="/api/auth"
            )
        """

        if self._app is None:

            logger.error(
                "Cannot register router - web app was not initialized."
            )

            return

        if router is None:

            logger.warning(
                f"Skipping empty router with prefix '{prefix}'."
            )

            return

        try:

            self._app.include_router(
                router,
                prefix=prefix
            )

            logger.info(
                f"Registered router with prefix "
                f"'{prefix or '/'}'"
            )

        except Exception as exc:

            logger.exception(
                f"Failed to register router "
                f"'{prefix or '/'}': {exc}"
            )

    # =========================================================================
    # STATUS CALLBACK
    # =========================================================================

    def set_status_callback(
        self,
        callback: Callable[[bool, str], None]
    ) -> None:
        """
        Register a callback invoked whenever the server's running state
        changes.

        Callback signature:

            callback(is_running, message)
        """

        self._status_callback = callback

    # =========================================================================
    # AVAILABILITY
    # =========================================================================

    def is_available(self) -> bool:
        """
        Return True if FastAPI/Uvicorn are installed and the application
        was successfully initialized.
        """

        return (
            _WEB_DEPENDENCIES_AVAILABLE
            and self._app is not None
        )

    # =========================================================================
    # RUNNING STATE
    # =========================================================================

    def is_running(self) -> bool:
        """
        Return True if the Uvicorn server is currently running.
        """

        return self._is_running

    # =========================================================================
    # LOCAL IP DETECTION
    # =========================================================================

    def get_local_ip(self) -> str:
        """
        Best-effort detection of this machine's LAN IP.

        This IP is used to display the remote dashboard URL.

        Returns:
            LAN IP address or 127.0.0.1 if detection fails.
        """

        try:

            with socket.socket(
                socket.AF_INET,
                socket.SOCK_DGRAM
            ) as s:

                # No actual application data needs to be sent.
                # This lets Windows determine the preferred interface.

                s.connect(
                    ("8.8.8.8", 80)
                )

                return s.getsockname()[0]

        except OSError:

            return "127.0.0.1"

    # =========================================================================
    # START SERVER
    # =========================================================================

    def start(
        self,
        host: str = "0.0.0.0",
        port: int = 8443
    ) -> bool:
        """
        Start the web server in a background thread.

        Args:
            host:
                Interface to bind to.

                0.0.0.0 = all network interfaces.

            port:
                TCP port used by the dashboard.

        Returns:
            True if started successfully or already running.
            False if web dependencies are unavailable.
        """

        # --------------------------------------------------------------------
        # Check dependencies
        # --------------------------------------------------------------------

        if not self.is_available():

            logger.error(
                "Cannot start web server - dependencies unavailable."
            )

            self._notify_status(
                False,
                "FastAPI/Uvicorn not installed."
            )

            return False

        # --------------------------------------------------------------------
        # Already running
        # --------------------------------------------------------------------

        if self._is_running:

            logger.info(
                "Web server already running."
            )

            return True

        # --------------------------------------------------------------------
        # Store configuration
        # --------------------------------------------------------------------

        self._host = host
        self._port = port

        # --------------------------------------------------------------------
        # Uvicorn configuration
        # --------------------------------------------------------------------

        config = uvicorn.Config(
            app=self._app,
            host=host,
            port=port,
            log_level="warning",
            loop="asyncio",
        )

        # --------------------------------------------------------------------
        # Create Uvicorn server
        # --------------------------------------------------------------------

        self._server = uvicorn.Server(
            config
        )

        # --------------------------------------------------------------------
        # Create background thread
        # --------------------------------------------------------------------

        self._server_thread = threading.Thread(
            target=self._run_server,
            name="CyberSentinelWebServerThread",
            daemon=True,
        )

        # --------------------------------------------------------------------
        # Start thread
        # --------------------------------------------------------------------

        self._server_thread.start()

        self._is_running = True

        # --------------------------------------------------------------------
        # Build dashboard URL
        # --------------------------------------------------------------------

        local_ip = self.get_local_ip()

        url = (
            f"http://{local_ip}:{port}"
        )

        logger.info(
            f"Web server starting at {url}"
        )

        self._notify_status(
            True,
            f"Remote dashboard live at {url}"
        )

        return True

    # =========================================================================
    # SERVER THREAD
    # =========================================================================

    def _run_server(self) -> None:
        """
        Background thread target.

        Runs Uvicorn's asyncio event loop independently from Qt's event loop.
        """

        try:

            if self._server is not None:

                self._server.run()

        except Exception as exc:

            logger.exception(
                f"Web server crashed: {exc}"
            )

        finally:

            self._is_running = False

            self._notify_status(
                False,
                "Web server stopped."
            )

    # =========================================================================
    # STOP SERVER
    # =========================================================================

    def stop(self) -> None:
        """
        Gracefully stop the running Uvicorn server.
        """

        if (
            not self._is_running
            or self._server is None
        ):

            logger.info(
                "Web server is not running."
            )

            return

        # --------------------------------------------------------------------
        # Signal Uvicorn to exit
        # --------------------------------------------------------------------

        self._server.should_exit = True

        # --------------------------------------------------------------------
        # Wait for background thread
        # --------------------------------------------------------------------

        if self._server_thread is not None:

            self._server_thread.join(
                timeout=5.0
            )

        self._is_running = False

        logger.info(
            "Web server stopped."
        )

        self._notify_status(
            False,
            "Remote dashboard stopped."
        )

    # =========================================================================
    # STATUS CALLBACK HELPER
    # =========================================================================

    def _notify_status(
        self,
        is_running: bool,
        message: str
    ) -> None:
        """
        Notify the optional desktop UI callback.

        UI callback exceptions are caught so that they cannot crash the
        web server.
        """

        if self._status_callback is None:

            return

        try:

            self._status_callback(
                is_running,
                message
            )

        except Exception as exc:

            logger.error(
                f"Status callback failed: {exc}"
            )


# ============================================================================
# SINGLETON ACCESSOR
# ============================================================================

def get_web_server_manager() -> WebServerManager:
    """
    Return the singleton WebServerManager instance.
    """

    return WebServerManager()


# ============================================================================
# STANDALONE SMOKE TEST
# ============================================================================

if __name__ == "__main__":

    import time

    # ------------------------------------------------------------------------
    # Create manager
    # ------------------------------------------------------------------------

    manager = get_web_server_manager()

    # ------------------------------------------------------------------------
    # Status callback
    # ------------------------------------------------------------------------

    manager.set_status_callback(
        lambda running, msg:
            print(
                f"[STATUS] running={running} | {msg}"
            )
    )

    # ------------------------------------------------------------------------
    # Start server
    # ------------------------------------------------------------------------

    if not manager.start(
        host="0.0.0.0",
        port=8443
    ):

        print(
            "Failed to start CyberSentinel AI web server."
        )

        raise SystemExit(1)

    # ------------------------------------------------------------------------
    # Keep process alive
    # ------------------------------------------------------------------------

    print(
        "Press Ctrl+C to stop."
    )

    try:

        while True:

            time.sleep(1)

    except KeyboardInterrupt:

        print(
            "\nStopping CyberSentinel AI web server..."
        )

        manager.stop()

        print(
            "Web server stopped."
        )