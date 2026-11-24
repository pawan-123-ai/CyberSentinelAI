"""
app/ui/login_panel.py

Embedded login/signup panel for CyberSentinel AI.

Features:
    - Sign In using username OR email
    - Create Account using username + email + password
    - First-run signup mode when no users exist
    - Password visibility toggle
    - Keyboard-friendly form
    - Delegates authentication entirely to AuthManager
    - Emits:
          authenticated(username, role, session_token)

This file contains UI only.
Authentication rules and MongoDB persistence remain in
web.auth_manager.AuthManager.
"""

from typing import Optional

from PySide6.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    Qt,
    Signal,
)
from PySide6.QtGui import (
    QFont,
    QShowEvent,
)
from PySide6.QtWidgets import (
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from utils.logger import get_logger
from web.auth_manager import get_auth_manager

logger = get_logger(__name__)


class LoginPanel(QWidget):
    """
    Full-page login/signup widget.

    Signals:
        authenticated(str, str, str):
            Emits (username, role, session_token) after successful
            authentication.
    """

    authenticated = Signal(str, str, str)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)

        self.setAttribute(
            Qt.WidgetAttribute.WA_StyledBackground,
            True,
        )

        # ------------------------------------------------------------------
        # Authentication manager
        # ------------------------------------------------------------------

        self._auth = get_auth_manager()

        # ------------------------------------------------------------------
        # Animation state
        # ------------------------------------------------------------------

        self._entrance_played = False
        self._entrance_anim = None

        # ------------------------------------------------------------------
        # First-run behavior
        #
        # If MongoDB contains no users, the application starts directly
        # in Create Account mode.
        # ------------------------------------------------------------------

        self._signup_mode: bool = (
            not self._auth.has_any_users()
        )

        # ------------------------------------------------------------------
        # Build interface
        # ------------------------------------------------------------------

        self._build_ui()
        self._apply_styles()
        self._wire_events()
        self._refresh_mode()

    # ======================================================================
    # UI construction
    # ======================================================================

    def _build_ui(self) -> None:
        """Construct the complete login/signup interface."""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        outer.addStretch(1)

        center_row = QHBoxLayout()
        center_row.addStretch(1)

        # ------------------------------------------------------------------
        # Main authentication card
        # ------------------------------------------------------------------

        self.card = QWidget(
            objectName="card"
        )

        self.card.setFixedWidth(420)

        card_layout = QVBoxLayout(
            self.card
        )

        card_layout.setContentsMargins(
            36,
            30,
            36,
            28,
        )

        card_layout.setSpacing(10)

        center_row.addWidget(self.card)
        center_row.addStretch(1)

        outer.addLayout(center_row)

        outer.addStretch(1)

        # ==================================================================
        # Branding
        # ==================================================================

        self.icon_label = QLabel(
            "🛡",
            objectName="iconLabel",
        )

        self.icon_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.title_label = QLabel(
            objectName="titleLabel"
        )

        self.title_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.subtitle_label = QLabel(
            objectName="subtitleLabel"
        )

        self.subtitle_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        card_layout.addWidget(
            self.icon_label
        )

        card_layout.addWidget(
            self.title_label
        )

        card_layout.addWidget(
            self.subtitle_label
        )

        card_layout.addSpacing(8)

        # ==================================================================
        # Error banner
        # ==================================================================

        self.error_label = QLabel(
            "",
            objectName="errorLabel",
        )

        self.error_label.setWordWrap(True)

        self.error_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.error_label.hide()

        card_layout.addWidget(
            self.error_label
        )

        # ==================================================================
        # Username / login identifier
        # ==================================================================

        self.username_label = self._field_label(
            "USERNAME"
        )

        self.username_input = QLineEdit(
            objectName="inputField"
        )

        self.username_input.setPlaceholderText(
            "Username or email address"
        )

        self.username_input.setClearButtonEnabled(
            True
        )

        card_layout.addWidget(
            self.username_label
        )

        card_layout.addWidget(
            self.username_input
        )

        # ==================================================================
        # Email address
        #
        # Visible only during account creation.
        # ==================================================================

        self.email_label = self._field_label(
            "EMAIL ADDRESS"
        )

        self.email_input = QLineEdit(
            objectName="inputField"
        )

        self.email_input.setPlaceholderText(
            "Enter your email address"
        )

        self.email_input.setClearButtonEnabled(
            True
        )

        # Hidden initially; _refresh_mode() controls visibility.
        self.email_label.hide()
        self.email_input.hide()

        card_layout.addWidget(
            self.email_label
        )

        card_layout.addWidget(
            self.email_input
        )

        # ==================================================================
        # Password
        # ==================================================================

        password_label = self._field_label(
            "PASSWORD"
        )

        card_layout.addWidget(
            password_label
        )

        password_row = QHBoxLayout()
        password_row.setSpacing(0)

        self.password_input = QLineEdit(
            objectName="inputField"
        )

        self.password_input.setPlaceholderText(
            "Enter your password"
        )

        self.password_input.setEchoMode(
            QLineEdit.EchoMode.Password
        )

        self.toggle_pw_btn = QPushButton(
            "Show",
            objectName="toggleBtn",
        )

        self.toggle_pw_btn.setFixedWidth(
            58
        )

        self.toggle_pw_btn.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        password_row.addWidget(
            self.password_input
        )

        password_row.addWidget(
            self.toggle_pw_btn
        )

        card_layout.addLayout(
            password_row
        )

        # ==================================================================
        # Confirm password
        #
        # Signup only.
        # ==================================================================

        self.confirm_label = self._field_label(
            "CONFIRM PASSWORD"
        )

        self.confirm_input = QLineEdit(
            objectName="inputField"
        )

        self.confirm_input.setPlaceholderText(
            "Re-enter your password"
        )

        self.confirm_input.setEchoMode(
            QLineEdit.EchoMode.Password
        )

        card_layout.addWidget(
            self.confirm_label
        )

        card_layout.addWidget(
            self.confirm_input
        )

        card_layout.addSpacing(4)

        # ==================================================================
        # Submit button
        # ==================================================================

        self.submit_btn = QPushButton(
            objectName="submitBtn"
        )

        self.submit_btn.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        self.submit_btn.setFixedHeight(
            42
        )

        card_layout.addWidget(
            self.submit_btn
        )

        # ==================================================================
        # Login / signup switch
        # ==================================================================

        self.toggle_mode_btn = QPushButton(
            objectName="modeToggleBtn"
        )

        self.toggle_mode_btn.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        self.toggle_mode_btn.setFlat(
            True
        )

        card_layout.addWidget(
            self.toggle_mode_btn,
            alignment=Qt.AlignmentFlag.AlignCenter,
        )

        card_layout.addSpacing(6)

        # ==================================================================
        # Footer
        # ==================================================================

        self.footer_label = QLabel(
            "Protected access · Secure session",
            objectName="footerLabel",
        )

        self.footer_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        card_layout.addWidget(
            self.footer_label
        )

    @staticmethod
    def _field_label(
        text: str,
    ) -> QLabel:
        """Create a small uppercase form label."""

        return QLabel(
            text,
            objectName="fieldLabel",
        )

    # ======================================================================
    # Styling
    # ======================================================================

    def _apply_styles(self) -> None:
        """Apply CyberSentinel's dark neon-blue authentication theme."""

        self.setFont(
            QFont(
                "Segoe UI",
                10,
            )
        )

        self.setStyleSheet(
            """
            LoginPanel {
                background-color: #070b12;
            }

            QWidget#card {
                background-color: #0d1420;
                border: 1px solid rgba(56, 189, 248, 0.25);
                border-radius: 16px;
            }

            QLabel#iconLabel {
                font-size: 32px;
                margin-top: 4px;
            }

            QLabel#titleLabel {
                color: #e6edf5;
                font-size: 20px;
                font-weight: 600;
            }

            QLabel#subtitleLabel {
                color: #8291a8;
                font-size: 11px;
            }

            QLabel#fieldLabel {
                color: #8291a8;
                font-size: 11px;
                font-weight: 600;
                margin-top: 4px;
            }

            QLineEdit#inputField {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 8px;
                padding: 9px 10px;
                color: #e6edf5;
                font-size: 13px;
                selection-background-color: #155e75;
            }

            QLineEdit#inputField:focus {
                border: 1px solid #22d3ee;
                background-color: rgba(34, 211, 238, 0.05);
            }

            QPushButton#toggleBtn {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-left: none;
                border-top-right-radius: 8px;
                border-bottom-right-radius: 8px;
                color: #8291a8;
                font-size: 11px;
            }

            QPushButton#toggleBtn:hover {
                color: #22d3ee;
            }

            QPushButton#submitBtn {
                background-color: #22d3ee;
                color: #04141c;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
            }

            QPushButton#submitBtn:hover {
                background-color: #38bdf8;
            }

            QPushButton#submitBtn:disabled {
                background-color: rgba(34, 211, 238, 0.35);
                color: rgba(4, 20, 28, 0.6);
            }

            QPushButton#modeToggleBtn {
                background: transparent;
                color: #22d3ee;
                font-size: 12px;
                border: none;
                padding: 4px;
            }

            QPushButton#modeToggleBtn:hover {
                color: #67e8f9;
                text-decoration: underline;
            }

            QLabel#errorLabel {
                background-color: rgba(244, 63, 94, 0.1);
                border: 1px solid rgba(244, 63, 94, 0.35);
                color: #fda4af;
                border-radius: 8px;
                padding: 8px;
                font-size: 12px;
            }

            QLabel#footerLabel {
                color: #4b5768;
                font-size: 10px;
            }
            """
        )

    # ======================================================================
    # Events
    # ======================================================================

    def _wire_events(self) -> None:
        """Connect UI signals to handlers."""

        # Password visibility.
        self.toggle_pw_btn.clicked.connect(
            self._toggle_password_visibility
        )

        # Login/signup mode.
        self.toggle_mode_btn.clicked.connect(
            self._toggle_mode
        )

        # Submit.
        self.submit_btn.clicked.connect(
            self._handle_submit
        )

        # Keyboard submission.
        self.username_input.returnPressed.connect(
            self._handle_submit
        )

        self.email_input.returnPressed.connect(
            self._handle_submit
        )

        self.password_input.returnPressed.connect(
            self._handle_submit
        )

        self.confirm_input.returnPressed.connect(
            self._handle_submit
        )

    # ======================================================================
    # Mode switching
    # ======================================================================

    def _toggle_mode(self) -> None:
        """Switch between Sign In and Create Account."""

        self._signup_mode = not self._signup_mode

        self._clear_error()
        self._reset_fields()
        self._refresh_mode()

        self.username_input.setFocus()

    def _refresh_mode(self) -> None:
        """Refresh the UI according to the active authentication mode."""

        if self._signup_mode:

            # ----------------------------------------------------------------
            # CREATE ACCOUNT
            # ----------------------------------------------------------------

            self.title_label.setText(
                "Create Account"
            )

            self.subtitle_label.setText(
                "Set up your CyberSentinel AI account"
            )

            self.submit_btn.setText(
                "Create Account"
            )

            self.toggle_mode_btn.setText(
                "Already have an account? Sign in"
            )

            self.username_label.setText(
                "USERNAME"
            )

            self.username_input.setPlaceholderText(
                "Choose a username"
            )

            # Email visible during signup.
            self.email_label.show()
            self.email_input.show()

            self.confirm_label.show()
            self.confirm_input.show()

            self.footer_label.setText(
                "Your credentials are securely protected"
            )

        else:

            # ----------------------------------------------------------------
            # SIGN IN
            # ----------------------------------------------------------------

            self.title_label.setText(
                "CyberSentinel AI"
            )

            self.subtitle_label.setText(
                "AI-Powered Network Defense"
            )

            self.submit_btn.setText(
                "Sign In"
            )

            self.toggle_mode_btn.setText(
                "New here? Create an account"
            )

            self.username_label.setText(
                "USERNAME OR EMAIL"
            )

            self.username_input.setPlaceholderText(
                "Username or email address"
            )

            # Email field isn't required during login.
            self.email_label.hide()
            self.email_input.hide()

            # Confirm password isn't required during login.
            self.confirm_label.hide()
            self.confirm_input.hide()

            self.footer_label.setText(
                "Protected access · Secure session"
            )

        # ------------------------------------------------------------------
        # On a fresh installation, there is no account to sign in to.
        #
        # Therefore the mode-switch button is hidden until at least one
        # account exists.
        # ------------------------------------------------------------------

        self.toggle_mode_btn.setVisible(
            self._auth.has_any_users()
        )

    # ======================================================================
    # Password visibility
    # ======================================================================

    def _toggle_password_visibility(self) -> None:
        """Show/hide password fields."""

        is_hidden = (
            self.password_input.echoMode()
            == QLineEdit.EchoMode.Password
        )

        new_mode = (
            QLineEdit.EchoMode.Normal
            if is_hidden
            else QLineEdit.EchoMode.Password
        )

        self.password_input.setEchoMode(
            new_mode
        )

        self.confirm_input.setEchoMode(
            new_mode
        )

        self.toggle_pw_btn.setText(
            "Hide"
            if is_hidden
            else "Show"
        )

    # ======================================================================
    # Submission
    # ======================================================================

    def _handle_submit(self) -> None:
        """Route the form to signup or login."""

        self._clear_error()

        identifier = (
            self.username_input.text().strip()
        )

        password = self.password_input.text()

        # ------------------------------------------------------------------
        # Basic required fields
        # ------------------------------------------------------------------

        if not identifier or not password:
            self._show_error(
                "Please enter your username/email and password."
            )
            return

        # ------------------------------------------------------------------
        # Signup
        # ------------------------------------------------------------------

        if self._signup_mode:

            email = (
                self.email_input.text().strip()
            )

            if not email:
                self._show_error(
                    "Please enter your email address."
                )
                self.email_input.setFocus()
                return

            self._handle_signup(
                identifier,
                email,
                password,
            )

        # ------------------------------------------------------------------
        # Login
        # ------------------------------------------------------------------

        else:

            self._handle_login(
                identifier,
                password,
            )

    # ======================================================================
    # Signup
    # ======================================================================

    def _handle_signup(
        self,
        username: str,
        email: str,
        password: str,
    ) -> None:
        """
        Create a MongoDB-backed account and automatically sign in.
        """

        confirm_password = (
            self.confirm_input.text()
        )

        # ------------------------------------------------------------------
        # Confirm password
        # ------------------------------------------------------------------

        if password != confirm_password:
            self._show_error(
                "Passwords do not match."
            )

            self.confirm_input.setFocus()
            return

        # ------------------------------------------------------------------
        # Disable button during registration
        # ------------------------------------------------------------------

        self.submit_btn.setEnabled(
            False
        )

        self.submit_btn.setText(
            "Creating account..."
        )

        try:

            error = self._auth.register_user(
                username=username,
                email=email,
                password=password,
            )

        except TypeError:
            # Compatibility guard for an older AuthManager.
            #
            # The current AuthManager should expose:
            # register_user(username, email, password)
            #
            # This fallback intentionally does NOT alter the MongoDB
            # architecture.
            try:
                error = self._auth.register_user(
                    username,
                    email,
                    password,
                )
            except Exception:
                logger.exception(
                    "Unexpected account registration error."
                )

                error = (
                    "Unable to create the account right now."
                )

        except Exception:
            logger.exception(
                "Unexpected account registration error."
            )

            error = (
                "Unable to create the account right now."
            )

        # ------------------------------------------------------------------
        # Registration failed
        # ------------------------------------------------------------------

        if error:

            self.submit_btn.setEnabled(
                True
            )

            self.submit_btn.setText(
                "Create Account"
            )

            self._show_error(
                error
            )

            return

        # ------------------------------------------------------------------
        # Registration succeeded.
        #
        # Automatically authenticate the newly created account using its
        # username. This also works with the AuthManager's username/email
        # login implementation.
        # ------------------------------------------------------------------

        try:

            token = self._auth.verify_login(
                username,
                password,
            )

        except Exception:
            logger.exception(
                "Automatic sign-in failed after registration."
            )

            token = None

        self.submit_btn.setEnabled(
            True
        )

        self.submit_btn.setText(
            "Create Account"
        )

        self._complete_login(
            token
        )

    # ======================================================================
    # Login
    # ======================================================================

    def _handle_login(
        self,
        identifier: str,
        password: str,
    ) -> None:
        """
        Sign in using either username or email.
        """

        self.submit_btn.setEnabled(
            False
        )

        self.submit_btn.setText(
            "Signing in..."
        )

        try:

            token = self._auth.verify_login(
                identifier,
                password,
            )

        except Exception:
            logger.exception(
                "Unexpected login error."
            )

            token = None

        self.submit_btn.setEnabled(
            True
        )

        self.submit_btn.setText(
            "Sign In"
        )

        if token is None:

            self._show_error(
                "Invalid username/email or password, "
                "or the account is temporarily locked."
            )

            return

        self._complete_login(
            token
        )

    # ======================================================================
    # Login completion
    # ======================================================================

    def _complete_login(
        self,
        token: Optional[str],
    ) -> None:
        """
        Complete authentication and emit the existing signal.
        """

        # ------------------------------------------------------------------
        # No session token
        # ------------------------------------------------------------------

        if token is None:

            self._show_error(
                "Account created, but automatic sign-in failed. "
                "Please sign in manually."
            )

            self._signup_mode = False

            self._refresh_mode()

            return

        # ------------------------------------------------------------------
        # Retrieve session information
        # ------------------------------------------------------------------

        username = (
            self._auth.get_username_for_session(
                token
            )
        )

        role = (
            self._auth.get_role_for_session(
                token
            )
        )

        # ------------------------------------------------------------------
        # Session could not be loaded
        # ------------------------------------------------------------------

        if not username:

            self._show_error(
                "Authentication completed, but the user session "
                "could not be loaded."
            )

            return

        logger.info(
            "Desktop login successful for '%s' (role=%s).",
            username,
            role,
        )

        # ------------------------------------------------------------------
        # Clear form
        # ------------------------------------------------------------------

        self._reset_fields()

        # ------------------------------------------------------------------
        # IMPORTANT:
        #
        # Do NOT change this signal.
        #
        # MainWindow already expects:
        #
        #     authenticated(username, role, session_token)
        # ------------------------------------------------------------------

        self.authenticated.emit(
            username,
            role or "standard",
            token,
        )

    # ======================================================================
    # Form reset
    # ======================================================================

    def _reset_fields(self) -> None:
        """Clear authentication form fields."""

        self.username_input.clear()
        self.email_input.clear()
        self.password_input.clear()
        self.confirm_input.clear()

        self._clear_error()

        # ------------------------------------------------------------------
        # Always return password fields to secure mode.
        # ------------------------------------------------------------------

        self.password_input.setEchoMode(
            QLineEdit.EchoMode.Password
        )

        self.confirm_input.setEchoMode(
            QLineEdit.EchoMode.Password
        )

        self.toggle_pw_btn.setText(
            "Show"
        )

    def reset_for_next_login(self) -> None:
        """
        Reset the authentication page after logout/switch account.
        """

        self._signup_mode = False

        self._reset_fields()

        self._refresh_mode()

        self.username_input.setFocus()

    # ======================================================================
    # Error handling
    # ======================================================================

    def _show_error(
        self,
        message: str,
    ) -> None:
        """Display an authentication error."""

        self.error_label.setText(
            message
        )

        self.error_label.show()

        self.password_input.clear()
        self.confirm_input.clear()

        self.password_input.setFocus()

    def _clear_error(self) -> None:
        """Hide the error banner."""

        self.error_label.clear()

        self.error_label.hide()

    # ======================================================================
    # Entrance animation
    # ======================================================================

    def showEvent(
        self,
        event: QShowEvent,
    ) -> None:
        """
        Fade the authentication card in the first time it appears.
        """

        super().showEvent(
            event
        )

        if not self._entrance_played:

            self._entrance_played = True

            effect = QGraphicsOpacityEffect(
                self.card
            )

            self.card.setGraphicsEffect(
                effect
            )

            self._entrance_anim = QPropertyAnimation(
                effect,
                b"opacity",
            )

            self._entrance_anim.setDuration(
                280
            )

            self._entrance_anim.setStartValue(
                0.0
            )

            self._entrance_anim.setEndValue(
                1.0
            )

            self._entrance_anim.setEasingCurve(
                QEasingCurve.Type.OutCubic
            )

            self._entrance_anim.start()


# ==========================================================================
# Standalone smoke test
# ==========================================================================

if __name__ == "__main__":

    import sys

    from PySide6.QtWidgets import (
        QApplication,
        QMainWindow,
    )

    app = QApplication(
        sys.argv
    )

    window = QMainWindow()

    window.setWindowTitle(
        "CyberSentinel AI - Login Panel Test"
    )

    window.resize(
        1000,
        700,
    )

    panel = LoginPanel()

    panel.authenticated.connect(
        lambda username, role, token: print(
            f"AUTHENTICATED: "
            f"username={username} "
            f"role={role} "
            f"token={token[:16]}..."
        )
    )

    window.setCentralWidget(
        panel
    )

    window.show()

    sys.exit(
        app.exec()
    )