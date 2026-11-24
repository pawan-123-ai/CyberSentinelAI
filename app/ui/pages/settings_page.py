"""
app/ui/pages/settings_page.py

Settings page for CyberSentinel AI's sidebar navigation.

Responsibilities:
    - AI Model: select and apply/retrain the active AI backend via
      ai.model_loader.ModelLoader, running the (potentially
      multi-second) load/train operation on a background QThread so
      the UI never freezes.
    - Network: select a preferred capture interface, applied via
      app.core.engine.MonitoringEngine.set_preferred_interface(),
      used the next time Start Monitoring is clicked.
    - Alerts & Notifications: set the minimum severity required to
      raise a user-facing alert (security.alert_manager.AlertManager)
      and toggle desktop toast notifications
      (app.core.notifier.DesktopNotifier) on/off.
    - Database: show the current MongoDB connection status and
      provide a manual "Purge Old Records" maintenance action.

Unlike every other page so far, this page receives BOTH the
MonitoringEngine and the DesktopNotifier via constructor injection
from MainWindow, since it needs to control state owned by both.
"""

from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ai.model_loader import ModelBundle, ModelLoader
from ai.predictor import get_predictor
from app.core.engine import MonitoringEngine
from app.core.notifier import DesktopNotifier
from config.settings import AIConfig, ThemeConfig
from database.db_manager import get_database_manager
from security.alert_manager import get_alert_manager
from security.risk_analyzer import SeverityLevel
from utils.logger import get_logger

logger = get_logger(__name__)

# Mirrors the display_name class attribute defined on each backend in
# ai/model_loader.py (RandomForestBackend, XGBoostBackend, etc.).
# Duplicated locally as a plain string map rather than importing the
# backend classes just to read a class attribute, since this mapping
# is purely a UI display concern.
MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "deep_learning": "Deep Learning (Neural Network)",
}

_SEVERITY_OPTIONS: list[SeverityLevel] = [
    SeverityLevel.LOW,
    SeverityLevel.MEDIUM,
    SeverityLevel.HIGH,
    SeverityLevel.CRITICAL,
]


class _ModelReloadWorker(QThread):
    """
    Background worker thread that loads or trains an AI model without
    blocking the Qt event loop.

    ai.model_loader.ModelLoader.load_or_train() is synchronous and can
    take several seconds (synthetic dataset generation + scikit-learn
    training on first use of a given model type) — running it
    directly on the GUI thread would freeze the entire application
    for that duration. This is the first background QThread used
    anywhere in the UI layer; every other live-data path in the app
    instead uses the plain threading.Thread + Qt-signal-bridge pattern
    established in app/core/engine.py, which fits continuous
    background work better than this one-shot operation.
    """

    finished_with_result = Signal(object, str)  # (ModelBundle | None, error_message)

    def __init__(self, model_name: str, force_retrain: bool, parent: Optional[QThread] = None) -> None:
        """
        Args:
            model_name: The model backend key to load/train (one of
                config.settings.AIConfig.AVAILABLE_MODELS).
            force_retrain: If True, skips any cached model on disk and
                trains a fresh one.
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self._model_name = model_name
        self._force_retrain = force_retrain

    def run(self) -> None:
        """Executes the blocking load/train operation on this background thread."""
        try:
            loader = ModelLoader(model_name=self._model_name)
            bundle = loader.load_or_train(force_retrain=self._force_retrain)
            self.finished_with_result.emit(bundle, "")
        except Exception as error:  # broad: surfaces any failure back to the UI thread safely
            logger.error("Model reload worker failed: %s", error)
            self.finished_with_result.emit(None, str(error))


class _SettingsSection(QFrame):
    """
    A rounded card container for one settings section (AI Model,
    Network, Alerts & Notifications, Database), providing a
    consistent title + content layout shared across all four sections.
    """

    def __init__(self, title: str, parent: Optional[QWidget] = None) -> None:
        """
        Args:
            title: The section's heading (e.g., "AI Model").
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.setObjectName("settingsSection")
        self.setStyleSheet(
            f"""
            QFrame#settingsSection {{
                background-color: {ThemeConfig.BACKGROUND_CARD};
                border: 1px solid {ThemeConfig.BORDER_COLOR};
                border-radius: 10px;
            }}
            """
        )

        self._outer_layout = QVBoxLayout(self)
        self._outer_layout.setContentsMargins(20, 16, 20, 18)
        self._outer_layout.setSpacing(12)

        title_label = QLabel(title)
        title_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 13, QFont.Weight.DemiBold))
        title_label.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY}; background: transparent;")
        self._outer_layout.addWidget(title_label)

    def add_row(self, layout: QHBoxLayout) -> None:
        """
        Appends a row (typically a label + control + button) to this
        section's content area.

        Args:
            layout: The row layout to add.
        """
        self._outer_layout.addLayout(layout)

    def add_widget(self, widget: QWidget) -> None:
        """
        Appends a standalone widget (e.g., a status label) to this
        section's content area.

        Args:
            widget: The widget to add.
        """
        self._outer_layout.addWidget(widget)


class SettingsPage(QWidget):
    """
    Sidebar page providing runtime control over the AI model, capture
    interface, alert behavior, and database maintenance.

    Usage:
        page = SettingsPage(engine=self.engine, notifier=self.notifier)
    """

    def __init__(
        self,
        engine: MonitoringEngine,
        notifier: DesktopNotifier,
        parent: Optional[QWidget] = None,
    ) -> None:
        """
        Args:
            engine: The application's shared MonitoringEngine instance.
            notifier: The application's shared DesktopNotifier instance.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._engine = engine
        self._notifier = notifier
        self._db_manager = get_database_manager()
        self._reload_worker: Optional[_ModelReloadWorker] = None

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(24, 20, 24, 20)
        outer_layout.setSpacing(16)

        heading = QLabel("Settings")
        heading.setFont(QFont(ThemeConfig.FONT_FAMILY, 16, QFont.Weight.Bold))
        heading.setStyleSheet(f"color: {ThemeConfig.TEXT_PRIMARY};")
        outer_layout.addWidget(heading)

        outer_layout.addWidget(self._build_model_section())
        outer_layout.addWidget(self._build_network_section())
        outer_layout.addWidget(self._build_alerts_section())
        outer_layout.addWidget(self._build_database_section())
        outer_layout.addStretch()

        logger.debug("SettingsPage initialized.")

    # ------------------------------------------------------------------
    # AI Model section
    # ------------------------------------------------------------------

    def _build_model_section(self) -> _SettingsSection:
        """
        Builds the AI Model settings section: a model selector, an
        Apply button (loads/switches to the selected model), a
        Retrain button (forces fresh training of the currently
        selected model), and a status label reflecting the currently
        active model's metadata.

        Returns:
            The constructed section widget.
        """
        section = _SettingsSection("AI Model")

        self._model_combo = QComboBox()
        for model_key in AIConfig.AVAILABLE_MODELS:
            self._model_combo.addItem(MODEL_DISPLAY_NAMES.get(model_key, model_key), model_key)
        self._style_combo(self._model_combo)

        self._apply_model_button = QPushButton("Apply Model")
        self._apply_model_button.clicked.connect(self._on_apply_model_clicked)
        self._style_button(self._apply_model_button, ThemeConfig.NEON_BLUE)

        self._retrain_model_button = QPushButton("Retrain")
        self._retrain_model_button.clicked.connect(self._on_retrain_model_clicked)
        self._style_button(self._retrain_model_button, ThemeConfig.WARNING_ORANGE)

        row = QHBoxLayout()
        row.addWidget(QLabel("Active model:"))
        row.addWidget(self._model_combo, stretch=1)
        row.addWidget(self._apply_model_button)
        row.addWidget(self._retrain_model_button)
        section.add_row(row)

        self._model_status_label = QLabel()
        self._model_status_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 9))
        self._model_status_label.setStyleSheet(f"color: {ThemeConfig.TEXT_SECONDARY}; background: transparent;")
        self._model_status_label.setWordWrap(True)
        section.add_widget(self._model_status_label)
        self._refresh_model_status_label()

        return section

    def _refresh_model_status_label(self) -> None:
        """Updates the model status label from the predictor's current metadata."""
        metadata = get_predictor().model_metadata
        self._model_status_label.setText(
            f"Currently active: {metadata['model_name']}  |  "
            f"Trained: {metadata['trained_at'][:19].replace('T', ' ')}  |  "
            f"Training accuracy: {metadata['training_accuracy'] * 100:.1f}%  |  "
            f"Samples: {metadata['num_training_samples']:,}"
        )

    def _on_apply_model_clicked(self) -> None:
        """Loads (or trains, if not cached) the currently selected model on a background thread."""
        selected_key = self._model_combo.currentData()
        self._start_model_reload(selected_key, force_retrain=False, busy_text="Loading model...")

    def _on_retrain_model_clicked(self) -> None:
        """Forces fresh training of the currently selected model on a background thread."""
        selected_key = self._model_combo.currentData()
        reply = QMessageBox.question(
            self,
            "Retrain Model",
            f"Retrain the {MODEL_DISPLAY_NAMES.get(selected_key, selected_key)} model from scratch?\n"
            "This discards the cached model on disk and may take a few seconds.",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._start_model_reload(selected_key, force_retrain=True, busy_text="Retraining model...")

    def _start_model_reload(self, model_key: str, force_retrain: bool, busy_text: str) -> None:
        """
        Starts a background _ModelReloadWorker and disables the model
        controls until it completes.

        Args:
            model_key: The model backend key to load/train.
            force_retrain: Whether to force fresh training.
            busy_text: Status text to show while the worker runs.
        """
        self._set_model_controls_busy(True, busy_text)

        self._reload_worker = _ModelReloadWorker(model_name=model_key, force_retrain=force_retrain)
        self._reload_worker.finished_with_result.connect(self._on_model_reload_finished)
        self._reload_worker.start()

    def _on_model_reload_finished(self, bundle: Optional[ModelBundle], error_message: str) -> None:
        """
        Handles _ModelReloadWorker.finished_with_result: applies the
        new bundle to the shared ThreatPredictor singleton on success,
        or shows an error dialog on failure. Re-enables the model
        controls either way.

        Args:
            bundle: The loaded/trained ModelBundle, or None on failure.
            error_message: The error description if bundle is None.
        """
        self._set_model_controls_busy(False)

        if bundle is None:
            QMessageBox.warning(self, "AI Model", f"Failed to load model:\n{error_message}")
            return

        get_predictor().reload_model(model_bundle=bundle)
        self._refresh_model_status_label()
        QMessageBox.information(
            self, "AI Model",
            f"{bundle.metadata.model_name} is now active "
            f"(accuracy: {bundle.metadata.training_accuracy * 100:.1f}%).",
        )

    def _set_model_controls_busy(self, busy: bool, status_text: str = "") -> None:
        """
        Toggles the AI Model section's controls between busy and idle
        states, so the user can't trigger overlapping reload operations.

        Args:
            busy: True to disable controls and show status_text; False
                to re-enable them.
            status_text: Text to show while busy (ignored when busy=False).
        """
        self._model_combo.setEnabled(not busy)
        self._apply_model_button.setEnabled(not busy)
        self._retrain_model_button.setEnabled(not busy)
        if busy:
            self._model_status_label.setText(status_text)
        else:
            self._refresh_model_status_label()

    # ------------------------------------------------------------------
    # Network section
    # ------------------------------------------------------------------

    def _build_network_section(self) -> _SettingsSection:
        """
        Builds the Network settings section: a capture interface
        selector and an Apply/Refresh control pair.

        Returns:
            The constructed section widget.
        """
        section = _SettingsSection("Network")

        self._interface_combo = QComboBox()
        self._style_combo(self._interface_combo)
        self._populate_interfaces()

        apply_button = QPushButton("Apply Interface")
        apply_button.clicked.connect(self._on_apply_interface_clicked)
        self._style_button(apply_button, ThemeConfig.NEON_BLUE)

        refresh_button = QPushButton("⟳")
        refresh_button.setFixedWidth(36)
        refresh_button.clicked.connect(self._populate_interfaces)
        self._style_button(refresh_button, ThemeConfig.NEON_CYAN)

        row = QHBoxLayout()
        row.addWidget(QLabel("Capture interface:"))
        row.addWidget(self._interface_combo, stretch=1)
        row.addWidget(refresh_button)
        row.addWidget(apply_button)
        section.add_row(row)

        note_label = QLabel(
            "Takes effect the next time monitoring is started. Live packet "
            "capture requires Npcap and administrator privileges on Windows."
        )
        note_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 8))
        note_label.setStyleSheet(f"color: {ThemeConfig.TEXT_DISABLED}; background: transparent;")
        note_label.setWordWrap(True)
        section.add_widget(note_label)

        return section

    def _populate_interfaces(self) -> None:
        """Refreshes the interface dropdown from the engine's live interface list."""
        self._interface_combo.clear()
        self._interface_combo.addItem("Auto-detect", None)
        for interface_name in self._engine.get_available_interfaces():
            self._interface_combo.addItem(interface_name, interface_name)

        preferred = self._engine.preferred_interface
        if preferred:
            index = self._interface_combo.findData(preferred)
            if index >= 0:
                self._interface_combo.setCurrentIndex(index)

    def _on_apply_interface_clicked(self) -> None:
        """Applies the selected interface as the engine's preferred capture interface."""
        selected_interface = self._interface_combo.currentData()
        self._engine.set_preferred_interface(selected_interface)
        QMessageBox.information(
            self, "Network",
            f"Preferred interface set to: {selected_interface or 'Auto-detect'}\n"
            "This will be used the next time monitoring starts.",
        )

    # ------------------------------------------------------------------
    # Alerts & Notifications section
    # ------------------------------------------------------------------

    def _build_alerts_section(self) -> _SettingsSection:
        """
        Builds the Alerts & Notifications settings section: a minimum
        alert severity selector and a desktop notification toggle.

        Returns:
            The constructed section widget.
        """
        section = _SettingsSection("Alerts & Notifications")

        alert_manager = get_alert_manager()

        self._severity_combo = QComboBox()
        for level in _SEVERITY_OPTIONS:
            self._severity_combo.addItem(level.value, level)
        current_index = self._severity_combo.findData(alert_manager.get_severity_threshold())
        if current_index >= 0:
            self._severity_combo.setCurrentIndex(current_index)
        self._style_combo(self._severity_combo)

        apply_severity_button = QPushButton("Apply")
        apply_severity_button.clicked.connect(self._on_apply_severity_clicked)
        self._style_button(apply_severity_button, ThemeConfig.NEON_BLUE)

        severity_row = QHBoxLayout()
        severity_row.addWidget(QLabel("Minimum alert severity:"))
        severity_row.addWidget(self._severity_combo, stretch=1)
        severity_row.addWidget(apply_severity_button)
        section.add_row(severity_row)

        self._notifications_checkbox = QCheckBox("Show desktop toast notifications for HIGH/CRITICAL alerts")
        self._notifications_checkbox.setChecked(self._notifier.notifications_enabled)
        self._notifications_checkbox.setFont(QFont(ThemeConfig.FONT_FAMILY, 9))
        self._notifications_checkbox.toggled.connect(self._on_notifications_toggled)
        section.add_widget(self._notifications_checkbox)

        return section

    def _on_apply_severity_clicked(self) -> None:
        """Applies the selected minimum severity threshold to AlertManager."""
        selected_level: SeverityLevel = self._severity_combo.currentData()
        get_alert_manager().set_severity_threshold(selected_level)
        QMessageBox.information(
            self, "Alerts", f"Minimum alert severity set to: {selected_level.value}"
        )

    def _on_notifications_toggled(self, checked: bool) -> None:
        """
        Handles the desktop notifications checkbox: enables or
        disables DesktopNotifier's toast notifications immediately.

        Args:
            checked: The new checkbox state.
        """
        self._notifier.set_notifications_enabled(checked)

    # ------------------------------------------------------------------
    # Database section
    # ------------------------------------------------------------------

    def _build_database_section(self) -> _SettingsSection:
        """
        Builds the Database settings section: a connection status
        label and a manual "Purge Old Records" maintenance action.

        Returns:
            The constructed section widget.
        """
        section = _SettingsSection("Database")

        is_connected = self._db_manager.is_connected()
        status_color = ThemeConfig.SAFE_GREEN if is_connected else ThemeConfig.ALERT_RED
        status_text = "Connected" if is_connected else "Disconnected"

        status_label = QLabel(f"MongoDB status: <b style='color:{status_color};'>{status_text}</b>")
        status_label.setFont(QFont(ThemeConfig.FONT_FAMILY, 10))
        status_label.setStyleSheet("background: transparent;")
        section.add_widget(status_label)

        purge_button = QPushButton("Purge Records Older Than 30 Days")
        purge_button.clicked.connect(self._on_purge_clicked)
        self._style_button(purge_button, ThemeConfig.ALERT_RED)

        row = QHBoxLayout()
        row.addWidget(purge_button)
        row.addStretch()
        section.add_row(row)

        return section

    def _on_purge_clicked(self) -> None:
        """
        Handles the Purge Records button: confirms with the user, then
        calls DatabaseManager.clear_old_records() to delete threats
        and alerts older than the configured retention window.
        """
        reply = QMessageBox.question(
            self, "Purge Old Records",
            "Permanently delete threats and alerts older than 30 days?\n"
            "This cannot be undone.",
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        self._db_manager.clear_old_records()
        QMessageBox.information(self, "Purge Old Records", "Old records have been purged.")

    # ------------------------------------------------------------------
    # Shared styling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _style_combo(combo: QComboBox) -> None:
        """
        Applies consistent dark-theme styling to a QComboBox.

        Args:
            combo: The combo box to style.
        """
        combo.setFont(QFont(ThemeConfig.FONT_FAMILY, 9))
        combo.setStyleSheet(
            f"""
            QComboBox {{
                color: {ThemeConfig.TEXT_PRIMARY};
                background-color: {ThemeConfig.BACKGROUND_SECONDARY};
                border: 1px solid {ThemeConfig.BORDER_COLOR};
                border-radius: 6px;
                padding: 6px 10px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {ThemeConfig.BACKGROUND_SECONDARY};
                color: {ThemeConfig.TEXT_PRIMARY};
                selection-background-color: {ThemeConfig.BACKGROUND_CARD};
            }}
            """
        )

    @staticmethod
    def _style_button(button: QPushButton, color: str) -> None:
        """
        Applies consistent outlined-pill styling to an action button,
        matching the pattern used in app/ui/pages/logs_page.py and
        app/ui/pages/reports_page.py.

        Args:
            button: The button to style.
            color: Hex accent color for the border/text/hover fill.
        """
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setFont(QFont(ThemeConfig.FONT_FAMILY, 9, QFont.Weight.DemiBold))
        button.setStyleSheet(
            f"""
            QPushButton {{
                color: {color};
                border: 1px solid {color};
                border-radius: 6px;
                padding: 6px 14px;
                background-color: {ThemeConfig.BACKGROUND_CARD};
            }}
            QPushButton:hover {{
                background-color: {color};
                color: {ThemeConfig.BACKGROUND_PRIMARY};
            }}
            QPushButton:disabled {{
                color: {ThemeConfig.TEXT_DISABLED};
                border: 1px solid {ThemeConfig.BORDER_COLOR};
                background-color: {ThemeConfig.BACKGROUND_CARD};
            }}
            """
        )