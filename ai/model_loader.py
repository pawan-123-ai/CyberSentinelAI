"""
ai/model_loader.py

AI model lifecycle management for CyberSentinel AI.

Responsibilities:
    - Define a common backend interface (BaseModelBackend) so the
      active ML model can be swapped (Random Forest, XGBoost,
      LightGBM, Deep Learning) via config.settings.AIConfig.ACTIVE_MODEL
      without changing any other module.
    - Load a previously trained model + scaler from disk if available,
      per model type (each backend gets its own cached artifact file,
      so switching models in Settings never overwrites another
      model's saved state).
    - Otherwise, bootstrap a working model by training on a generated
      synthetic dataset, so the application is fully demoable without
      requiring a real intrusion-detection dataset up front.
    - Persist trained models, scalers, and training metadata to the
      models/ directory via joblib (or, for the Deep Learning backend,
      Keras's native .keras format).

Each non-scikit-learn backend (XGBoost, LightGBM, Deep Learning)
imports its underlying library lazily, inside build(), rather than at
module level. If that library isn't installed, build() raises
NotImplementedError, which ModelBackendFactory catches to gracefully
fall back to Random Forest with a logged warning — consistent with
the fail-safe pattern used throughout the rest of the codebase
(database/db_manager.py, network/packet_capture.py, etc.).

This module does not perform live inference (see ai/predictor.py) —
it only prepares a ready-to-use ModelBundle for the predictor to consume.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ai.feature_extractor import PROTOCOL_ENCODING
from config.settings import AIConfig, PathConfig
from utils.logger import get_logger

logger = get_logger(__name__)

# Number of synthetic samples generated per attack class when
# bootstrapping a fresh model (no real dataset available yet).
_SYNTHETIC_SAMPLES_PER_CLASS: int = 400

# File extension used for each model type's saved artifact. All
# scikit-learn-API-compatible backends (Random Forest, XGBoost,
# LightGBM) are joblib-picklable and use .pkl; the Deep Learning
# backend uses Keras's own .keras single-file format, which is NOT
# a plain pickle and must be saved/loaded via tf.keras APIs directly.
_MODEL_FILE_EXTENSIONS: Dict[str, str] = {
    "random_forest": ".pkl",
    "xgboost": ".pkl",
    "lightgbm": ".pkl",
    "deep_learning": ".keras",
}


# ==========================================================================
# Backend interface + implementations
# ==========================================================================


class BaseModelBackend(abc.ABC):
    """
    Abstract interface every AI model backend must implement.

    This is the extension point referenced throughout the project
    spec: to add a new model type, create a new subclass implementing
    these methods and register it in ModelBackendFactory. No other
    module (predictor, UI, security layer) needs to change.
    """

    display_name: str = "Base Model"

    @abc.abstractmethod
    def build(self) -> None:
        """Instantiates the underlying untrained model object."""
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Trains the model on the given feature matrix and integer labels.

        Args:
            features: 2D array of shape (n_samples, n_features).
            labels: 1D array of shape (n_samples,) of integer class labels.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predicts integer class labels for the given feature matrix.

        Args:
            features: 2D array of shape (n_samples, n_features).

        Returns:
            1D array of shape (n_samples,) of predicted integer labels.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predicts class probabilities for the given feature matrix.

        Args:
            features: 2D array of shape (n_samples, n_features).

        Returns:
            2D array of shape (n_samples, n_classes) of class probabilities.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        """Persists the trained model to disk at the given path."""
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, path: Path) -> bool:
        """
        Loads a previously trained model from disk.

        Returns:
            True if loading succeeded, False otherwise (e.g., file
            missing or incompatible).
        """
        raise NotImplementedError


class RandomForestBackend(BaseModelBackend):
    """
    Random Forest classifier backend, using scikit-learn. The default
    backend per the project spec (AIConfig.ACTIVE_MODEL ==
    "random_forest"). Hyperparameters are sourced from
    AIConfig.RANDOM_FOREST_PARAMS.
    """

    display_name = "Random Forest"

    def __init__(self) -> None:
        self._model: Optional[RandomForestClassifier] = None

    def build(self) -> None:
        """Instantiates a fresh, untrained RandomForestClassifier."""
        self._model = RandomForestClassifier(**AIConfig.RANDOM_FOREST_PARAMS)
        logger.info("RandomForestClassifier built with params: %s", AIConfig.RANDOM_FOREST_PARAMS)

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        if self._model is None:
            self.build()
        self._model.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._ensure_ready()
        return self._model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self._ensure_ready()
        return self._model.predict_proba(features)

    def save(self, path: Path) -> None:
        self._ensure_ready()
        joblib.dump(self._model, path)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            self._model = joblib.load(path)
            return True
        except Exception as error:  # broad: joblib/pickle can raise varied errors
            logger.error("Failed to load Random Forest model from %s: %s", path, error)
            return False

    def _ensure_ready(self) -> None:
        if self._model is None:
            raise RuntimeError(
                "RandomForestBackend used before build()/train()/load() was called."
            )


class XGBoostBackend(BaseModelBackend):
    """
    Gradient-boosted tree classifier backend, using the xgboost
    library. Uses the same scikit-learn-compatible API surface as
    RandomForestBackend, so it's fully joblib-picklable and needs no
    special save/load handling.
    """

    display_name = "XGBoost"

    def __init__(self) -> None:
        self._model = None

    def build(self) -> None:
        """
        Instantiates a fresh, untrained XGBClassifier. Raises
        NotImplementedError (caught by ModelBackendFactory) if xgboost
        is not installed, triggering graceful fallback to Random Forest.
        """
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise NotImplementedError(f"xgboost is not installed: {error}") from error

        self._model = XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        logger.info("XGBClassifier built.")

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        if self._model is None:
            self.build()
        self._model.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._ensure_ready()
        return self._model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self._ensure_ready()
        return self._model.predict_proba(features)

    def save(self, path: Path) -> None:
        self._ensure_ready()
        joblib.dump(self._model, path)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            self._model = joblib.load(path)
            return True
        except Exception as error:
            logger.error("Failed to load XGBoost model from %s: %s", path, error)
            return False

    def _ensure_ready(self) -> None:
        if self._model is None:
            raise RuntimeError("XGBoostBackend used before build()/train()/load() was called.")


class LightGBMBackend(BaseModelBackend):
    """
    Gradient-boosted tree classifier backend, using the lightgbm
    library — generally faster to train than XGBoost on the same
    data, at a small accuracy trade-off. Same scikit-learn-compatible
    API surface as the other tree-based backends.
    """

    display_name = "LightGBM"

    def __init__(self) -> None:
        self._model = None

    def build(self) -> None:
        """
        Instantiates a fresh, untrained LGBMClassifier. Raises
        NotImplementedError (caught by ModelBackendFactory) if
        lightgbm is not installed, triggering graceful fallback to
        Random Forest.
        """
        try:
            from lightgbm import LGBMClassifier
        except ImportError as error:
            raise NotImplementedError(f"lightgbm is not installed: {error}") from error

        self._model = LGBMClassifier(
            n_estimators=250,
            max_depth=-1,
            num_leaves=31,
            learning_rate=0.1,
            objective="multiclass",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        logger.info("LGBMClassifier built.")

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        if self._model is None:
            self.build()
        self._model.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._ensure_ready()
        return self._model.predict(features)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self._ensure_ready()
        return self._model.predict_proba(features)

    def save(self, path: Path) -> None:
        self._ensure_ready()
        joblib.dump(self._model, path)

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            self._model = joblib.load(path)
            return True
        except Exception as error:
            logger.error("Failed to load LightGBM model from %s: %s", path, error)
            return False

    def _ensure_ready(self) -> None:
        if self._model is None:
            raise RuntimeError("LightGBMBackend used before build()/train()/load() was called.")


class DeepLearningBackend(BaseModelBackend):
    """
    Feedforward neural network classifier backend, using
    TensorFlow/Keras. A small fully-connected network (two hidden
    layers with dropout) is sufficient given the low-dimensional
    tabular feature set (AIConfig.FEATURE_COLUMNS has 8 entries) —
    this is a classification network, not a deep architecture, but
    fulfils the project spec's "TensorFlow or PyTorch (future ready)"
    requirement with a genuinely trained model rather than a stub.

    Unlike the other backends, the underlying model is NOT
    joblib-picklable — Keras models are saved/loaded via their own
    .keras single-file format, handled explicitly in save()/load().
    """

    display_name = "Deep Learning (Neural Network)"

    def __init__(self) -> None:
        self._model = None
        self._num_features: int = len(AIConfig.FEATURE_COLUMNS)
        self._num_classes: int = len(AIConfig.ATTACK_CLASSES)

    def build(self) -> None:
        """
        Instantiates a fresh, untrained Keras Sequential model. Raises
        NotImplementedError (caught by ModelBackendFactory) if
        tensorflow is not installed, triggering graceful fallback to
        Random Forest.
        """
        try:
            import tensorflow as tf
        except ImportError as error:
            raise NotImplementedError(f"tensorflow is not installed: {error}") from error

        self._model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(self._num_features,)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(self._num_classes, activation="softmax"),
            ]
        )
        self._model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        logger.info(
            "Deep learning model built (%d input features -> %d classes).",
            self._num_features,
            self._num_classes,
        )

    def train(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Trains the network. Labels must be integer-encoded class
        indices (matching sparse_categorical_crossentropy), which is
        exactly what ModelLoader._train_new_model() already provides
        via its shared LabelEncoder — no special-casing needed there.
        """
        if self._model is None:
            self.build()
        # verbose=0 keeps training silent; this runs on a background
        # QThread (see app/ui/pages/settings_page.py's
        # _ModelReloadWorker) so console noise isn't user-visible anyway.
        self._model.fit(features, labels, epochs=30, batch_size=32, verbose=0)

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._ensure_ready()
        probabilities = self._model.predict(features, verbose=0)
        return np.argmax(probabilities, axis=1)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Returns the network's softmax output directly — it already
        sums to 1.0 per row by construction, matching the same
        semantics as the tree-based backends' predict_proba().
        """
        self._ensure_ready()
        return self._model.predict(features, verbose=0)

    def save(self, path: Path) -> None:
        """
        Persists the model using Keras's native .keras format, NOT
        joblib — a Keras model is not a plain picklable Python object.

        Args:
            path: The target path (expected to already have a .keras
                extension, resolved by ModelLoader per
                _MODEL_FILE_EXTENSIONS).
        """
        self._ensure_ready()
        self._model.save(str(path))

    def load(self, path: Path) -> bool:
        if not path.exists():
            return False
        try:
            import tensorflow as tf

            self._model = tf.keras.models.load_model(str(path))
            return True
        except Exception as error:
            logger.error("Failed to load deep learning model from %s: %s", path, error)
            return False

    def _ensure_ready(self) -> None:
        if self._model is None:
            raise RuntimeError("DeepLearningBackend used before build()/train()/load() was called.")


class ModelBackendFactory:
    """
    Resolves a model name string (from AIConfig.ACTIVE_MODEL) into a
    concrete BaseModelBackend instance.

    If the requested backend's underlying library is not installed
    (build() raises NotImplementedError), falls back to Random Forest
    and logs a clear warning, so a missing optional dependency never
    crashes the application — only that one model choice becomes
    unavailable until the library is installed.
    """

    _REGISTRY: Dict[str, type[BaseModelBackend]] = {
        "random_forest": RandomForestBackend,
        "xgboost": XGBoostBackend,
        "lightgbm": LightGBMBackend,
        "deep_learning": DeepLearningBackend,
    }

    @classmethod
    def create(cls, model_name: str) -> BaseModelBackend:
        """
        Args:
            model_name: One of AIConfig.AVAILABLE_MODELS.

        Returns:
            An instantiated BaseModelBackend subclass. Falls back to
            RandomForestBackend if the requested backend is unknown
            or its underlying library isn't installed.
        """
        backend_class = cls._REGISTRY.get(model_name)
        if backend_class is None:
            logger.warning(
                "Unknown model '%s' requested; falling back to Random Forest.",
                model_name,
            )
            return RandomForestBackend()

        try:
            backend = backend_class()
            if not isinstance(backend, RandomForestBackend):
                # Proactively probe availability for non-default
                # backends so we fall back BEFORE training is
                # attempted, not during it.
                backend.build()
            return backend
        except NotImplementedError:
            logger.warning(
                "Model backend '%s' is unavailable (missing dependency); "
                "falling back to Random Forest.",
                model_name,
            )
            return RandomForestBackend()


# ==========================================================================
# Model bundle + metadata
# ==========================================================================


@dataclass
class ModelMetadata:
    """
    Descriptive information about a trained model, used for display on
    the dashboard's "AI Status" panel and the Settings/About pages.
    """
    model_name: str
    trained_at: datetime
    training_accuracy: float
    num_training_samples: int
    feature_columns: list[str] = field(default_factory=lambda: list(AIConfig.FEATURE_COLUMNS))
    attack_classes: list[str] = field(default_factory=lambda: list(AIConfig.ATTACK_CLASSES))

    def to_dict(self) -> dict:
        """Returns a plain-dict representation for display/logging."""
        return {
            "model_name": self.model_name,
            "trained_at": self.trained_at.isoformat(),
            "training_accuracy": round(self.training_accuracy, 4),
            "num_training_samples": self.num_training_samples,
            "feature_columns": self.feature_columns,
            "attack_classes": self.attack_classes,
        }


@dataclass
class ModelBundle:
    """
    Everything ai/predictor.py needs to run live inference: the
    trained backend, the fitted feature scaler, the label encoder
    mapping class names <-> integer labels, and descriptive metadata.
    """
    backend: BaseModelBackend
    scaler: StandardScaler
    label_encoder: LabelEncoder
    metadata: ModelMetadata


# ==========================================================================
# ModelLoader: orchestrates load-from-disk vs. train-fresh
# ==========================================================================


class ModelLoader:
    """
    Orchestrates preparing a ready-to-use ModelBundle: attempts to
    load a previously trained model + scaler from disk (per model
    type — each backend has its own cached artifact file), and if
    none is found (or force_retrain is requested), trains a fresh
    model on a generated synthetic dataset and persists it.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        """
        Args:
            model_name: The model backend to use. Defaults to
                AIConfig.ACTIVE_MODEL if not provided.
        """
        self._model_name = model_name or AIConfig.ACTIVE_MODEL

    def load_or_train(self, force_retrain: bool = False) -> ModelBundle:
        """
        Main entry point: returns a ready-to-use ModelBundle, loading
        from disk when possible or training fresh otherwise.

        Args:
            force_retrain: If True, skips the disk cache and always
                trains a new model (used by the Settings page's
                "Retrain" button).

        Returns:
            A fully populated ModelBundle ready for inference.
        """
        if not force_retrain:
            cached_bundle = self._load_from_disk()
            if cached_bundle is not None:
                logger.info(
                    "Loaded existing '%s' model from disk (trained %s, accuracy %.2f%%).",
                    cached_bundle.metadata.model_name,
                    cached_bundle.metadata.trained_at.isoformat(),
                    cached_bundle.metadata.training_accuracy * 100,
                )
                return cached_bundle

        logger.info("Training a new '%s' model...", self._model_name)
        return self._train_new_model()

    # ------------------------------------------------------------------
    # File path resolution
    # ------------------------------------------------------------------

    def _resolve_model_file_path(self) -> Path:
        """
        Returns:
            The model artifact file path for this loader's model
            type, e.g. "models/xgboost_model.pkl" or
            "models/deep_learning_model.keras". Each model type gets
            its own file so switching models in Settings never
            overwrites a different model's saved state.
        """
        extension = _MODEL_FILE_EXTENSIONS.get(self._model_name, ".pkl")
        return PathConfig.MODELS_DIR / f"{self._model_name}_model{extension}"

    def _resolve_scaler_file_path(self) -> Path:
        """Returns: The fitted-scaler file path for this loader's model type."""
        return PathConfig.MODELS_DIR / f"{self._model_name}_scaler.pkl"

    def _resolve_metadata_file_path(self) -> Path:
        """Returns: The training-metadata file path for this loader's model type."""
        return PathConfig.MODELS_DIR / f"{self._model_name}_metadata.pkl"

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> Optional[ModelBundle]:
        """
        Attempts to load a previously persisted backend, scaler, and
        metadata from this loader's model-specific file paths.

        Returns:
            A ModelBundle if all artifacts loaded successfully, else None.
        """
        backend = ModelBackendFactory.create(self._model_name)

        model_path = self._resolve_model_file_path()
        if not backend.load(model_path):
            return None

        scaler_path = self._resolve_scaler_file_path()
        metadata_path = self._resolve_metadata_file_path()
        if not scaler_path.exists() or not metadata_path.exists():
            logger.warning(
                "Model file found but scaler/metadata missing for '%s'; retraining instead.",
                self._model_name,
            )
            return None

        try:
            scaler: StandardScaler = joblib.load(scaler_path)
            metadata: ModelMetadata = joblib.load(metadata_path)
        except Exception as error:  # broad: joblib/pickle can raise varied errors
            logger.error("Failed to load scaler/metadata from disk: %s", error)
            return None

        return ModelBundle(
            backend=backend,
            scaler=scaler,
            label_encoder=self._build_label_encoder(),
            metadata=metadata,
        )

    def _save_to_disk(self, bundle: ModelBundle) -> None:
        """
        Persists the backend, scaler, and metadata to this loader's
        model-specific file paths so future application launches (or
        switches back to this model in Settings) can skip retraining.

        Args:
            bundle: The freshly trained ModelBundle to persist.
        """
        try:
            bundle.backend.save(self._resolve_model_file_path())
            joblib.dump(bundle.scaler, self._resolve_scaler_file_path())
            joblib.dump(bundle.metadata, self._resolve_metadata_file_path())
            logger.info("Model artifacts for '%s' saved to %s", self._model_name, PathConfig.MODELS_DIR)
        except Exception as error:  # broad: disk I/O / permissions issues
            logger.error("Failed to persist model artifacts: %s", error)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_new_model(self) -> ModelBundle:
        """
        Generates a synthetic training dataset, fits a scaler and the
        selected backend, evaluates holdout accuracy, and persists
        the result.

        Returns:
            The newly trained ModelBundle.
        """
        raw_features, string_labels = self._generate_synthetic_dataset()

        label_encoder = self._build_label_encoder()
        encoded_labels = label_encoder.transform(string_labels)

        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(raw_features)

        x_train, x_test, y_train, y_test = train_test_split(
            scaled_features,
            encoded_labels,
            test_size=0.2,
            random_state=42,
            stratify=encoded_labels,
        )

        backend = ModelBackendFactory.create(self._model_name)
        backend.build()
        backend.train(x_train, y_train)

        predictions = backend.predict(x_test)
        accuracy = float(accuracy_score(y_test, predictions))

        metadata = ModelMetadata(
            model_name=getattr(backend, "display_name", self._model_name),
            trained_at=datetime.now(timezone.utc),
            training_accuracy=accuracy,
            num_training_samples=int(raw_features.shape[0]),
        )

        bundle = ModelBundle(
            backend=backend,
            scaler=scaler,
            label_encoder=label_encoder,
            metadata=metadata,
        )

        logger.info(
            "Model training complete: %s | samples=%d | holdout accuracy=%.2f%%",
            metadata.model_name,
            metadata.num_training_samples,
            accuracy * 100,
        )

        self._save_to_disk(bundle)
        return bundle

    @staticmethod
    def _build_label_encoder() -> LabelEncoder:
        """
        Builds a LabelEncoder fitted on the fixed, ordered class list
        from AIConfig.ATTACK_CLASSES, rather than deriving classes
        from whatever data happens to be available. This guarantees
        the integer <-> class-name mapping is always identical across
        training runs, application restarts, and model switches.

        Returns:
            A fitted LabelEncoder covering all AIConfig.ATTACK_CLASSES.
        """
        encoder = LabelEncoder()
        encoder.fit(AIConfig.ATTACK_CLASSES)
        return encoder

    @staticmethod
    def _generate_synthetic_dataset(
        samples_per_class: int = _SYNTHETIC_SAMPLES_PER_CLASS,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a plausible synthetic dataset covering every class
        in AIConfig.ATTACK_CLASSES, with per-class feature
        distributions loosely modeled on well-known traffic patterns.
        Used identically regardless of which backend is training, so
        all four algorithms are compared on the exact same data.

        Args:
            samples_per_class: Number of synthetic samples to generate
                for each attack class.

        Returns:
            A tuple (features, labels):
                features: 2D array of shape (n_samples, n_feature_columns).
                labels: 1D array of string class names.
        """
        rng = np.random.default_rng(seed=42)
        tcp = PROTOCOL_ENCODING["TCP"]
        udp = PROTOCOL_ENCODING["UDP"]

        feature_rows: list[np.ndarray] = []
        label_rows: list[str] = []

        def add_samples(
            class_name: str,
            packet_size_range: tuple[float, float],
            protocol_choices: list[int],
            port_range: tuple[int, int],
            duration_range: tuple[float, float],
            packet_rate_range: tuple[float, float],
            byte_rate_range: tuple[float, float],
            flag_count_range: tuple[float, float],
        ) -> None:
            """Draws `samples_per_class` rows for one attack class."""
            n = samples_per_class
            packet_size = rng.uniform(*packet_size_range, size=n)
            protocol_type = rng.choice(protocol_choices, size=n).astype(float)
            src_port = rng.uniform(*port_range, size=n)
            dst_port = rng.uniform(*port_range, size=n)
            duration = rng.uniform(*duration_range, size=n)
            packet_rate = rng.uniform(*packet_rate_range, size=n)
            byte_rate = rng.uniform(*byte_rate_range, size=n)
            flag_count = rng.uniform(*flag_count_range, size=n)

            rows = np.column_stack(
                [packet_size, protocol_type, src_port, dst_port, duration, packet_rate, byte_rate, flag_count]
            )
            feature_rows.append(rows)
            label_rows.extend([class_name] * n)

        add_samples(
            "Normal",
            packet_size_range=(64, 1500), protocol_choices=[tcp, udp], port_range=(1, 49151),
            duration_range=(1.0, 60.0), packet_rate_range=(0.5, 20.0),
            byte_rate_range=(50.0, 5000.0), flag_count_range=(0.0, 4.0),
        )
        add_samples(
            "DoS",
            packet_size_range=(40, 300), protocol_choices=[tcp], port_range=(1, 1024),
            duration_range=(0.1, 5.0), packet_rate_range=(500.0, 3000.0),
            byte_rate_range=(20000.0, 200000.0), flag_count_range=(2.0, 10.0),
        )
        add_samples(
            "Probe",
            packet_size_range=(40, 100), protocol_choices=[tcp], port_range=(1, 65535),
            duration_range=(0.05, 2.0), packet_rate_range=(5.0, 100.0),
            byte_rate_range=(50.0, 2000.0), flag_count_range=(1.0, 2.0),
        )
        add_samples(
            "R2L",
            packet_size_range=(100, 800), protocol_choices=[tcp], port_range=(20, 1024),
            duration_range=(5.0, 120.0), packet_rate_range=(1.0, 15.0),
            byte_rate_range=(100.0, 4000.0), flag_count_range=(2.0, 6.0),
        )
        add_samples(
            "U2R",
            packet_size_range=(40, 400), protocol_choices=[tcp], port_range=(1, 1024),
            duration_range=(0.5, 30.0), packet_rate_range=(0.1, 5.0),
            byte_rate_range=(20.0, 1000.0), flag_count_range=(0.0, 3.0),
        )
        add_samples(
            "Port Scan",
            packet_size_range=(40, 80), protocol_choices=[tcp], port_range=(1, 65535),
            duration_range=(0.01, 0.5), packet_rate_range=(20.0, 200.0),
            byte_rate_range=(100.0, 3000.0), flag_count_range=(1.0, 1.0),
        )
        add_samples(
            "DDoS",
            packet_size_range=(40, 200), protocol_choices=[tcp, udp], port_range=(1, 65535),
            duration_range=(0.01, 1.0), packet_rate_range=(2000.0, 10000.0),
            byte_rate_range=(100000.0, 1000000.0), flag_count_range=(3.0, 15.0),
        )

        features = np.vstack(feature_rows).astype(np.float32)
        labels = np.array(label_rows, dtype=object)

        shuffle_indices = rng.permutation(len(labels))
        return features[shuffle_indices], labels[shuffle_indices]


# ==========================================================================
# Module-level singleton accessor
# ==========================================================================

_model_bundle_instance: Optional[ModelBundle] = None


def get_model_bundle(force_retrain: bool = False) -> ModelBundle:
    """
    Returns the shared, application-wide ModelBundle, loading or
    training it on first call.

    Args:
        force_retrain: If True, discards any cached bundle and trains
            a fresh model.

    Returns:
        The singleton ModelBundle instance.
    """
    global _model_bundle_instance
    if _model_bundle_instance is None or force_retrain:
        loader = ModelLoader()
        _model_bundle_instance = loader.load_or_train(force_retrain=force_retrain)
    return _model_bundle_instance