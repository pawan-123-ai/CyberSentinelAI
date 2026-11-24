"""
ai/predictor.py

Live AI inference engine for CyberSentinel AI.

Supported backends:

    - Random Forest
    - XGBoost
    - LightGBM
    - Deep Learning (MLP)

The predictor is backend-independent.

It communicates only with BaseModelBackend through ModelBundle.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from ai.feature_extractor import FeatureVector
from ai.model_loader import (
    ModelBundle,
    get_model_bundle,
)
from config.settings import AIConfig
from network.packet_capture import CapturedPacket
from utils.logger import get_logger


logger = get_logger(__name__)


# ==========================================================================
# Constants
# ==========================================================================


NORMAL_CLASS_NAME: str = "Normal"


# ==========================================================================
# Prediction
# ==========================================================================


@dataclass
class Prediction:
    """
    Result of classifying one network FeatureVector.
    """

    timestamp: datetime

    predicted_class: str

    confidence: float

    is_threat: bool

    class_probabilities: Dict[str, float]

    model_name: str

    feature_vector: FeatureVector = field(
        repr=False
    )

    source_packet: CapturedPacket = field(
        repr=False
    )

    def to_dict(self) -> dict:

        return {
            "timestamp": self.timestamp,

            "predicted_class": self.predicted_class,

            "confidence": round(
                self.confidence,
                4,
            ),

            "is_threat": self.is_threat,

            "class_probabilities": {
                name: round(
                    probability,
                    4,
                )
                for name, probability
                in self.class_probabilities.items()
            },

            "model_name": self.model_name,

            "src_ip": self.source_packet.src_ip,

            "dst_ip": self.source_packet.dst_ip,

            "src_port": self.source_packet.src_port,

            "dst_port": self.source_packet.dst_port,

            "protocol": self.source_packet.protocol,

            "features": self.feature_vector.to_dict(),
        }


# ==========================================================================
# Prediction Statistics
# ==========================================================================


@dataclass
class PredictionStats:
    """
    Rolling prediction statistics for the dashboard.
    """

    total_predictions: int = 0

    threat_count: int = 0

    safe_count: int = 0

    _confidence_sum: float = 0.0

    @property
    def average_confidence(self) -> float:

        if self.total_predictions == 0:

            return 0.0

        return (
            self._confidence_sum
            / self.total_predictions
        )

    def record(
        self,
        prediction: Prediction,
    ) -> None:

        self.total_predictions += 1

        self._confidence_sum += (
            prediction.confidence
        )

        if prediction.is_threat:

            self.threat_count += 1

        else:

            self.safe_count += 1

    def reset(self) -> None:

        self.total_predictions = 0

        self.threat_count = 0

        self.safe_count = 0

        self._confidence_sum = 0.0

    def to_dict(self) -> dict:

        return {
            "total_predictions":
                self.total_predictions,

            "threat_count":
                self.threat_count,

            "safe_count":
                self.safe_count,

            "average_confidence":
                round(
                    self.average_confidence,
                    4,
                ),
        }


# ==========================================================================
# Threat Predictor
# ==========================================================================


class ThreatPredictor:
    """
    Backend-independent live AI inference engine.
    """

    def __init__(
        self,
        model_bundle: Optional[
            ModelBundle
        ] = None,
    ) -> None:

        self._lock = threading.Lock()

        self._bundle = (
            model_bundle
            or get_model_bundle()
        )

        self.stats = PredictionStats()

        logger.info(
            "ThreatPredictor initialized with model '%s'. "
            "Training accuracy: %.2f%%",
            self._bundle.metadata.model_name,
            self._bundle.metadata.training_accuracy
            * 100,
        )

    # ------------------------------------------------------------------
    # Single prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        feature_vector: FeatureVector,
    ) -> Prediction:

        with self._lock:

            try:

                scaled_input = (
                    self._prepare_input(
                        feature_vector
                    )
                )

                probabilities = (
                    self._bundle.backend
                    .predict_proba(
                        scaled_input
                    )[0]
                )

                probabilities = (
                    self._normalize_probabilities(
                        probabilities
                    )
                )

                prediction = (
                    self._build_prediction(
                        feature_vector,
                        probabilities,
                    )
                )

            except Exception as error:

                logger.error(
                    "AI prediction failed. "
                    "Using safe fallback: %s",
                    error,
                )

                prediction = (
                    self._fallback_prediction(
                        feature_vector
                    )
                )

            self.stats.record(
                prediction
            )

            return prediction

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(
        self,
        feature_vectors: List[
            FeatureVector
        ],
    ) -> List[Prediction]:

        if not feature_vectors:

            return []

        with self._lock:

            try:

                raw_arrays = [
                    fv.to_array(
                        self._bundle.metadata
                        .feature_columns
                    )
                    for fv in feature_vectors
                ]

                stacked_input = np.vstack(
                    raw_arrays
                )

                scaled_input = (
                    self._bundle.scaler
                    .transform(
                        stacked_input
                    )
                )

                all_probabilities = (
                    self._bundle.backend
                    .predict_proba(
                        scaled_input
                    )
                )

                predictions = []

                for fv, probabilities in zip(
                    feature_vectors,
                    all_probabilities,
                ):

                    probabilities = (
                        self._normalize_probabilities(
                            probabilities
                        )
                    )

                    predictions.append(
                        self._build_prediction(
                            fv,
                            probabilities,
                        )
                    )

            except Exception as error:

                logger.error(
                    "Batch AI prediction failed. "
                    "Using safe fallback: %s",
                    error,
                )

                predictions = [
                    self._fallback_prediction(
                        fv
                    )
                    for fv in feature_vectors
                ]

            for prediction in predictions:

                self.stats.record(
                    prediction
                )

            return predictions

    # ------------------------------------------------------------------
    # Model reload
    # ------------------------------------------------------------------

    def reload_model(
        self,
        model_bundle: Optional[
            ModelBundle
        ] = None,
        force_retrain: bool = False,
    ) -> None:

        with self._lock:

            self._bundle = (
                model_bundle
                or get_model_bundle(
                    force_retrain=force_retrain
                )
            )

            self.stats.reset()

        logger.info(
            "ThreatPredictor model reloaded: "
            "'%s'",
            self._bundle.metadata.model_name,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_model_name(self) -> str:

        return (
            self._bundle.metadata.model_name
        )

    @property
    def model_identifier(self) -> str:

        return (
            self._bundle.backend
            .__class__.__name__
        )

    @property
    def model_metadata(self) -> dict:

        return (
            self._bundle.metadata.to_dict()
        )

    # ------------------------------------------------------------------
    # Internal input preparation
    # ------------------------------------------------------------------

    def _prepare_input(
        self,
        feature_vector: FeatureVector,
    ) -> np.ndarray:

        raw_array = (
            feature_vector.to_array(
                self._bundle.metadata
                .feature_columns
            )
        )

        reshaped = raw_array.reshape(
            1,
            -1,
        )

        return (
            self._bundle.scaler
            .transform(
                reshaped
            )
        )

    # ------------------------------------------------------------------
    # Probability normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_probabilities(
        probabilities: np.ndarray,
    ) -> np.ndarray:

        probabilities = np.asarray(
            probabilities,
            dtype=np.float64,
        ).flatten()

        if probabilities.size == 0:

            raise ValueError(
                "Model returned empty probability array."
            )

        if not np.all(
            np.isfinite(
                probabilities
            )
        ):

            raise ValueError(
                "Model returned invalid probabilities."
            )

        probabilities = np.clip(
            probabilities,
            0.0,
            1.0,
        )

        total = probabilities.sum()

        if total <= 0:

            raise ValueError(
                "Model returned zero probability."
            )

        return probabilities / total

    # ------------------------------------------------------------------
    # Build prediction
    # ------------------------------------------------------------------

    def _build_prediction(
        self,
        feature_vector: FeatureVector,
        probabilities: np.ndarray,
    ) -> Prediction:

        class_names = (
            self._bundle
            .label_encoder
            .classes_
        )

        if len(probabilities) != len(
            class_names
        ):

            raise ValueError(
                "Model probability count does not "
                "match configured attack classes."
            )

        class_probabilities = {
            str(name): float(probability)
            for name, probability
            in zip(
                class_names,
                probabilities,
            )
        }

        best_index = int(
            np.argmax(
                probabilities
            )
        )

        predicted_class = str(
            class_names[
                best_index
            ]
        )

        confidence = float(
            probabilities[
                best_index
            ]
        )

        is_threat = (
            predicted_class
            != NORMAL_CLASS_NAME
            and confidence
            >= AIConfig.CONFIDENCE_THRESHOLD
        )

        return Prediction(
            timestamp=datetime.now(
                timezone.utc
            ),

            predicted_class=predicted_class,

            confidence=confidence,

            is_threat=is_threat,

            class_probabilities=(
                class_probabilities
            ),

            model_name=(
                self._bundle
                .metadata
                .model_name
            ),

            feature_vector=feature_vector,

            source_packet=(
                feature_vector
                .source_packet
            ),
        )

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    def _fallback_prediction(
        self,
        feature_vector: FeatureVector,
    ) -> Prediction:

        class_names = (
            self._bundle
            .label_encoder
            .classes_
        )

        neutral_probabilities = {
            str(name): 0.0
            for name in class_names
        }

        neutral_probabilities[
            NORMAL_CLASS_NAME
        ] = 1.0

        return Prediction(
            timestamp=datetime.now(
                timezone.utc
            ),

            predicted_class=(
                NORMAL_CLASS_NAME
            ),

            confidence=0.0,

            is_threat=False,

            class_probabilities=(
                neutral_probabilities
            ),

            model_name=(
                self._bundle
                .metadata
                .model_name
            ),

            feature_vector=feature_vector,

            source_packet=(
                feature_vector
                .source_packet
            ),
        )


# ==========================================================================
# Singleton
# ==========================================================================


_predictor_instance: Optional[
    ThreatPredictor
] = None


def get_predictor() -> ThreatPredictor:
    """
    Returns the application-wide predictor.
    """

    global _predictor_instance

    if _predictor_instance is None:

        _predictor_instance = (
            ThreatPredictor()
        )

    return _predictor_instance


def reset_predictor() -> None:
    """
    Clears the predictor singleton.

    Useful when switching models during development/testing.
    """

    global _predictor_instance

    _predictor_instance = None