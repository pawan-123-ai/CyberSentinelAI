"""
ai/feature_extractor.py

Converts raw CapturedPacket objects into numeric FeatureVector
instances for the AI threat detection system.

Traffic state and behavioral tracking are delegated to
network.traffic_analyzer.TrafficAnalyzer.

Responsibilities:
    - Encode network protocols into stable numeric values.
    - Convert TrafficSnapshot data into FeatureVector.
    - Preserve the original packet for traceability.
    - Provide AI-ready numeric features.

This module does NOT:
    - Maintain its own flow table.
    - Maintain source-IP tracking.
    - Perform ML inference.
    - Calculate risk scores.
    - Generate alerts.

Architecture:

    CapturedPacket
          |
          v
    TrafficAnalyzer
          |
          v
    TrafficSnapshot
          |
          v
    FeatureExtractor
          |
          v
    FeatureVector
          |
          v
    ThreatPredictor
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config.settings import AIConfig
from network.packet_capture import CapturedPacket
from network.traffic_analyzer import (
    TrafficAnalyzer,
    TrafficSnapshot,
    get_traffic_analyzer,
)
from utils.logger import get_logger


logger = get_logger(__name__)


# ==========================================================================
# Protocol Encoding
# ==========================================================================

# IMPORTANT:
# This mapping is an AI-specific concern.
# It must remain here because model training and live inference
# must use exactly the same encoding.

PROTOCOL_ENCODING: Dict[str, int] = {
    "TCP": 0,
    "UDP": 1,
    "ICMP": 2,
    "OTHER": 3,
}


# ==========================================================================
# Feature Vector
# ==========================================================================


@dataclass
class FeatureVector:
    """
    Numeric representation of one network packet plus its derived
    traffic-behavior features.

    The field names intentionally match AIConfig.FEATURE_COLUMNS.

    Current AI feature schema:

        1. packet_size
        2. protocol_type
        3. src_port
        4. dst_port
        5. duration
        6. packet_rate
        7. byte_rate
        8. flag_count
        9. unique_dst_ports
        10. unique_dst_ips
        11. connection_rate

    The original CapturedPacket is retained separately so the security
    layer can report source/destination information.
    """

    packet_size: float

    protocol_type: float

    src_port: float

    dst_port: float

    duration: float

    packet_rate: float

    byte_rate: float

    flag_count: float

    unique_dst_ports: float

    unique_dst_ips: float

    connection_rate: float

    source_packet: CapturedPacket = field(
        repr=False
    )

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_array(
        self,
        feature_columns: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Converts the FeatureVector into a 1D numpy array.

        The ordering is controlled by AIConfig.FEATURE_COLUMNS.
        """

        columns = (
            feature_columns
            or AIConfig.FEATURE_COLUMNS
        )

        values = [
            float(
                getattr(self, column)
            )
            for column in columns
        ]

        return np.array(
            values,
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Dictionary
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Converts the AI features into a plain dictionary.

        The original CapturedPacket is intentionally excluded.
        """

        return {
            "packet_size": self.packet_size,

            "protocol_type": self.protocol_type,

            "src_port": self.src_port,

            "dst_port": self.dst_port,

            "duration": self.duration,

            "packet_rate": self.packet_rate,

            "byte_rate": self.byte_rate,

            "flag_count": self.flag_count,

            "unique_dst_ports": (
                self.unique_dst_ports
            ),

            "unique_dst_ips": (
                self.unique_dst_ips
            ),

            "connection_rate": (
                self.connection_rate
            ),
        }


# ==========================================================================
# Feature Extractor
# ==========================================================================


class FeatureExtractor:
    """
    Converts CapturedPacket objects into FeatureVector instances.

    All traffic state is delegated to the shared TrafficAnalyzer.

    This guarantees that:

        AI feature extraction
        Behavior analysis
        Source-IP analysis
        Flow analysis

    all observe the same underlying traffic state.
    """

    def __init__(
        self,
        traffic_analyzer: Optional[
            TrafficAnalyzer
        ] = None,
    ) -> None:
        """
        Args:
            traffic_analyzer:
                Optional TrafficAnalyzer instance.

                If omitted, the application-wide singleton is used.
        """

        self._traffic_analyzer = (
            traffic_analyzer
            or get_traffic_analyzer()
        )

        logger.info(
            "FeatureExtractor initialized."
        )

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def extract(
        self,
        packet: CapturedPacket,
    ) -> FeatureVector:
        """
        Processes a captured packet.

        TrafficAnalyzer performs the stateful analysis.
        FeatureExtractor maps the resulting snapshot to the AI
        feature schema.

        Args:
            packet:
                Parsed packet from PacketCapture.

        Returns:
            FeatureVector ready for model inference.
        """

        snapshot: TrafficSnapshot = (
            self._traffic_analyzer.analyze(
                packet
            )
        )

        return FeatureVector(
            packet_size=float(
                packet.size
            ),

            protocol_type=float(
                self._encode_protocol(
                    packet.protocol
                )
            ),

            src_port=float(
                packet.src_port
            ),

            dst_port=float(
                packet.dst_port
            ),

            duration=float(
                snapshot.flow_duration
            ),

            packet_rate=float(
                snapshot.flow_packet_rate
            ),

            byte_rate=float(
                snapshot.flow_byte_rate
            ),

            flag_count=float(
                snapshot.flow_flag_count
            ),

            unique_dst_ports=float(
                snapshot.unique_dst_ports
            ),

            unique_dst_ips=float(
                snapshot.unique_dst_ips
            ),

            connection_rate=float(
                snapshot.connection_rate
            ),

            source_packet=packet,
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Resets the underlying TrafficAnalyzer.

        This should be called when monitoring is stopped and restarted.
        """

        self._traffic_analyzer.reset()

        logger.info(
            "FeatureExtractor reset."
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def active_flow_count(self) -> int:
        """
        Returns the number of active flows tracked by TrafficAnalyzer.
        """

        return (
            self._traffic_analyzer.active_flow_count
        )

    @property
    def tracked_source_ip_count(self) -> int:
        """
        Returns the number of currently tracked source IPs.
        """

        return (
            self._traffic_analyzer
            .tracked_source_ip_count
        )

    # ------------------------------------------------------------------
    # Protocol encoding
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_protocol(
        protocol: str,
    ) -> int:
        """
        Converts a protocol string into the canonical numeric encoding.
        """

        return PROTOCOL_ENCODING.get(
            protocol,
            PROTOCOL_ENCODING["OTHER"],
        ) 