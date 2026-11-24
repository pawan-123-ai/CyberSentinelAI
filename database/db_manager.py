"""
database/db_manager.py

MongoDB persistence layer for CyberSentinel AI.

This module is the single point of contact between the application
and the database. No other module should import pymongo directly —
they should always go through DatabaseManager, so the underlying
database technology can be swapped later without touching business
logic (e.g., migrating to MongoDB Atlas for the future Cloud
Dashboard / Multi-PC Monitoring feature).

The connection is thread-safe (pymongo's MongoClient manages its own
internal connection pool), so a single shared DatabaseManager instance
can safely be used from the packet capture thread, the AI inference
thread, and the Qt UI thread simultaneously.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError

from config.settings import DatabaseConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """
    Manages the MongoDB connection lifecycle and provides a typed,
    high-level API for reading and writing CyberSentinel AI data.

    Responsibilities:
        - Establish and hold a single MongoDB connection.
        - Create indexes (including TTL indexes for log retention).
        - Provide insert/query methods for each data domain
          (packets, threats, alerts, logs, system stats, predictions).
        - Fail gracefully (never raise into UI/network code) if
          MongoDB is unreachable, so the desktop app keeps running
          in a degraded (no-persistence) mode.
    """

    def __init__(self) -> None:
        """
        Initializes the DatabaseManager. Does not connect immediately;
        call connect() explicitly (typically once, at app startup).
        """
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._connected: bool = False

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Establishes the connection to MongoDB and prepares indexes.

        Returns:
            True if the connection succeeded, False otherwise.
            The application should remain usable even if this returns
            False — persistence simply becomes a no-op.
        """
        try:
            self._client = MongoClient(
                DatabaseConfig.MONGO_URI,
                serverSelectionTimeoutMS=DatabaseConfig.SERVER_SELECTION_TIMEOUT_MS,
            )
            # Forces a round trip to confirm the server is reachable.
            self._client.admin.command("ping")

            self._db = self._client[DatabaseConfig.MONGO_DB_NAME]
            self._connected = True
            self._create_indexes()

            logger.info(
                "Connected to MongoDB at %s (database: %s)",
                DatabaseConfig.MONGO_URI,
                DatabaseConfig.MONGO_DB_NAME,
            )
            return True

        except ServerSelectionTimeoutError:
            self._connected = False
            logger.warning(
                "Could not connect to MongoDB at %s. "
                "The application will continue running without persistence "
                "until a database connection is available.",
                DatabaseConfig.MONGO_URI,
            )
            return False

        except PyMongoError as error:
            self._connected = False
            logger.error("Unexpected MongoDB connection error: %s", error)
            return False

    def disconnect(self) -> None:
        """
        Closes the MongoDB connection cleanly. Should be called on
        application shutdown (main.py exit handler).
        """
        if self._client is not None:
            self._client.close()
            self._connected = False
            logger.info("MongoDB connection closed.")

    def is_connected(self) -> bool:
        """
        Returns:
            True if a live MongoDB connection is currently active.
        """
        return self._connected

    def _create_indexes(self) -> None:
        """
        Creates indexes required for query performance and automatic
        data retention. Called once after a successful connect().

        TTL (Time-To-Live) indexes are applied to high-volume,
        low-long-term-value collections (raw packets, system stats,
        general logs) so old documents self-delete after
        MAX_LOG_RETENTION_DAYS. Threats and alerts are NOT auto-expired,
        since they matter for the Reports feature and security audits.
        """
        if self._db is None:
            return

        retention_seconds = DatabaseConfig.MAX_LOG_RETENTION_DAYS * 24 * 60 * 60

        # Descending timestamp indexes for fast "most recent N" queries
        self._db[DatabaseConfig.COLLECTION_PACKETS].create_index(
            [("timestamp", DESCENDING)]
        )
        self._db[DatabaseConfig.COLLECTION_THREATS].create_index(
            [("timestamp", DESCENDING)]
        )
        self._db[DatabaseConfig.COLLECTION_ALERTS].create_index(
            [("timestamp", DESCENDING)]
        )
        self._db[DatabaseConfig.COLLECTION_LOGS].create_index(
            [("timestamp", DESCENDING)]
        )
        self._db[DatabaseConfig.COLLECTION_SYSTEM_STATS].create_index(
            [("timestamp", DESCENDING)]
        )

        # TTL indexes: MongoDB auto-deletes documents once their
        # "timestamp" field is older than expireAfterSeconds.
        self._db[DatabaseConfig.COLLECTION_PACKETS].create_index(
            "timestamp", expireAfterSeconds=retention_seconds
        )
        self._db[DatabaseConfig.COLLECTION_SYSTEM_STATS].create_index(
            "timestamp", expireAfterSeconds=retention_seconds
        )
        self._db[DatabaseConfig.COLLECTION_LOGS].create_index(
            "timestamp", expireAfterSeconds=retention_seconds
        )

        logger.info("MongoDB indexes verified/created.")

    def _get_collection(self, name: str) -> Optional[Collection]:
        """
        Internal helper to safely retrieve a collection handle.

        Args:
            name: The collection name (use DatabaseConfig.COLLECTION_*).

        Returns:
            The Collection object, or None if not connected.
        """
        if not self._connected or self._db is None:
            return None
        return self._db[name]

    @staticmethod
    def _utc_now() -> datetime:
        """Returns the current UTC timestamp, used to stamp all records."""
        return datetime.now(timezone.utc)

    # ------------------------------------------------------------------
    # Insert operations
    # ------------------------------------------------------------------

    def insert_packet_record(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Stores a summarized packet capture record.

        Args:
            data: Dictionary containing packet metadata, e.g.
                  {src_ip, dst_ip, src_port, dst_port, protocol,
                   size, flags}.

        Returns:
            The inserted document's ID as a string, or None if the
            write failed or the database is offline.
        """
        return self._insert_one(DatabaseConfig.COLLECTION_PACKETS, data)

    def insert_threat_record(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Stores a detected threat record produced by the AI engine.

        Args:
            data: Dictionary containing threat details, e.g.
                  {src_ip, dst_ip, attack_type, confidence,
                   risk_score, severity}.

        Returns:
            The inserted document's ID as a string, or None on failure.
        """
        return self._insert_one(DatabaseConfig.COLLECTION_THREATS, data)

    def insert_alert_record(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Stores a user-facing alert for the Notification Center.

        Args:
            data: Dictionary containing alert details, e.g.
                  {title, message, severity, is_read}.

        Returns:
            The inserted document's ID as a string, or None on failure.
        """
        data.setdefault("is_read", False)
        return self._insert_one(DatabaseConfig.COLLECTION_ALERTS, data)

    def insert_log_record(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Stores a general security/event log entry (for the Logs page
        and Export Logs feature).

        Args:
            data: Dictionary containing log details, e.g.
                  {level, source, message}.

        Returns:
            The inserted document's ID as a string, or None on failure.
        """
        return self._insert_one(DatabaseConfig.COLLECTION_LOGS, data)

    def insert_system_stat(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Stores a point-in-time system resource snapshot
        (CPU / RAM / Disk / Network usage) for historical graphing.

        Args:
            data: Dictionary containing usage percentages, e.g.
                  {cpu_percent, ram_percent, disk_percent, net_sent,
                   net_recv}.

        Returns:
            The inserted document's ID as a string, or None on failure.
        """
        return self._insert_one(DatabaseConfig.COLLECTION_SYSTEM_STATS, data)

    def insert_ai_prediction(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Stores a raw AI model prediction (useful for later model
        evaluation, retraining, and confidence-score auditing).

        Args:
            data: Dictionary containing prediction details, e.g.
                  {model_name, features, predicted_class, confidence}.

        Returns:
            The inserted document's ID as a string, or None on failure.
        """
        return self._insert_one(DatabaseConfig.COLLECTION_AI_PREDICTIONS, data)

    def _insert_one(self, collection_name: str, data: Dict[str, Any]) -> Optional[str]:
        """
        Shared internal insert logic used by all insert_* methods.
        Automatically stamps every document with a UTC "timestamp"
        field unless one is already provided.

        Args:
            collection_name: Target collection name.
            data: The document to insert.

        Returns:
            The inserted document's ID as a string, or None if the
            database is offline or the write failed.
        """
        collection = self._get_collection(collection_name)
        if collection is None:
            return None

        data.setdefault("timestamp", self._utc_now())

        try:
            result = collection.insert_one(data)
            return str(result.inserted_id)
        except PyMongoError as error:
            logger.error("Failed to insert into '%s': %s", collection_name, error)
            return None

    # ------------------------------------------------------------------
    # Query operations
    # ------------------------------------------------------------------

    def get_recent_threats(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent threat detections, newest first.

        Args:
            limit: Maximum number of records to return.

        Returns:
            A list of threat documents (empty list if offline or none found).
        """
        return self._find_recent(DatabaseConfig.COLLECTION_THREATS, limit)

    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent alerts, newest first, for the
        Notification Center.

        Args:
            limit: Maximum number of records to return.

        Returns:
            A list of alert documents (empty list if offline or none found).
        """
        return self._find_recent(DatabaseConfig.COLLECTION_ALERTS, limit)

    def get_recent_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent security/event log entries, newest first.

        Args:
            limit: Maximum number of records to return.

        Returns:
            A list of log documents (empty list if offline or none found).
        """
        return self._find_recent(DatabaseConfig.COLLECTION_LOGS, limit)

    def get_recent_system_stats(self, limit: int = 60) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent system resource snapshots, newest
        first — used to backfill live graphs on dashboard startup.

        Args:
            limit: Maximum number of records to return.

        Returns:
            A list of system stat documents (empty list if offline or none found).
        """
        return self._find_recent(DatabaseConfig.COLLECTION_SYSTEM_STATS, limit)

    def _find_recent(self, collection_name: str, limit: int) -> List[Dict[str, Any]]:
        """
        Shared internal query logic for "most recent N documents".

        Args:
            collection_name: Target collection name.
            limit: Maximum number of documents to return.

        Returns:
            A list of documents sorted by timestamp descending.
        """
        collection = self._get_collection(collection_name)
        if collection is None:
            return []

        try:
            cursor = collection.find().sort("timestamp", DESCENDING).limit(limit)
            results = list(cursor)
            for doc in results:
                doc["_id"] = str(doc["_id"])
            return results
        except PyMongoError as error:
            logger.error("Failed to query '%s': %s", collection_name, error)
            return []

    def get_threat_count(self, since: Optional[datetime] = None) -> int:
        """
        Counts detected threats, optionally since a given timestamp.

        Args:
            since: If provided, only counts threats detected after this time.

        Returns:
            The count of matching threat documents (0 if offline).
        """
        return self._count(DatabaseConfig.COLLECTION_THREATS, since)

    def get_packet_count(self, since: Optional[datetime] = None) -> int:
        """
        Counts captured packet records, optionally since a given timestamp.

        Args:
            since: If provided, only counts packets captured after this time.

        Returns:
            The count of matching packet documents (0 if offline).
        """
        return self._count(DatabaseConfig.COLLECTION_PACKETS, since)

    def get_unread_alert_count(self) -> int:
        """
        Counts alerts that have not yet been marked as read, for the
        Notification Center badge.

        Returns:
            The count of unread alert documents (0 if offline).
        """
        collection = self._get_collection(DatabaseConfig.COLLECTION_ALERTS)
        if collection is None:
            return 0
        try:
            return collection.count_documents({"is_read": False})
        except PyMongoError as error:
            logger.error("Failed to count unread alerts: %s", error)
            return 0

    def get_threat_statistics(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Aggregates summary statistics over the "threats" collection,
        optionally scoped to threats detected since a given timestamp.
        Powers app/ui/pages/reports_page.py's summary cards, severity
        breakdown chart, and attack-type breakdown table.

        Args:
            since: If provided, only aggregates threats detected after
                this time. None aggregates the full collection.

        Returns:
            A dictionary with keys:
                "total": int
                "avg_risk_score": float
                "avg_confidence": float
                "severity_counts": Dict[str, int]
                "attack_type_counts": Dict[str, int]
            All values default to empty/zero if the database is
            offline or no matching threats exist.
        """
        empty_result: Dict[str, Any] = {
            "total": 0,
            "avg_risk_score": 0.0,
            "avg_confidence": 0.0,
            "severity_counts": {},
            "attack_type_counts": {},
        }

        collection = self._get_collection(DatabaseConfig.COLLECTION_THREATS)
        if collection is None:
            return empty_result

        match_stage: Dict[str, Any] = {}
        if since is not None:
            match_stage["timestamp"] = {"$gte": since}

        try:
            summary_pipeline = [
                {"$match": match_stage},
                {
                    "$group": {
                        "_id": None,
                        "total": {"$sum": 1},
                        "avg_risk_score": {"$avg": "$risk_score"},
                        "avg_confidence": {"$avg": "$confidence"},
                    }
                },
            ]
            summary_docs = list(collection.aggregate(summary_pipeline))

            severity_pipeline = [
                {"$match": match_stage},
                {"$group": {"_id": "$severity", "count": {"$sum": 1}}},
            ]
            severity_docs = list(collection.aggregate(severity_pipeline))

            attack_type_pipeline = [
                {"$match": match_stage},
                {"$group": {"_id": "$predicted_class", "count": {"$sum": 1}}},
            ]
            attack_type_docs = list(collection.aggregate(attack_type_pipeline))

        except PyMongoError as error:
            logger.error("Failed to aggregate threat statistics: %s", error)
            return empty_result

        if not summary_docs:
            return empty_result

        summary = summary_docs[0]
        return {
            "total": summary.get("total", 0),
            "avg_risk_score": summary.get("avg_risk_score") or 0.0,
            "avg_confidence": summary.get("avg_confidence") or 0.0,
            "severity_counts": {doc["_id"]: doc["count"] for doc in severity_docs if doc["_id"]},
            "attack_type_counts": {doc["_id"]: doc["count"] for doc in attack_type_docs if doc["_id"]},
        }

    def _count(self, collection_name: str, since: Optional[datetime]) -> int:
        """
        Shared internal counting logic, optionally filtered by time.

        Args:
            collection_name: Target collection name.
            since: Optional lower-bound timestamp filter.

        Returns:
            The document count matching the filter.
        """
        collection = self._get_collection(collection_name)
        if collection is None:
            return 0

        query: Dict[str, Any] = {}
        if since is not None:
            query["timestamp"] = {"$gte": since}

        try:
            return collection.count_documents(query)
        except PyMongoError as error:
            logger.error("Failed to count documents in '%s': %s", collection_name, error)
            return 0

    # ------------------------------------------------------------------
    # Update operations
    # ------------------------------------------------------------------

    def mark_alert_as_read(self, alert_id: str) -> bool:
        """
        Marks a specific alert as read in the Notification Center.

        Args:
            alert_id: The string representation of the alert's ObjectId.

        Returns:
            True if the update succeeded, False otherwise.
        """
        from bson import ObjectId  # local import: keeps bson usage isolated

        collection = self._get_collection(DatabaseConfig.COLLECTION_ALERTS)
        if collection is None:
            return False

        try:
            result = collection.update_one(
                {"_id": ObjectId(alert_id)}, {"$set": {"is_read": True}}
            )
            return result.modified_count > 0
        except PyMongoError as error:
            logger.error("Failed to mark alert '%s' as read: %s", alert_id, error)
            return False

    # ------------------------------------------------------------------
    # Maintenance / export operations
    # ------------------------------------------------------------------

    def clear_old_records(self, days: int = DatabaseConfig.MAX_LOG_RETENTION_DAYS) -> None:
        """
        Manually purges records older than the given number of days
        from non-TTL-managed collections (threats, alerts). TTL
        collections (packets, logs, system_stats) clean themselves
        automatically and don't need this.

        Args:
            days: Age threshold in days; records older than this are deleted.
        """
        cutoff = self._utc_now() - timedelta(days=days)

        for collection_name in (
            DatabaseConfig.COLLECTION_THREATS,
            DatabaseConfig.COLLECTION_ALERTS,
        ):
            collection = self._get_collection(collection_name)
            if collection is None:
                continue
            try:
                result = collection.delete_many({"timestamp": {"$lt": cutoff}})
                logger.info(
                    "Purged %d old record(s) from '%s'.",
                    result.deleted_count,
                    collection_name,
                )
            except PyMongoError as error:
                logger.error("Failed to purge old records from '%s': %s", collection_name, error)

    def export_collection(self, collection_name: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Retrieves documents from any collection for the "Export Logs"
        UI feature (e.g., to save as CSV/JSON from the Logs page).

        Args:
            collection_name: Target collection name (use DatabaseConfig.COLLECTION_*).
            limit: Maximum number of documents to export.

        Returns:
            A list of documents ready for serialization.
        """
        return self._find_recent(collection_name, limit)


# ----------------------------------------------------------------------
# Module-level singleton accessor
# ----------------------------------------------------------------------

_db_manager_instance: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """
    Returns the shared, application-wide DatabaseManager instance,
    creating and connecting it on first call.

    This mirrors the pattern used in utils/logger.py — all modules
    (network, ai, security, ui) should call this function rather than
    instantiating DatabaseManager directly, ensuring a single shared
    MongoDB connection pool across the whole application.

    Returns:
        The singleton DatabaseManager instance.
    """
    global _db_manager_instance
    if _db_manager_instance is None:
        _db_manager_instance = DatabaseManager()
        _db_manager_instance.connect()
    return _db_manager_instance