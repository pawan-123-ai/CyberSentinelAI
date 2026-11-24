"""
web/auth_manager.py

MongoDB-backed authentication manager for CyberSentinel AI.

Features:
    - Multiple users
    - Username + email
    - Login using username OR email
    - Unique username
    - Unique email
    - PBKDF2-HMAC-SHA256 password hashing
    - Per-user random salt
    - Session tokens
    - MongoDB-persisted sessions
    - Session expiration
    - Brute-force protection
    - User roles
    - Account activation/deactivation
    - Account deletion
    - Password changes
    - FastAPI authentication router

MongoDB:
    mongodb://localhost:27017

    database: cybersentinel_db

    collections:
        users
        sessions

Existing users are preserved.

Existing users created before email support may not have an email.
Those users can continue logging in with their username.

New accounts require an email address.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
import time

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


# ----------------------------------------------------------------------
# Optional MongoDB dependency
# ----------------------------------------------------------------------

try:
    from pymongo import ASCENDING, DESCENDING, MongoClient
    from pymongo.errors import DuplicateKeyError, PyMongoError

    _PYMONGO_AVAILABLE = True

except ImportError:
    MongoClient = None  # type: ignore
    DuplicateKeyError = Exception  # type: ignore
    PyMongoError = Exception  # type: ignore

    ASCENDING = 1
    DESCENDING = -1

    _PYMONGO_AVAILABLE = False


# ----------------------------------------------------------------------
# Optional FastAPI dependency
# ----------------------------------------------------------------------

try:
    from fastapi import (
        APIRouter,
        Cookie,
        HTTPException,
        Response,
        status,
    )
    from pydantic import BaseModel

    _FASTAPI_AVAILABLE = True

except ImportError:
    APIRouter = None  # type: ignore
    Cookie = None  # type: ignore
    HTTPException = None  # type: ignore
    Response = None  # type: ignore
    BaseModel = object  # type: ignore

    _FASTAPI_AVAILABLE = False


# ======================================================================
# AuthManager
# ======================================================================

class AuthManager:
    """
    Singleton MongoDB authentication manager.

    Authentication data is persisted in MongoDB rather than local JSON.
    """

    _instance: Optional["AuthManager"] = None
    _instance_lock = threading.Lock()

    # ------------------------------------------------------------------
    # MongoDB configuration
    # ------------------------------------------------------------------

    MONGODB_URI = "mongodb://localhost:27017"
    DATABASE_NAME = "cybersentinel_db"

    USERS_COLLECTION = "users"
    SESSIONS_COLLECTION = "sessions"

    # ------------------------------------------------------------------
    # Security configuration
    # ------------------------------------------------------------------

    SESSION_TTL_SECONDS: int = 8 * 3600

    PBKDF2_ITERATIONS: int = 260_000

    MAX_FAILED_ATTEMPTS: int = 5
    LOCKOUT_SECONDS: int = 60

    MIN_USERNAME_LENGTH: int = 3
    MAX_USERNAME_LENGTH: int = 64

    MIN_PASSWORD_LENGTH: int = 6
    MAX_PASSWORD_LENGTH: int = 256

    MAX_EMAIL_LENGTH: int = 254

    USERNAME_PATTERN = re.compile(
        r"^[a-zA-Z0-9_.]+$"
    )

    # Fixed email validation pattern.
    EMAIL_PATTERN = re.compile(
        r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    )

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    def __new__(cls) -> "AuthManager":
        with cls._instance_lock:

            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False

        return cls._instance

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def __init__(self) -> None:

        if self._initialized:
            return

        self._initialized = True

        self._state_lock = threading.RLock()

        # Local failed-login cache.
        self._failed_attempts: Dict[str, List[float]] = {}

        self._client = None
        self._db = None
        self._users = None
        self._sessions = None

        self._connect_mongodb()
        self._ensure_indexes()

    # ------------------------------------------------------------------
    # MongoDB connection
    # ------------------------------------------------------------------

    def _connect_mongodb(self) -> None:
        """
        Connect to MongoDB.

        Authentication remains usable only when MongoDB is available.
        """

        if not _PYMONGO_AVAILABLE:

            logger.error(
                "PyMongo is not installed. "
                "Install it with: pip install pymongo"
            )

            return

        try:

            self._client = MongoClient(
                self.MONGODB_URI,
                serverSelectionTimeoutMS=3000,
                connectTimeoutMS=3000,
            )

            # Force a connection check.
            self._client.admin.command("ping")

            self._db = self._client[
                self.DATABASE_NAME
            ]

            self._users = self._db[
                self.USERS_COLLECTION
            ]

            self._sessions = self._db[
                self.SESSIONS_COLLECTION
            ]

            logger.info(
                "AuthManager connected to MongoDB "
                "'%s/%s'.",
                self.DATABASE_NAME,
                self.USERS_COLLECTION,
            )

        except Exception as exc:

            self._client = None
            self._db = None
            self._users = None
            self._sessions = None

            logger.exception(
                "Unable to connect AuthManager to MongoDB: %s",
                exc,
            )

    # ------------------------------------------------------------------
    # MongoDB indexes
    # ------------------------------------------------------------------

    def _ensure_indexes(self) -> None:
        """
        Create and verify authentication indexes.

        Existing indexes are detected by their key pattern before a new
        index is created. This prevents MongoDB IndexOptionsConflict
        errors when an older version of CyberSentinel AI created the
        same index under a different name.

        username_key:
            Case-insensitive unique username.

        email_key:
            Case-insensitive unique email.

        The email index is sparse so legacy users without email
        continue to work.
        """

        if self._users is None:
            return

        def ensure_index(
            collection,
            keys,
            *,
            name: str,
            unique: bool = False,
            sparse: bool = False,
            expire_after_seconds=None,
        ) -> None:
            """
            Create an index only when an equivalent key pattern does not
            already exist.

            MongoDB identifies an index by its key pattern as well as its
            options. If an equivalent key pattern already exists under a
            different name (for example created_at_desc), reusing that
            index is safer than attempting to create a duplicate name.
            """
            existing_indexes = list(collection.list_indexes())

            wanted_keys = dict(keys)

            for index in existing_indexes:
                existing_keys = dict(index.get("key", {}))

                if existing_keys != wanted_keys:
                    continue

                # An equivalent index already exists. Do not create a
                # second index with the same key pattern under another name.
                logger.debug(
                    "Authentication index '%s' is already covered by "
                    "existing index '%s'.",
                    name,
                    index.get("name", "<unnamed>"),
                )
                return

            options = {
                "name": name,
                "unique": unique,
                "sparse": sparse,
            }

            if expire_after_seconds is not None:
                options["expireAfterSeconds"] = expire_after_seconds

            collection.create_index(keys, **options)

        try:
            # ----------------------------------------------------------
            # Username
            # ----------------------------------------------------------
            ensure_index(
                self._users,
                [("username_key", ASCENDING)],
                unique=True,
                name="username_key_unique",
            )

            # ----------------------------------------------------------
            # Email
            # ----------------------------------------------------------
            ensure_index(
                self._users,
                [("email_key", ASCENDING)],
                unique=True,
                sparse=True,
                name="email_key_unique_sparse",
            )

            # ----------------------------------------------------------
            # User metadata
            # ----------------------------------------------------------
            ensure_index(
                self._users,
                [("created_at", DESCENDING)],
                name="created_at_index",
            )

            ensure_index(
                self._users,
                [("role", ASCENDING)],
                name="role_index",
            )

            logger.debug(
                "Authentication user indexes verified."
            )

        except Exception as exc:
            logger.exception(
                "Unable to create user authentication indexes: %s",
                exc,
            )

        # --------------------------------------------------------------
        # Session indexes
        # --------------------------------------------------------------

        if self._sessions is not None:
            try:
                ensure_index(
                    self._sessions,
                    [("token_hash", ASCENDING)],
                    unique=True,
                    name="token_hash_unique",
                )

                ensure_index(
                    self._sessions,
                    [("expires_at", ASCENDING)],
                    expire_after_seconds=0,
                    name="session_expiry_ttl",
                )

            except Exception as exc:
                logger.exception(
                    "Unable to create session indexes: %s",
                    exc,
                )

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_username(username: str) -> str:
        return username.strip()

    @staticmethod
    def _username_key(username: str) -> str:
        return username.strip().lower()

    @staticmethod
    def _normalize_email(email: str) -> str:
        return email.strip().lower()

    @staticmethod
    def _email_key(email: str) -> str:
        return email.strip().lower()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_new_account(
        self,
        username: str,
        email: str,
        password: str,
    ) -> Optional[str]:
        """
        Validate new-account information.

        Returns:
            None if valid.
            Error string otherwise.
        """

        username = self._normalize_username(username)
        email = self._normalize_email(email)

        # --------------------------------------------------------------
        # Username
        # --------------------------------------------------------------

        if not username:
            return "Username is required."

        if len(username) < self.MIN_USERNAME_LENGTH:
            return (
                f"Username must contain at least "
                f"{self.MIN_USERNAME_LENGTH} characters."
            )

        if len(username) > self.MAX_USERNAME_LENGTH:
            return (
                f"Username must not exceed "
                f"{self.MAX_USERNAME_LENGTH} characters."
            )

        if not self.USERNAME_PATTERN.fullmatch(username):
            return (
                "Username may contain only letters, "
                "numbers, underscore (_) and dot (.)."
            )

        # --------------------------------------------------------------
        # Email
        # --------------------------------------------------------------

        if not email:
            return "Email address is required."

        if len(email) > self.MAX_EMAIL_LENGTH:
            return "Email address is too long."

        if not self.EMAIL_PATTERN.fullmatch(email):
            return "Please enter a valid email address."

        # --------------------------------------------------------------
        # Password
        # --------------------------------------------------------------

        if not password:
            return "Password is required."

        if len(password) < self.MIN_PASSWORD_LENGTH:
            return (
                f"Password must contain at least "
                f"{self.MIN_PASSWORD_LENGTH} characters."
            )

        if len(password) > self.MAX_PASSWORD_LENGTH:
            return "Password is too long."

        return None

    # ------------------------------------------------------------------
    # Password hashing
    # ------------------------------------------------------------------

    @classmethod
    def _hash_password(
        cls,
        password: str,
        salt: bytes,
    ) -> str:
        """
        PBKDF2-HMAC-SHA256 password derivation.
        """

        derived = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            cls.PBKDF2_ITERATIONS,
        )

        return derived.hex()

    # ------------------------------------------------------------------
    # Account existence
    # ------------------------------------------------------------------

    def has_any_users(self) -> bool:
        """Return True if at least one account exists."""

        if self._users is None:
            return False

        try:

            return (
                self._users.count_documents(
                    {},
                    limit=1,
                )
                > 0
            )

        except Exception as exc:

            logger.error(
                "Unable to check user existence: %s",
                exc,
            )

            return False

    def user_exists(
        self,
        username: str,
    ) -> bool:
        """Check whether a username already exists."""

        if self._users is None:
            return False

        key = self._username_key(username)

        try:

            return (
                self._users.find_one(
                    {"username_key": key}
                )
                is not None
            )

        except Exception:

            logger.exception(
                "Unable to check username."
            )

            return False

    def email_exists(
        self,
        email: str,
    ) -> bool:
        """Check whether an email already exists."""

        if self._users is None:
            return False

        key = self._email_key(email)

        try:

            return (
                self._users.find_one(
                    {"email_key": key}
                )
                is not None
            )

        except Exception:

            logger.exception(
                "Unable to check email."
            )

            return False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_user(
        self,
        username: str,
        email: str,
        password: str,
    ) -> Optional[str]:
        """
        Create a new MongoDB user.

        First registered account becomes admin.
        Later accounts become standard users.

        Returns:
            None on success.
            Error string on failure.
        """

        if self._users is None:
            return (
                "Authentication database is unavailable. "
                "Please make sure MongoDB is running."
            )

        username = self._normalize_username(username)
        email = self._normalize_email(email)

        error = self.validate_new_account(
            username,
            email,
            password,
        )

        if error:
            return error

        username_key = self._username_key(username)
        email_key = self._email_key(email)

        try:

            # ----------------------------------------------------------
            # Duplicate username
            # ----------------------------------------------------------

            if self._users.find_one(
                {"username_key": username_key}
            ):

                return (
                    "That username is already registered."
                )

            # ----------------------------------------------------------
            # Duplicate email
            # ----------------------------------------------------------

            if self._users.find_one(
                {"email_key": email_key}
            ):

                return (
                    "That email address is already registered."
                )

            # ----------------------------------------------------------
            # First account = admin
            # ----------------------------------------------------------

            role = (
                "admin"
                if not self.has_any_users()
                else "standard"
            )

            # ----------------------------------------------------------
            # Password hashing
            # ----------------------------------------------------------

            salt = secrets.token_bytes(16)

            password_hash = self._hash_password(
                password,
                salt,
            )

            now = datetime.now(timezone.utc)

            # ----------------------------------------------------------
            # User document
            # ----------------------------------------------------------

            document = {
                "username": username,
                "username_key": username_key,

                "email": email,
                "email_key": email_key,

                "password_hash": password_hash,
                "salt": salt.hex(),

                "role": role,
                "active": True,

                "created_at": now,
                "updated_at": now,
                "last_login": None,
            }

            self._users.insert_one(document)

            logger.info(
                "New user '%s' registered with role '%s'.",
                username,
                role,
            )

            return None

        except DuplicateKeyError as exc:

            logger.warning(
                "Duplicate account registration attempt: %s",
                exc,
            )

            # Race-condition-safe duplicate detection.

            if self.user_exists(username):
                return (
                    "That username is already registered."
                )

            if self.email_exists(email):
                return (
                    "That email address is already registered."
                )

            return (
                "An account with these details "
                "already exists."
            )

        except Exception as exc:

            logger.exception(
                "Account registration failed: %s",
                exc,
            )

            return (
                "Unable to create the account. "
                "Please try again."
            )

    # ------------------------------------------------------------------
    # Failed-login protection
    # ------------------------------------------------------------------

    def _is_locked_out(
        self,
        identifier: str,
    ) -> bool:
        """Return True if an identifier is temporarily locked."""

        now = time.time()

        attempts = self._failed_attempts.get(
            identifier,
            [],
        )

        recent = [
            timestamp
            for timestamp in attempts
            if now - timestamp < self.LOCKOUT_SECONDS
        ]

        self._failed_attempts[identifier] = recent

        return (
            len(recent)
            >= self.MAX_FAILED_ATTEMPTS
        )

    def _record_failed_attempt(
        self,
        identifier: str,
    ) -> None:
        """Record a failed authentication attempt."""

        self._failed_attempts.setdefault(
            identifier,
            [],
        ).append(
            time.time()
        )

    # ------------------------------------------------------------------
    # Login lookup
    # ------------------------------------------------------------------

    def _find_user_for_login(
        self,
        identifier: str,
    ) -> Optional[dict]:
        """
        Find a user using either username or email.

        Existing legacy users that do not have email_key are still
        searchable by username_key.
        """

        if self._users is None:
            return None

        identifier = identifier.strip()
        key = identifier.lower()

        if not key:
            return None

        try:

            user = self._users.find_one(
                {
                    "$or": [
                        {
                            "username_key": key
                        },
                        {
                            "email_key": key
                        },
                    ]
                }
            )

            return user

        except Exception:

            logger.exception(
                "Unable to find login user."
            )

            return None

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    def verify_login(
        self,
        identifier: str,
        password: str,
    ) -> Optional[str]:
        """
        Verify username/email + password.

        Returns:
            Raw session token on success.
            None on failure.
        """

        if self._users is None:
            logger.error(
                "Login attempted while MongoDB "
                "authentication is unavailable."
            )
            return None

        identifier = identifier.strip()

        if not identifier:
            return None

        lock_key = identifier.lower()

        with self._state_lock:

            # ----------------------------------------------------------
            # Brute-force protection
            # ----------------------------------------------------------

            if self._is_locked_out(lock_key):

                logger.warning(
                    "Login blocked for '%s' - "
                    "temporarily locked.",
                    identifier,
                )

                return None

            # ----------------------------------------------------------
            # Find account
            # ----------------------------------------------------------

            user = self._find_user_for_login(
                identifier
            )

            if user is None:

                self._record_failed_attempt(
                    lock_key
                )

                return None

            # ----------------------------------------------------------
            # Account status
            # ----------------------------------------------------------

            if not user.get("active", True):

                logger.warning(
                    "Login blocked for disabled "
                    "account '%s'.",
                    user.get(
                        "username",
                        identifier,
                    ),
                )

                return None

            # ----------------------------------------------------------
            # Password data
            # ----------------------------------------------------------

            stored_hash = user.get(
                "password_hash"
            )

            salt_hex = user.get(
                "salt"
            )

            if not stored_hash or not salt_hex:

                self._record_failed_attempt(
                    lock_key
                )

                return None

            try:

                salt = bytes.fromhex(
                    salt_hex
                )

            except ValueError:

                self._record_failed_attempt(
                    lock_key
                )

                return None

            # ----------------------------------------------------------
            # Hash candidate password
            # ----------------------------------------------------------

            candidate_hash = self._hash_password(
                password,
                salt,
            )

            # Constant-time comparison.
            if not hmac.compare_digest(
                candidate_hash,
                stored_hash,
            ):

                self._record_failed_attempt(
                    lock_key
                )

                logger.warning(
                    "Failed login attempt for '%s'.",
                    identifier,
                )

                return None

            # ----------------------------------------------------------
            # Successful authentication
            # ----------------------------------------------------------

            self._failed_attempts.pop(
                lock_key,
                None,
            )

            username = user["username"]

            email = user.get("email")

            role = user.get(
                "role",
                "standard",
            )

            # ----------------------------------------------------------
            # Generate session token
            # ----------------------------------------------------------

            token = secrets.token_urlsafe(32)

            token_hash = self._hash_session_token(
                token
            )

            now = datetime.now(timezone.utc)

            expires_at = (
                now
                + timedelta(
                    seconds=self.SESSION_TTL_SECONDS
                )
            )

            # ----------------------------------------------------------
            # Session document
            # ----------------------------------------------------------

            session_document = {
                "token_hash": token_hash,

                "username": username,
                "username_key": self._username_key(
                    username
                ),

                # Stored for convenient account/session access.
                # Legacy users may have None here.
                "email": email,

                "role": role,

                "created_at": now,
                "expires_at": expires_at,
            }

            try:

                if self._sessions is None:
                    logger.error(
                        "Session collection is unavailable."
                    )
                    return None

                self._sessions.insert_one(
                    session_document
                )

                self._users.update_one(
                    {
                        "_id": user["_id"]
                    },
                    {
                        "$set": {
                            "last_login": now,
                            "updated_at": now,
                        }
                    },
                )

            except Exception:

                logger.exception(
                    "Unable to persist login session."
                )

                return None

            logger.info(
                "User '%s' logged in successfully.",
                username,
            )

            return token

    # ------------------------------------------------------------------
    # Session token hashing
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_session_token(
        token: str,
    ) -> str:
        """Hash a session token before storing it."""

        return hashlib.sha256(
            token.encode("utf-8")
        ).hexdigest()

    # ------------------------------------------------------------------
    # Session lookup
    # ------------------------------------------------------------------

    def _get_session(
        self,
        token: str,
    ) -> Optional[dict]:
        """Retrieve a valid session."""

        if (
            not token
            or self._sessions is None
        ):
            return None

        token_hash = self._hash_session_token(
            token
        )

        try:

            session = self._sessions.find_one(
                {
                    "token_hash": token_hash
                }
            )

            if session is None:
                return None

            expires_at = session.get(
                "expires_at"
            )

            if expires_at is None:
                return None

            # MongoDB normally returns timezone-aware UTC datetimes,
            # but this also safely handles legacy naive datetimes.
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(
                    tzinfo=timezone.utc
                )

            if expires_at <= datetime.now(
                timezone.utc
            ):

                self._sessions.delete_one(
                    {
                        "_id": session["_id"]
                    }
                )

                return None

            return session

        except Exception:

            logger.exception(
                "Unable to retrieve session."
            )

            return None

    # ------------------------------------------------------------------
    # Session validation
    # ------------------------------------------------------------------

    def validate_session(
        self,
        token: str,
    ) -> bool:
        """Return True if a session is valid."""

        return (
            self._get_session(token)
            is not None
        )

    # ------------------------------------------------------------------
    # Session username
    # ------------------------------------------------------------------

    def get_username_for_session(
        self,
        token: str,
    ) -> Optional[str]:
        """Return username associated with a session."""

        session = self._get_session(token)

        if session is None:
            return None

        return session.get(
            "username"
        )

    # ------------------------------------------------------------------
    # Session role
    # ------------------------------------------------------------------

    def get_role_for_session(
        self,
        token: str,
    ) -> Optional[str]:
        """Return role associated with a session."""

        session = self._get_session(token)

        if session is None:
            return None

        return session.get(
            "role",
            "standard",
        )

    # ------------------------------------------------------------------
    # Session email
    # ------------------------------------------------------------------

    def get_email_for_session(
        self,
        token: str,
    ) -> Optional[str]:
        """
        Return email associated with a session.

        For newer sessions the email is stored directly in the session.

        For legacy sessions, fall back to the users collection.
        """

        session = self._get_session(token)

        if session is None:
            return None

        # New session format. Only use the session value when an
        # actual email is present. Legacy users may have email=None.
        email = session.get("email")
        if email:
            return email

        # Legacy session fallback.
        username = session.get("username")

        if not username:
            return None

        return self.get_email_for_user(
            username
        )

    # ------------------------------------------------------------------
    # Session user information
    # ------------------------------------------------------------------

    def get_user_for_session(
        self,
        token: str,
    ) -> Optional[dict]:
        """
        Return safe user information for a valid session.
        """

        username = self.get_username_for_session(
            token
        )

        if not username:
            return None

        return self.get_user(
            username
        )

    # ------------------------------------------------------------------
    # Logout
    # ------------------------------------------------------------------

    def logout(
        self,
        token: str,
    ) -> None:
        """Invalidate a session immediately."""

        if (
            not token
            or self._sessions is None
        ):
            return

        try:

            self._sessions.delete_one(
                {
                    "token_hash":
                        self._hash_session_token(
                            token
                        )
                }
            )

            logger.info(
                "Session logged out."
            )

        except Exception:

            logger.exception(
                "Unable to logout session."
            )

    # ------------------------------------------------------------------
    # Session cleanup
    # ------------------------------------------------------------------

    def cleanup_expired_sessions(
        self,
    ) -> None:
        """Remove expired sessions manually."""

        if self._sessions is None:
            return

        try:

            result = self._sessions.delete_many(
                {
                    "expires_at": {
                        "$lte":
                            datetime.now(
                                timezone.utc
                            )
                    }
                }
            )

            if result.deleted_count:

                logger.info(
                    "Cleaned up %d expired session(s).",
                    result.deleted_count,
                )

        except Exception:

            logger.exception(
                "Unable to clean expired sessions."
            )

    # ------------------------------------------------------------------
    # User information
    # ------------------------------------------------------------------

    def get_user(
        self,
        username: str,
    ) -> Optional[dict]:
        """
        Return safe account information.

        Password hash and salt are never returned.
        """

        if self._users is None:
            return None

        try:

            user = self._users.find_one(
                {
                    "username_key":
                        self._username_key(
                            username
                        )
                }
            )

            if user is None:
                return None

            return {
                "username": user.get(
                    "username"
                ),

                "email": user.get(
                    "email"
                ),

                "role": user.get(
                    "role",
                    "standard",
                ),

                "active": user.get(
                    "active",
                    True,
                ),

                "created_at": user.get(
                    "created_at"
                ),

                "last_login": user.get(
                    "last_login"
                ),
            }

        except Exception:

            logger.exception(
                "Unable to retrieve user '%s'.",
                username,
            )

            return None

    # ------------------------------------------------------------------
    # User lookup by identifier
    # ------------------------------------------------------------------

    def get_user_by_identifier(
        self,
        identifier: str,
    ) -> Optional[dict]:
        """
        Return safe user information using either
        username or email.
        """

        if self._users is None:
            return None

        identifier = identifier.strip()

        if not identifier:
            return None

        key = identifier.lower()

        try:

            user = self._users.find_one(
                {
                    "$or": [
                        {
                            "username_key": key
                        },
                        {
                            "email_key": key
                        },
                    ]
                }
            )

            if user is None:
                return None

            return {
                "username": user.get(
                    "username"
                ),

                "email": user.get(
                    "email"
                ),

                "role": user.get(
                    "role",
                    "standard",
                ),

                "active": user.get(
                    "active",
                    True,
                ),

                "created_at": user.get(
                    "created_at"
                ),

                "last_login": user.get(
                    "last_login"
                ),
            }

        except Exception:

            logger.exception(
                "Unable to retrieve user by identifier."
            )

            return None

    # ------------------------------------------------------------------
    # Email retrieval
    # ------------------------------------------------------------------

    def get_email_for_user(
        self,
        username: str,
    ) -> Optional[str]:
        """Return the email address of a user."""

        user = self.get_user(
            username
        )

        if not user:
            return None

        return user.get(
            "email"
        )

    # ------------------------------------------------------------------
    # Add/update email for legacy user
    # ------------------------------------------------------------------

    def set_user_email(
        self,
        username: str,
        email: str,
    ) -> Optional[str]:
        """
        Add or update the email address of an existing user.

        Useful for accounts created before email support.
        """

        if self._users is None:
            return (
                "Authentication database is unavailable."
            )

        email = self._normalize_email(
            email
        )

        if len(email) > self.MAX_EMAIL_LENGTH:
            return "Email address is too long."

        if not self.EMAIL_PATTERN.fullmatch(
            email
        ):
            return (
                "Please enter a valid email address."
            )

        username_key = self._username_key(
            username
        )

        email_key = self._email_key(
            email
        )

        try:

            # ----------------------------------------------------------
            # Verify user exists
            # ----------------------------------------------------------

            user = self._users.find_one(
                {
                    "username_key":
                        username_key
                }
            )

            if user is None:
                return "User account not found."

            # ----------------------------------------------------------
            # Check email ownership
            # ----------------------------------------------------------

            existing = self._users.find_one(
                {
                    "email_key": email_key,
                    "username_key": {
                        "$ne": username_key
                    },
                }
            )

            if existing:

                return (
                    "That email address is already "
                    "associated with another account."
                )

            # ----------------------------------------------------------
            # Update email
            # ----------------------------------------------------------

            self._users.update_one(
                {
                    "_id": user["_id"]
                },
                {
                    "$set": {
                        "email": email,
                        "email_key": email_key,
                        "updated_at":
                            datetime.now(
                                timezone.utc
                            ),
                    }
                },
            )

            logger.info(
                "Email updated for user '%s'.",
                username,
            )

            return None

        except DuplicateKeyError:

            return (
                "That email address is already "
                "associated with another account."
            )

        except Exception:

            logger.exception(
                "Unable to update user email."
            )

            return (
                "Unable to update email address."
            )

    # ------------------------------------------------------------------
    # Change password
    # ------------------------------------------------------------------

    def change_password(
        self,
        username: str,
        new_password: str,
    ) -> Optional[str]:
        """Change a user's password."""

        if self._users is None:
            return (
                "Authentication database is unavailable."
            )

        if not new_password:
            return "Password is required."

        if len(new_password) < self.MIN_PASSWORD_LENGTH:
            return (
                f"Password must contain at least "
                f"{self.MIN_PASSWORD_LENGTH} characters."
            )

        if len(new_password) > self.MAX_PASSWORD_LENGTH:
            return "Password is too long."

        salt = secrets.token_bytes(16)

        password_hash = self._hash_password(
            new_password,
            salt,
        )

        try:

            result = self._users.update_one(
                {
                    "username_key":
                        self._username_key(
                            username
                        )
                },
                {
                    "$set": {
                        "password_hash":
                            password_hash,

                        "salt":
                            salt.hex(),

                        "updated_at":
                            datetime.now(
                                timezone.utc
                            ),
                    }
                },
            )

            if result.matched_count == 0:
                return "User account not found."

            logger.info(
                "Password changed for '%s'.",
                username,
            )

            return None

        except Exception:

            logger.exception(
                "Unable to change password."
            )

            return (
                "Unable to change password."
            )

    # ------------------------------------------------------------------
    # List usernames
    # ------------------------------------------------------------------

    def list_usernames(
        self,
    ) -> List[str]:
        """Return usernames without exposing credentials."""

        if self._users is None:
            return []

        try:

            users = self._users.find(
                {},
                {
                    "username": 1,
                    "_id": 0,
                },
            ).sort(
                "username",
                ASCENDING,
            )

            return [
                user["username"]
                for user in users
                if user.get("username")
            ]

        except Exception:

            logger.exception(
                "Unable to list users."
            )

            return []

    # ------------------------------------------------------------------
    # Admin: activate/deactivate
    # ------------------------------------------------------------------

    def set_user_active(
        self,
        username: str,
        active: bool,
    ) -> bool:
        """Enable or disable a user account."""

        if self._users is None:
            return False

        try:

            result = self._users.update_one(
                {
                    "username_key":
                        self._username_key(
                            username
                        )
                },
                {
                    "$set": {
                        "active": bool(active),
                        "updated_at":
                            datetime.now(
                                timezone.utc
                            ),
                    }
                },
            )

            return result.matched_count > 0

        except Exception:

            logger.exception(
                "Unable to update account state."
            )

            return False

    # ------------------------------------------------------------------
    # Admin: delete user
    # ------------------------------------------------------------------

    def delete_user(
        self,
        username: str,
    ) -> bool:
        """
        Delete a user.

        Sessions belonging to the user are also removed.
        """

        if self._users is None:
            return False

        username_key = self._username_key(
            username
        )

        try:

            result = self._users.delete_one(
                {
                    "username_key":
                        username_key
                }
            )

            if result.deleted_count == 0:
                return False

            if self._sessions is not None:

                self._sessions.delete_many(
                    {
                        "username_key":
                            username_key
                    }
                )

            logger.warning(
                "User '%s' deleted.",
                username,
            )

            return True

        except Exception:

            logger.exception(
                "Unable to delete user."
            )

            return False

    # ------------------------------------------------------------------
    # Database status
    # ------------------------------------------------------------------

    def database_available(self) -> bool:
        """
        Return True when MongoDB authentication storage is ready.
        """

        return (
            self._users is not None
            and self._sessions is not None
        )


# ======================================================================
# Singleton accessor
# ======================================================================

def get_auth_manager() -> AuthManager:
    """Return the application-wide AuthManager."""

    return AuthManager()


# ======================================================================
# FastAPI
# ======================================================================

auth_router: Optional["APIRouter"] = None


if _FASTAPI_AVAILABLE:

    # ------------------------------------------------------------------
    # Request models
    # ------------------------------------------------------------------

    class LoginRequest(BaseModel):
        """
        Login request.

        `identifier` supports both username and email.

        `username` is retained as an optional compatibility field so
        older clients sending {"username": "...", "password": "..."}
        continue to work.
        """

        identifier: Optional[str] = None
        username: Optional[str] = None
        password: str

    class SignupRequest(BaseModel):
        username: str
        email: str
        password: str

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    auth_router = APIRouter(
        tags=["Authentication"]
    )

    # ------------------------------------------------------------------
    # Current user dependency
    # ------------------------------------------------------------------

    def get_current_user(
        session_token: Optional[str] = Cookie(
            default=None,
            alias="session_token",
        ),
    ) -> dict:
        """
        FastAPI authentication dependency.

        Raises HTTP 401 if the session is invalid.
        """

        auth = get_auth_manager()

        if not session_token:

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated.",
            )

        user = auth.get_user_for_session(
            session_token
        )

        if user is None:

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired or invalid.",
            )

        return user

    # ------------------------------------------------------------------
    # Signup
    # ------------------------------------------------------------------

    @auth_router.post("/signup")
    def signup(
        request: SignupRequest,
        response: Response,
    ) -> dict:

        auth = get_auth_manager()

        error = auth.register_user(
            username=request.username,
            email=request.email,
            password=request.password,
        )

        if error:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error,
            )

        # Automatically sign the newly created user in.
        token = auth.verify_login(
            request.username,
            request.password,
        )

        if token is None:

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Account created, but automatic "
                    "sign-in failed."
                ),
            )

        response.set_cookie(
            key="session_token",
            value=token,
            httponly=True,
            samesite="lax",
            secure=False,
            max_age=auth.SESSION_TTL_SECONDS,
        )

        username = (
            auth.get_username_for_session(
                token
            )
        )

        role = (
            auth.get_role_for_session(
                token
            )
        )

        email = (
            auth.get_email_for_session(
                token
            )
        )

        return {
            "authenticated": True,
            "username": username,
            "email": email,
            "role": role,
        }

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------

    @auth_router.post("/login")
    def login(
        request: LoginRequest,
        response: Response,
    ) -> dict:

        auth = get_auth_manager()

        # New clients use identifier.
        # Older clients may still send username.
        identifier = (
            request.identifier
            if request.identifier
            else request.username
        )

        if not identifier:

            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username or email is required.",
            )

        token = auth.verify_login(
            identifier,
            request.password,
        )

        if token is None:

            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=(
                    "Invalid username/email or password, "
                    "or account temporarily locked."
                ),
            )

        response.set_cookie(
            key="session_token",
            value=token,
            httponly=True,
            samesite="lax",
            secure=False,
            max_age=auth.SESSION_TTL_SECONDS,
        )

        username = (
            auth.get_username_for_session(
                token
            )
        )

        role = (
            auth.get_role_for_session(
                token
            )
        )

        email = (
            auth.get_email_for_session(
                token
            )
        )

        return {
            "authenticated": True,
            "username": username,
            "email": email,
            "role": role,
        }

    # ------------------------------------------------------------------
    # Logout
    # ------------------------------------------------------------------

    @auth_router.post("/logout")
    def logout(
        response: Response,
        session_token: Optional[str] = Cookie(
            default=None,
            alias="session_token",
        ),
    ) -> dict:

        auth = get_auth_manager()

        if session_token:

            auth.logout(
                session_token
            )

        response.delete_cookie(
            key="session_token"
        )

        return {
            "authenticated": False
        }

    # ------------------------------------------------------------------
    # Session status
    # ------------------------------------------------------------------

    @auth_router.get("/status")
    def status_endpoint(
        session_token: Optional[str] = Cookie(
            default=None,
            alias="session_token",
        ),
    ) -> dict:

        auth = get_auth_manager()

        if not session_token:

            return {
                "authenticated": False,
                "username": None,
                "email": None,
                "role": None,
            }

        user = auth.get_user_for_session(
            session_token
        )

        if user is None:

            return {
                "authenticated": False,
                "username": None,
                "email": None,
                "role": None,
            }

        return {
            "authenticated": True,
            "username": user.get(
                "username"
            ),
            "email": user.get(
                "email"
            ),
            "role": user.get(
                "role",
                "standard",
            ),
        }