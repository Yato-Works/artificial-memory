from __future__ import annotations

import json
import secrets
from datetime import datetime, timedelta

from artificial_memory.auth.models import (
    APIKey,
    Session,
    User,
    UserRole,
    generate_api_key,
    hash_api_key,
    hash_password,
    verify_password,
)
from artificial_memory.core.interfaces import MemoryStore


class AuthService:
    """Authentication and authorization service."""

    def __init__(self, store: MemoryStore, jwt_secret: str | None = None):
        self.store = store
        if jwt_secret is None:
            import secrets
            import warnings
            warnings.warn(
                "AuthService: no jwt_secret provided; generating an ephemeral "
                "secret. Sessions will be invalidated on restart. Provide "
                "jwt_secret explicitly for production.",
                stacklevel=2,
            )
            jwt_secret = secrets.token_urlsafe(32)
        self.jwt_secret = jwt_secret
        self._init_auth_tables()

    def _init_auth_tables(self):
        """Initialize auth tables in SQLite."""
        # This would normally be done via migrations
        # For now, we'll create tables if they don't exist
        with self.store._transaction() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    email TEXT NOT NULL UNIQUE,
                    hashed_password TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL DEFAULT 'user',
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS api_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    name TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    key_prefix TEXT NOT NULL,
                    scopes TEXT DEFAULT '[]',
                    is_active BOOLEAN DEFAULT 1,
                    expires_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id),
                    token TEXT NOT NULL,
                    expires_at TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    ip_address TEXT,
                    user_agent TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
                CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
                CREATE INDEX IF NOT EXISTS idx_api_keys_prefix ON api_keys(key_prefix);
                CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(token);
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
            """)

    # User management
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER,
    ) -> User:
        """Create a new user."""
        hashed, salt = hash_password(password)

        with self.store._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO users (username, email, hashed_password, salt, role)
                   VALUES (?, ?, ?, ?, ?)""",
                (username, email, hashed, salt, role.value)
            )
            user_id = cursor.lastrowid

        return User(
            id=user_id,
            username=username,
            email=email,
            hashed_password=hashed,
            role=role,
        )

    def get_user(self, user_id: int) -> User | None:
        """Get user by ID."""
        row = self.store._conn.execute(
            "SELECT * FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_user(row)

    def get_user_by_username(self, username: str) -> User | None:
        """Get user by username."""
        row = self.store._conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_user(row)

    def get_user_by_email(self, email: str) -> User | None:
        """Get user by email."""
        row = self.store._conn.execute(
            "SELECT * FROM users WHERE email = ?", (email,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_user(row)

    def _row_to_user(self, row) -> User:
        return User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            hashed_password=row["hashed_password"],
            role=UserRole(row["role"]),
            is_active=bool(row["is_active"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_login=datetime.fromisoformat(row["last_login"]) if row["last_login"] else None,
        )

    def authenticate(self, username: str, password: str) -> User | None:
        """Authenticate user with username/password."""
        user = self.get_user_by_username(username)
        if not user or not user.is_active:
            return None

        # Get salt
        row = self.store._conn.execute(
            "SELECT salt FROM users WHERE id = ?", (user.id,)
        ).fetchone()
        if not row:
            return None

        if verify_password(password, user.hashed_password, row["salt"]):
            # Update last login
            self.store._conn.execute(
                "UPDATE users SET last_login = ? WHERE id = ?",
                (datetime.now().isoformat(), user.id)
            )
            return user
        return None

    def update_user(self, user: User) -> User:
        """Update user info."""
        user.updated_at = datetime.now()
        with self.store._transaction() as conn:
            conn.execute(
                """UPDATE users SET username = ?, email = ?, role = ?,
                   is_active = ?, updated_at = ? WHERE id = ?""",
                (user.username, user.email, user.role.value,
                 int(user.is_active), user.updated_at.isoformat(), user.id)
            )
        return user

    def delete_user(self, user_id: int) -> bool:
        """Delete a user."""
        with self.store._transaction() as conn:
            cursor = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
            return cursor.rowcount > 0

    # API Key management
    def create_api_key(
        self,
        user_id: int,
        name: str,
        scopes: list[str] = None,
        expires_days: int = 365,
    ) -> tuple[APIKey, str]:
        """Create an API key for a user. Returns (APIKey, raw_key)."""
        raw_key, key_hash = generate_api_key()
        key_prefix = raw_key[:12]
        expires_at = datetime.now() + timedelta(days=expires_days)

        with self.store._transaction() as conn:
            cursor = conn.execute(
                """INSERT INTO api_keys (user_id, name, key_hash, key_prefix, scopes, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, name, key_hash, key_prefix,
                 json.dumps(scopes or []), expires_at.isoformat())
            )
            key_id = cursor.lastrowid

        api_key = APIKey(
            id=key_id,
            user_id=user_id,
            name=name,
            key_hash=key_hash,
            key_prefix=key_prefix,
            scopes=scopes or [],
            expires_at=expires_at,
        )
        return api_key, raw_key

    def get_api_key(self, key_id: int) -> APIKey | None:
        """Get API key by ID."""
        row = self.store._conn.execute(
            "SELECT * FROM api_keys WHERE id = ?", (key_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_api_key(row)

    def get_api_key_by_hash(self, key_hash: str) -> APIKey | None:
        """Get API key by hash."""
        row = self.store._conn.execute(
            "SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_api_key(row)

    def _row_to_api_key(self, row) -> APIKey:
        import json
        return APIKey(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            key_hash=row["key_hash"],
            key_prefix=row["key_prefix"],
            scopes=json.loads(row["scopes"]),
            is_active=bool(row["is_active"]),
            expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            last_used=datetime.fromisoformat(row["last_used"]) if row["last_used"] else None,
        )

    def list_user_api_keys(self, user_id: int) -> list[APIKey]:
        """List all API keys for a user."""
        rows = self.store._conn.execute(
            "SELECT * FROM api_keys WHERE user_id = ?", (user_id,)
        ).fetchall()
        return [self._row_to_api_key(row) for row in rows]

    def revoke_api_key(self, key_id: int) -> bool:
        """Revoke an API key."""
        with self.store._transaction() as conn:
            cursor = conn.execute(
                "UPDATE api_keys SET is_active = 0 WHERE id = ?", (key_id,)
            )
            return cursor.rowcount > 0

    def validate_api_key(self, key: str) -> APIKey | None:
        """Validate an API key and return associated APIKey if valid."""
        key_hash = hash_api_key(key)
        api_key = self.get_api_key_by_hash(key_hash)

        if not api_key or not api_key.is_active:
            return None

        if api_key.expires_at and api_key.expires_at < datetime.now():
            return None

        # Update last used
        self.store._conn.execute(
            "UPDATE api_keys SET last_used = ? WHERE id = ?",
            (datetime.now().isoformat(), api_key.id)
        )

        return api_key

    # Session management
    def create_session(self, user_id: int, ip: str = None, user_agent: str = None) -> Session:
        """Create a new session."""
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now() + timedelta(days=30)

        session = Session(
            id=secrets.token_urlsafe(16),
            user_id=user_id,
            token=token,
            expires_at=expires_at,
            ip_address=ip,
            user_agent=user_agent,
        )

        with self.store._transaction() as conn:
            conn.execute(
                """INSERT INTO sessions (id, user_id, token, expires_at, ip_address, user_agent)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session.id, user_id, session.token, session.expires_at.isoformat(),
                 session.ip_address, session.user_agent)
            )

        return session

    def get_session(self, token: str) -> Session | None:
        """Get session by token."""
        row = self.store._conn.execute(
            "SELECT * FROM sessions WHERE token = ?", (token,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_session(row)

    def _row_to_session(self, row) -> Session:
        return Session(
            id=row["id"],
            user_id=row["user_id"],
            token=row["token"],
            expires_at=datetime.fromisoformat(row["expires_at"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            ip_address=row["ip_address"],
            user_agent=row["user_agent"],
        )

    def validate_session(self, token: str) -> Session | None:
        """Validate session token."""
        session = self.get_session(token)
        if not session or session.expires_at < datetime.now():
            return None
        return session

    def revoke_session(self, token: str) -> bool:
        """Revoke a session."""
        with self.store._transaction() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
            return cursor.rowcount > 0

    def revoke_user_sessions(self, user_id: int) -> int:
        """Revoke all sessions for a user."""
        with self.store._transaction() as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
            return cursor.rowcount


def create_auth_service(store: MemoryStore, jwt_secret: str = "change-me") -> AuthService:
    """Factory function to create auth service."""
    return AuthService(store, jwt_secret)
