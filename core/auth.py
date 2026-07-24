"""
core/auth.py — SQLite-backed user authentication and profile management.
"""
from __future__ import annotations

import hashlib
import logging
import secrets
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

SESSION_TTL_DAYS = 30.0


# ─────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def _hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 260_000)
    return h.hex(), salt


def _verify_password(password: str, stored_hash: str, salt: str) -> bool:
    computed, _ = _hash_password(password, salt)
    return secrets.compare_digest(computed, stored_hash)


# ─────────────────────────────────────────────────────────────────────
# AuthDB
# ─────────────────────────────────────────────────────────────────────

class AuthDB:
    """
    Manages users and sessions in a local SQLite database.

    Tables
    ------
    users    — profile data (display_name, email, password hash, bio, avatar)
    sessions — bearer tokens with TTL
    """

    def __init__(self, storage_dir: str = "data/faiss_store", avatars_dir: str = "data/avatars") -> None:
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        Path(avatars_dir).mkdir(parents=True, exist_ok=True)
        self._db_path = str(Path(storage_dir) / "users.db")
        self._avatars_dir = str(Path(avatars_dir))
        self._init_db()

    # ── init ─────────────────────────────────────────────────────────


    def _init_db(self) -> None:
        with _connect(self._db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    username      TEXT    NOT NULL UNIQUE,
                    display_name  TEXT    NOT NULL,
                    email         TEXT    NOT NULL UNIQUE,
                    password_hash TEXT    NOT NULL,
                    password_salt TEXT    NOT NULL,
                    bio           TEXT    NOT NULL DEFAULT '',
                    avatar_file   TEXT,
                    created_at    REAL    NOT NULL,
                    updated_at    REAL    NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token         TEXT    PRIMARY KEY,
                    user_id       INTEGER NOT NULL,
                    created_at    REAL    NOT NULL,
                    expires_at    REAL    NOT NULL,
                    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user
                    ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_expires
                    ON sessions(expires_at);
            """)
            conn.commit()

    def _ensure_default_user(self) -> None:
        # Intentionally a no-op. The old seeded "hanane" demo account was removed
        # because it surfaced as a hard-coded name in the dashboard. Accounts now
        # come only from real sign-ups (Supabase auth on the frontend).
        return

    # ── user CRUD ─────────────────────────────────────────────────────

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        with _connect(self._db_path) as conn:
            row = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None

    def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        with _connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email=?", (email.lower().strip(),)
            ).fetchone()
        return dict(row) if row else None

    def get_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        with _connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username=?", (username.lower().strip(),)
            ).fetchone()
        return dict(row) if row else None

    def create_user(
        self,
        username: str,
        display_name: str,
        email: str,
        password: str,
    ) -> Dict[str, Any]:
        h, s = _hash_password(password)
        now = time.time()
        with _connect(self._db_path) as conn:
            cur = conn.execute(
                "INSERT INTO users "
                "(username, display_name, email, password_hash, password_salt, bio, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, '', ?, ?)",
                (username.lower().strip(), display_name.strip(),
                 email.lower().strip(), h, s, now, now),
            )
            conn.commit()
            uid = cur.lastrowid
        return self.get_user(uid)

    def update_profile(
        self,
        user_id: int,
        *,
        display_name: Optional[str] = None,
        username: Optional[str] = None,
        email: Optional[str] = None,
        bio: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        clauses, vals = [], []
        if display_name is not None:
            clauses.append("display_name=?"); vals.append(display_name.strip())
        if username is not None:
            clauses.append("username=?"); vals.append(username.lower().strip())
        if email is not None:
            clauses.append("email=?"); vals.append(email.lower().strip())
        if bio is not None:
            clauses.append("bio=?"); vals.append(bio)
        if not clauses:
            return self.get_user(user_id)
        clauses.append("updated_at=?"); vals.append(time.time())
        vals.append(user_id)
        with _connect(self._db_path) as conn:
            conn.execute(f"UPDATE users SET {', '.join(clauses)} WHERE id=?", vals)
            conn.commit()
        return self.get_user(user_id)

    def change_password(
        self, user_id: int, current_password: str, new_password: str
    ) -> bool:
        user = self.get_user(user_id)
        if user is None:
            return False
        if not _verify_password(current_password, user["password_hash"], user["password_salt"]):
            return False
        h, s = _hash_password(new_password)
        with _connect(self._db_path) as conn:
            conn.execute(
                "UPDATE users SET password_hash=?, password_salt=?, updated_at=? WHERE id=?",
                (h, s, time.time(), user_id),
            )
            conn.commit()
        return True

    def save_avatar(self, user_id: int, data: bytes, content_type: str = "image/jpeg") -> str:
        ext = ".png" if "png" in content_type else ".jpg"
        filename = f"avatar_{user_id}{ext}"
        path = Path(self._avatars_dir) / filename
        path.write_bytes(data)
        with _connect(self._db_path) as conn:
            conn.execute(
                "UPDATE users SET avatar_file=?, updated_at=? WHERE id=?",
                (filename, time.time(), user_id),
            )
            conn.commit()
        return filename

    def get_avatar_path(self, filename: str) -> Optional[Path]:
        p = Path(self._avatars_dir) / filename
        return p if p.exists() else None

    # ── sessions ─────────────────────────────────────────────────────

    def create_session(self, user_id: int) -> str:
        token = secrets.token_hex(32)
        now = time.time()
        with _connect(self._db_path) as conn:
            conn.execute(
                "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token, user_id, now, now + SESSION_TTL_DAYS * 86_400),
            )
            conn.commit()
        return token

    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        if not token or len(token) < 10:
            return None
        now = time.time()
        with _connect(self._db_path) as conn:
            row = conn.execute(
                "SELECT user_id, expires_at FROM sessions WHERE token=?", (token,)
            ).fetchone()
        if row is None or now > row["expires_at"]:
            return None
        return self.get_user(row["user_id"])

    def delete_session(self, token: str) -> None:
        with _connect(self._db_path) as conn:
            conn.execute("DELETE FROM sessions WHERE token=?", (token,))
            conn.commit()

    def purge_expired_sessions(self) -> int:
        with _connect(self._db_path) as conn:
            cur = conn.execute("DELETE FROM sessions WHERE expires_at < ?", (time.time(),))
            conn.commit()
            return cur.rowcount

    # ── login ─────────────────────────────────────────────────────────

    def login(self, email: str, password: str) -> Optional[Tuple[Dict[str, Any], str]]:
        user = self.get_by_email(email)
        if user is None:
            # constant-time: still hash so timing doesn't reveal whether the user exists
            _hash_password(password)
            return None
        if not _verify_password(password, user["password_hash"], user["password_salt"]):
            return None
        token = self.create_session(user["id"])
        return user, token

    def stats(self) -> Dict[str, Any]:
        with _connect(self._db_path) as conn:
            users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
            sessions = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE expires_at > ?", (time.time(),)
            ).fetchone()[0]
        return {"total_users": users, "active_sessions": sessions}


# ─────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────

_auth_db: Optional[AuthDB] = None


def get_auth_db(
    storage_dir: str = "data/faiss_store",
    avatars_dir: str = "data/avatars",
) -> AuthDB:
    global _auth_db
    if _auth_db is None:
        _auth_db = AuthDB(storage_dir=storage_dir, avatars_dir=avatars_dir)
    return _auth_db


def public_user(user: Dict[str, Any]) -> Dict[str, Any]:
    """Return only safe-to-send user fields (no hashes)."""
    avatar = user.get("avatar_file")
    return {
        "id": user["id"],
        "username": user["username"],
        "display_name": user["display_name"],
        "email": user["email"],
        "bio": user.get("bio", ""),
        "avatar_url": f"/avatars/{avatar}" if avatar else None,
        "created_at": user["created_at"],
    }
