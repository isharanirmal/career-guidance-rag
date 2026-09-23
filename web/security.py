import base64
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

DB_PATH = Path(os.getenv("AUTH_DB_PATH", "data/careerguide_auth.db"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", str(8 * 60 * 60)))
PBKDF2_ITERATIONS = int(os.getenv("PBKDF2_ITERATIONS", "310000"))
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_lock = threading.Lock()
_sessions: dict[str, dict] = {}


@dataclass
class User:
    id: int
    username: str
    created_at: int
    updated_at: int


def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL
            )
            """
        )
        conn.commit()


def user_count() -> int:
    init_db()
    with _connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])


def validate_username(username: str) -> str:
    value = (username or "").strip()
    if not USERNAME_RE.fullmatch(value):
        raise ValueError("Username must be 3-32 characters and use only letters, numbers, dot, dash or underscore.")
    return value


def validate_password(password: str):
    if len(password or "") < 10:
        raise ValueError("Password must contain at least 10 characters.")
    checks = [
        re.search(r"[A-Z]", password),
        re.search(r"[a-z]", password),
        re.search(r"\d", password),
        re.search(r"[^A-Za-z0-9]", password),
    ]
    if sum(bool(x) for x in checks) < 3:
        raise ValueError("Password must include at least 3 of: uppercase, lowercase, number, special character.")


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${base64.b64encode(salt).decode()}${base64.b64encode(digest).decode()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt_b64, digest_b64 = stored.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        salt = base64.b64decode(salt_b64)
        expected = base64.b64decode(digest_b64)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def create_user(username: str, password: str) -> User:
    username = validate_username(username)
    validate_password(password)
    now = int(time.time())
    try:
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO users(username,password_hash,created_at,updated_at) VALUES(?,?,?,?)",
                (username, _hash_password(password), now, now),
            )
            conn.commit()
            return User(cur.lastrowid, username, now, now)
    except sqlite3.IntegrityError as exc:
        raise ValueError("That username is already in use.") from exc


def authenticate(username: str, password: str) -> User | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=? COLLATE NOCASE", ((username or "").strip(),)).fetchone()
    if not row or not _verify_password(password or "", row["password_hash"]):
        return None
    return User(row["id"], row["username"], row["created_at"], row["updated_at"])


def get_user(user_id: int) -> User | None:
    with _connect() as conn:
        row = conn.execute("SELECT id,username,created_at,updated_at FROM users WHERE id=?", (user_id,)).fetchone()
    return User(**dict(row)) if row else None


def change_username(user_id: int, password: str, new_username: str) -> User:
    user = get_user(user_id)
    if not user or not authenticate(user.username, password):
        raise ValueError("Current password is incorrect.")
    new_username = validate_username(new_username)
    now = int(time.time())
    try:
        with _connect() as conn:
            conn.execute("UPDATE users SET username=?, updated_at=? WHERE id=?", (new_username, now, user_id))
            conn.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError("That username is already in use.") from exc
    return get_user(user_id)


def change_password(user_id: int, current_password: str, new_password: str):
    user = get_user(user_id)
    if not user or not authenticate(user.username, current_password):
        raise ValueError("Current password is incorrect.")
    validate_password(new_password)
    if hmac.compare_digest(current_password or "", new_password or ""):
        raise ValueError("New password must be different from the current password.")
    now = int(time.time())
    with _connect() as conn:
        conn.execute("UPDATE users SET password_hash=?, updated_at=? WHERE id=?", (_hash_password(new_password), now, user_id))
        conn.commit()
    revoke_user_sessions(user_id)


def create_session(user_id: int) -> tuple[str, str]:
    token = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(24)
    now = int(time.time())
    with _lock:
        _cleanup_sessions(now)
        _sessions[token] = {"user_id": user_id, "csrf": csrf, "expires": now + SESSION_TTL_SECONDS}
    return token, csrf


def _cleanup_sessions(now: int | None = None):
    now = now or int(time.time())
    expired = [k for k, v in _sessions.items() if v["expires"] <= now]
    for key in expired:
        _sessions.pop(key, None)


def get_session(token: str | None):
    if not token:
        return None
    now = int(time.time())
    with _lock:
        _cleanup_sessions(now)
        session = _sessions.get(token)
        if not session:
            return None
        session["expires"] = now + SESSION_TTL_SECONDS
        return dict(session)


def revoke_session(token: str | None):
    if not token:
        return
    with _lock:
        _sessions.pop(token, None)


def revoke_user_sessions(user_id: int):
    with _lock:
        doomed = [k for k, v in _sessions.items() if v["user_id"] == user_id]
        for key in doomed:
            _sessions.pop(key, None)
