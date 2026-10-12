"""
data/database.py — EcoSort AI
================================
SQLite helpers for user authentication and classification history.

Why SQLite?
  sqlite3 is part of Python's standard library — no extra install needed.
  The database file (history.db) is created automatically in the project
  root the first time the app runs.

Schema
------
Table: users
  id            INTEGER  Primary key, auto-incremented.
  username      TEXT     Unique, stored lowercase.
  password_hash TEXT     SHA-256 hex digest of the password.
  created_at    TEXT     ISO-8601 datetime string (UTC).

Table: classifications
  id            INTEGER  Primary key, auto-incremented.
  username      TEXT     The user who made this classification.
  timestamp     TEXT     ISO-8601 datetime string (UTC).
  filename      TEXT     Original uploaded filename.
  predicted_label TEXT   The waste category returned by the model.
  confidence    REAL     Model confidence score, 0.0 – 1.0.
  recommendation TEXT    The recycling tip shown to the user.

Public API — Authentication
---------------------------
  init_db()                          → None   Create both tables if missing.
  register_user(username, password)  → bool   True if registered, False if username taken.
  verify_user(username, password)    → bool   True if credentials match.
  user_exists(username)              → bool   Check if a username is already taken.

Public API — Classification history
------------------------------------
  insert_record(...)    → None   Save one classification result.
  fetch_history(n)      → list   Return the n most recent records as dicts.
  clear_history()       → None   Delete all records.
  record_count()        → int    Total number of records stored.
"""

import hashlib
import sqlite3
import os
from datetime import datetime, timezone

# ── Database file location ────────────────────────────────────────────────────
# Stored in the project root, next to app.py.
# os.path.dirname(__file__) points to the data/ folder, so we go up one level.
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "history.db")


def _connect() -> sqlite3.Connection:
    """
    Open a connection to the SQLite database.

    check_same_thread=False is required because Streamlit may call database
    functions from different threads. SQLite handles this safely for the
    read-heavy, low-concurrency usage pattern of this app.
    """
    return sqlite3.connect(_DB_PATH, check_same_thread=False)


def _hash_password(password: str) -> str:
    """Return the SHA-256 hex digest of a plain-text password."""
    return hashlib.sha256(password.encode()).hexdigest()


# ── Table creation ────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create the users and classifications tables if they do not already exist.

    This is safe to call every time the app starts — it is a no-op if the
    tables are already present (CREATE TABLE IF NOT EXISTS).

    Also seeds a default admin account on first run so the app is never
    left with zero users.
    """
    with _connect() as conn:
        # Users table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                username      TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                created_at    TEXT    NOT NULL
            )
            """
        )

        # Classifications table — now includes username column
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS classifications (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                username          TEXT    NOT NULL DEFAULT '',
                timestamp         TEXT    NOT NULL,
                filename          TEXT    NOT NULL,
                predicted_label   TEXT    NOT NULL,
                confidence        REAL    NOT NULL,
                recommendation    TEXT    NOT NULL
            )
            """
        )
        conn.commit()

    # Seed a default admin account if no users exist yet
    if not user_exists("admin"):
        register_user("admin", "ecosort123")


# ── Authentication helpers ────────────────────────────────────────────────────

def user_exists(username: str) -> bool:
    """Return True if a user with this username already exists."""
    with _connect() as conn:
        cursor = conn.execute(
            "SELECT 1 FROM users WHERE username = ? LIMIT 1",
            (username.strip().lower(),),
        )
        return cursor.fetchone() is not None


def register_user(username: str, password: str) -> bool:
    """
    Create a new user account.

    Parameters
    ----------
    username : Desired username (will be stored lowercase, stripped).
    password : Plain-text password (stored as SHA-256 hash — never plain text).

    Returns
    -------
    True  — account created successfully.
    False — username is already taken.
    """
    username = username.strip().lower()
    if user_exists(username):
        return False

    created_at = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, _hash_password(password), created_at),
        )
        conn.commit()
    return True


def verify_user(username: str, password: str) -> bool:
    """
    Check if the username + password combination is correct.

    Parameters
    ----------
    username : Username to look up (case-insensitive).
    password : Plain-text password to verify against the stored hash.

    Returns
    -------
    True if the credentials match, False otherwise.
    """
    username = username.strip().lower()
    with _connect() as conn:
        cursor = conn.execute(
            "SELECT password_hash FROM users WHERE username = ? LIMIT 1",
            (username,),
        )
        row = cursor.fetchone()
    if row is None:
        return False
    return row[0] == _hash_password(password)


# ── Classification history helpers ────────────────────────────────────────────

def insert_record(
    filename: str,
    predicted_label: str,
    confidence: float,
    recommendation: str,
    username: str = "",
) -> None:
    """
    Insert one classification result into the database.

    Parameters
    ----------
    filename        : Original uploaded file name (e.g. "bottle.jpg").
    predicted_label : The waste category string from the model (e.g. "plastic").
    confidence      : Float between 0.0 and 1.0 from the model's softmax output.
    recommendation  : The recycling tip string from recommendations.py.
    username        : The logged-in user (stored for per-user history filtering).
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO classifications
                (username, timestamp, filename, predicted_label, confidence, recommendation)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (username.strip().lower(), timestamp, filename,
             predicted_label, float(confidence), recommendation),
        )
        conn.commit()


def fetch_history(limit: int = 100) -> list[dict]:
    """
    Return the most recent classification records as a list of dicts.

    Parameters
    ----------
    limit : Maximum number of rows to return (default 100). Most recent first.

    Returns
    -------
    A list of dicts with keys:
      id, username, timestamp, filename, predicted_label, confidence, recommendation
    """
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            """
            SELECT id, username, timestamp, filename, predicted_label,
                   confidence, recommendation
            FROM classifications
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]


def clear_history() -> None:
    """
    Delete all classification records from the database.

    This action is irreversible. The UI should ask the user to confirm
    before calling this function.
    """
    with _connect() as conn:
        conn.execute("DELETE FROM classifications")
        conn.commit()


def record_count() -> int:
    """Return the total number of classification records stored."""
    with _connect() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM classifications")
        return cursor.fetchone()[0]
