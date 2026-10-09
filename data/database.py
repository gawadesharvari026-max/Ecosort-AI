"""
data/database.py — EcoSort AI
================================
SQLite helpers for storing and retrieving classification history.

Why SQLite?
  sqlite3 is part of Python's standard library — no extra install needed.
  The database file (history.db) is created automatically in the project
  root the first time the app runs.

Schema
------
Table: classifications
  id            INTEGER  Primary key, auto-incremented.
  timestamp     TEXT     ISO-8601 datetime string (UTC).
  filename      TEXT     Original uploaded filename (or "camera" etc.).
  predicted_label TEXT   The waste category returned by the model.
  confidence    REAL     Model confidence score, 0.0 – 1.0.
  recommendation TEXT    The recycling tip shown to the user.

Public API
----------
  init_db()             → None   Create the table if it does not exist.
  insert_record(...)    → None   Save one classification result.
  fetch_history(n)      → list   Return the n most recent records as dicts.
  clear_history()       → None   Delete all records (used from the History page).
  record_count()        → int    Total number of records stored.
"""

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


def init_db() -> None:
    """
    Create the classifications table if it does not already exist.

    This is safe to call every time the app starts — it is a no-op if the
    table is already present (CREATE TABLE IF NOT EXISTS).
    """
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS classifications (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp         TEXT    NOT NULL,
                filename          TEXT    NOT NULL,
                predicted_label   TEXT    NOT NULL,
                confidence        REAL    NOT NULL,
                recommendation    TEXT    NOT NULL
            )
            """
        )
        conn.commit()


def insert_record(
    filename: str,
    predicted_label: str,
    confidence: float,
    recommendation: str,
) -> None:
    """
    Insert one classification result into the database.

    Parameters
    ----------
    filename        : Original uploaded file name (e.g. "bottle.jpg").
    predicted_label : The waste category string from the model (e.g. "plastic").
    confidence      : Float between 0.0 and 1.0 from the model's softmax output.
    recommendation  : The recycling tip string from recommendations.py.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO classifications
                (timestamp, filename, predicted_label, confidence, recommendation)
            VALUES (?, ?, ?, ?, ?)
            """,
            (timestamp, filename, predicted_label, float(confidence), recommendation),
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
      id, timestamp, filename, predicted_label, confidence, recommendation
    """
    with _connect() as conn:
        conn.row_factory = sqlite3.Row  # enables dict-like column access
        cursor = conn.execute(
            """
            SELECT id, timestamp, filename, predicted_label, confidence, recommendation
            FROM classifications
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        # Convert sqlite3.Row objects to plain dicts for easy use with pandas
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
