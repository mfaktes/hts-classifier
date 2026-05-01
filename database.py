"""
Audit Trail Database
--------------------
Compliance work is built on documentation. Every classification decision needs
to be traceable: who classified it, when, what code they picked, and why.
This module provides a simple SQLite-backed audit log.

CBP and most internal compliance programs require classification records to
be retained for at least 5 years. We store:
  - The original product description the user entered
  - The HTS code they ultimately accepted
  - Whether they accepted the top suggestion or overrode it (signal of
    classifier accuracy over time)
  - A free-text note field for the reasoning behind the choice
  - Timestamp
"""

import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "classifications.db"


def init_db():
    """Create the database schema if it doesn't exist yet."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS classifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            product_description TEXT NOT NULL,
            chosen_hts_code TEXT NOT NULL,
            chosen_description TEXT,
            confidence TEXT,
            score REAL,
            top_suggestion_code TEXT,
            was_top_suggestion INTEGER,
            notes TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_classification(
    product_description: str,
    chosen_hts_code: str,
    chosen_description: str,
    confidence: str,
    score: float,
    top_suggestion_code: str,
    notes: str = ""
):
    """Save one classification decision to the audit log."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO classifications (
            timestamp, product_description, chosen_hts_code, chosen_description,
            confidence, score, top_suggestion_code, was_top_suggestion, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(timespec="seconds") + "Z",
        product_description,
        chosen_hts_code,
        chosen_description,
        confidence,
        score,
        top_suggestion_code,
        1 if chosen_hts_code == top_suggestion_code else 0,
        notes,
    ))
    conn.commit()
    conn.close()


def get_all_classifications():
    """Return all saved classifications, newest first."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, timestamp, product_description, chosen_hts_code,
               chosen_description, confidence, score, top_suggestion_code,
               was_top_suggestion, notes
        FROM classifications
        ORDER BY id DESC
    """)
    rows = cur.fetchall()
    conn.close()
    cols = ["id", "timestamp", "product_description", "chosen_hts_code",
            "chosen_description", "confidence", "score",
            "top_suggestion_code", "was_top_suggestion", "notes"]
    return [dict(zip(cols, r)) for r in rows]


def delete_classification(record_id: int):
    """Remove an audit entry (for cleanup during testing)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM classifications WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
