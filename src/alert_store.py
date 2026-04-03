"""
Alert State Store
Persists alert investigation states to a local SQLite database so
status survives page reloads and session changes.

States: Unreviewed | Investigating | False Positive | Escalated
"""
import sqlite3
import os
from datetime import datetime, timezone
from typing import Dict

_DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'alert_state.db')

VALID_STATES = ["Unreviewed", "Investigating", "False Positive", "Escalated"]


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init() -> None:
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS alert_states (
                account_id TEXT PRIMARY KEY,
                state      TEXT NOT NULL DEFAULT 'Unreviewed',
                note       TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL
            )
        """)
        c.commit()


def get_alert_state(account_id: str) -> dict:
    """Return the alert state dict for one account (defaults to Unreviewed)."""
    _init()
    with _conn() as c:
        row = c.execute(
            "SELECT * FROM alert_states WHERE account_id = ?", (str(account_id),)
        ).fetchone()
    if row:
        return dict(row)
    return {'account_id': account_id, 'state': 'Unreviewed', 'note': '', 'updated_at': None}


def set_alert_state(account_id: str, state: str, note: str = '') -> None:
    """Upsert the alert state for one account."""
    _init()
    if state not in VALID_STATES:
        raise ValueError(f"Invalid state '{state}'. Must be one of {VALID_STATES}")
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute("""
            INSERT INTO alert_states (account_id, state, note, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(account_id) DO UPDATE SET
                state      = excluded.state,
                note       = excluded.note,
                updated_at = excluded.updated_at
        """, (str(account_id), state, note, now))
        c.commit()


def get_all_states() -> Dict[str, dict]:
    """Return a dict mapping account_id → state dict for all tracked accounts."""
    _init()
    with _conn() as c:
        rows = c.execute("SELECT * FROM alert_states").fetchall()
    return {r['account_id']: dict(r) for r in rows}
