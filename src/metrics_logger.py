"""
Metrics Logger for AML Inference API

Logs inference metrics to SQLite database for real-time dashboard visualization.
Also maintains an append-only audit_log table for security and compliance.
"""

import hashlib
import sqlite3
import json
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

class MetricsLogger:
    """Thread-safe metrics logger using SQLite."""
    
    def __init__(self, db_path: str = "metrics.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for safe concurrent writes
            conn.execute("PRAGMA journal_mode=WAL")

            cursor = conn.cursor()

            # Audit log — append-only, never UPDATE or DELETE rows here
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_time  TEXT    NOT NULL,
                    event_type  TEXT    NOT NULL,
                    client_ip   TEXT,
                    key_hash    TEXT,
                    endpoint    TEXT,
                    input_hash  TEXT,
                    outcome     TEXT,
                    detail      TEXT
                )
            """)

            # Inference logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    account_id TEXT,
                    endpoint TEXT,
                    engine TEXT,
                    latency_ms REAL,
                    risk_score REAL,
                    risk_level TEXT,
                    component_scores TEXT,
                    status TEXT,
                    error TEXT,
                    requester_key_hash TEXT,
                    input_payload_hash TEXT,
                    client_ip TEXT
                )
            """)

            # Migrate existing inference_logs tables that predate Phase 1 security columns
            existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(inference_logs)")}
            for col, typedef in [
                ("requester_key_hash", "TEXT"),
                ("input_payload_hash", "TEXT"),
                ("client_ip",          "TEXT"),
            ]:
                if col not in existing_cols:
                    cursor.execute(f"ALTER TABLE inference_logs ADD COLUMN {col} {typedef}")
            
            # Engine throughput table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS engine_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    engine TEXT NOT NULL,
                    operation TEXT,
                    count INTEGER DEFAULT 1,
                    latency_ms REAL
                )
            """)
            
            # KPI aggregates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS kpi_aggregates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_accounts INTEGER DEFAULT 0,
                    total_transactions INTEGER DEFAULT 0,
                    total_events INTEGER DEFAULT 0,
                    high_risk_count INTEGER DEFAULT 0,
                    medium_risk_count INTEGER DEFAULT 0,
                    low_risk_count INTEGER DEFAULT 0,
                    clean_count INTEGER DEFAULT 0,
                    total_amount_at_risk REAL DEFAULT 0.0,
                    avg_latency_ms REAL DEFAULT 0.0
                )
            """)
            
            # Link predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS link_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source_account TEXT,
                    target_account TEXT,
                    probability REAL,
                    risk_score REAL
                )
            """)
            
            conn.commit()
    
    @staticmethod
    def _sha256(value: str) -> str:
        """Return hex SHA-256 of a string."""
        return hashlib.sha256(value.encode()).hexdigest()

    def log_audit_event(
        self,
        event_type: str,
        client_ip: Optional[str] = None,
        key_hash: Optional[str] = None,
        endpoint: Optional[str] = None,
        input_hash: Optional[str] = None,
        outcome: Optional[str] = None,
        detail: Optional[str] = None,
    ):
        """
        Append a row to the immutable audit_log table.

        This method only ever INSERTs — rows are never modified or deleted,
        providing the append-only audit trail required for AML compliance.
        """
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    INSERT INTO audit_log
                        (event_time, event_type, client_ip, key_hash,
                         endpoint, input_hash, outcome, detail)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.utcnow().isoformat(),
                        event_type,
                        client_ip,
                        key_hash,
                        endpoint,
                        input_hash,
                        outcome,
                        detail,
                    ),
                )
                conn.commit()

    def log_inference(self, data: Dict[str, Any]):
        """Log an inference request."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO inference_logs
                    (timestamp, account_id, endpoint, engine, latency_ms,
                     risk_score, risk_level, component_scores, status, error,
                     requester_key_hash, input_payload_hash, client_ip)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    data.get('timestamp', datetime.utcnow().isoformat()),
                    data.get('account_id'),
                    data.get('endpoint'),
                    data.get('engine'),
                    data.get('latency_ms'),
                    data.get('risk_score'),
                    data.get('risk_level'),
                    json.dumps(data.get('component_scores', {})),
                    data.get('status', 'success'),
                    data.get('error'),
                    data.get('requester_key_hash'),
                    data.get('input_payload_hash'),
                    data.get('client_ip'),
                ))
                conn.commit()
    
    def log_engine_activity(self, engine: str, operation: str, latency_ms: float = None, count: int = 1):
        """Log engine-specific activity."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO engine_stats 
                    (timestamp, engine, operation, count, latency_ms)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    datetime.utcnow().isoformat(),
                    engine,
                    operation,
                    count,
                    latency_ms
                ))
                conn.commit()
    
    def log_link_prediction(self, source: str, target: str, probability: float, risk_score: float = None):
        """Log a link prediction."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO link_predictions 
                    (timestamp, source_account, target_account, probability, risk_score)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    datetime.utcnow().isoformat(),
                    source,
                    target,
                    probability,
                    risk_score
                ))
                conn.commit()
    
    def get_recent_inferences(self, limit: int = 100):
        """Get recent inference logs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM inference_logs 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_kpi_stats(self, minutes: int = 60):
        """Get KPI statistics for the last N minutes, falling back to all-time if empty."""
        query = """
            SELECT
                COUNT(DISTINCT account_id) as total_accounts,
                COUNT(*) as total_transactions,
                SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END) as high_risk_count,
                SUM(CASE WHEN risk_level = 'MEDIUM' THEN 1 ELSE 0 END) as medium_risk_count,
                SUM(CASE WHEN risk_level = 'LOW' THEN 1 ELSE 0 END) as low_risk_count,
                SUM(CASE WHEN risk_level = 'CLEAN' THEN 1 ELSE 0 END) as clean_count,
                AVG(latency_ms) as avg_latency_ms
            FROM inference_logs
            {where}
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Try windowed query first
            cursor.execute(
                query.format(where="WHERE datetime(timestamp) >= datetime('now', '-' || ? || ' minutes')"),
                (minutes,)
            )
            row = cursor.fetchone()
            # Fall back to all-time when the window has fewer than 10 transactions
            # (prevents sparse recent data making the dashboard look empty)
            if not row or (row[1] or 0) < 10:
                cursor.execute(query.format(where=""))
                row = cursor.fetchone()
            return {
                'total_accounts': row[0] or 0,
                'total_transactions': row[1] or 0,
                'high_risk_count': row[2] or 0,
                'medium_risk_count': row[3] or 0,
                'low_risk_count': row[4] or 0,
                'clean_count': row[5] or 0,
                'avg_latency_ms': round(row[6] or 0, 2)
            }
    
    def get_engine_stats(self, minutes: int = 60):
        """Get engine throughput statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    engine,
                    COUNT(*) as total_operations,
                    AVG(latency_ms) as avg_latency_ms,
                    MAX(latency_ms) as max_latency_ms
                FROM engine_stats
                WHERE datetime(timestamp) >= datetime('now', '-' || ? || ' minutes')
                GROUP BY engine
            """, (minutes,))
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_top_links(self, limit: int = 10):
        """Get top emerging links by probability."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM link_predictions 
                ORDER BY probability DESC, timestamp DESC
                LIMIT ?
            """, (limit,))
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_latency_trends(self, engine: str = None, limit: int = 50):
        """Get latency trends for visualization."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if engine:
                cursor.execute("""
                    SELECT timestamp, latency_ms, engine
                    FROM engine_stats
                    WHERE engine = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (engine, limit))
            else:
                cursor.execute("""
                    SELECT timestamp, latency_ms, engine
                    FROM engine_stats
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (limit,))
            columns = [desc[0] for desc in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def clear_old_data(self, days: int = 7):
        """Clear data older than N days."""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM inference_logs 
                    WHERE datetime(timestamp) < datetime('now', '-' || ? || ' days')
                """, (days,))
                cursor.execute("""
                    DELETE FROM engine_stats 
                    WHERE datetime(timestamp) < datetime('now', '-' || ? || ' days')
                """, (days,))
                cursor.execute("""
                    DELETE FROM link_predictions 
                    WHERE datetime(timestamp) < datetime('now', '-' || ? || ' days')
                """, (days,))
                conn.commit()


# Global singleton instance
_metrics_logger = None

def get_metrics_logger(db_path: str = "metrics.db") -> MetricsLogger:
    """Get or create the global metrics logger instance."""
    global _metrics_logger
    if _metrics_logger is None:
        _metrics_logger = MetricsLogger(db_path)
    return _metrics_logger
