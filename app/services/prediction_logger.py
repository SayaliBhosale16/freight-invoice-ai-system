import json
import logging
import sqlite3
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PredictionLogger:
    """Logs predictions to a SQLite database for monitoring and drift detection."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                input_data TEXT NOT NULL,
                prediction TEXT NOT NULL,
                latency_ms REAL
            )
        """)
        conn.commit()
        conn.close()

    def log(
        self,
        model_name: str,
        model_version: str,
        input_data: dict,
        prediction: dict,
        latency_ms: float,
    ):
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """INSERT INTO predictions
               (timestamp, model_name, model_version, input_data, prediction, latency_ms)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().isoformat(),
                model_name,
                model_version,
                json.dumps(input_data),
                json.dumps(prediction),
                latency_ms,
            ),
        )
        conn.commit()
        conn.close()

    def get_count_since(self, hours: int = 24) -> int:
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        conn = sqlite3.connect(self.db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE timestamp > ?", (cutoff,)
        ).fetchone()[0]
        conn.close()
        return count

    def get_recent_inputs(self, model_name: str, limit: int = 100) -> list[dict]:
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            """SELECT input_data FROM predictions
               WHERE model_name = ?
               ORDER BY timestamp DESC LIMIT ?""",
            (model_name, limit),
        ).fetchall()
        conn.close()
        return [json.loads(row[0]) for row in rows]
