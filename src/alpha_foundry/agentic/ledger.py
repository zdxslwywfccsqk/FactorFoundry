from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from .schemas import utc_now
from .utils import to_jsonable


class EventLedger:
    """Append-only event log for agent, tool, and split-guard activity."""

    def __init__(self, run_dir: Path, run_id: str):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.db_path = self.run_dir / "ledger.sqlite"
        self.events_path = self.run_dir / "events.jsonl"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    task_id TEXT,
                    artifact_id TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(run_id, event_type);
                """
            )

    def append(
        self,
        event_type: str,
        actor: str,
        payload: dict[str, Any] | None = None,
        task_id: str | None = None,
        artifact_id: str | None = None,
    ) -> str:
        event_id = f"evt-{uuid.uuid4().hex[:12]}"
        created_at = utc_now()
        clean_payload = to_jsonable(payload or {})
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO events(event_id, run_id, created_at, actor, event_type, task_id, artifact_id, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    self.run_id,
                    created_at,
                    actor,
                    event_type,
                    task_id,
                    artifact_id,
                    json.dumps(clean_payload, ensure_ascii=False),
                ),
            )
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "event_id": event_id,
                        "run_id": self.run_id,
                        "created_at": created_at,
                        "actor": actor,
                        "event_type": event_type,
                        "task_id": task_id,
                        "artifact_id": artifact_id,
                        "payload": clean_payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return event_id

    def query(self, event_type: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
        query = "SELECT * FROM events WHERE run_id = ?"
        params: list[Any] = [self.run_id]
        if event_type is not None:
            query += " AND event_type = ?"
            params.append(event_type)
        query += " ORDER BY created_at DESC, event_id DESC LIMIT ?"
        params.append(int(limit))
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "created_at": row["created_at"],
                "actor": row["actor"],
                "event_type": row["event_type"],
                "task_id": row["task_id"],
                "artifact_id": row["artifact_id"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]
