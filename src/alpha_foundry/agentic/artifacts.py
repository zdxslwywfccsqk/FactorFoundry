from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .schemas import ArtifactKind, ArtifactRecord, ArtifactStatus, SplitName, utc_now
from .utils import content_hash, to_jsonable


class ArtifactStore:
    """Small, append-only artifact store backed by SQLite and JSON files.

    Artifacts are the public contract between agents. Agents never mutate an
    existing artifact; a revision is a new artifact with the old id in parents.
    """

    def __init__(self, run_dir: Path, run_id: str):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.artifacts_dir = self.run_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.run_dir / "ledger.sqlite"
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    status TEXT NOT NULL,
                    producer TEXT NOT NULL,
                    parents_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    schema_version TEXT NOT NULL,
                    split TEXT,
                    content_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_artifacts_kind ON artifacts(run_id, kind);
                CREATE INDEX IF NOT EXISTS idx_artifacts_status ON artifacts(run_id, status);
                """
            )

    def _next_sequence(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM artifacts WHERE run_id = ?", (self.run_id,)
            ).fetchone()
        return int(row["n"]) + 1

    @staticmethod
    def _kind(kind: ArtifactKind | str) -> ArtifactKind:
        return kind if isinstance(kind, ArtifactKind) else ArtifactKind(kind)

    @staticmethod
    def _status(status: ArtifactStatus | str) -> ArtifactStatus:
        return status if isinstance(status, ArtifactStatus) else ArtifactStatus(status)

    @staticmethod
    def _split(split: SplitName | str | None) -> SplitName | None:
        if split is None:
            return None
        return split if isinstance(split, SplitName) else SplitName(split)

    def put(
        self,
        kind: ArtifactKind | str,
        payload: dict[str, Any],
        producer: str,
        status: ArtifactStatus | str = ArtifactStatus.COMPLETED,
        parents: Iterable[str] | None = None,
        split: SplitName | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        kind_value = self._kind(kind)
        status_value = self._status(status)
        split_value = self._split(split)
        clean_payload = to_jsonable(payload)
        record = ArtifactRecord(
            artifact_id=f"{self.run_id}-{kind_value.value}-{self._next_sequence():04d}",
            run_id=self.run_id,
            kind=kind_value,
            status=status_value,
            producer=producer,
            parents=[str(item) for item in (parents or [])],
            created_at=utc_now(),
            split=split_value,
            content_hash=content_hash(clean_payload),
            payload=clean_payload,
            metadata=to_jsonable(metadata or {}),
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO artifacts(
                    artifact_id, run_id, kind, status, producer, parents_json,
                    created_at, schema_version, split, content_hash,
                    payload_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.artifact_id,
                    record.run_id,
                    record.kind.value,
                    record.status.value,
                    record.producer,
                    json.dumps(record.parents, ensure_ascii=False),
                    record.created_at,
                    record.schema_version,
                    record.split.value if record.split else None,
                    record.content_hash,
                    json.dumps(record.payload, ensure_ascii=False),
                    json.dumps(record.metadata, ensure_ascii=False),
                ),
            )
        path = self.artifacts_dir / f"{record.artifact_id}.json"
        path.write_text(
            json.dumps(record.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return record

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=row["artifact_id"],
            run_id=row["run_id"],
            kind=ArtifactKind(row["kind"]),
            status=ArtifactStatus(row["status"]),
            producer=row["producer"],
            parents=json.loads(row["parents_json"]),
            created_at=row["created_at"],
            schema_version=row["schema_version"],
            split=SplitName(row["split"]) if row["split"] else None,
            content_hash=row["content_hash"],
            payload=json.loads(row["payload_json"]),
            metadata=json.loads(row["metadata_json"]),
        )

    def get(self, artifact_id: str) -> ArtifactRecord:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_id = ?", (artifact_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown artifact: {artifact_id}")
        return self._row_to_record(row)

    def list(
        self,
        kind: ArtifactKind | str | None = None,
        status: ArtifactStatus | str | None = None,
    ) -> list[ArtifactRecord]:
        clauses = ["run_id = ?"]
        params: list[Any] = [self.run_id]
        if kind is not None:
            clauses.append("kind = ?")
            params.append(self._kind(kind).value)
        if status is not None:
            clauses.append("status = ?")
            params.append(self._status(status).value)
        query = "SELECT * FROM artifacts WHERE " + " AND ".join(clauses) + " ORDER BY created_at, artifact_id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def latest(
        self,
        kind: ArtifactKind | str,
        status: ArtifactStatus | str | None = None,
    ) -> ArtifactRecord | None:
        records = self.list(kind=kind, status=status)
        return records[-1] if records else None

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM artifacts WHERE run_id = ?", (self.run_id,)
            ).fetchone()
        return int(row["n"])

    def lineage(self, artifact_id: str) -> list[ArtifactRecord]:
        """Return ancestors in breadth-first order, oldest dependencies first."""
        ordered: list[ArtifactRecord] = []
        seen: set[str] = set()
        queue = [artifact_id]
        while queue:
            current_id = queue.pop(0)
            if current_id in seen:
                continue
            seen.add(current_id)
            record = self.get(current_id)
            ordered.append(record)
            queue.extend(record.parents)
        ordered.reverse()
        return ordered
