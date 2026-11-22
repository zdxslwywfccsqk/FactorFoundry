from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .factors import FactorSpec


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value) if isinstance(value, float) else False:
        return None
    return value


class ExperimentMemory:
    def __init__(self, experiments_dir: Path):
        self.experiments_dir = experiments_dir
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = experiments_dir / "memory.sqlite"
        self.events_path = experiments_dir / "agent_events.jsonl"
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def init_db(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    data_fingerprint TEXT,
                    status TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS agent_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    action TEXT NOT NULL,
                    payload_json TEXT
                );
                CREATE TABLE IF NOT EXISTS factor_results (
                    run_id TEXT NOT NULL,
                    factor_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    family TEXT NOT NULL,
                    category TEXT NOT NULL,
                    hypothesis TEXT,
                    formula TEXT,
                    direction REAL,
                    window INTEGER,
                    status TEXT,
                    metrics_json TEXT,
                    train_json TEXT,
                    validation_json TEXT,
                    test_json TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, factor_id)
                );
                CREATE TABLE IF NOT EXISTS portfolio_results (
                    run_id TEXT NOT NULL,
                    portfolio_id TEXT NOT NULL,
                    factors_json TEXT,
                    metrics_json TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, portfolio_id)
                );
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def start_run(self, run_id: str, config: dict, data_fingerprint: str) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO runs(run_id, created_at, config_json, data_fingerprint, status) VALUES (?, ?, ?, ?, ?)",
                (run_id, self._now(), json.dumps(_jsonable(config), ensure_ascii=False), data_fingerprint, "running"),
            )

    def finish_run(self, run_id: str, status: str = "completed") -> None:
        with self.connect() as connection:
            connection.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))

    def log_event(self, run_id: str, agent: str, action: str, payload: dict | None = None) -> None:
        created_at = self._now()
        payload = _jsonable(payload or {})
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO agent_events(run_id, created_at, agent, action, payload_json) VALUES (?, ?, ?, ?, ?)",
                (run_id, created_at, agent, action, json.dumps(payload, ensure_ascii=False)),
            )
        with self.events_path.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "created_at": created_at,
                        "agent": agent,
                        "action": action,
                        "payload": payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    def save_factor(
        self,
        run_id: str,
        spec: FactorSpec,
        status: str,
        metrics: dict,
        train_metrics: dict,
        validation_metrics: dict,
        test_metrics: dict,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO factor_results(
                    run_id, factor_id, name, family, category, hypothesis, formula,
                    direction, window, status, metrics_json, train_json,
                    validation_json, test_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    spec.spec_id,
                    spec.name,
                    spec.family,
                    spec.category,
                    spec.hypothesis,
                    spec.formula,
                    float(spec.direction),
                    int(spec.window),
                    status,
                    json.dumps(_jsonable(metrics), ensure_ascii=False),
                    json.dumps(_jsonable(train_metrics), ensure_ascii=False),
                    json.dumps(_jsonable(validation_metrics), ensure_ascii=False),
                    json.dumps(_jsonable(test_metrics), ensure_ascii=False),
                    self._now(),
                ),
            )

    def save_portfolio(
        self,
        run_id: str,
        portfolio_id: str,
        factors: list[str],
        metrics: dict,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO portfolio_results(run_id, portfolio_id, factors_json, metrics_json, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    portfolio_id,
                    json.dumps(factors, ensure_ascii=False),
                    json.dumps(_jsonable(metrics), ensure_ascii=False),
                    self._now(),
                ),
            )

    def events_frame(self, run_id: str) -> pd.DataFrame:
        with self.connect() as connection:
            return pd.read_sql_query(
                "SELECT * FROM agent_events WHERE run_id = ? ORDER BY id", connection, params=(run_id,)
            )
