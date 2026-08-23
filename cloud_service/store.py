from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    status: str
    created_at: str
    updated_at: str
    request: dict[str, Any]
    artifact_path: str | None = None
    error: str | None = None


class ExperimentStore:
    def __init__(self, path: str | Path = "artifacts/experiments.db") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    artifact_path TEXT,
                    error TEXT
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def create(self, experiment_id: str, request: dict[str, Any]) -> ExperimentRecord:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            con.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?)",
                (experiment_id, "queued", now, now, json.dumps(request), None, None),
            )
        return self.get(experiment_id)

    def update(
        self,
        experiment_id: str,
        *,
        status: str,
        artifact_path: str | None = None,
        error: str | None = None,
    ) -> ExperimentRecord:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as con:
            con.execute(
                """
                UPDATE experiments
                SET status = ?, updated_at = ?, artifact_path = COALESCE(?, artifact_path), error = ?
                WHERE experiment_id = ?
                """,
                (status, now, artifact_path, error, experiment_id),
            )
        return self.get(experiment_id)

    def get(self, experiment_id: str) -> ExperimentRecord:
        with self._connect() as con:
            row = con.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        if row is None:
            raise KeyError(experiment_id)
        return ExperimentRecord(
            experiment_id=row["experiment_id"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            request=json.loads(row["request_json"]),
            artifact_path=row["artifact_path"],
            error=row["error"],
        )

    def list(self, limit: int = 50) -> list[ExperimentRecord]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT experiment_id FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self.get(row["experiment_id"]) for row in rows]
