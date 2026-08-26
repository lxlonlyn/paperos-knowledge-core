"""Low-concurrency SQLite operational job queue."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from paperos_core.domain.documents import utc_now
from paperos_core.errors import JobQueueError, public_diagnostic
from paperos_core.paths import DataPaths
from paperos_core.storage.path_refs import DataPathCodec


class OperationalJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    job_type: str
    payload: dict[str, Any]
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    error: str | None = None
    result: dict[str, Any] | None = None


class JobQueue:
    def __init__(self, paths: DataPaths) -> None:
        self.paths = paths
        self.path_codec = DataPathCodec(paths.root)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.paths.registry_db, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def public_dict(self, job: OperationalJob) -> dict[str, Any]:
        payload = job.model_dump(mode="json")
        job_payload = payload.get("payload")
        if isinstance(job_payload, dict):
            job_payload.pop("path", None)
        payload["error"] = (
            public_diagnostic("operational_job_failed")
            if job.status == "failed"
            else None
        )
        return payload

    def enqueue(self, job_type: str, payload: dict[str, Any] | None = None) -> OperationalJob:
        job = OperationalJob(
            id=f"opjob_{uuid.uuid4().hex}",
            job_type=job_type,
            payload=payload or {},
        )
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO operational_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.id,
                    job.job_type,
                    json.dumps(self._persistent_payload(job)),
                    job.status,
                    job.created_at.isoformat(),
                    job.updated_at.isoformat(),
                    None,
                    None,
                ),
            )
        return job

    def claim_next(self) -> OperationalJob | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM operational_jobs WHERE status='pending' "
                "ORDER BY created_at, id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            now = utc_now().isoformat()
            connection.execute(
                "UPDATE operational_jobs SET status='running', updated_at=? WHERE id=?",
                (now, row["id"]),
            )
        return self.get(str(row["id"]))

    def complete(self, job_id: str, result: dict[str, Any]) -> OperationalJob:
        return self._finish(job_id, "completed", result=result)

    def fail(self, job_id: str, error: str) -> OperationalJob:
        return self._finish(job_id, "failed", error=error)

    def _finish(
        self,
        job_id: str,
        status: Literal["completed", "failed"],
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> OperationalJob:
        with self._connect() as connection:
            connection.execute(
                "UPDATE operational_jobs SET status=?, updated_at=?, error=?, result=? "
                "WHERE id=?",
                (
                    status,
                    utc_now().isoformat(),
                    error,
                    json.dumps(result) if result is not None else None,
                    job_id,
                ),
            )
        return self.get(job_id)

    def get(self, job_id: str) -> OperationalJob:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM operational_jobs WHERE id=?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobQueueError(
                f"Operational job '{job_id}' does not exist.", affected=job_id
            )
        return self._from_row(row)

    def list_jobs(self, *, limit: int = 100) -> list[OperationalJob]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM operational_jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def _from_row(self, row: sqlite3.Row) -> OperationalJob:
        return OperationalJob(
            id=row["id"],
            job_type=row["job_type"],
            payload=self._runtime_payload(json.loads(row["payload"])),
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            error=row["error"],
            result=json.loads(row["result"]) if row["result"] else None,
        )
    def _persistent_payload(self, job: OperationalJob) -> dict[str, Any]:
        payload = dict(job.payload)
        if job.job_type == "ingest" and "path" in payload:
            payload["path"] = self.path_codec.encode(Path(str(payload["path"])))
        return payload

    def _runtime_payload(self, payload: object) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise JobQueueError("Operational job payload is not a JSON object.")
        result = {str(key): value for key, value in payload.items()}
        if "path" in result:
            result["path"] = str(self.path_codec.decode(str(result["path"])))
        return result
