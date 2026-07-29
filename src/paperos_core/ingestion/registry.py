"""SQLite source registry and immutable raw-PDF repository."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from paperos_core.domain.documents import IngestionJob, SourceFile, utc_now
from paperos_core.domain.enums import IngestionJobStatus
from paperos_core.domain.ids import ingestion_job_id, source_file_id
from paperos_core.errors import (
    JobNotFoundError,
    SourceChangedError,
    SourceNotFoundError,
    SourceRegistryError,
    StorageIntegrityError,
)
from paperos_core.ingestion.validation import ValidatedPDF, calculate_sha256
from paperos_core.jobs.state import validate_transition
from paperos_core.paths import DataPaths

_COPY_CHUNK_SIZE = 1024 * 1024


class SourceRegistry:
    """Own SourceFile records, ingestion jobs, and immutable source bytes."""

    def __init__(self, paths: DataPaths) -> None:
        self.paths = paths
        self.paths.initialize()
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.paths.registry_db, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_schema(self) -> None:
        try:
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS source_files (
                        id TEXT PRIMARY KEY,
                        sha256 TEXT NOT NULL UNIQUE,
                        original_filename TEXT NOT NULL,
                        stored_filename TEXT NOT NULL,
                        media_type TEXT NOT NULL,
                        size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
                        storage_path TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL,
                        schema_version TEXT NOT NULL,
                        id_version TEXT NOT NULL,
                        source_url TEXT,
                        user_metadata TEXT,
                        dataset_id TEXT
                    );
                    CREATE TABLE IF NOT EXISTS ingestion_jobs (
                        id TEXT PRIMARY KEY,
                        source_file_id TEXT NOT NULL,
                        dataset_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        current_operation TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        error_code TEXT,
                        error_message TEXT,
                        completed_at TEXT,
                        requested_options TEXT,
                        schema_version TEXT NOT NULL,
                        id_version TEXT NOT NULL,
                        FOREIGN KEY (source_file_id) REFERENCES source_files(id)
                    );
                    CREATE INDEX IF NOT EXISTS ingestion_jobs_source_idx
                        ON ingestion_jobs(source_file_id);
                    CREATE INDEX IF NOT EXISTS ingestion_jobs_status_idx
                        ON ingestion_jobs(status);
                    """
                )
        except sqlite3.Error as exc:
            raise SourceRegistryError(
                f"Unable to initialize source registry: {exc}",
                affected=self.paths.registry_db,
            ) from exc

    @staticmethod
    def _json_dump(value: dict[str, Any] | None) -> str | None:
        return json.dumps(value, ensure_ascii=False, sort_keys=True) if value is not None else None

    @staticmethod
    def _json_load(value: str | None) -> dict[str, Any] | None:
        return json.loads(value) if value is not None else None

    @classmethod
    def _source_from_row(cls, row: sqlite3.Row) -> SourceFile:
        return SourceFile(
            id=row["id"],
            sha256=row["sha256"],
            original_filename=row["original_filename"],
            stored_filename=row["stored_filename"],
            media_type=row["media_type"],
            size_bytes=row["size_bytes"],
            storage_path=Path(row["storage_path"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            schema_version=row["schema_version"],
            id_version=row["id_version"],
            source_url=row["source_url"],
            user_metadata=cls._json_load(row["user_metadata"]),
            dataset_id=row["dataset_id"],
        )

    @classmethod
    def _job_from_row(cls, row: sqlite3.Row) -> IngestionJob:
        return IngestionJob(
            id=row["id"],
            source_file_id=row["source_file_id"],
            dataset_id=row["dataset_id"],
            status=IngestionJobStatus(row["status"]),
            current_operation=row["current_operation"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            attempt_count=row["attempt_count"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
            requested_options=cls._json_load(row["requested_options"]),
            schema_version=row["schema_version"],
            id_version=row["id_version"],
        )

    def get_source(self, source_id: str) -> SourceFile:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM source_files WHERE id = ?", (source_id,)
            ).fetchone()
        if row is None:
            raise SourceNotFoundError(
                f"SourceFile '{source_id}' is not registered.", affected=source_id
            )
        return self._source_from_row(row)

    def find_source_by_sha256(self, sha256: str) -> SourceFile | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM source_files WHERE sha256 = ?", (sha256.lower(),)
            ).fetchone()
        return self._source_from_row(row) if row is not None else None

    def _verify_stored_source(self, source: SourceFile) -> None:
        stored = source.storage_path
        self.paths.assert_within_root(stored)
        if not stored.is_file():
            raise StorageIntegrityError(
                "Registered immutable source PDF is missing.", affected=stored
            )
        if stored.stat().st_size != source.size_bytes:
            raise StorageIntegrityError(
                "Registered immutable source PDF has an unexpected size.",
                affected=stored,
            )
        actual = calculate_sha256(stored)
        if actual != source.sha256:
            raise StorageIntegrityError(
                "Registered immutable source PDF checksum does not match its SourceFile record.",
                affected=stored,
                details={"expected_sha256": source.sha256, "actual_sha256": actual},
            )

    def _persist_immutable_pdf(self, validated: ValidatedPDF, target: Path) -> None:
        self.paths.assert_within_root(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not target.is_file() or target.stat().st_size != validated.size_bytes:
                raise StorageIntegrityError(
                    "Immutable source target already exists with different content.",
                    affected=target,
                )
            if calculate_sha256(target) != validated.sha256:
                raise StorageIntegrityError(
                    "Immutable source target already exists with a different checksum.",
                    affected=target,
                )
            return

        temp_name: str | None = None
        try:
            digest = hashlib.sha256()
            copied = 0
            with (
                validated.path.open("rb") as source,
                tempfile.NamedTemporaryFile(
                    mode="wb", prefix=".source-", suffix=".tmp", dir=target.parent, delete=False
                ) as temporary,
            ):
                temp_name = temporary.name
                for block in iter(lambda: source.read(_COPY_CHUNK_SIZE), b""):
                    temporary.write(block)
                    digest.update(block)
                    copied += len(block)
                temporary.flush()
                os.fsync(temporary.fileno())
            if copied != validated.size_bytes or digest.hexdigest() != validated.sha256:
                raise SourceChangedError(
                    "Source PDF changed while it was being copied; no raw PDF was registered.",
                    affected=validated.path,
                )
            os.chmod(temp_name, 0o444)
            try:
                os.link(temp_name, target)
            except FileExistsError:
                if (
                    target.stat().st_size != validated.size_bytes
                    or calculate_sha256(target) != validated.sha256
                ):
                    raise StorageIntegrityError(
                        "Concurrent immutable source registration produced conflicting content.",
                        affected=target,
                    )
        except OSError as exc:
            if isinstance(exc, (SourceChangedError, StorageIntegrityError)):
                raise
            raise SourceRegistryError(
                f"Unable to preserve immutable source PDF: {exc}", affected=target
            ) from exc
        finally:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)

    def register_source(
        self,
        validated: ValidatedPDF,
        *,
        dataset_id: str,
        user_metadata: dict[str, Any] | None = None,
    ) -> tuple[SourceFile, bool]:
        existing = self.find_source_by_sha256(validated.sha256)
        if existing is not None:
            self._verify_stored_source(existing)
            return existing, True

        source_id = source_file_id(validated.sha256)
        target = (self.paths.raw / source_id / "source.pdf").resolve(strict=False)
        source = SourceFile(
            id=source_id,
            sha256=validated.sha256,
            original_filename=validated.original_filename,
            media_type=validated.media_type,
            size_bytes=validated.size_bytes,
            storage_path=target,
            user_metadata=user_metadata,
            dataset_id=dataset_id,
        )
        self._persist_immutable_pdf(validated, target)
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO source_files (
                        id, sha256, original_filename, stored_filename, media_type,
                        size_bytes, storage_path, created_at, schema_version, id_version,
                        source_url, user_metadata, dataset_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        source.id,
                        source.sha256,
                        source.original_filename,
                        source.stored_filename,
                        source.media_type,
                        source.size_bytes,
                        str(source.storage_path),
                        source.created_at.isoformat(),
                        source.schema_version,
                        source.id_version,
                        source.source_url,
                        self._json_dump(source.user_metadata),
                        source.dataset_id,
                    ),
                )
        except sqlite3.IntegrityError:
            existing = self.find_source_by_sha256(validated.sha256)
            if existing is None:
                raise SourceRegistryError(
                    "Source registration conflicted without a recoverable SourceFile record.",
                    affected=self.paths.registry_db,
                )
            self._verify_stored_source(existing)
            return existing, True
        except sqlite3.Error as exc:
            raise SourceRegistryError(
                f"Unable to register SourceFile: {exc}", affected=self.paths.registry_db
            ) from exc
        return source, False

    def create_job(
        self,
        source_file_id: str,
        *,
        dataset_id: str,
        requested_options: dict[str, Any] | None = None,
    ) -> IngestionJob:
        self.get_source(source_file_id)
        job = IngestionJob(
            id=ingestion_job_id(),
            source_file_id=source_file_id,
            dataset_id=dataset_id,
            requested_options=requested_options,
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO ingestion_jobs (
                        id, source_file_id, dataset_id, status, current_operation,
                        created_at, updated_at, attempt_count, error_code, error_message,
                        completed_at, requested_options, schema_version, id_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.id,
                        job.source_file_id,
                        job.dataset_id,
                        job.status.value,
                        job.current_operation,
                        job.created_at.isoformat(),
                        job.updated_at.isoformat(),
                        job.attempt_count,
                        job.error_code,
                        job.error_message,
                        None,
                        self._json_dump(job.requested_options),
                        job.schema_version,
                        job.id_version,
                    ),
                )
        except sqlite3.Error as exc:
            raise SourceRegistryError(
                f"Unable to create IngestionJob: {exc}", affected=self.paths.registry_db
            ) from exc
        return job

    def get_job(self, job_id: str) -> IngestionJob:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM ingestion_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise JobNotFoundError(f"IngestionJob '{job_id}' does not exist.", affected=job_id)
        return self._job_from_row(row)

    def update_job(
        self,
        job_id: str,
        *,
        status: IngestionJobStatus,
        current_operation: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> IngestionJob:
        current = self.get_job(job_id)
        validate_transition(current.status, status)
        updated_at = utc_now()
        completed_at = updated_at if status == IngestionJobStatus.COMPLETED else None
        attempt_count = current.attempt_count + (
            1
            if current.status == IngestionJobStatus.FAILED and status == IngestionJobStatus.PENDING
            else 0
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE ingestion_jobs
                SET status = ?, current_operation = ?, updated_at = ?, attempt_count = ?,
                    error_code = ?, error_message = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    status.value,
                    current_operation,
                    updated_at.isoformat(),
                    attempt_count,
                    error_code,
                    error_message,
                    completed_at.isoformat() if completed_at else None,
                    job_id,
                ),
            )
        return self.get_job(job_id)

    def list_jobs(self, *, limit: int = 20) -> list[IngestionJob]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM ingestion_jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def status(self, *, limit: int = 20) -> dict[str, Any]:
        with self._connect() as connection:
            source_count = connection.execute("SELECT COUNT(*) FROM source_files").fetchone()[0]
            job_count = connection.execute("SELECT COUNT(*) FROM ingestion_jobs").fetchone()[0]
            status_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM ingestion_jobs GROUP BY status"
            ).fetchall()
        return {
            "data_dir": str(self.paths.root),
            "registry_db": str(self.paths.registry_db),
            "source_file_count": source_count,
            "ingestion_job_count": job_count,
            "jobs_by_status": {row["status"]: row["count"] for row in status_rows},
            "recent_jobs": [job.model_dump(mode="json") for job in self.list_jobs(limit=limit)],
        }
