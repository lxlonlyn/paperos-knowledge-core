"""Shared source-file and ingestion-job domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from paperos_core.domain.enums import IngestionJobStatus
from paperos_core.domain.ids import (
    INGESTION_JOB_ID_VERSION,
    INGESTION_JOB_SCHEMA_VERSION,
    SOURCE_FILE_ID_VERSION,
    SOURCE_FILE_SCHEMA_VERSION,
    normalize_sha256,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class DomainModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("created_at", "updated_at", "completed_at", check_fields=False)
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.tzinfo is None:
            raise ValueError("persisted timestamps must include a timezone")
        return value


class SourceFile(DomainModel):
    id: str
    sha256: str
    original_filename: str
    stored_filename: str = "source.pdf"
    media_type: str = "application/pdf"
    size_bytes: int = Field(ge=1)
    storage_path: Path
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = SOURCE_FILE_SCHEMA_VERSION
    id_version: str = SOURCE_FILE_ID_VERSION
    source_url: str | None = None
    user_metadata: dict[str, Any] | None = None
    dataset_id: str | None = None

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return normalize_sha256(value)

    @field_validator("original_filename", "stored_filename", "media_type")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source-file text fields must not be empty")
        return value


class IngestionJob(DomainModel):
    id: str
    source_file_id: str
    dataset_id: str
    status: IngestionJobStatus = IngestionJobStatus.PENDING
    current_operation: str = "awaiting_parse"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    attempt_count: int = Field(default=0, ge=0)
    error_code: str | None = None
    error_message: str | None = None
    completed_at: datetime | None = None
    requested_options: dict[str, Any] | None = None
    schema_version: str = INGESTION_JOB_SCHEMA_VERSION
    id_version: str = INGESTION_JOB_ID_VERSION

    @field_validator("dataset_id", "current_operation")
    @classmethod
    def require_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("job dataset and current operation must not be empty")
        return value


class IngestionResult(DomainModel):
    source_file: SourceFile
    job: IngestionJob
    duplicate: bool

    @property
    def source_file_id(self) -> str:
        return self.source_file.id

    @property
    def job_id(self) -> str:
        return self.job.id

    @property
    def status(self) -> IngestionJobStatus:
        return self.job.status

    def public_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job.id,
            "source_file_id": self.source_file.id,
            "status": self.job.status.value,
            "duplicate": self.duplicate,
            "source_file": self.source_file.model_dump(mode="json"),
            "job": self.job.model_dump(mode="json"),
        }
