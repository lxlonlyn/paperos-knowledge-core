"""Provider-neutral parser execution and immutable artifact models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from paperos_core.domain.documents import DomainModel, utc_now
from paperos_core.domain.enums import ParserArtifactType, ParseRunStatus
from paperos_core.domain.ids import (
    PARSE_RUN_SCHEMA_VERSION,
    PARSER_ARTIFACT_ID_VERSION,
    normalize_sha256,
)


class ParseRun(DomainModel):
    id: str
    source_file_id: str
    provider: str
    backend: str
    status: ParseRunStatus
    request_options: dict[str, Any]
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    artifact_manifest_path: Path
    schema_version: str = PARSE_RUN_SCHEMA_VERSION
    pipeline_version: str = "gate2.1"
    provider_task_id: str | None = None
    provider_version: str | None = None
    provider_model: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    raw_metadata: dict[str, Any] | None = None


class ParserArtifact(DomainModel):
    id: str
    parse_run_id: str
    artifact_type: ParserArtifactType
    storage_path: Path
    sha256: str
    size_bytes: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    media_type: str | None = None
    page: int | None = Field(default=None, ge=1)
    provider_name: str | None = None
    provider_metadata: dict[str, Any] | None = None
    id_version: str = PARSER_ARTIFACT_ID_VERSION

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return normalize_sha256(value)


class ParsedIngestionResult(DomainModel):
    source_file_id: str
    ingestion_job_id: str
    duplicate_source: bool
    parse_run: ParseRun
    artifacts: list[ParserArtifact]

    def public_dict(self) -> dict[str, Any]:
        return {
            "source_file_id": self.source_file_id,
            "job_id": self.ingestion_job_id,
            "duplicate": self.duplicate_source,
            "status": self.parse_run.status.value,
            "parse_run": self.parse_run.model_dump(mode="json"),
            "artifacts": [artifact.model_dump(mode="json") for artifact in self.artifacts],
        }
