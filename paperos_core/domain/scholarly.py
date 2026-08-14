"""Stable scholarly-work identities, separate from parsed Documents."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from paperos_core.domain.documents import DomainModel, utc_now
from paperos_core.domain.ids import (
    SCHOLARLY_WORK_ID_VERSION,
    SCHOLARLY_WORK_SCHEMA_VERSION,
)


class WorkIdentityStatus(StrEnum):
    PROVISIONAL = "provisional"
    IDENTIFIED = "identified"
    INGESTED = "ingested"


class WorkIdentifierKind(StrEnum):
    DOI = "doi"
    ARXIV = "arxiv"
    TITLE = "title"


class ScholarlyWork(DomainModel):
    """A permanent internal identity for a scholarly result."""

    id: str
    title: str
    normalized_title: str
    identity_status: WorkIdentityStatus
    identity_confidence: float = Field(ge=0, le=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    schema_version: str = SCHOLARLY_WORK_SCHEMA_VERSION
    id_version: str = SCHOLARLY_WORK_ID_VERSION
    doi: str | None = None
    arxiv_id: str | None = None
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    normalized_first_author: str | None = None


class ReferenceWorkResolution(DomainModel):
    reference_id: str
    source_document_id: str
    work_id: str | None = None
    resolution_status: str
    confidence: float = Field(ge=0, le=1)
    source_chunk_ids: list[str] = Field(default_factory=list)


class ScholarlyContext(DomainModel):
    document_work: ScholarlyWork
    works: list[ScholarlyWork]
    reference_resolutions: list[ReferenceWorkResolution]

    def resolution_by_reference(self) -> dict[str, ReferenceWorkResolution]:
        return {item.reference_id: item for item in self.reference_resolutions}
