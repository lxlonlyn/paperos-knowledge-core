"""Versioned provider-neutral canonical document models."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from paperos_core.domain.documents import DomainModel, utc_now
from paperos_core.domain.enums import ElementType, ReferenceResolutionStatus
from paperos_core.domain.ids import (
    CANONICAL_ID_VERSION,
    CANONICAL_PIPELINE_VERSION,
    CANONICAL_SCHEMA_VERSION,
    CHUNKING_VERSION,
    CLASSIFICATION_VERSION,
    CLEANING_VERSION,
    REFERENCE_PROCESSING_VERSION,
)
from paperos_core.domain.parsing import ParsedIngestionResult


class SourceSpan(DomainModel):
    artifact_id: str
    item_index: int = Field(ge=0)
    page: int | None = Field(default=None, ge=1)
    bounding_box: tuple[float, float, float, float] | None = None


class Person(DomainModel):
    id: str
    display_name: str
    schema_version: str = CANONICAL_SCHEMA_VERSION
    id_version: str = CANONICAL_ID_VERSION
    given_name: str | None = None
    family_name: str | None = None
    name_parts: list[str] | None = None
    orcid: str | None = None
    aliases: list[str] | None = None
    raw_name: str | None = None


class Document(DomainModel):
    id: str
    source_file_id: str
    parse_run_id: str
    canonical_snapshot_id: str
    document_type: str = "research_paper"
    language: str
    title: str
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = CANONICAL_SCHEMA_VERSION
    id_version: str = CANONICAL_ID_VERSION
    subtitle: str | None = None
    abstract: str | None = None
    publication_date: date | None = None
    year: int | None = None
    venue: str | None = None
    publisher: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    isbn: str | None = None
    issn: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    keywords: list[str] | None = None
    authors: list[Person] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)


class Section(DomainModel):
    id: str
    document_id: str
    canonical_snapshot_id: str
    title: str
    level: int = Field(ge=1)
    order: int = Field(ge=0)
    path: str
    schema_version: str = CANONICAL_SCHEMA_VERSION
    id_version: str = CANONICAL_ID_VERSION
    parent_section_id: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    source_span: SourceSpan | None = None
    section_type: str | None = None
    raw_title: str | None = None


class Element(DomainModel):
    id: str
    document_id: str
    canonical_snapshot_id: str
    element_type: ElementType
    order: int = Field(ge=0)
    schema_version: str = CANONICAL_SCHEMA_VERSION
    id_version: str = CANONICAL_ID_VERSION
    section_id: str | None = None
    parent_element_id: str | None = None
    text: str | None = None
    raw_text: str | None = None
    markdown: str | None = None
    latex: str | None = None
    html: str | None = None
    asset_path: Path | None = None
    page: int | None = Field(default=None, ge=1)
    bounding_box: tuple[float, float, float, float] | None = None
    source_span: SourceSpan | None = None
    caption_element_ids: list[str] = Field(default_factory=list)
    footnote_element_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(DomainModel):
    id: str
    document_id: str
    canonical_snapshot_id: str
    text: str
    order: int = Field(ge=0)
    schema_version: str = CANONICAL_SCHEMA_VERSION
    id_version: str = CANONICAL_ID_VERSION
    chunking_version: str = CHUNKING_VERSION
    element_ids: list[str] = Field(min_length=1)
    section_id: str | None = None
    section_path: str | None = None
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    token_count: int | None = Field(default=None, ge=1)
    character_start: int | None = Field(default=None, ge=0)
    character_end: int | None = Field(default=None, ge=0)
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    overlap_source_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReferenceEntry(DomainModel):
    id: str
    document_id: str
    canonical_snapshot_id: str
    raw_text: str
    order: int = Field(ge=0)
    schema_version: str = CANONICAL_SCHEMA_VERSION
    id_version: str = CANONICAL_ID_VERSION
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    publisher: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    volume: str | None = None
    issue: str | None = None
    pages: str | None = None
    source_element_id: str | None = None
    resolved_document_id: str | None = None
    resolution_status: ReferenceResolutionStatus = ReferenceResolutionStatus.UNRESOLVED
    resolution_confidence: float | None = Field(default=None, ge=0, le=1)
    parsed_fields: dict[str, Any] = Field(default_factory=dict)


class CanonicalSnapshot(DomainModel):
    id: str
    source_file_id: str
    parse_run_id: str
    document_id: str
    manifest_path: Path
    dataset_id: str = "papers"
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = CANONICAL_SCHEMA_VERSION
    id_version: str = CANONICAL_ID_VERSION
    pipeline_version: str = CANONICAL_PIPELINE_VERSION
    cleaning_version: str = CLEANING_VERSION
    classification_version: str = CLASSIFICATION_VERSION
    chunking_version: str = CHUNKING_VERSION
    reference_processing_version: str = REFERENCE_PROCESSING_VERSION


class CanonicalBundle(DomainModel):
    snapshot: CanonicalSnapshot
    document: Document
    sections: list[Section]
    elements: list[Element]
    chunks: list[Chunk]
    references: list[ReferenceEntry]
    warnings: list[str] = Field(default_factory=list)

    def public_dict(self) -> dict[str, Any]:
        return {
            "status": "completed",
            "canonical_snapshot": self.snapshot.model_dump(mode="json"),
            "document": self.document.model_dump(mode="json"),
            "counts": {
                "sections": len(self.sections),
                "elements": len(self.elements),
                "chunks": len(self.chunks),
                "references": len(self.references),
            },
            "element_types": sorted({element.element_type.value for element in self.elements}),
            "warnings": self.warnings,
        }


class CanonicalIngestionResult(DomainModel):
    parsed: ParsedIngestionResult
    canonical: CanonicalBundle

    def public_dict(self) -> dict[str, Any]:
        payload = self.parsed.public_dict()
        payload.update(self.canonical.public_dict())
        return payload
