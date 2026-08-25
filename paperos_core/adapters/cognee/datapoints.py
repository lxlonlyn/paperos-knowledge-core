"""Cognee DataPoint declarations for canonical objects and Work projections."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from paperos_core.adapters.cognee.compat import DataPoint


class PaperOSGraphDataPoint(DataPoint):  # type: ignore[misc]
    canonical_id: str
    derived_from_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = {"index_fields": []}  # noqa: RUF012


class CanonicalBackedDataPoint(PaperOSGraphDataPoint):
    canonical_snapshot_id: str
    source_file_id: str
    parse_run_id: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class ScholarlyWorkDataPoint(PaperOSGraphDataPoint):
    title: str
    normalized_title: str
    doi: str | None = None
    arxiv_id: str | None = None
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    identity_status: str
    identity_confidence: float
    metadata: dict[str, Any] = {"index_fields": ["title"]}  # noqa: RUF012


class DocumentDataPoint(CanonicalBackedDataPoint):
    work_id: str
    title: str
    document_type: str
    language: str
    doi: str | None = None
    year: int | None = None
    metadata: dict[str, Any] = {"index_fields": ["title"]}  # noqa: RUF012


class SectionDataPoint(CanonicalBackedDataPoint):
    document_id: str
    title: str
    path: str
    level: int
    metadata: dict[str, Any] = {"index_fields": ["title"]}  # noqa: RUF012


class ChunkDataPoint(CanonicalBackedDataPoint):
    document_id: str
    section_id: str | None = None
    section_path: str | None = None
    text: str
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = {"index_fields": ["text"]}  # noqa: RUF012


class ElementDataPoint(CanonicalBackedDataPoint):
    document_id: str
    section_id: str | None = None
    element_type: str
    text: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = {"index_fields": []}  # noqa: RUF012


class ReferenceDataPoint(CanonicalBackedDataPoint):
    document_id: str
    raw_text: str
    doi: str | None = None
    year: int | None = None
    resolved_work_id: str | None = None
    resolution_status: str
    metadata: dict[str, Any] = {"index_fields": []}  # noqa: RUF012


class EntityDataPoint(CanonicalBackedDataPoint):
    entity_type: str
    name: str
    description: str | None = None
    status: str
    confidence: float | None = None
    metadata: dict[str, Any] = {"index_fields": ["name", "description"]}  # noqa: RUF012


class ClaimDataPoint(CanonicalBackedDataPoint):
    text: str
    claim_type: str | None = None
    status: str
    confidence: float | None = None
    source_document_id: str | None = None
    source_work_id: str | None = None
    metadata: dict[str, Any] = {"index_fields": ["text"]}  # noqa: RUF012


class ConceptRelationDataPoint(CanonicalBackedDataPoint):
    relation_type: str
    source_object_id: str
    target_object_id: str
    description: str | None = None
    status: str
    confidence: float | None = None
    metadata: dict[str, Any] = {"index_fields": ["description"]}  # noqa: RUF012


class TripletDataPoint(CanonicalBackedDataPoint):
    """Searchable typed-edge node with canonical provenance."""

    relation_type: str
    source_object_id: str
    target_object_id: str
    text: str
    metadata: dict[str, Any] = {"index_fields": ["text"]}  # noqa: RUF012
