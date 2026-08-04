"""Centralized Cognee DataPoint declarations for all PaperOS graph writes."""

from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid5

from cognee.infrastructure.engine import DataPoint  # type: ignore[import-untyped]
from pydantic import Field


def cognee_uuid(canonical_id: str, *, mapping_version: str = "1") -> UUID:
    return uuid5(NAMESPACE_URL, f"paperos:cognee:{mapping_version}:{canonical_id}")


class PaperOSDataPoint(DataPoint):  # type: ignore[misc]
    canonical_id: str
    canonical_snapshot_id: str
    source_file_id: str
    parse_run_id: str
    source_chunk_ids: list[str] = Field(default_factory=list)
    derived_from_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = {"index_fields": []}  # noqa: RUF012


class DocumentDataPoint(PaperOSDataPoint):
    title: str
    document_type: str
    language: str
    doi: str | None = None
    year: int | None = None
    metadata: dict[str, Any] = {"index_fields": ["title"]}  # noqa: RUF012


class SectionDataPoint(PaperOSDataPoint):
    document_id: str
    title: str
    path: str
    level: int
    metadata: dict[str, Any] = {"index_fields": ["title"]}  # noqa: RUF012


class ChunkDataPoint(PaperOSDataPoint):
    document_id: str
    section_id: str | None = None
    section_path: str | None = None
    text: str
    page_start: int | None = None
    page_end: int | None = None
    metadata: dict[str, Any] = {"index_fields": ["text"]}  # noqa: RUF012


class ElementDataPoint(PaperOSDataPoint):
    document_id: str
    section_id: str | None = None
    element_type: str
    text: str | None = None
    page: int | None = None
    metadata: dict[str, Any] = {"index_fields": []}  # noqa: RUF012


class ReferenceDataPoint(PaperOSDataPoint):
    document_id: str
    raw_text: str
    doi: str | None = None
    year: int | None = None
    resolved_document_id: str | None = None
    resolution_status: str
    metadata: dict[str, Any] = {"index_fields": []}  # noqa: RUF012


class EntityDataPoint(PaperOSDataPoint):
    entity_type: str
    name: str
    description: str | None = None
    status: str
    confidence: float | None = None
    metadata: dict[str, Any] = {  # noqa: RUF012
        "index_fields": ["name", "description"]
    }


class ClaimDataPoint(PaperOSDataPoint):
    text: str
    claim_type: str | None = None
    status: str
    confidence: float | None = None
    metadata: dict[str, Any] = {"index_fields": ["text"]}  # noqa: RUF012


class ConceptRelationDataPoint(PaperOSDataPoint):
    relation_type: str
    source_object_id: str
    target_object_id: str
    description: str | None = None
    status: str
    confidence: float | None = None
    metadata: dict[str, Any] = {"index_fields": ["description"]}  # noqa: RUF012


class SummaryDataPoint(PaperOSDataPoint):
    summary_type: str
    text: str
    status: str
    metadata: dict[str, Any] = {"index_fields": ["text"]}  # noqa: RUF012


class TripletDataPoint(PaperOSDataPoint):
    """Searchable typed-edge node with canonical provenance."""

    relation_type: str
    source_object_id: str
    target_object_id: str
    text: str
    metadata: dict[str, Any] = {"index_fields": ["text"]}  # noqa: RUF012
