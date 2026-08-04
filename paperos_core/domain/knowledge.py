"""Semantic knowledge derived from canonical evidence."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from paperos_core.domain.documents import DomainModel
from paperos_core.domain.ids import CANONICAL_ID_VERSION, CANONICAL_SCHEMA_VERSION


class KnowledgeStatus(StrEnum):
    EXTRACTED = "extracted"
    INFERRED = "inferred"
    USER_CONFIRMED = "user_confirmed"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class Entity(DomainModel):
    id: str
    canonical_snapshot_id: str
    entity_type: str
    name: str
    status: KnowledgeStatus
    derived_from_ids: list[str] = Field(min_length=1)
    source_chunk_ids: list[str] = Field(min_length=1)
    schema_version: str = CANONICAL_SCHEMA_VERSION
    id_version: str = CANONICAL_ID_VERSION
    description: str | None = None
    aliases: list[str] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0, le=1)
    model: str | None = None
    model_version: str | None = None


class Claim(DomainModel):
    id: str
    canonical_snapshot_id: str
    text: str
    status: KnowledgeStatus
    derived_from_ids: list[str] = Field(min_length=1)
    source_chunk_ids: list[str] = Field(min_length=1)
    schema_version: str = CANONICAL_SCHEMA_VERSION
    id_version: str = CANONICAL_ID_VERSION
    claim_type: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    model: str | None = None
    model_version: str | None = None


class ConceptRelation(DomainModel):
    id: str
    canonical_snapshot_id: str
    relation_type: str
    source_object_id: str
    target_object_id: str
    status: KnowledgeStatus
    derived_from_ids: list[str] = Field(min_length=1)
    source_chunk_ids: list[str] = Field(min_length=1)
    schema_version: str = CANONICAL_SCHEMA_VERSION
    id_version: str = CANONICAL_ID_VERSION
    description: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    model: str | None = None
    model_version: str | None = None


class Summary(DomainModel):
    id: str
    canonical_snapshot_id: str
    summary_type: str
    text: str
    status: KnowledgeStatus
    derived_from_ids: list[str] = Field(min_length=1)
    source_chunk_ids: list[str] = Field(min_length=1)
    schema_version: str = CANONICAL_SCHEMA_VERSION
    id_version: str = CANONICAL_ID_VERSION
    model: str | None = None
    model_version: str | None = None


class SemanticEnrichment(DomainModel):
    entities: list[Entity]
    claims: list[Claim]
    relations: list[ConceptRelation]
    summaries: list[Summary]
    model: str
