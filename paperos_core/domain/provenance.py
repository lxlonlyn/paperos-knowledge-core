"""Central relation and provenance declarations."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class RelationType(StrEnum):
    CONTAINS = "CONTAINS"
    HAS_SECTION = "HAS_SECTION"
    HAS_CHUNK = "HAS_CHUNK"
    HAS_ELEMENT = "HAS_ELEMENT"
    HAS_REFERENCE = "HAS_REFERENCE"
    DERIVED_FROM = "DERIVED_FROM"
    REPRESENTS_WORK = "REPRESENTS_WORK"
    RESOLVES_TO = "RESOLVES_TO"
    CITES = "CITES"
    ABOUT = "ABOUT"
    MENTIONS = "MENTIONS"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    USES = "USES"
    EXTENDS = "EXTENDS"
    COMPARES_WITH = "COMPARES_WITH"
    EVALUATES_ON = "EVALUATES_ON"
    PROPOSES = "PROPOSES"
    RELATED_TO = "RELATED_TO"


SEMANTIC_RELATION_TYPES: frozenset[RelationType] = frozenset(
    {
        RelationType.MENTIONS,
        RelationType.SUPPORTS,
        RelationType.CONTRADICTS,
        RelationType.USES,
        RelationType.EXTENDS,
        RelationType.COMPARES_WITH,
        RelationType.EVALUATES_ON,
        RelationType.PROPOSES,
        RelationType.RELATED_TO,
    }
)


class RelationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    relation_type: RelationType
    source_chunk_ids: list[str] = Field(default_factory=list)
    derived_from_ids: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
