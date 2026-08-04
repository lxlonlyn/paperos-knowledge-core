"""Versioned manifests for rebuildable Gate 4 projections."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from paperos_core.domain.documents import utc_now

INDEX_SCHEMA_VERSION = "1.0"
LEXICAL_INDEX_VERSION = "1"
VECTOR_INDEX_VERSION = "2"
COGNEE_MAPPING_VERSION = "2"


class IndexManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_snapshot_id: str
    document_id: str
    schema_version: str = INDEX_SCHEMA_VERSION
    lexical_index_version: str = LEXICAL_INDEX_VERSION
    vector_index_version: str = VECTOR_INDEX_VERSION
    cognee_mapping_version: str = COGNEE_MAPPING_VERSION
    embedding_model: str
    embedding_dimensions: int = Field(gt=0)
    vector_backend: str = "cognee"
    lexical_database: Path
    vector_database: Path
    cognee_manifest: Path
    lexical_object_ids: list[str]
    vector_object_ids: list[str]
    cognee_object_ids: list[str]
    relation_count: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)


class IndexingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_snapshot_id: str
    document_id: str
    manifest_path: Path
    cognee_manifest_path: Path
    lexical_database: Path
    vector_database: Path
    vector_backend: str = "cognee"
    cognee_object_count: int = Field(ge=0)
    relation_count: int = Field(ge=0)
    lexical_object_count: int = Field(ge=0)
    vector_object_count: int = Field(ge=0)
    embedding_dimensions: int = Field(gt=0)
    semantic_entity_count: int = Field(ge=0)
    semantic_claim_count: int = Field(ge=0)
    semantic_relation_count: int = Field(ge=0)
    summary_count: int = Field(ge=0)
    consistency_valid: bool
    rebuilt: bool = False
    warnings: list[str] = Field(default_factory=list)
