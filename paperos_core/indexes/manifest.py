"""PaperOS-owned FTS and chunk-projection manifests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from paperos_core.domain.documents import utc_now

INDEX_SCHEMA_VERSION = "1.0"
LEXICAL_INDEX_VERSION = "1"


class IndexManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_snapshot_id: str
    document_id: str
    schema_version: str = INDEX_SCHEMA_VERSION
    lexical_index_version: str = LEXICAL_INDEX_VERSION
    lexical_database: Path
    lexical_object_ids: list[str]
    chunk_projection_ids: list[str]
    created_at: datetime = Field(default_factory=utc_now)


class IndexingReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_snapshot_id: str
    document_id: str
    dataset_name: str
    cognee_dataset_id: str
    cognee_data_id: str
    cognee_pipeline_run_id: str
    cognee_provenance_backend: str
    manifest_path: Path
    cognee_manifest_path: Path
    lexical_database: Path
    vector_database: str
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
    def public_dict(self) -> dict[str, object]:
        return self.model_dump(
            mode="json",
            exclude={
                "manifest_path",
                "cognee_manifest_path",
                "lexical_database",
                "vector_database",
            },
        )
