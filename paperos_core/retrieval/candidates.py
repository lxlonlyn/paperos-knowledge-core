"""Shared query, candidate, evidence, and response models."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


@dataclass(slots=True)
class VectorSearchDiagnostics:
    request_limits: list[int] = dataclass_field(default_factory=list)
    raw_hit_counts: list[int] = dataclass_field(default_factory=list)
    filtered_hit_counts: list[int] = dataclass_field(default_factory=list)
    backend_exhausted: bool = False
    safety_limit_reached: bool = False


class RerankDiagnostics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_token_count: int = Field(gt=0)
    input_token_count: int = Field(gt=0)
    model_max_input_tokens: int = Field(gt=0)
    query_token_count: int = Field(gt=0)
    truncated: bool
    window_count: int = Field(gt=0)
    winning_window_document_token_count: int = Field(gt=0)
    winning_window_index: int = Field(ge=0)
    winning_window_score: float = Field(ge=0, le=1)
    winning_window_text: str = Field(min_length=1)


class RerankTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_token_count: int = Field(gt=0)
    input_token_count: int = Field(gt=0)
    model_max_input_tokens: int = Field(gt=0)
    query_token_count: int = Field(gt=0)
    truncated: bool
    window_count: int = Field(gt=0)
    winning_window_document_token_count: int = Field(gt=0)
    winning_window_index: int = Field(ge=0)
    winning_window_score: float = Field(ge=0, le=1)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    dataset: str | None = Field(default=None, min_length=1)
    top_k: int | None = Field(default=None, gt=0, le=100)
    document_ids: list[str] | None = None
    work_ids: list[str] | None = None
    expand_context: bool = False
    expand_graph: bool = False

    @field_validator("dataset")
    @classmethod
    def normalize_dataset(cls, value: str | None) -> str | None:
        if value is None:
            return None
        selected = value.strip()
        if not selected:
            raise ValueError("dataset must not be blank")
        return selected


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    object_id: str
    object_type: str
    document_id: str
    source_file_id: str
    source_filename: str
    canonical_snapshot_id: str
    chunk_id: str
    section_id: str | None = None
    section_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    text: str
    channels: list[str]
    channel_ranks: dict[str, int] = Field(default_factory=dict)
    channel_scores: dict[str, float] = Field(default_factory=dict)
    fused_score: float = 0
    rerank_score: float | None = None
    rerank_diagnostics: RerankDiagnostics | None = None
    final_rank: int | None = None
    knowledge_kind: Literal[
        "source_fact",
        "structured_relation",
        "system_inference",
        "user_confirmed",
    ] = "source_fact"
    derived_from_ids: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
    source_work_id: str | None = None


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    chunk_id: str
    document_id: str
    source_file_id: str
    source_filename: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    section_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    text: str
    channels: list[str]
    knowledge_kind: str
    derived_from_ids: list[str]
    source_work_id: str | None = None


class QueryReplay(BaseModel):
    """Exact final synthesis input, or empty replay_text when no LLM was called."""

    model_config = ConfigDict(extra="forbid")

    original_query: str
    replay_text: str


class RetrievalTrace(BaseModel):
    """Compact, auditable trace of the single production retrieval pipeline."""

    model_config = ConfigDict(extra="forbid")

    requested_document_ids: list[str] = Field(default_factory=list)
    requested_work_ids: list[str] = Field(default_factory=list)
    resolved_work_document_ids: list[str] = Field(default_factory=list)
    applied_document_ids: list[str] = Field(default_factory=list)
    applied_snapshot_ids: list[str] = Field(default_factory=list)
    candidate_pool_sizes: list[int] = Field(default_factory=list)
    lexical_request_limits: list[int] = Field(default_factory=list)
    lexical_filtered_counts: list[int] = Field(default_factory=list)
    vector_request_limits: list[int] = Field(default_factory=list)
    vector_raw_hit_counts: list[int] = Field(default_factory=list)
    vector_filtered_counts: list[int] = Field(default_factory=list)
    vector_backend_exhausted: list[bool] = Field(default_factory=list)
    vector_safety_limit_reached: list[bool] = Field(default_factory=list)
    first_stage_chunk_ids: list[str] = Field(default_factory=list)
    first_reranked_chunk_ids: list[str] = Field(default_factory=list)
    first_rerank_diagnostics: list[RerankTrace] = Field(default_factory=list)
    local_expanded_chunk_ids: list[str] = Field(default_factory=list)
    local_new_chunk_ids: list[str] = Field(default_factory=list)
    semantic_expanded_chunk_ids: list[str] = Field(default_factory=list)
    semantic_new_chunk_ids: list[str] = Field(default_factory=list)
    seed_chunk_ids: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
    derived_from_ids: list[str] = Field(default_factory=list)
    second_reranked_chunk_ids: list[str] = Field(default_factory=list)
    second_rerank_diagnostics: list[RerankTrace] = Field(default_factory=list)
    second_rerank_candidate_ids: list[str] = Field(default_factory=list)
    final_selected_chunk_ids: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    dataset: str
    answer: str
    answer_model: str
    stages: list[str]
    channels_used: list[str]
    evidence: list[Evidence]
    replay: QueryReplay
    candidates: list[Candidate]
    distinct_documents: int
    provenance_complete: bool
    trace: RetrievalTrace
