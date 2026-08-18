"""Shared query, candidate, evidence, and response models."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievalProfile(StrEnum):
    COMPREHENSIVE = "comprehensive"
    TRUTH = "truth"
    ASSOCIATIVE = "associative"


class QueryScopeInput(BaseModel):
    """Optional explicit scope supplied by a client or agent."""

    model_config = ConfigDict(extra="forbid")

    source_work_ids: list[str] | None = None
    exclude_source_work_ids: list[str] | None = None
    subject_work_ids: list[str] | None = None
    work_set_work_ids: list[str] | None = None
    topic_queries: list[str] | None = None


class ResolvedQueryScope(BaseModel):
    """Resolved source / subject / work-set / topic scope for one query."""

    model_config = ConfigDict(extra="forbid")

    source_work_ids: list[str] = Field(default_factory=list)
    exclude_source_work_ids: list[str] = Field(default_factory=list)
    subject_work_ids: list[str] = Field(default_factory=list)
    work_set_work_ids: list[str] = Field(default_factory=list)
    topic_queries: list[str] = Field(default_factory=list)

    @property
    def has_hard_work_scope(self) -> bool:
        return bool(
            self.source_work_ids
            or self.exclude_source_work_ids
            or self.subject_work_ids
            or self.work_set_work_ids
        )


class QueryScopeTrace(BaseModel):
    """Trace of how scope was resolved and applied."""

    model_config = ConfigDict(extra="forbid")

    resolution: Literal["explicit", "deterministic", "llm", "fallback_unscoped"] = (
        "fallback_unscoped"
    )
    warnings: list[str] = Field(default_factory=list)
    mentioned_work_ids: list[str] = Field(default_factory=list)
    planner_notes: str | None = None
    applied_document_ids: list[str] = Field(default_factory=list)
    recall_context_disabled: bool = False


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    profile: RetrievalProfile = RetrievalProfile.COMPREHENSIVE
    dataset: str | None = Field(default=None, min_length=1)
    top_k: int | None = Field(default=None, gt=0, le=100)
    document_ids: list[str] | None = None
    scope: QueryScopeInput | None = None

    @field_validator("dataset")
    @classmethod
    def normalize_dataset(cls, value: str | None) -> str | None:
        if value is None:
            return None
        selected = value.strip()
        if not selected:
            raise ValueError("dataset must not be blank")
        return selected


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: RetrievalProfile
    channels: list[str]
    search_types: dict[str, str]
    top_k: int
    candidate_pool_size: int
    graph_depth: int
    weights: dict[str, float]


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
    final_rank: int | None = None
    knowledge_kind: Literal[
        "source_fact",
        "structured_relation",
        "system_inference",
        "user_confirmed",
    ] = "source_fact"
    derived_from_ids: list[str] = Field(default_factory=list)
    source_work_id: str | None = None
    subject_work_ids: list[str] = Field(default_factory=list)


class Evidence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    chunk_id: str
    document_id: str
    source_file_id: str
    source_filename: str
    title: str
    section_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    text: str
    channels: list[str]
    knowledge_kind: str
    derived_from_ids: list[str]
    source_work_id: str | None = None
    subject_work_ids: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    query: str
    profile: RetrievalProfile
    dataset: str
    answer: str
    answer_model: str
    stages: list[str]
    channels_used: list[str]
    evidence: list[Evidence]
    candidates: list[Candidate]
    distinct_documents: int
    provenance_complete: bool
    resolved_scope: ResolvedQueryScope = Field(default_factory=ResolvedQueryScope)
    scope_trace: QueryScopeTrace = Field(default_factory=QueryScopeTrace)
