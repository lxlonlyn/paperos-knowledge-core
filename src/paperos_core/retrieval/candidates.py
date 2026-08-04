"""Shared query, candidate, evidence, and response models."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetrievalProfile(StrEnum):
    COMPREHENSIVE = "comprehensive"
    TRUTH = "truth"
    ASSOCIATIVE = "associative"


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    profile: RetrievalProfile = RetrievalProfile.COMPREHENSIVE
    dataset: str | None = Field(default=None, min_length=1)
    top_k: int | None = Field(default=None, gt=0, le=100)
    document_ids: list[str] | None = None

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
    top_k: int
    candidate_pool_size: int
    graph_depth: int
    weights: dict[str, float]


class ExpansionTrace(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    lexical_queries: list[str]
    semantic_queries: list[str]
    entity_queries: list[str]
    relation_queries: list[str]
    hyde_text: str
    raw_output: str
    planner_model: str | None = None
    planner_raw_output: str | None = None


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
    expansion: ExpansionTrace
    evidence: list[Evidence]
    candidates: list[Candidate]
    distinct_documents: int
    provenance_complete: bool
