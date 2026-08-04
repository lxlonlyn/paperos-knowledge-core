"""Typed private local inference requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class InferenceModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmbeddingRequest(InferenceModel):
    input: list[str] = Field(min_length=1)
    model: str = "default"


class EmbeddingDatum(InferenceModel):
    object: str
    index: int = Field(ge=0)
    embedding: list[float] = Field(min_length=1)


class EmbeddingResponse(InferenceModel):
    object: str
    model: str
    data: list[EmbeddingDatum] = Field(min_length=1)
    usage: dict[str, int]


class RerankRequest(InferenceModel):
    query: str
    candidate_ids: list[str] = Field(min_length=1)
    texts: list[str] = Field(min_length=1)
    limit: int = Field(gt=0)


class RerankResult(InferenceModel):
    candidate_id: str
    original_index: int = Field(ge=0)
    relevance_score: float = Field(ge=0, le=1)
    final_rank: int = Field(gt=0)


class RerankResponse(InferenceModel):
    model: str
    results: list[RerankResult]


class QueryExpansionRequest(InferenceModel):
    query: str
    profile: str


class QueryExpansionResponse(InferenceModel):
    model: str
    lexical_queries: list[str] = Field(min_length=1)
    semantic_queries: list[str] = Field(min_length=1)
    entity_queries: list[str] = Field(min_length=1)
    relation_queries: list[str] = Field(min_length=1)
    hyde_text: str
    raw_output: str
