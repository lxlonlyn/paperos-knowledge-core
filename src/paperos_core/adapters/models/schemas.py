"""Typed local-model gateway requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class GatewayModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EmbeddingRequest(GatewayModel):
    input: list[str] = Field(min_length=1)
    model: str = "default"


class EmbeddingDatum(GatewayModel):
    object: str
    index: int = Field(ge=0)
    embedding: list[float] = Field(min_length=1)


class EmbeddingResponse(GatewayModel):
    object: str
    model: str
    data: list[EmbeddingDatum] = Field(min_length=1)
    usage: dict[str, int]


class RerankRequest(GatewayModel):
    query: str
    candidate_ids: list[str] = Field(min_length=1)
    texts: list[str] = Field(min_length=1)
    limit: int = Field(gt=0)


class RerankResult(GatewayModel):
    candidate_id: str
    original_index: int = Field(ge=0)
    relevance_score: float = Field(ge=0, le=1)
    final_rank: int = Field(gt=0)


class RerankResponse(GatewayModel):
    model: str
    results: list[RerankResult]


class QueryExpansionRequest(GatewayModel):
    query: str
    profile: str


class QueryExpansionResponse(GatewayModel):
    model: str
    lexical_queries: list[str] = Field(min_length=1)
    semantic_queries: list[str] = Field(min_length=1)
    entity_queries: list[str] = Field(min_length=1)
    relation_queries: list[str] = Field(min_length=1)
    hyde_text: str
    raw_output: str
