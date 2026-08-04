"""Typed DeepSeek adapter for evidence-bound semantic enrichment."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperos_core.config import DeepSeekSettings
from paperos_core.domain.canonical import CanonicalBundle
from paperos_core.domain.ids import semantic_object_id
from paperos_core.domain.knowledge import (
    Claim,
    ConceptRelation,
    Entity,
    KnowledgeStatus,
    SemanticEnrichment,
    Summary,
)
from paperos_core.errors import SemanticEnrichmentError
from paperos_core.prompt_repository import PromptDescriptor, PromptRepository


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _EntityExtraction(_StrictModel):
    key: str
    name: str
    entity_type: str
    description: str | None = None
    source_chunk_ids: list[str] = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class _ClaimExtraction(_StrictModel):
    key: str
    text: str
    claim_type: str | None = None
    source_chunk_ids: list[str] = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class _RelationExtraction(_StrictModel):
    source_key: str
    target_key: str
    relation_type: str
    description: str | None = None
    source_chunk_ids: list[str] = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class _SummaryExtraction(_StrictModel):
    text: str
    source_chunk_ids: list[str] = Field(min_length=1)


class _EnrichmentExtraction(_StrictModel):
    entities: list[_EntityExtraction]
    claims: list[_ClaimExtraction]
    relations: list[_RelationExtraction]
    summary: _SummaryExtraction


class _QueryPlanExtraction(_StrictModel):
    lexical_queries: list[str] = Field(min_length=1)
    semantic_queries: list[str] = Field(min_length=1)
    entity_queries: list[str] = Field(min_length=1)
    relation_queries: list[str] = Field(min_length=1)
    hyde_text: str


class DeepSeekClient:
    """Call only the configured DeepSeek-compatible endpoint."""

    def __init__(
        self,
        config: DeepSeekSettings,
        prompts: PromptRepository,
    ) -> None:
        self.config = config
        self.prompts = prompts
        self.max_attempts = config.max_attempts
        headers = {"Content-Type": "application/json"}
        if config.api_key_value():
            headers["Authorization"] = f"Bearer {config.api_key_value()}"
        self.client = httpx.AsyncClient(
            base_url=config.endpoint,
            headers=headers,
            timeout=config.timeout_seconds,
            trust_env=False,
        )

    async def health_check(self) -> dict[str, Any]:
        if not self.config.endpoint or not self.config.model or not self.config.api_key_value():
            raise SemanticEnrichmentError(
                "DeepSeek requires endpoint/model configuration and DEEPSEEK_API_KEY.",
                affected="DEEPSEEK_API_KEY",
                retryable=False,
            )
        try:
            response = await self.client.get(
                "/models", timeout=min(self.config.timeout_seconds, 10)
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise SemanticEnrichmentError(
                f"DeepSeek health check failed: {exc}",
                affected=f"{self.config.endpoint}/models",
            ) from exc
        if not isinstance(payload, dict):
            raise SemanticEnrichmentError(
                "DeepSeek health response is not a JSON object.",
                affected=f"{self.config.endpoint}/models",
            )
        return payload

    async def enrich(self, bundle: CanonicalBundle) -> SemanticEnrichment:
        chunks = _select_evidence(bundle)
        prompt = self.prompts.describe("semantic_enrichment")
        extraction = await self._generate_structured(
            system=prompt.text,
            user=json.dumps(
                {
                    "schema": {
                        "entities": [
                            {
                                "key": "unique local key",
                                "name": "string",
                                "entity_type": "string",
                                "description": "optional string",
                                "source_chunk_ids": ["chunk id"],
                                "confidence": "optional number 0..1",
                            }
                        ],
                        "claims": [
                            {
                                "key": "unique local key",
                                "text": "string",
                                "claim_type": "optional string",
                                "source_chunk_ids": ["chunk id"],
                                "confidence": "optional number 0..1",
                            }
                        ],
                        "relations": [
                            {
                                "source_key": "entity key",
                                "target_key": "entity key",
                                "relation_type": "string",
                                "description": "optional string",
                                "source_chunk_ids": ["chunk id"],
                                "confidence": "optional number 0..1",
                            }
                        ],
                        "summary": {
                            "text": "string",
                            "source_chunk_ids": ["chunk id"],
                        },
                    },
                    "document": {
                        "title": bundle.document.title,
                        "abstract": bundle.document.abstract,
                    },
                    "evidence": chunks,
                },
                ensure_ascii=False,
            ),
        )
        return _to_domain(bundle, extraction, self.config.model, prompt)

    async def _generate_structured(self, *, system: str, user: str) -> _EnrichmentExtraction:
        request = {
            "model": _deepseek_request_model(self.config.model),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 16_000,
            "stream": False,
        }
        failures: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self.client.post("/chat/completions", json=request)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        "retryable DeepSeek response",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                if not isinstance(content, str):
                    raise TypeError("completion content is not text")
                return _EnrichmentExtraction.model_validate_json(_strip_json_fence(content))
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                ValidationError,
            ) as exc:
                failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt < self.max_attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
        raise SemanticEnrichmentError(
            "DeepSeek structured semantic enrichment failed after finite retries.",
            affected=f"{self.config.endpoint}/chat/completions",
            details={"attempts": failures},
        )

    async def synthesize_answer(
        self,
        *,
        query: str,
        profile: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        """Synthesize one evidence-bound answer through the configured provider."""
        compact_evidence: list[dict[str, Any]] = []
        for item in evidence:
            compact = dict(item)
            text = compact.get("text")
            if isinstance(text, str):
                compact["text"] = text[:3_000]
            compact_evidence.append(compact)
        request = {
            "model": _deepseek_request_model(self.config.model),
            "messages": [
                {
                    "role": "system",
                    "content": self.prompts.load("answer_synthesis"),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "profile": profile,
                            "question": query,
                            "evidence": compact_evidence,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "temperature": 0.1,
            "max_tokens": 16_000,
            "stream": False,
        }
        failures: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self.client.post("/chat/completions", json=request)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        "retryable DeepSeek response",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise TypeError("completion content is empty or not text")
                return content.strip()
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
            ) as exc:
                failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt < self.max_attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
        raise SemanticEnrichmentError(
            "DeepSeek answer synthesis failed after finite retries.",
            affected=f"{self.config.endpoint}/chat/completions",
            details={"attempts": failures},
        )

    async def plan_query(
        self, *, query: str, profile: str
    ) -> tuple[_QueryPlanExtraction, str]:
        """Generate retrieval-only bilingual expansions with a validated schema."""
        request = {
            "model": _deepseek_request_model(self.config.model),
            "messages": [
                {
                    "role": "system",
                    "content": self.prompts.load("query_planning"),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "profile": profile,
                            "query": query,
                            "schema": {
                                "lexical_queries": ["string"],
                                "semantic_queries": ["string"],
                                "entity_queries": ["string"],
                                "relation_queries": ["string"],
                                "hyde_text": "short hypothetical relevant passage",
                            },
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
            "max_tokens": 2_000,
            "stream": False,
        }
        failures: list[str] = []
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self.client.post("/chat/completions", json=request)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise httpx.HTTPStatusError(
                        "retryable DeepSeek response",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                payload = response.json()
                content = payload["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise TypeError("completion content is empty or not text")
                return (
                    _QueryPlanExtraction.model_validate_json(
                        _strip_json_fence(content)
                    ),
                    content,
                )
            except (
                httpx.HTTPError,
                KeyError,
                IndexError,
                TypeError,
                ValueError,
                ValidationError,
            ) as exc:
                failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt < self.max_attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
        raise SemanticEnrichmentError(
            "DeepSeek query planning failed after finite retries.",
            affected=f"{self.config.endpoint}/chat/completions",
            details={"attempts": failures},
        )

    async def aclose(self) -> None:
        await self.client.aclose()


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```"):
        first_newline = stripped.find("\n")
        if first_newline >= 0:
            stripped = stripped[first_newline + 1 :]
        stripped = stripped.removesuffix("```")
    return stripped.strip()


def _deepseek_request_model(configured_model: str) -> str:
    """Cognee/LiteLLM uses a provider prefix; DeepSeek's own API does not."""
    prefix = "deepseek/"
    return configured_model.removeprefix(prefix)


def _select_evidence(bundle: CanonicalBundle) -> list[dict[str, str]]:
    """Keep all real chunks represented while bounding one provider request."""
    selected: list[dict[str, str]] = []
    character_budget = 24_000
    for chunk in bundle.chunks:
        if character_budget <= 0:
            break
        text = chunk.text[: min(len(chunk.text), 4_000, character_budget)]
        if not text.strip():
            continue
        selected.append(
            {
                "chunk_id": chunk.id,
                "section_path": chunk.section_path or "",
                "text": text,
            }
        )
        character_budget -= len(text)
    if not selected:
        raise SemanticEnrichmentError(
            "Canonical snapshot contains no non-empty chunks for semantic enrichment.",
            affected=bundle.snapshot.id,
        )
    return selected


def _to_domain(
    bundle: CanonicalBundle,
    extracted: _EnrichmentExtraction,
    model: str,
    prompt: PromptDescriptor,
) -> SemanticEnrichment:
    valid_chunks = {chunk.id for chunk in bundle.chunks}

    def validated(ids: list[str], *, object_key: str) -> list[str]:
        resolved: list[str] = []
        invalid: list[str] = []
        for value in dict.fromkeys(ids):
            if value in valid_chunks:
                resolved.append(value)
                continue
            prefix_matches = sorted(
                chunk_id for chunk_id in valid_chunks if chunk_id.startswith(value)
            )
            if value.startswith("chunk_") and len(prefix_matches) == 1:
                resolved.append(prefix_matches[0])
            else:
                invalid.append(value)
        if invalid:
            raise SemanticEnrichmentError(
                "DeepSeek returned source chunk IDs outside the canonical snapshot.",
                affected=object_key,
                details={"invalid_or_ambiguous_chunk_ids": sorted(invalid)},
            )
        return list(dict.fromkeys(resolved))

    key_to_id: dict[str, str] = {}
    entities: list[Entity] = []
    for entity_item in extracted.entities:
        source_ids = validated(entity_item.source_chunk_ids, object_key=entity_item.key)
        entity_id = semantic_object_id(
            "entity",
            bundle.snapshot.id,
            f"{entity_item.entity_type}:{entity_item.name}",
            source_ids,
        )
        if entity_item.key in key_to_id:
            raise SemanticEnrichmentError(
                "DeepSeek returned duplicate entity keys.", affected=entity_item.key
            )
        key_to_id[entity_item.key] = entity_id
        entities.append(
            Entity(
                id=entity_id,
                canonical_snapshot_id=bundle.snapshot.id,
                entity_type=entity_item.entity_type,
                name=entity_item.name,
                description=entity_item.description,
                status=KnowledgeStatus.EXTRACTED,
                source_chunk_ids=source_ids,
                derived_from_ids=source_ids,
                confidence=entity_item.confidence,
                model=model,
                model_version=model,
            )
        )

    claims: list[Claim] = []
    for claim_item in extracted.claims:
        source_ids = validated(claim_item.source_chunk_ids, object_key=claim_item.key)
        claim_id = semantic_object_id("claim", bundle.snapshot.id, claim_item.text, source_ids)
        claims.append(
            Claim(
                id=claim_id,
                canonical_snapshot_id=bundle.snapshot.id,
                text=claim_item.text,
                claim_type=claim_item.claim_type,
                status=KnowledgeStatus.EXTRACTED,
                source_chunk_ids=source_ids,
                derived_from_ids=source_ids,
                confidence=claim_item.confidence,
                model=model,
                model_version=model,
            )
        )

    relations: list[ConceptRelation] = []
    for relation_item in extracted.relations:
        if relation_item.source_key not in key_to_id or relation_item.target_key not in key_to_id:
            raise SemanticEnrichmentError(
                "DeepSeek relation references an unknown entity key.",
                affected=f"{relation_item.source_key}->{relation_item.target_key}",
            )
        source_ids = validated(
            relation_item.source_chunk_ids,
            object_key=f"{relation_item.source_key}->{relation_item.target_key}",
        )
        source_object_id = key_to_id[relation_item.source_key]
        target_object_id = key_to_id[relation_item.target_key]
        relation_id = semantic_object_id(
            "relation",
            bundle.snapshot.id,
            f"{source_object_id}:{relation_item.relation_type}:{target_object_id}",
            source_ids,
        )
        relations.append(
            ConceptRelation(
                id=relation_id,
                canonical_snapshot_id=bundle.snapshot.id,
                relation_type=relation_item.relation_type.upper().replace(" ", "_"),
                source_object_id=source_object_id,
                target_object_id=target_object_id,
                description=relation_item.description,
                status=KnowledgeStatus.INFERRED,
                source_chunk_ids=source_ids,
                derived_from_ids=source_ids,
                confidence=relation_item.confidence,
                model=model,
                model_version=model,
            )
        )

    summary_sources = validated(extracted.summary.source_chunk_ids, object_key="document_summary")
    summary_id = semantic_object_id(
        "summary",
        bundle.snapshot.id,
        extracted.summary.text,
        summary_sources,
    )
    summary = Summary(
        id=summary_id,
        canonical_snapshot_id=bundle.snapshot.id,
        summary_type="document",
        text=extracted.summary.text,
        status=KnowledgeStatus.INFERRED,
        source_chunk_ids=summary_sources,
        derived_from_ids=summary_sources,
        model=model,
        model_version=model,
    )
    return SemanticEnrichment(
        entities=entities,
        claims=claims,
        relations=relations,
        summaries=[summary],
        model=model,
        model_version=model,
        prompt_name=prompt.name,
        prompt_version=prompt.version,
        prompt_sha256=prompt.sha256,
    )
