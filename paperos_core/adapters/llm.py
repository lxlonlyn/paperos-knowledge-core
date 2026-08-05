"""Provider-neutral LLM client backed exclusively by Cognee's LLMGateway.

PaperOS never talks to a specific LLM vendor or endpoint. It supplies the
prompt, the Pydantic response schema, the canonical-domain mapping, and the
source-chunk validation; Cognee's LLMGateway resolves the configured provider,
model, endpoint, retries, and structured-output framework. Switching providers
is a configuration-only change.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperos_core.config import LLMSettings
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


class LLMClient:
    """Run PaperOS prompts through Cognee's configured LLMGateway."""

    def __init__(
        self,
        config: LLMSettings,
        prompts: PromptRepository,
    ) -> None:
        self.config = config
        self.prompts = prompts

    async def health_check(self) -> dict[str, Any]:
        if not self.config.model or not self.config.api_key_value():
            raise SemanticEnrichmentError(
                "LLM requires model configuration and LLM_API_KEY.",
                affected="LLM_API_KEY",
                retryable=False,
            )
        from cognee.infrastructure.llm.utils import (  # type: ignore[import-untyped]
            test_llm_connection,
        )

        try:
            await asyncio.wait_for(
                test_llm_connection(),
                timeout=min(self.config.timeout_seconds, 15),
            )
        except Exception as exc:
            raise SemanticEnrichmentError(
                f"LLM health check failed: {exc}",
                affected=self.config.endpoint,
            ) from exc
        return {
            "status": "healthy",
            "provider": self.config.provider,
            "model": self.config.model,
        }

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
            response_model=_EnrichmentExtraction,
        )
        return _to_domain(bundle, extraction, self.config.model, prompt)

    async def _generate_structured(
        self,
        *,
        system: str,
        user: str,
        response_model: type[_EnrichmentExtraction],
    ) -> _EnrichmentExtraction:
        from cognee.infrastructure.llm import LLMGateway  # type: ignore[import-untyped]

        failures: list[str] = []
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = await LLMGateway.acreate_structured_output(
                    text_input=user,
                    system_prompt=system,
                    response_model=response_model,
                    temperature=0.1,
                    max_tokens=16_000,
                )
                if not isinstance(result, _EnrichmentExtraction):
                    raise TypeError("structured completion is not the expected schema")
                return result
            except (ValidationError, TypeError, ValueError) as exc:
                failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt < self.config.max_attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
        raise SemanticEnrichmentError(
            "LLM structured semantic enrichment failed after finite retries.",
            affected=self.config.endpoint,
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
        from cognee.infrastructure.llm import LLMGateway  # type: ignore[import-untyped]

        compact_evidence: list[dict[str, Any]] = []
        for item in evidence:
            compact = dict(item)
            text = compact.get("text")
            if isinstance(text, str):
                compact["text"] = text[:3_000]
            compact_evidence.append(compact)
        failures: list[str] = []
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                content = await LLMGateway.acreate_structured_output(
                    text_input=json.dumps(
                        {
                            "profile": profile,
                            "question": query,
                            "evidence": compact_evidence,
                        },
                        ensure_ascii=False,
                    ),
                    system_prompt=self.prompts.load("answer_synthesis"),
                    response_model=str,
                    temperature=0.1,
                    max_tokens=16_000,
                )
                if not isinstance(content, str) or not content.strip():
                    raise TypeError("completion content is empty or not text")
                return content.strip()
            except (TypeError, ValueError) as exc:
                failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt < self.config.max_attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
        raise SemanticEnrichmentError(
            "LLM answer synthesis failed after finite retries.",
            affected=self.config.endpoint,
            details={"attempts": failures},
        )

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
                "LLM returned source chunk IDs outside the canonical snapshot.",
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
                "LLM returned duplicate entity keys.", affected=entity_item.key
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
                "LLM relation references an unknown entity key.",
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
