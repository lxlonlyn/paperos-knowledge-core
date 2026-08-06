"""Provider-neutral LLM client backed exclusively by Cognee's LLMGateway.

PaperOS never talks to a specific LLM vendor or endpoint. It supplies the
prompt, the Pydantic response schema, the canonical-domain mapping, and the
source-chunk validation; Cognee's LLMGateway resolves the configured provider,
model, endpoint, retries, and structured-output framework. Switching providers
is a configuration-only change.

Semantic enrichment is section-aware: chunks are grouped by section, each
section is extracted in bounded batches, the section results are merged, and
the document summary is produced from bounded document-level evidence. The
manifest records exactly which chunks were input to the LLM
(``covered_chunk_ids``), which were not (``uncovered_chunk_ids``), and the
coverage ratio; chunks that were never fed to the model are never claimed as
covered.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperos_core.config import LLMSettings
from paperos_core.domain.canonical import CanonicalBundle, Chunk
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

_BATCH_CHARACTER_BUDGET = 20_000
_CHUNK_TEXT_LIMIT = 4_000
_SUMMARY_CHARACTER_BUDGET = 16_000


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


class _SectionExtraction(_StrictModel):
    entities: list[_EntityExtraction]
    claims: list[_ClaimExtraction]
    relations: list[_RelationExtraction]


class _SummaryExtraction(_StrictModel):
    text: str
    source_chunk_ids: list[str] = Field(min_length=1)


class AnswerOutput(_StrictModel):
    """Explicit Pydantic schema for evidence-bound answer synthesis."""

    answer: str


_T = TypeVar("_T", bound=BaseModel)


@dataclass(slots=True)
class _MergedEntity:
    entity_type: str
    name: str
    description: str | None
    confidence: float | None
    source_chunk_ids: list[str]


@dataclass(slots=True)
class _MergedClaim:
    text: str
    claim_type: str | None
    confidence: float | None
    source_chunk_ids: list[str]


@dataclass(slots=True)
class _MergedRelation:
    source_object_id: str
    target_object_id: str
    relation_type: str
    description: str | None
    confidence: float | None
    source_chunk_ids: list[str]


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
        if not self.config.model:
            raise SemanticEnrichmentError(
                "LLM requires model configuration; the provider validates keys.",
                affected="llm.model",
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
        """Section-grouped, batch-local semantic enrichment with coverage."""
        prompt = self.prompts.describe("semantic_enrichment")
        raw_sections: list[tuple[str | None, list[Chunk], _SectionExtraction]] = []
        covered: list[str] = []
        for section_id, section_chunks in _chunks_by_section(bundle):
            for batch in _chunk_batches(section_chunks):
                covered.extend(chunk.id for chunk in batch)
                extraction = await self._extract_batch(
                    bundle, prompt, section_id=section_id, chunks=batch
                )
                raw_sections.append((section_id, batch, extraction))
        entities, claims, relations = _merge_section_extractions(
            bundle, raw_sections, model=self.config.model
        )
        summary = await self._summarize_document(bundle, prompt)
        covered_ids = list(dict.fromkeys(covered))
        covered_set = set(covered_ids)
        uncovered_ids = [
            chunk.id for chunk in bundle.chunks if chunk.id not in covered_set
        ]
        total = len(bundle.chunks)
        return SemanticEnrichment(
            entities=entities,
            claims=claims,
            relations=relations,
            summaries=[summary],
            model=self.config.model,
            model_version=self.config.model,
            prompt_name=prompt.name,
            prompt_version=prompt.version,
            prompt_sha256=prompt.sha256,
            covered_chunk_ids=covered_ids,
            uncovered_chunk_ids=uncovered_ids,
            coverage_ratio=round(len(covered_ids) / total, 4) if total else 0.0,
        )

    async def _extract_batch(
        self,
        bundle: CanonicalBundle,
        prompt: PromptDescriptor,
        *,
        section_id: str | None,
        chunks: list[Chunk],
    ) -> _SectionExtraction:
        return await self._generate_structured(
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
                    },
                    "document": {
                        "title": bundle.document.title,
                        "abstract": bundle.document.abstract,
                    },
                    "section": section_id or "(front matter)",
                    "evidence": _chunk_evidence(chunks),
                },
                ensure_ascii=False,
            ),
            response_model=_SectionExtraction,
        )

    async def _summarize_document(
        self,
        bundle: CanonicalBundle,
        prompt: PromptDescriptor,
    ) -> Summary:
        summary_evidence = _chunk_evidence(
            _summary_evidence_chunks(bundle)
        )
        if not summary_evidence:
            raise SemanticEnrichmentError(
                "Canonical snapshot contains no non-empty chunks for a summary.",
                affected=bundle.snapshot.id,
            )
        extraction = await self._generate_structured(
            system=prompt.text,
            user=json.dumps(
                {
                    "schema": {
                        "text": "string",
                        "source_chunk_ids": ["chunk id"],
                    },
                    "document": {
                        "title": bundle.document.title,
                        "abstract": bundle.document.abstract,
                    },
                    "task": "produce one four-sentence document summary",
                    "evidence": summary_evidence,
                },
                ensure_ascii=False,
            ),
            response_model=_SummaryExtraction,
        )
        valid_chunks = {item["chunk_id"] for item in summary_evidence}
        source_ids = _validate_chunk_ids(
            extraction.source_chunk_ids,
            valid_chunks,
            object_key="document_summary",
            snapshot_id=bundle.snapshot.id,
        )
        summary_id = semantic_object_id(
            "summary",
            bundle.snapshot.id,
            extraction.text,
            source_ids,
        )
        return Summary(
            id=summary_id,
            canonical_snapshot_id=bundle.snapshot.id,
            summary_type="document",
            text=extraction.text,
            status=KnowledgeStatus.INFERRED,
            source_chunk_ids=source_ids,
            derived_from_ids=source_ids,
            model=self.config.model,
            model_version=self.config.model,
        )

    async def _generate_structured(
        self,
        *,
        system: str,
        user: str,
        response_model: type[_T],
    ) -> _T:
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
                if not isinstance(result, response_model):
                    raise TypeError("structured completion is not the expected schema")
                return result
            except (ValidationError, TypeError, ValueError) as exc:
                failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt < self.config.max_attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
        raise SemanticEnrichmentError(
            "LLM structured output failed after finite retries.",
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
        """Synthesize one evidence-bound answer with an explicit Pydantic schema."""
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
                    response_model=AnswerOutput,
                    temperature=0.1,
                    max_tokens=16_000,
                )
                if not isinstance(content, AnswerOutput) or not content.answer.strip():
                    raise TypeError("completion content is empty or not an AnswerOutput")
                return content.answer.strip()
            except (TypeError, ValueError) as exc:
                failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt < self.config.max_attempts:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
        raise SemanticEnrichmentError(
            "LLM answer synthesis failed after finite retries.",
            affected=self.config.endpoint,
            details={"attempts": failures},
        )


def _chunks_by_section(
    bundle: CanonicalBundle,
) -> list[tuple[str | None, list[Chunk]]]:
    groups: dict[str | None, list[Chunk]] = {}
    for chunk in bundle.chunks:
        groups.setdefault(chunk.section_id, []).append(chunk)
    ordered: list[tuple[str | None, list[Chunk]]] = []
    for section_id in [None, *(section.id for section in bundle.sections)]:
        if groups.get(section_id):
            ordered.append((section_id, groups[section_id]))
    return ordered


def _chunk_batches(
    chunks: list[Chunk],
    *,
    character_budget: int = _BATCH_CHARACTER_BUDGET,
) -> list[list[Chunk]]:
    """Bound each extraction request without dropping any chunk."""
    batches: list[list[Chunk]] = []
    current: list[Chunk] = []
    budget = 0
    for chunk in chunks:
        size = min(len(chunk.text), _CHUNK_TEXT_LIMIT) + 80
        if current and budget + size > character_budget:
            batches.append(current)
            current = []
            budget = 0
        current.append(chunk)
        budget += size
    if current:
        batches.append(current)
    return batches


def _summary_evidence_chunks(bundle: CanonicalBundle) -> list[Chunk]:
    selected: list[Chunk] = []
    budget = _SUMMARY_CHARACTER_BUDGET
    for chunk in bundle.chunks:
        if budget <= 0:
            break
        text = chunk.text[: min(len(chunk.text), _CHUNK_TEXT_LIMIT, budget)]
        if not text.strip():
            continue
        selected.append(chunk)
        budget -= len(text)
    return selected


def _chunk_evidence(chunks: list[Chunk]) -> list[dict[str, str]]:
    return [
        {
            "chunk_id": chunk.id,
            "section_path": chunk.section_path or "",
            "text": chunk.text[:_CHUNK_TEXT_LIMIT],
        }
        for chunk in chunks
        if chunk.text.strip()
    ]


def _validate_chunk_ids(
    ids: list[str],
    valid_chunks: set[str],
    *,
    object_key: str,
    snapshot_id: str,
) -> list[str]:
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
            "LLM returned source chunk IDs outside the supplied evidence.",
            affected=object_key,
            details={
                "snapshot_id": snapshot_id,
                "invalid_or_ambiguous_chunk_ids": sorted(invalid),
            },
        )
    return list(dict.fromkeys(resolved))


def _merge_section_extractions(
    bundle: CanonicalBundle,
    raw_sections: list[tuple[str | None, list[Chunk], _SectionExtraction]],
    *,
    model: str,
) -> tuple[list[Entity], list[Claim], list[ConceptRelation]]:
    """Merge batch-local extractions into canonical, deduplicated objects."""
    snapshot_id = bundle.snapshot.id
    entity_by_key: dict[tuple[str, str], _MergedEntity] = {}
    local_entity_keys: dict[tuple[int, str], tuple[str, str]] = {}

    for batch_index, (_section_id, chunks, extraction) in enumerate(raw_sections):
        batch_valid = {chunk.id for chunk in chunks}
        for item in extraction.entities:
            source_ids = _validate_chunk_ids(
                item.source_chunk_ids,
                batch_valid,
                object_key=item.key,
                snapshot_id=snapshot_id,
            )
            key = (item.entity_type.casefold(), item.name.casefold())
            merged = entity_by_key.get(key)
            if merged is None:
                merged = _MergedEntity(
                    entity_type=item.entity_type,
                    name=item.name,
                    description=item.description,
                    confidence=item.confidence,
                    source_chunk_ids=[],
                )
                entity_by_key[key] = merged
            merged.source_chunk_ids = list(
                dict.fromkeys([*merged.source_chunk_ids, *source_ids])
            )
            if merged.description is None and item.description:
                merged.description = item.description
            if item.confidence is not None and (
                merged.confidence is None or item.confidence > merged.confidence
            ):
                merged.confidence = item.confidence
            local_entity_keys[(batch_index, item.key)] = key

    entity_id_by_key: dict[tuple[str, str], str] = {}
    entities: list[Entity] = []
    for key, merged in entity_by_key.items():
        entity_id = semantic_object_id(
            "entity",
            snapshot_id,
            f"{merged.entity_type}:{merged.name}",
            merged.source_chunk_ids,
        )
        entity_id_by_key[key] = entity_id
        entities.append(
            Entity(
                id=entity_id,
                canonical_snapshot_id=snapshot_id,
                entity_type=merged.entity_type,
                name=merged.name,
                description=merged.description,
                status=KnowledgeStatus.EXTRACTED,
                source_chunk_ids=merged.source_chunk_ids,
                derived_from_ids=list(merged.source_chunk_ids),
                confidence=merged.confidence,
                model=model,
                model_version=model,
            )
        )

    claim_by_key: dict[str, _MergedClaim] = {}
    for batch_index, (_section_id, chunks, extraction) in enumerate(raw_sections):
        batch_valid = {chunk.id for chunk in chunks}
        for item in extraction.claims:
            source_ids = _validate_chunk_ids(
                item.source_chunk_ids,
                batch_valid,
                object_key=item.key,
                snapshot_id=snapshot_id,
            )
            key = item.text.casefold()
            merged = claim_by_key.get(key)
            if merged is None:
                merged = _MergedClaim(
                    text=item.text,
                    claim_type=item.claim_type,
                    confidence=item.confidence,
                    source_chunk_ids=[],
                )
                claim_by_key[key] = merged
            merged.source_chunk_ids = list(
                dict.fromkeys([*merged.source_chunk_ids, *source_ids])
            )
            if item.confidence is not None and (
                merged.confidence is None or item.confidence > merged.confidence
            ):
                merged.confidence = item.confidence

    claims: list[Claim] = []
    for merged in claim_by_key.values():
        claim_id = semantic_object_id(
            "claim", snapshot_id, merged.text, merged.source_chunk_ids
        )
        claims.append(
            Claim(
                id=claim_id,
                canonical_snapshot_id=snapshot_id,
                text=merged.text,
                claim_type=merged.claim_type,
                status=KnowledgeStatus.EXTRACTED,
                source_chunk_ids=merged.source_chunk_ids,
                derived_from_ids=list(merged.source_chunk_ids),
                confidence=merged.confidence,
                model=model,
                model_version=model,
            )
        )

    relation_by_key: dict[
        tuple[str, str, str], _MergedRelation
    ] = {}
    for batch_index, (_section_id, chunks, extraction) in enumerate(raw_sections):
        batch_valid = {chunk.id for chunk in chunks}
        for item in extraction.relations:
            source_key = local_entity_keys.get((batch_index, item.source_key))
            target_key = local_entity_keys.get((batch_index, item.target_key))
            if source_key is None or target_key is None:
                raise SemanticEnrichmentError(
                    "LLM relation references an unknown entity key.",
                    affected=f"{item.source_key}->{item.target_key}",
                )
            source_ids = _validate_chunk_ids(
                item.source_chunk_ids,
                batch_valid,
                object_key=f"{item.source_key}->{item.target_key}",
                snapshot_id=snapshot_id,
            )
            source_object_id = entity_id_by_key[source_key]
            target_object_id = entity_id_by_key[target_key]
            relation_type = item.relation_type.upper().replace(" ", "_")
            relation_key = (source_object_id, relation_type, target_object_id)
            merged = relation_by_key.get(relation_key)
            if merged is None:
                merged = _MergedRelation(
                    source_object_id=source_object_id,
                    target_object_id=target_object_id,
                    relation_type=relation_type,
                    description=item.description,
                    confidence=item.confidence,
                    source_chunk_ids=[],
                )
                relation_by_key[relation_key] = merged
            merged.source_chunk_ids = list(
                dict.fromkeys([*merged.source_chunk_ids, *source_ids])
            )
            if merged.description is None and item.description:
                merged.description = item.description
            if item.confidence is not None and (
                merged.confidence is None or item.confidence > merged.confidence
            ):
                merged.confidence = item.confidence

    relations: list[ConceptRelation] = []
    for merged in relation_by_key.values():
        relation_id = semantic_object_id(
            "relation",
            snapshot_id,
            f"{merged.source_object_id}:{merged.relation_type}:{merged.target_object_id}",
            merged.source_chunk_ids,
        )
        relations.append(
            ConceptRelation(
                id=relation_id,
                canonical_snapshot_id=snapshot_id,
                relation_type=merged.relation_type,
                source_object_id=merged.source_object_id,
                target_object_id=merged.target_object_id,
                description=merged.description,
                status=KnowledgeStatus.INFERRED,
                source_chunk_ids=merged.source_chunk_ids,
                derived_from_ids=list(merged.source_chunk_ids),
                confidence=merged.confidence,
                model=model,
                model_version=model,
            )
        )
    return entities, claims, relations
