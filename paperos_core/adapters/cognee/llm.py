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
import logging
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from paperos_core.adapters.cognee.runtime_config import CogneeRuntimeConfigReader
from paperos_core.domain.canonical import CanonicalBundle, Chunk, Section
from paperos_core.domain.ids import semantic_object_id
from paperos_core.domain.knowledge import (
    AboutRole,
    Claim,
    ClaimAboutTarget,
    ConceptRelation,
    Entity,
    KnowledgeStatus,
    SemanticEnrichment,
    Summary,
)
from paperos_core.domain.scholarly import ScholarlyContext, ScholarlyWork
from paperos_core.errors import SemanticEnrichmentError
from paperos_core.prompt_repository import PromptDescriptor, PromptRepository

_BATCH_CHARACTER_BUDGET = 20_000
_SELF_WORK_KEY = "SELF"
_ABOUT_ROLES = {role.value for role in AboutRole}
_LOGGER = logging.getLogger(__name__)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _EntityExtraction(_StrictModel):
    key: str
    name: str
    entity_type: str
    description: str | None = None
    source_chunk_ids: list[str] = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)


class _ClaimAboutExtraction(_StrictModel):
    work_key: str
    role: str
    source_chunk_ids: list[str] | None = None


class _ClaimExtraction(_StrictModel):
    key: str
    text: str
    claim_type: str | None = None
    source_chunk_ids: list[str] = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    about: list[_ClaimAboutExtraction] = Field(default_factory=list)


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
    cited_chunk_ids: list[str]


class ScopePlannerOutput(_StrictModel):
    """Bounded work-catalog scope plan. Never invents work keys or answers."""

    source_work_keys: list[str] = Field(default_factory=list)
    exclude_source_work_keys: list[str] = Field(default_factory=list)
    subject_work_keys: list[str] = Field(default_factory=list)
    work_set_work_keys: list[str] = Field(default_factory=list)
    topic_queries: list[str] = Field(default_factory=list)
    notes: str | None = None
    confident: bool = True


_T = TypeVar("_T", bound=BaseModel)


@dataclass(slots=True)
class _MergedEntity:
    entity_type: str
    name: str
    description: str | None
    confidence: float | None
    source_chunk_ids: list[str]


@dataclass(slots=True)
class _MergedAbout:
    work_id: str
    roles: list[str]
    source_chunk_ids: list[str]


@dataclass(slots=True)
class _MergedClaim:
    text: str
    claim_type: str | None
    confidence: float | None
    source_chunk_ids: list[str]
    about_by_work: dict[str, _MergedAbout]


@dataclass(slots=True)
class _MergedRelation:
    source_object_id: str
    target_object_id: str
    relation_type: str
    description: str | None
    confidence: float | None
    source_chunk_ids: list[str]


@dataclass(slots=True)
class _WorkCatalog:
    key_to_work_id: dict[str, str]
    entries: list[dict[str, Any]]


class LLMClient:
    """Run PaperOS prompts through Cognee's configured LLMGateway."""

    def __init__(
        self,
        prompts: PromptRepository,
        runtime_config: CogneeRuntimeConfigReader,
    ) -> None:
        self.prompts = prompts
        self.runtime_config = runtime_config

    @property
    def provider(self) -> str:
        return self.runtime_config.read().llm_provider

    @property
    def model(self) -> str:
        return self.runtime_config.read().llm_model

    async def health_check(self) -> dict[str, Any]:
        config = self.runtime_config.read()
        try:
            await asyncio.wait_for(
                self.runtime_config.test_llm_connection(),
                timeout=15,
            )
        except Exception as exc:
            raise SemanticEnrichmentError(
                f"LLM health check failed: {exc}",
                affected=config.llm_endpoint,
            ) from exc
        return {
            "status": "healthy",
            "provider": config.llm_provider,
            "model": config.llm_model,
        }

    async def enrich(
        self,
        bundle: CanonicalBundle,
        chunks: list[Chunk],
        scholarly: ScholarlyContext,
    ) -> SemanticEnrichment:
        """Section-grouped, batch-local semantic enrichment with coverage."""
        prompt = self.prompts.describe("semantic_enrichment")
        catalog = _build_work_catalog(bundle, scholarly)
        raw_sections: list[tuple[str | None, list[Chunk], _SectionExtraction]] = []
        covered: list[str] = []
        for section_id, section_chunks in _chunks_by_section(chunks, bundle.sections):
            for batch in _chunk_batches(section_chunks):
                covered.extend(chunk.id for chunk in batch)
                extraction = await self._extract_batch(
                    bundle,
                    prompt,
                    section_id=section_id,
                    chunks=batch,
                    catalog=catalog,
                )
                raw_sections.append((section_id, batch, extraction))
        config = self.runtime_config.read()
        entities, claims, relations = _merge_section_extractions(
            bundle,
            raw_sections,
            scholarly=scholarly,
            catalog=catalog,
            model=config.llm_model,
        )
        summary = await self._summarize_document(bundle, chunks, prompt)
        covered_ids = list(dict.fromkeys(covered))
        covered_set = set(covered_ids)
        uncovered_ids = [chunk.id for chunk in chunks if chunk.id not in covered_set]
        total = len(chunks)
        return SemanticEnrichment(
            entities=entities,
            claims=claims,
            relations=relations,
            summaries=[summary],
            model=config.llm_model,
            provider=config.llm_provider,
            model_version=config.llm_model,
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
        catalog: _WorkCatalog,
    ) -> _SectionExtraction:
        request = {
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
                        "about": [
                            {
                                "work_key": "SELF or CITED_NNN from work_catalog",
                                "role": "self|subject|comparison_target|topic",
                                "source_chunk_ids": [
                                    "optional; defaults to claim source_chunk_ids"
                                ],
                            }
                        ],
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
            "work_catalog": catalog.entries,
            "section": section_id or "(front matter)",
            "evidence": _chunk_evidence(chunks),
        }
        failures: list[dict[str, Any]] = []
        extraction: _SectionExtraction | None = None
        for attempt in range(1, 4):
            extraction = await self._generate_structured(
                system=prompt.text,
                user=json.dumps(request, ensure_ascii=False),
                response_model=_SectionExtraction,
            )
            try:
                return _normalize_section_extraction(
                    extraction,
                    valid_chunks={chunk.id for chunk in chunks},
                    valid_work_keys=set(catalog.key_to_work_id),
                    snapshot_id=bundle.snapshot.id,
                )
            except SemanticEnrichmentError as exc:
                failures.append(
                    {
                        "attempt": attempt,
                        "affected": exc.affected,
                        **exc.details,
                    }
                )
                if attempt < 3:
                    request["validation_feedback"] = {
                        "instruction": (
                            "Regenerate the complete response. Every source_chunk_ids value "
                            "must exactly equal a chunk_id present in evidence; relation keys "
                            "must reference entities in this response; claim about.work_key "
                            "must exactly equal a work_key from work_catalog; about.role must "
                            "be one of self, subject, comparison_target, topic."
                        ),
                        "allowed_work_keys": sorted(catalog.key_to_work_id),
                        "previous_error": exc.details,
                    }
        if extraction is None:
            raise SemanticEnrichmentError(
                "LLM semantic extraction produced no structured response.",
                affected=section_id or "(front matter)",
                details={"snapshot_id": bundle.snapshot.id, "attempts": failures},
            )
        return _sanitize_section_extraction(
            extraction,
            valid_chunks={chunk.id for chunk in chunks},
            valid_work_keys=set(catalog.key_to_work_id),
            snapshot_id=bundle.snapshot.id,
        )

    async def _summarize_document(
        self,
        bundle: CanonicalBundle,
        chunks: list[Chunk],
        prompt: PromptDescriptor,
    ) -> Summary:
        grouped = _chunks_by_section(chunks, bundle.sections)
        if not grouped:
            raise SemanticEnrichmentError(
                "Canonical snapshot contains no non-empty chunks for a summary.",
                affected=bundle.snapshot.id,
            )
        section_summaries: list[dict[str, Any]] = []
        for section_id, section_chunks in grouped:
            batches: list[_SummaryExtraction] = []
            for batch_index, batch in enumerate(_chunk_batches(section_chunks)):
                batches.append(
                    await self._summarize_evidence(
                        bundle,
                        prompt,
                        task=(
                            f"summarize section {section_id or '(front matter)'} "
                            f"batch {batch_index + 1}"
                        ),
                        evidence=_chunk_evidence(batch),
                        valid_chunks={chunk.id for chunk in batch},
                        object_key=(
                            f"section_summary:{section_id or '(front matter)'}:"
                            f"batch:{batch_index + 1}"
                        ),
                    )
                )
            source_ids = list(
                dict.fromkeys(chunk_id for item in batches for chunk_id in item.source_chunk_ids)
            )
            if len(batches) == 1:
                section_text = batches[0].text
            else:
                merged = await self._summarize_evidence(
                    bundle,
                    prompt,
                    task=f"merge section {section_id or '(front matter)'} summaries",
                    evidence=[
                        {"text": item.text, "source_chunk_ids": item.source_chunk_ids}
                        for item in batches
                    ],
                    valid_chunks=set(source_ids),
                    object_key=f"section_summary:{section_id or '(front matter)'}:merge",
                )
                section_text = merged.text
            section_summaries.append(
                {
                    "section_id": section_id,
                    "text": section_text,
                    "source_chunk_ids": source_ids,
                }
            )
        extraction = await self._summarize_evidence(
            bundle,
            prompt,
            task="merge all section summaries into one four-sentence document summary",
            evidence=section_summaries,
            valid_chunks={chunk.id for chunk in chunks},
            object_key="document_summary",
        )
        summary_id = semantic_object_id(
            "summary",
            bundle.snapshot.id,
            extraction.text,
            extraction.source_chunk_ids,
        )
        return Summary(
            id=summary_id,
            canonical_snapshot_id=bundle.snapshot.id,
            summary_type="document",
            text=extraction.text,
            status=KnowledgeStatus.INFERRED,
            source_chunk_ids=extraction.source_chunk_ids,
            derived_from_ids=extraction.source_chunk_ids,
            model=self.model,
            model_version=self.model,
        )

    async def _summarize_evidence(
        self,
        bundle: CanonicalBundle,
        prompt: PromptDescriptor,
        *,
        task: str,
        evidence: list[dict[str, Any]],
        valid_chunks: set[str],
        object_key: str,
    ) -> _SummaryExtraction:
        request: dict[str, Any] = {
            "schema": {"text": "string", "source_chunk_ids": ["chunk id"]},
            "document": {
                "title": bundle.document.title,
                "abstract": bundle.document.abstract,
            },
            "task": task,
            "evidence": evidence,
        }
        failures: list[dict[str, Any]] = []
        for attempt in range(1, 4):
            extraction = await self._generate_structured(
                system=prompt.text,
                user=json.dumps(request, ensure_ascii=False),
                response_model=_SummaryExtraction,
            )
            try:
                source_ids = _validate_chunk_ids(
                    extraction.source_chunk_ids,
                    valid_chunks,
                    object_key=object_key,
                    snapshot_id=bundle.snapshot.id,
                )
                return extraction.model_copy(update={"source_chunk_ids": source_ids})
            except SemanticEnrichmentError as exc:
                failures.append(
                    {
                        "attempt": attempt,
                        "affected": exc.affected,
                        **exc.details,
                    }
                )
                if attempt < 3:
                    request["validation_feedback"] = {
                        "instruction": (
                            "Regenerate the complete summary. Every source_chunk_ids value "
                            "must exactly equal a chunk_id present in evidence."
                        ),
                        "allowed_source_chunk_ids": sorted(valid_chunks),
                        "previous_error": exc.details,
                    }
        raise SemanticEnrichmentError(
            "LLM summary provenance remained invalid after finite retries.",
            affected=object_key,
            details={"snapshot_id": bundle.snapshot.id, "attempts": failures},
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
        for attempt in range(1, 4):
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
                if attempt < 3:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
        raise SemanticEnrichmentError(
            "LLM structured output failed after finite retries.",
            affected=self.runtime_config.read().llm_endpoint,
            details={"attempts": failures},
        )

    async def plan_query_scope(
        self,
        *,
        query: str,
        catalog_entries: list[dict[str, Any]],
    ) -> ScopePlannerOutput:
        """Classify source/subject/work-set/topic using a bounded Work catalog."""
        from cognee.infrastructure.llm import LLMGateway

        content = await LLMGateway.acreate_structured_output(
            text_input=json.dumps(
                {
                    "question": query,
                    "work_catalog": catalog_entries,
                    "instructions": {
                        "return_only_catalog_keys": True,
                        "do_not_answer": True,
                        "do_not_retrieve": True,
                    },
                },
                ensure_ascii=False,
            ),
            system_prompt=self.prompts.load("query_scope"),
            response_model=ScopePlannerOutput,
            temperature=0.0,
            max_tokens=2_000,
        )
        if not isinstance(content, ScopePlannerOutput):
            raise TypeError("scope planner returned unexpected type")
        return content

    async def synthesize_answer(
        self,
        *,
        query: str,
        profile: str,
        evidence: list[dict[str, Any]],
        recall_context: list[str] | None = None,
        resolved_scope: dict[str, Any] | None = None,
    ) -> str:
        """Synthesize one evidence-bound answer with an explicit Pydantic schema."""
        from cognee.infrastructure.llm import LLMGateway

        compact_evidence: list[dict[str, Any]] = []
        for item in evidence:
            compact = dict(item)
            text = compact.get("text")
            if isinstance(text, str):
                compact["text"] = text[:3_000]
            compact_evidence.append(compact)
        failures: list[str] = []
        valid_chunk_ids = {
            str(item["chunk_id"]) for item in compact_evidence if item.get("chunk_id")
        }
        evidence_to_chunk = {
            str(item["evidence_id"]): str(item["chunk_id"])
            for item in compact_evidence
            if item.get("evidence_id") and item.get("chunk_id")
        }

        def resolve_cited_chunk_id(raw_id: str) -> str | None:
            selected = raw_id.strip().strip("[]［］【】")
            if selected in valid_chunk_ids:
                return selected
            if selected in evidence_to_chunk:
                return evidence_to_chunk[selected]
            candidates = [chunk_id for chunk_id in valid_chunk_ids if chunk_id.startswith(selected)]
            return candidates[0] if len(candidates) == 1 else None

        for attempt in range(1, 4):
            try:
                content = await LLMGateway.acreate_structured_output(
                    text_input=json.dumps(
                        {
                            "profile": profile,
                            "question": query,
                            "resolved_scope": resolved_scope or {},
                            "evidence": compact_evidence,
                            "recall_context": recall_context or [],
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
                normalized_citations = [
                    resolve_cited_chunk_id(chunk_id) for chunk_id in content.cited_chunk_ids
                ]
                if not normalized_citations or any(
                    chunk_id is None for chunk_id in normalized_citations
                ):
                    raise ValueError(
                        "answer cites chunks outside the supplied evidence: "
                        f"returned={content.cited_chunk_ids!r}"
                    )
                cited_chunk_ids = [
                    chunk_id for chunk_id in normalized_citations if chunk_id is not None
                ]
                answer = content.answer.strip()
                # ``cited_chunk_ids`` is the structured source of truth.  Some
                # providers satisfy the schema but omit inline markers from the
                # prose, so materialize those genuine model-selected citations
                # deterministically for the public evidence-bound answer.
                missing_citations = [
                    chunk_id
                    for chunk_id in dict.fromkeys(cited_chunk_ids)
                    if chunk_id not in answer
                ]
                if missing_citations:
                    answer = f"{answer} " + " ".join(
                        f"[{chunk_id}]" for chunk_id in missing_citations
                    )
                return answer
            except (TypeError, ValueError) as exc:
                failures.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
                if attempt < 3:
                    await asyncio.sleep(min(2 ** (attempt - 1), 4))
        raise SemanticEnrichmentError(
            "LLM answer synthesis failed after finite retries.",
            affected=self.runtime_config.read().llm_endpoint,
            details={"attempts": failures},
        )


def _chunks_by_section(
    chunks: list[Chunk],
    sections: list[Section],
) -> list[tuple[str | None, list[Chunk]]]:
    groups: dict[str | None, list[Chunk]] = {}
    for chunk in chunks:
        groups.setdefault(chunk.section_id, []).append(chunk)
    ordered: list[tuple[str | None, list[Chunk]]] = []
    for section_id in [None, *(section.id for section in sections)]:
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
        size = len(chunk.text) + 80
        if current and budget + size > character_budget:
            batches.append(current)
            current = []
            budget = 0
        current.append(chunk)
        budget += size
    if current:
        batches.append(current)
    return batches


def _chunk_evidence(chunks: list[Chunk]) -> list[dict[str, str]]:
    return [
        {
            "chunk_id": chunk.id,
            "section_path": chunk.section_path or "",
            "text": chunk.text,
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
        candidate = value.strip().strip("`'\"[](){}<>").strip()
        if candidate in valid_chunks:
            resolved.append(candidate)
            continue
        prefix_matches = sorted(
            chunk_id for chunk_id in valid_chunks if chunk_id.startswith(candidate)
        )
        if candidate.startswith("chunk_") and len(prefix_matches) == 1:
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


def _normalize_about_items(
    about_items: list[_ClaimAboutExtraction],
    *,
    claim_source_ids: list[str],
    valid_chunks: set[str],
    valid_work_keys: set[str],
    object_key: str,
    snapshot_id: str,
    raise_on_invalid_work: bool,
) -> list[_ClaimAboutExtraction]:
    normalized: list[_ClaimAboutExtraction] = []
    invalid_work_keys: list[str] = []
    invalid_roles: list[str] = []
    for item in about_items:
        work_key = item.work_key.strip().upper()
        if work_key not in valid_work_keys:
            invalid_work_keys.append(item.work_key)
            continue
        role = item.role.strip().casefold().replace("-", "_").replace(" ", "_")
        if role not in _ABOUT_ROLES:
            invalid_roles.append(item.role)
            continue
        raw_ids = item.source_chunk_ids if item.source_chunk_ids else claim_source_ids
        try:
            about_ids = _validate_chunk_ids(
                raw_ids,
                valid_chunks,
                object_key=f"{object_key}:about:{work_key}",
                snapshot_id=snapshot_id,
            )
        except SemanticEnrichmentError:
            if raise_on_invalid_work:
                raise
            continue
        if not about_ids:
            continue
        normalized.append(
            _ClaimAboutExtraction(
                work_key=work_key,
                role=role,
                source_chunk_ids=about_ids,
            )
        )
    if raise_on_invalid_work and (invalid_work_keys or invalid_roles):
        raise SemanticEnrichmentError(
            "LLM claim ABOUT grounding is invalid.",
            affected=object_key,
            details={
                "snapshot_id": snapshot_id,
                "invalid_work_keys": sorted(set(invalid_work_keys)),
                "invalid_roles": sorted(set(invalid_roles)),
                "allowed_work_keys": sorted(valid_work_keys),
                "allowed_roles": sorted(_ABOUT_ROLES),
            },
        )
    if invalid_work_keys or invalid_roles:
        _LOGGER.warning(
            "Dropped invalid claim ABOUT relations after finite retries: "
            "snapshot_id=%s claim=%s invalid_work_keys=%s invalid_roles=%s",
            snapshot_id,
            object_key,
            sorted(set(invalid_work_keys)),
            sorted(set(invalid_roles)),
        )
    return normalized


def _normalize_section_extraction(
    extraction: _SectionExtraction,
    *,
    valid_chunks: set[str],
    valid_work_keys: set[str],
    snapshot_id: str,
) -> _SectionExtraction:
    """Validate one response while it can still be retried, then canonicalize IDs."""
    entities = [
        item.model_copy(
            update={
                "source_chunk_ids": _validate_chunk_ids(
                    item.source_chunk_ids,
                    valid_chunks,
                    object_key=item.key,
                    snapshot_id=snapshot_id,
                )
            }
        )
        for item in extraction.entities
    ]
    claims: list[_ClaimExtraction] = []
    for item in extraction.claims:
        source_ids = _validate_chunk_ids(
            item.source_chunk_ids,
            valid_chunks,
            object_key=item.key,
            snapshot_id=snapshot_id,
        )
        about = _normalize_about_items(
            item.about,
            claim_source_ids=source_ids,
            valid_chunks=valid_chunks,
            valid_work_keys=valid_work_keys,
            object_key=item.key,
            snapshot_id=snapshot_id,
            raise_on_invalid_work=True,
        )
        claims.append(
            item.model_copy(update={"source_chunk_ids": source_ids, "about": about})
        )
    entity_keys = {item.key for item in entities}
    relations: list[_RelationExtraction] = []
    for item in extraction.relations:
        unknown_keys = sorted({item.source_key, item.target_key} - entity_keys)
        if unknown_keys:
            raise SemanticEnrichmentError(
                "LLM relation references an unknown entity key.",
                affected=f"{item.source_key}->{item.target_key}",
                details={
                    "snapshot_id": snapshot_id,
                    "unknown_entity_keys": unknown_keys,
                },
            )
        relations.append(
            item.model_copy(
                update={
                    "source_chunk_ids": _validate_chunk_ids(
                        item.source_chunk_ids,
                        valid_chunks,
                        object_key=f"{item.source_key}->{item.target_key}",
                        snapshot_id=snapshot_id,
                    )
                }
            )
        )
    return extraction.model_copy(
        update={"entities": entities, "claims": claims, "relations": relations}
    )


def _sanitize_section_extraction(
    extraction: _SectionExtraction,
    *,
    valid_chunks: set[str],
    valid_work_keys: set[str],
    snapshot_id: str,
) -> _SectionExtraction:
    """Drop ungrounded items after retries instead of persisting bad provenance."""

    def valid_ids(ids: list[str], *, object_key: str) -> list[str] | None:
        try:
            return _validate_chunk_ids(
                ids,
                valid_chunks,
                object_key=object_key,
                snapshot_id=snapshot_id,
            )
        except SemanticEnrichmentError:
            return None

    entities: list[_EntityExtraction] = []
    for item in extraction.entities:
        source_ids = valid_ids(item.source_chunk_ids, object_key=item.key)
        if source_ids:
            entities.append(item.model_copy(update={"source_chunk_ids": source_ids}))

    claims: list[_ClaimExtraction] = []
    for item in extraction.claims:
        source_ids = valid_ids(item.source_chunk_ids, object_key=item.key)
        if not source_ids:
            continue
        about = _normalize_about_items(
            item.about,
            claim_source_ids=source_ids,
            valid_chunks=valid_chunks,
            valid_work_keys=valid_work_keys,
            object_key=item.key,
            snapshot_id=snapshot_id,
            raise_on_invalid_work=False,
        )
        claims.append(
            item.model_copy(update={"source_chunk_ids": source_ids, "about": about})
        )

    entity_keys = {item.key for item in entities}
    relations: list[_RelationExtraction] = []
    for item in extraction.relations:
        if item.source_key not in entity_keys or item.target_key not in entity_keys:
            continue
        source_ids = valid_ids(
            item.source_chunk_ids,
            object_key=f"{item.source_key}->{item.target_key}",
        )
        if source_ids:
            relations.append(item.model_copy(update={"source_chunk_ids": source_ids}))

    dropped_entity_count = len(extraction.entities) - len(entities)
    dropped_claim_count = len(extraction.claims) - len(claims)
    dropped_relation_count = len(extraction.relations) - len(relations)
    if dropped_entity_count or dropped_claim_count or dropped_relation_count:
        _LOGGER.warning(
            "Dropped semantic objects with invalid provenance after finite retries: "
            "snapshot_id=%s dropped_entity_count=%d dropped_claim_count=%d "
            "dropped_relation_count=%d",
            snapshot_id,
            dropped_entity_count,
            dropped_claim_count,
            dropped_relation_count,
        )

    return extraction.model_copy(
        update={"entities": entities, "claims": claims, "relations": relations}
    )


def _build_work_catalog(
    bundle: CanonicalBundle,
    scholarly: ScholarlyContext,
) -> _WorkCatalog:
    """Build a bounded SELF/CITED_* catalog; LLM never sees real work_<uuid> IDs."""
    works_by_id = {work.id: work for work in scholarly.works}
    document_work = scholarly.document_work
    works_by_id[document_work.id] = document_work
    references_by_id = {reference.id: reference for reference in bundle.references}
    reference_texts: dict[str, list[str]] = {}
    for resolution in scholarly.reference_resolutions:
        if resolution.work_id is None:
            continue
        reference = references_by_id.get(resolution.reference_id)
        if reference is None:
            continue
        text = reference.raw_text.strip()
        if not text:
            continue
        reference_texts.setdefault(resolution.work_id, [])
        if text not in reference_texts[resolution.work_id]:
            reference_texts[resolution.work_id].append(text)

    key_to_work_id = {_SELF_WORK_KEY: document_work.id}
    entries: list[dict[str, Any]] = [
        _catalog_entry(_SELF_WORK_KEY, document_work, reference_texts.get(document_work.id, []))
    ]
    cited_works = sorted(
        (
            work
            for work_id, work in works_by_id.items()
            if work_id != document_work.id
        ),
        key=lambda work: work.id,
    )
    for index, work in enumerate(cited_works, start=1):
        key = f"CITED_{index:03d}"
        key_to_work_id[key] = work.id
        entries.append(
            _catalog_entry(key, work, reference_texts.get(work.id, []))
        )
    return _WorkCatalog(key_to_work_id=key_to_work_id, entries=entries)


def _catalog_entry(
    work_key: str,
    work: ScholarlyWork,
    reference_texts: list[str],
) -> dict[str, Any]:
    return {
        "work_key": work_key,
        "title": work.title,
        "authors": list(work.authors),
        "year": work.year,
        "doi": work.doi,
        "arxiv_id": work.arxiv_id,
        "reference_texts": reference_texts[:3],
    }


def _merge_section_extractions(
    bundle: CanonicalBundle,
    raw_sections: list[tuple[str | None, list[Chunk], _SectionExtraction]],
    *,
    scholarly: ScholarlyContext,
    catalog: _WorkCatalog,
    model: str,
) -> tuple[list[Entity], list[Claim], list[ConceptRelation]]:
    """Merge batch-local extractions into canonical, deduplicated objects."""
    snapshot_id = bundle.snapshot.id
    source_document_id = bundle.document.id
    source_work_id = scholarly.document_work.id
    entity_by_key: dict[tuple[str, str], _MergedEntity] = {}
    local_entity_keys: dict[tuple[int, str], tuple[str, str]] = {}

    for batch_index, (_section_id, chunks, extraction) in enumerate(raw_sections):
        batch_valid = {chunk.id for chunk in chunks}
        for entity_item in extraction.entities:
            source_ids = _validate_chunk_ids(
                entity_item.source_chunk_ids,
                batch_valid,
                object_key=entity_item.key,
                snapshot_id=snapshot_id,
            )
            key = (entity_item.entity_type.casefold(), entity_item.name.casefold())
            entity_merged = entity_by_key.get(key)
            if entity_merged is None:
                entity_merged = _MergedEntity(
                    entity_type=entity_item.entity_type,
                    name=entity_item.name,
                    description=entity_item.description,
                    confidence=entity_item.confidence,
                    source_chunk_ids=[],
                )
                entity_by_key[key] = entity_merged
            entity_merged.source_chunk_ids = list(
                dict.fromkeys([*entity_merged.source_chunk_ids, *source_ids])
            )
            if entity_merged.description is None and entity_item.description:
                entity_merged.description = entity_item.description
            if entity_item.confidence is not None and (
                entity_merged.confidence is None
                or entity_item.confidence > entity_merged.confidence
            ):
                entity_merged.confidence = entity_item.confidence
            local_entity_keys[(batch_index, entity_item.key)] = key

    entity_id_by_key: dict[tuple[str, str], str] = {}
    entities: list[Entity] = []
    for key, entity_merged in entity_by_key.items():
        entity_id = semantic_object_id(
            "entity",
            snapshot_id,
            f"{entity_merged.entity_type}:{entity_merged.name}",
            entity_merged.source_chunk_ids,
        )
        entity_id_by_key[key] = entity_id
        entities.append(
            Entity(
                id=entity_id,
                canonical_snapshot_id=snapshot_id,
                entity_type=entity_merged.entity_type,
                name=entity_merged.name,
                description=entity_merged.description,
                status=KnowledgeStatus.EXTRACTED,
                source_chunk_ids=entity_merged.source_chunk_ids,
                derived_from_ids=list(entity_merged.source_chunk_ids),
                confidence=entity_merged.confidence,
                model=model,
                model_version=model,
            )
        )

    claim_by_key: dict[str, _MergedClaim] = {}
    for _batch_index, (_section_id, chunks, extraction) in enumerate(raw_sections):
        batch_valid = {chunk.id for chunk in chunks}
        for claim_item in extraction.claims:
            source_ids = _validate_chunk_ids(
                claim_item.source_chunk_ids,
                batch_valid,
                object_key=claim_item.key,
                snapshot_id=snapshot_id,
            )
            claim_key = claim_item.text.casefold()
            claim_merged = claim_by_key.get(claim_key)
            if claim_merged is None:
                claim_merged = _MergedClaim(
                    text=claim_item.text,
                    claim_type=claim_item.claim_type,
                    confidence=claim_item.confidence,
                    source_chunk_ids=[],
                    about_by_work={},
                )
                claim_by_key[claim_key] = claim_merged
            claim_merged.source_chunk_ids = list(
                dict.fromkeys([*claim_merged.source_chunk_ids, *source_ids])
            )
            if claim_item.confidence is not None and (
                claim_merged.confidence is None or claim_item.confidence > claim_merged.confidence
            ):
                claim_merged.confidence = claim_item.confidence
            for about_item in claim_item.about:
                work_id = catalog.key_to_work_id.get(about_item.work_key.strip().upper())
                if work_id is None:
                    _LOGGER.warning(
                        "Dropped ABOUT with unknown work_key during merge: "
                        "snapshot_id=%s work_key=%s",
                        snapshot_id,
                        about_item.work_key,
                    )
                    continue
                about_ids = list(about_item.source_chunk_ids or source_ids)
                about_ids = _validate_chunk_ids(
                    about_ids,
                    batch_valid,
                    object_key=f"{claim_item.key}:about:{about_item.work_key}",
                    snapshot_id=snapshot_id,
                )
                role = about_item.role.strip().casefold().replace("-", "_").replace(" ", "_")
                if role not in _ABOUT_ROLES:
                    continue
                about_merged = claim_merged.about_by_work.get(work_id)
                if about_merged is None:
                    about_merged = _MergedAbout(
                        work_id=work_id,
                        roles=[],
                        source_chunk_ids=[],
                    )
                    claim_merged.about_by_work[work_id] = about_merged
                about_merged.roles = list(
                    dict.fromkeys([*about_merged.roles, role])
                )
                about_merged.source_chunk_ids = list(
                    dict.fromkeys([*about_merged.source_chunk_ids, *about_ids])
                )

    claims: list[Claim] = []
    for claim_merged in claim_by_key.values():
        claim_id = semantic_object_id(
            "claim", snapshot_id, claim_merged.text, claim_merged.source_chunk_ids
        )
        about_targets = [
            ClaimAboutTarget(
                work_id=about_merged.work_id,
                roles=list(about_merged.roles),
                source_chunk_ids=list(about_merged.source_chunk_ids),
                derived_from_ids=list(about_merged.source_chunk_ids),
            )
            for about_merged in claim_merged.about_by_work.values()
            if about_merged.source_chunk_ids
        ]
        claims.append(
            Claim(
                id=claim_id,
                canonical_snapshot_id=snapshot_id,
                text=claim_merged.text,
                claim_type=claim_merged.claim_type,
                status=KnowledgeStatus.EXTRACTED,
                source_chunk_ids=claim_merged.source_chunk_ids,
                derived_from_ids=list(claim_merged.source_chunk_ids),
                confidence=claim_merged.confidence,
                model=model,
                model_version=model,
                source_document_id=source_document_id,
                source_work_id=source_work_id,
                about=about_targets,
            )
        )

    relation_by_key: dict[tuple[str, str, str], _MergedRelation] = {}
    for batch_index, (_section_id, chunks, extraction) in enumerate(raw_sections):
        batch_valid = {chunk.id for chunk in chunks}
        for relation_item in extraction.relations:
            source_key = local_entity_keys.get((batch_index, relation_item.source_key))
            target_key = local_entity_keys.get((batch_index, relation_item.target_key))
            if source_key is None or target_key is None:
                raise SemanticEnrichmentError(
                    "LLM relation references an unknown entity key.",
                    affected=f"{relation_item.source_key}->{relation_item.target_key}",
                )
            source_ids = _validate_chunk_ids(
                relation_item.source_chunk_ids,
                batch_valid,
                object_key=f"{relation_item.source_key}->{relation_item.target_key}",
                snapshot_id=snapshot_id,
            )
            source_object_id = entity_id_by_key[source_key]
            target_object_id = entity_id_by_key[target_key]
            relation_type = relation_item.relation_type.upper().replace(" ", "_")
            if relation_type == "ABOUT":
                # ABOUT is Claim→Work only; Entity relations must not emit it.
                continue
            relation_key = (source_object_id, relation_type, target_object_id)
            relation_merged = relation_by_key.get(relation_key)
            if relation_merged is None:
                relation_merged = _MergedRelation(
                    source_object_id=source_object_id,
                    target_object_id=target_object_id,
                    relation_type=relation_type,
                    description=relation_item.description,
                    confidence=relation_item.confidence,
                    source_chunk_ids=[],
                )
                relation_by_key[relation_key] = relation_merged
            relation_merged.source_chunk_ids = list(
                dict.fromkeys([*relation_merged.source_chunk_ids, *source_ids])
            )
            if relation_merged.description is None and relation_item.description:
                relation_merged.description = relation_item.description
            if relation_item.confidence is not None and (
                relation_merged.confidence is None
                or relation_item.confidence > relation_merged.confidence
            ):
                relation_merged.confidence = relation_item.confidence

    relations: list[ConceptRelation] = []
    for relation_merged in relation_by_key.values():
        relation_id = semantic_object_id(
            "relation",
            snapshot_id,
            f"{relation_merged.source_object_id}:{relation_merged.relation_type}:{relation_merged.target_object_id}",
            relation_merged.source_chunk_ids,
        )
        relations.append(
            ConceptRelation(
                id=relation_id,
                canonical_snapshot_id=snapshot_id,
                relation_type=relation_merged.relation_type,
                source_object_id=relation_merged.source_object_id,
                target_object_id=relation_merged.target_object_id,
                description=relation_merged.description,
                status=KnowledgeStatus.INFERRED,
                source_chunk_ids=relation_merged.source_chunk_ids,
                derived_from_ids=list(relation_merged.source_chunk_ids),
                confidence=relation_merged.confidence,
                model=model,
                model_version=model,
            )
        )
    return entities, claims, relations
