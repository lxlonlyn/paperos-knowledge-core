"""Validation-only retrieval ablation policy.

Default production path never sets a policy. HTTP Query API and TOML are
unchanged; benchmarks inject a ContextVar for the duration of one query.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.domain.provenance import RelationType
from paperos_core.retrieval.candidates import Candidate, ResolvedQueryScope
from paperos_core.retrieval.corpus import CorpusView


@dataclass(frozen=True)
class RetrievalAblationPolicy:
    """Internal query-time switches for Claim/ABOUT necessity experiments."""

    configuration_id: str
    claim_nodes_visible: bool = True
    about_edges_visible: bool = True
    subject_claim_enabled: bool = True
    claim_hits_allowed_in_entity_claim: bool = True
    claim_seeds_allowed_in_graph: bool = True
    broad_chunk_rag: bool = True
    citation_scope: bool = False
    citation_anchor_expansion: bool = False
    candidate_pool_size: int | None = None
    final_top_k: int | None = None
    skip_synthesis: bool = False
    bypass_query_cache: bool = False

    @property
    def relax_subject_mention_filter(self) -> bool:
        """Citation-scoped chunk RAG must not require subject name in-chunk."""
        return self.citation_scope


@dataclass
class AblationTrace:
    """Filled by RetrievalService when an ablation run asks for instrumentation."""

    configuration_id: str = ""
    channel_candidate_ids: dict[str, list[str]] = field(default_factory=dict)
    channel_candidate_chunk_ids: dict[str, list[str]] = field(default_factory=dict)
    fused_before_rerank_chunk_ids: list[str] = field(default_factory=list)
    fused_before_rerank_candidate_ids: list[str] = field(default_factory=list)
    fused_candidate_channels: dict[str, list[str]] = field(default_factory=dict)
    fused_candidate_ranks: dict[str, int] = field(default_factory=dict)
    reranked_chunk_ids: list[str] = field(default_factory=list)
    selected_chunk_ids: list[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    citation_source_work_ids: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


_POLICY: ContextVar[RetrievalAblationPolicy | None] = ContextVar(
    "paperos_retrieval_ablation_policy", default=None
)
_TRACE: ContextVar[AblationTrace | None] = ContextVar(
    "paperos_retrieval_ablation_trace", default=None
)


def current_ablation_policy() -> RetrievalAblationPolicy | None:
    return _POLICY.get()


def current_ablation_trace() -> AblationTrace | None:
    return _TRACE.get()


@contextmanager
def ablation_policy_context(
    policy: RetrievalAblationPolicy,
    *,
    trace: AblationTrace | None = None,
) -> Iterator[AblationTrace]:
    active_trace = trace or AblationTrace(configuration_id=policy.configuration_id)
    active_trace.configuration_id = policy.configuration_id
    policy_token = _POLICY.set(policy)
    trace_token = _TRACE.set(active_trace)
    try:
        yield active_trace
    finally:
        _TRACE.reset(trace_token)
        _POLICY.reset(policy_token)


def policy_from_spec(config: dict[str, Any], **overrides: Any) -> RetrievalAblationPolicy:
    payload = {
        "configuration_id": str(config["id"]),
        "claim_nodes_visible": bool(config.get("claim_nodes_visible", True)),
        "about_edges_visible": bool(config.get("about_edges_visible", True)),
        "subject_claim_enabled": bool(config.get("subject_claim_enabled", True)),
        "claim_hits_allowed_in_entity_claim": bool(
            config.get("claim_hits_allowed_in_entity_claim", True)
        ),
        "claim_seeds_allowed_in_graph": bool(
            config.get("claim_seeds_allowed_in_graph", True)
        ),
        "broad_chunk_rag": bool(config.get("broad_chunk_rag", True)),
        "citation_scope": bool(config.get("citation_scope", False)),
        "citation_anchor_expansion": bool(
            config.get("citation_anchor_expansion", False)
        ),
        "skip_synthesis": True,
        "bypass_query_cache": True,
    }
    payload.update(overrides)
    return RetrievalAblationPolicy(**payload)


async def expand_citation_source_scope(
    compat: CogneeCompatibilityAdapter,
    corpus: CorpusView,
    scope: ResolvedQueryScope,
    *,
    dataset_name: str,
) -> ResolvedQueryScope:
    """Admit source Works via explicit scope and/or incoming CITES to subjects."""
    explicit_sources = list(scope.source_work_ids)
    admitted: set[str] = set(explicit_sources)
    if scope.subject_work_ids and not explicit_sources:
        relations = await compat.incoming_typed_relations(
            list(scope.subject_work_ids),
            dataset_name=dataset_name,
            relation_type=RelationType.CITES.value,
            depth=1,
            limit=500,
        )
        for relation in relations:
            if relation.source_work_id:
                admitted.add(relation.source_work_id)
        # Self queries with only subject set still keep the subject as source when
        # it is an ingested Work (no incoming citation required).
        for subject_id in scope.subject_work_ids:
            if subject_id in corpus.document_ids_by_work:
                admitted.add(subject_id)
    elif scope.subject_work_ids and explicit_sources:
        # Explicit source remains hard constraint; CITES does not widen it.
        admitted = set(explicit_sources)

    excluded = set(scope.exclude_source_work_ids)
    admitted -= excluded
    if not admitted and explicit_sources:
        admitted = set(explicit_sources) - excluded
    return scope.model_copy(update={"source_work_ids": sorted(admitted)})


async def citation_anchor_retrieve(
    compat: CogneeCompatibilityAdapter,
    corpus: CorpusView,
    scope: ResolvedQueryScope,
    *,
    dataset_name: str,
    limit: int,
) -> list[Candidate]:
    """CITES source_chunk anchors plus immediate prev/next canonical neighbors."""
    if not scope.subject_work_ids or limit <= 0:
        return []
    relations = await compat.incoming_typed_relations(
        list(scope.subject_work_ids),
        dataset_name=dataset_name,
        relation_type=RelationType.CITES.value,
        depth=1,
        limit=500,
    )
    allowed_sources = set(scope.source_work_ids) if scope.source_work_ids else None
    excluded = set(scope.exclude_source_work_ids)
    candidates: dict[str, Candidate] = {}
    for relation in relations:
        source_work_id = relation.source_work_id
        if source_work_id in excluded:
            continue
        if allowed_sources is not None and source_work_id not in allowed_sources:
            continue
        for anchor_id in relation.source_chunk_ids:
            for chunk_id in _anchor_neighborhood(corpus, anchor_id):
                chunk = corpus.chunks.get(chunk_id)
                if chunk is None:
                    continue
                work_id = corpus.work_id_by_document.get(chunk.document_id)
                if work_id in excluded:
                    continue
                if allowed_sources is not None and work_id not in allowed_sources:
                    continue
                score = 1.0 + (0.25 if chunk_id == anchor_id else 0.0)
                candidate = corpus.candidate_for_chunk(
                    chunk_id,
                    channel="citation_anchor",
                    score=score,
                    object_id=relation.source_canonical_id,
                    object_type="citation_anchor",
                    knowledge_kind="structured_relation",
                    derived_from_ids=[
                        relation.source_canonical_id,
                        relation.target_canonical_id,
                        *relation.derived_from_ids,
                        f"cites_anchor:{anchor_id}",
                    ],
                    source_work_id=work_id,
                    subject_work_ids=[relation.target_canonical_id],
                )
                existing = candidates.get(candidate.id)
                if (
                    existing is None
                    or score > existing.channel_scores["citation_anchor"]
                ):
                    candidates[candidate.id] = candidate
    ranked = sorted(
        candidates.values(),
        key=lambda item: (-item.channel_scores["citation_anchor"], item.id),
    )
    return ranked[:limit]


def _anchor_neighborhood(corpus: CorpusView, chunk_id: str) -> list[str]:
    chunk = corpus.chunks.get(chunk_id)
    if chunk is None:
        return []
    ordered = [chunk_id]
    if chunk.previous_chunk_id and chunk.previous_chunk_id in corpus.chunks:
        ordered.insert(0, chunk.previous_chunk_id)
    if chunk.next_chunk_id and chunk.next_chunk_id in corpus.chunks:
        ordered.append(chunk.next_chunk_id)
    return list(dict.fromkeys(ordered))
