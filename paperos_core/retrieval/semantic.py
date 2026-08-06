"""Semantic, entity/claim, and summary retrieval through Cognee public search."""

from __future__ import annotations

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.adapters.cognee.search import CogneeSearchAdapter, CogneeSearchHit
from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView

_CHUNK_TYPE = "ChunkDataPoint"
_SUMMARY_TYPE = "SummaryDataPoint"
_ENTITY_CLAIM_TYPES = {"EntityDataPoint", "ClaimDataPoint"}


async def semantic_retrieve(
    search: CogneeSearchAdapter,
    compat: CogneeCompatibilityAdapter,
    corpus: CorpusView,
    query: str,
    *,
    dataset_name: str,
    search_type: str,
    limit: int,
    document_ids: set[str],
    chunk_only: bool = False,
) -> list[Candidate]:
    hits = await search.graph_search(
        query,
        dataset=dataset_name,
        top_k=limit * 2,
        search_type=search_type,
    )
    return await _hits_to_candidates(
        hits,
        compat,
        corpus,
        channel="semantic",
        limit=limit,
        document_ids=document_ids,
        chunk_only=chunk_only,
    )


async def entity_claim_retrieve(
    search: CogneeSearchAdapter,
    compat: CogneeCompatibilityAdapter,
    corpus: CorpusView,
    query: str,
    *,
    dataset_name: str,
    search_type: str,
    limit: int,
    document_ids: set[str],
) -> list[Candidate]:
    hits = await search.graph_search(
        query,
        dataset=dataset_name,
        top_k=limit * 2,
        search_type=search_type,
    )
    filtered = [hit for hit in hits if hit.object_type in _ENTITY_CLAIM_TYPES]
    return await _hits_to_candidates(
        filtered,
        compat,
        corpus,
        channel="entity_claim",
        limit=limit,
        document_ids=document_ids,
    )


async def summary_retrieve(
    search: CogneeSearchAdapter,
    compat: CogneeCompatibilityAdapter,
    corpus: CorpusView,
    query: str,
    *,
    dataset_name: str,
    search_type: str,
    limit: int,
    document_ids: set[str],
) -> list[Candidate]:
    hits = await search.graph_search(
        query,
        dataset=dataset_name,
        top_k=limit,
        search_type=search_type,
    )
    filtered = [hit for hit in hits if hit.object_type == _SUMMARY_TYPE]
    return await _hits_to_candidates(
        filtered,
        compat,
        corpus,
        channel="global_context",
        limit=limit,
        document_ids=document_ids,
    )


async def _hits_to_candidates(
    hits: list[CogneeSearchHit],
    compat: CogneeCompatibilityAdapter,
    corpus: CorpusView,
    *,
    channel: str,
    limit: int,
    document_ids: set[str],
    chunk_only: bool = False,
) -> list[Candidate]:
    if chunk_only:
        hits = [hit for hit in hits if hit.object_type == _CHUNK_TYPE]
    non_chunk = [
        hit
        for hit in hits
        if hit.object_type != _CHUNK_TYPE
    ]
    resolved = await compat.resolve_graph_nodes(
        [hit.cognee_id for hit in non_chunk]
    )
    candidates: dict[str, Candidate] = {}
    for hit in hits:
        properties: dict[str, object] = {}
        if hit.object_type == _CHUNK_TYPE:
            chunk_ids = [hit.canonical_id]
        else:
            properties = resolved.get(hit.cognee_id, {})
            chunk_ids = _string_list(properties.get("source_chunk_ids"))
        for chunk_id in chunk_ids:
            chunk = corpus.chunks.get(chunk_id)
            if chunk is None or chunk.document_id not in document_ids:
                continue
            object_type = hit.object_type.removesuffix("DataPoint").casefold()
            candidate = corpus.candidate_for_chunk(
                chunk_id,
                channel=channel,
                score=hit.score,
                object_id=hit.canonical_id,
                object_type=object_type,
                knowledge_kind=(
                    "source_fact"
                    if hit.object_type == _CHUNK_TYPE
                    else "system_inference"
                ),
                derived_from_ids=[
                    hit.canonical_id,
                    *_string_list(properties.get("derived_from_ids")),
                ],
            )
            existing = candidates.get(chunk_id)
            if existing is None or hit.score > existing.channel_scores[channel]:
                candidates[chunk_id] = candidate
    return sorted(
        candidates.values(),
        key=lambda item: (-item.channel_scores[channel], item.id),
    )[:limit]


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []
