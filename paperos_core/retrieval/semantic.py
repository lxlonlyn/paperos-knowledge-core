"""Semantic retrieval through Cognee's public-first compatibility adapter."""

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
    from paperos_core.retrieval.ablation import (
        current_ablation_policy,
        current_ablation_trace,
        is_claim_object_type,
        record_claim_leak,
    )

    hits = await search.graph_search(
        query,
        dataset=dataset_name,
        top_k=limit * 2,
        search_type=search_type,
    )
    policy = current_ablation_policy()
    trace = current_ablation_trace()
    if trace is not None:
        trace.raw_hits["semantic"] = [
            {
                "object_id": hit.canonical_id,
                "object_type": hit.object_type,
                "score": hit.score,
            }
            for hit in hits
        ]
    if policy is not None and policy.claim_blind:
        cleaned = []
        for hit in hits:
            if is_claim_object_type(hit.object_type):
                record_claim_leak(
                    stage="semantic_raw_hits_filtered",
                    object_id=hit.canonical_id,
                    object_type=hit.object_type,
                )
                continue
            cleaned.append(hit)
        hits = cleaned
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
    from paperos_core.retrieval.ablation import (
        current_ablation_policy,
        current_ablation_trace,
        is_claim_object_type,
        record_claim_leak,
    )

    hits = await search.graph_search(
        query,
        dataset=dataset_name,
        top_k=limit * 2,
        search_type=search_type,
    )
    policy = current_ablation_policy()
    trace = current_ablation_trace()
    if trace is not None:
        trace.raw_hits["entity_claim"] = [
            {
                "object_id": hit.canonical_id,
                "object_type": hit.object_type,
                "score": hit.score,
            }
            for hit in hits
        ]
    allowed_types = set(_ENTITY_CLAIM_TYPES)
    if policy is not None and (
        not policy.claim_hits_allowed_in_entity_claim or policy.claim_blind
    ):
        allowed_types.discard("ClaimDataPoint")
    filtered = []
    for hit in hits:
        if hit.object_type not in allowed_types:
            if is_claim_object_type(hit.object_type) and policy is not None and policy.claim_blind:
                record_claim_leak(
                    stage="entity_claim_raw_hits_filtered",
                    object_id=hit.canonical_id,
                    object_type=hit.object_type,
                )
            continue
        filtered.append(hit)
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
    from paperos_core.retrieval.ablation import (
        current_ablation_policy,
        is_claim_object_type,
        record_claim_leak,
    )

    policy = current_ablation_policy()
    if chunk_only:
        hits = [hit for hit in hits if hit.object_type == _CHUNK_TYPE]
    if policy is not None and policy.claim_blind:
        kept = []
        for hit in hits:
            if is_claim_object_type(hit.object_type):
                record_claim_leak(
                    stage=f"{channel}_hits_to_candidates",
                    object_id=hit.canonical_id,
                    object_type=hit.object_type,
                )
                continue
            kept.append(hit)
        hits = kept
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
            if policy is not None and policy.claim_blind and is_claim_object_type(
                hit.object_type
            ):
                continue
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
