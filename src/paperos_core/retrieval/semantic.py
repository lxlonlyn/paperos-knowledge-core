"""Semantic and entity/claim retrieval through Cognee's vector indexes."""

from __future__ import annotations

from paperos_core.adapters.cognee.repository import (
    ENTITY_CLAIM_VECTOR_COLLECTIONS,
    SEMANTIC_VECTOR_COLLECTIONS,
    CogneeRepository,
    CogneeVectorHit,
)
from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView


async def semantic_retrieve(
    repository: CogneeRepository,
    corpus: CorpusView,
    queries: list[str],
    *,
    limit: int,
    document_ids: set[str],
) -> list[Candidate]:
    hits = await repository.search_vectors(
        queries[:4],
        collections=SEMANTIC_VECTOR_COLLECTIONS,
        limit=limit * 2,
    )
    return _hits_to_candidates(
        hits,
        corpus,
        channel="semantic",
        limit=limit,
        document_ids=document_ids,
    )


async def entity_claim_retrieve(
    repository: CogneeRepository,
    corpus: CorpusView,
    queries: list[str],
    *,
    limit: int,
    document_ids: set[str],
) -> list[Candidate]:
    hits = await repository.search_vectors(
        queries[:4],
        collections=ENTITY_CLAIM_VECTOR_COLLECTIONS,
        limit=limit * 2,
    )
    return _hits_to_candidates(
        hits,
        corpus,
        channel="entity_claim",
        limit=limit,
        document_ids=document_ids,
    )


def _hits_to_candidates(
    hits: list[CogneeVectorHit],
    corpus: CorpusView,
    *,
    channel: str,
    limit: int,
    document_ids: set[str],
) -> list[Candidate]:
    candidates: dict[str, Candidate] = {}
    for hit in hits:
        chunk_ids = list(hit.source_chunk_ids)
        if hit.object_type == "ChunkDataPoint" and hit.canonical_id not in chunk_ids:
            chunk_ids.insert(0, hit.canonical_id)
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
                    if hit.object_type == "ChunkDataPoint"
                    else "structured_relation"
                    if hit.object_type in {"TripletDataPoint", "ConceptRelationDataPoint"}
                    else "system_inference"
                ),
                derived_from_ids=[hit.canonical_id, *hit.derived_from_ids],
            )
            existing = candidates.get(chunk_id)
            if existing is None or hit.score > existing.channel_scores[channel]:
                candidates[chunk_id] = candidate
    return sorted(
        candidates.values(),
        key=lambda item: (-item.channel_scores[channel], item.id),
    )[:limit]
