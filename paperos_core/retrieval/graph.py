"""Cognee entity seeding, typed graph traversal, and chunk backtracking."""

from __future__ import annotations

from paperos_core.adapters.cognee.repository import (
    GRAPH_SEED_VECTOR_COLLECTIONS,
    CogneeRepository,
)
from paperos_core.domain.provenance import RelationType
from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView


async def graph_retrieve(
    repository: CogneeRepository,
    corpus: CorpusView,
    queries: list[str],
    *,
    limit: int,
    depth: int,
    document_ids: set[str],
) -> list[Candidate]:
    seed_hits = await repository.search_vectors(
        queries[:4],
        collections=GRAPH_SEED_VECTOR_COLLECTIONS,
        limit=limit,
    )
    allowed_seeds = [
        seed
        for seed in seed_hits
        if any(
            chunk_id in corpus.chunks
            and corpus.chunks[chunk_id].document_id in document_ids
            for chunk_id in seed.source_chunk_ids
        )
    ]
    traversed = await repository.traverse(
        allowed_seeds,
        depth=depth,
        edge_types={relation.value for relation in RelationType},
    )
    candidates: dict[str, Candidate] = {}
    for relation in traversed:
        for chunk_id in relation.source_chunk_ids:
            chunk = corpus.chunks.get(chunk_id)
            if chunk is None or chunk.document_id not in document_ids:
                continue
            candidate = corpus.candidate_for_chunk(
                chunk_id,
                channel="graph",
                score=relation.score,
                object_id=relation.source_canonical_id,
                object_type=f"graph_relation:{relation.relation_type}",
                knowledge_kind="structured_relation",
                derived_from_ids=[
                    relation.source_canonical_id,
                    relation.target_canonical_id,
                    *relation.derived_from_ids,
                ],
            )
            existing = candidates.get(chunk_id)
            if existing is None or relation.score > existing.channel_scores["graph"]:
                candidates[chunk_id] = candidate
    return sorted(
        candidates.values(),
        key=lambda item: (-item.channel_scores["graph"], item.id),
    )[:limit]
