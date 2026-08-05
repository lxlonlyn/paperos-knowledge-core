"""Typed graph traversal seeded by public Cognee search, with chunk backtracking."""

from __future__ import annotations

from paperos_core.adapters.cognee.compat import (
    CogneeCompatibilityAdapter,
    CogneeVectorHit,
)
from paperos_core.adapters.cognee.search import CogneeSearchAdapter
from paperos_core.domain.provenance import RelationType
from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView

_SEED_TYPES = {
    "EntityDataPoint",
    "ClaimDataPoint",
    "SummaryDataPoint",
    "TripletDataPoint",
    "ConceptRelationDataPoint",
}


async def graph_retrieve(
    search: CogneeSearchAdapter,
    compat: CogneeCompatibilityAdapter,
    corpus: CorpusView,
    query: str,
    *,
    dataset_name: str,
    limit: int,
    depth: int,
    document_ids: set[str],
) -> list[Candidate]:
    hits = await search.graph_search(
        query,
        dataset=dataset_name,
        top_k=limit,
    )
    resolved = await compat.resolve_graph_nodes(
        [hit.cognee_id for hit in hits if hit.object_type in _SEED_TYPES]
    )
    seeds = [
        CogneeVectorHit(
            cognee_id=hit.cognee_id,
            canonical_id=hit.canonical_id,
            object_type=hit.object_type,
            score=hit.score,
            source_chunk_ids=tuple(
                _string_list(
                    resolved.get(hit.cognee_id, {}).get("source_chunk_ids")
                )
            ),
            derived_from_ids=tuple(
                _string_list(
                    resolved.get(hit.cognee_id, {}).get("derived_from_ids")
                )
            ),
            canonical_snapshot_id=None,
        )
        for hit in hits
        if hit.object_type in _SEED_TYPES
    ]
    seeds = [seed for seed in seeds if seed.source_chunk_ids]
    allowed_seeds = [
        seed
        for seed in seeds
        if any(
            chunk_id in corpus.chunks
            and corpus.chunks[chunk_id].document_id in document_ids
            for chunk_id in seed.source_chunk_ids
        )
    ]
    traversed = await compat.typed_traverse(
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


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []
