"""Explicit post-hit expansion from canonical Chunk seeds."""

from __future__ import annotations

from paperos_core.adapters.cognee.compat import (
    CogneeCompatibilityAdapter,
    CogneeVectorHit,
    cognee_uuid,
)
from paperos_core.domain.provenance import RelationType
from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView
from paperos_core.retrieval.fusion import deduplicate_candidates_by_chunk

_GRAPH_RELATIONS = {
    RelationType.CITES.value,
    RelationType.USES.value,
    RelationType.EXTENDS.value,
    RelationType.COMPARES_WITH.value,
    RelationType.EVALUATES_ON.value,
    RelationType.SUPPORTS.value,
    RelationType.CONTRADICTS.value,
    RelationType.PROPOSES.value,
    RelationType.RELATED_TO.value,
}


def local_neighbor_expand(
    corpus: CorpusView,
    seeds: list[Candidate],
    *,
    document_ids: set[str],
) -> list[Candidate]:
    """Return ±1 chunks without crossing document, region, or major section."""
    expanded: list[Candidate] = []
    for seed in seeds:
        anchor = corpus.chunks[seed.chunk_id]
        for neighbor_id in (anchor.previous_chunk_id, anchor.next_chunk_id):
            if neighbor_id is None or neighbor_id not in corpus.chunks:
                continue
            neighbor = corpus.chunks[neighbor_id]
            if (
                neighbor.document_id != anchor.document_id
                or neighbor.document_id not in document_ids
                or neighbor.document_region != anchor.document_region
                or neighbor.major_section_id != anchor.major_section_id
            ):
                continue
            expanded.append(
                corpus.candidate_for_chunk(
                    neighbor.id,
                    channel="local_expansion",
                    score=_seed_score(seed) * 0.95,
                    object_id=anchor.id,
                    object_type="local_neighbor",
                    derived_from_ids=[anchor.id],
                )
            )
    return deduplicate_candidates_by_chunk(expanded)


async def citation_post_hit_expand(
    compat: CogneeCompatibilityAdapter,
    corpus: CorpusView,
    seeds: list[Candidate],
    *,
    dataset_name: str,
    document_ids: set[str],
    limit: int,
) -> list[Candidate]:
    """Expand Chunk→cited Work←CITES→source Chunk using edge provenance."""
    target_work_ids = list(
        dict.fromkeys(
            work_id
            for seed in seeds
            for work_id in sorted(
                corpus.cited_work_ids_by_chunk.get(seed.chunk_id, set())
            )
        )
    )
    relations = await compat.incoming_typed_relations(
        target_work_ids,
        dataset_name=dataset_name,
        relation_type=RelationType.CITES.value,
        depth=1,
        limit=max(limit * 4, limit),
    )
    candidates: list[Candidate] = []
    for relation in relations:
        for chunk_id in relation.source_chunk_ids:
            chunk = corpus.chunks.get(chunk_id)
            if chunk is None or chunk.document_id not in document_ids:
                continue
            candidates.append(
                corpus.candidate_for_chunk(
                    chunk_id,
                    channel="citation_expansion",
                    score=relation.score,
                    object_id=relation.source_canonical_id,
                    object_type="graph_relation:CITES",
                    knowledge_kind="structured_relation",
                    derived_from_ids=list(relation.derived_from_ids),
                    relation_types=[RelationType.CITES.value],
                    source_work_id=relation.source_work_id,
                    subject_work_ids=[relation.target_canonical_id],
                )
            )
    return deduplicate_candidates_by_chunk(candidates)[:limit]


async def graph_post_hit_expand(
    compat: CogneeCompatibilityAdapter,
    corpus: CorpusView,
    seeds: list[Candidate],
    *,
    depth: int,
    document_ids: set[str],
    limit: int,
    claim_enrichment_enabled: bool,
) -> list[Candidate]:
    """Run bounded Chunk→Graph→Chunk traversal; never search graph by query."""
    graph_seeds = [
        CogneeVectorHit(
            cognee_id=str(cognee_uuid(seed.chunk_id)),
            canonical_id=seed.chunk_id,
            object_type="ChunkDataPoint",
            text=corpus.chunks[seed.chunk_id].text,
            score=_seed_score(seed),
            source_chunk_ids=(seed.chunk_id,),
            derived_from_ids=(seed.chunk_id,),
            canonical_snapshot_id=seed.canonical_snapshot_id,
        )
        for seed in seeds
    ]
    edge_types = set(_GRAPH_RELATIONS)
    if claim_enrichment_enabled:
        edge_types.add(RelationType.ABOUT.value)
    relations = await compat.typed_traverse(
        graph_seeds,
        depth=depth,
        edge_types=edge_types,
        exclude_node_types=(None if claim_enrichment_enabled else {"ClaimDataPoint"}),
    )
    candidates: list[Candidate] = []
    for relation in relations:
        for chunk_id in relation.source_chunk_ids:
            chunk = corpus.chunks.get(chunk_id)
            if chunk is None or chunk.document_id not in document_ids:
                continue
            candidates.append(
                corpus.candidate_for_chunk(
                    chunk_id,
                    channel="graph_expansion",
                    score=relation.score,
                    object_id=relation.source_canonical_id,
                    object_type=f"graph_relation:{relation.relation_type}",
                    knowledge_kind="structured_relation",
                    derived_from_ids=list(relation.derived_from_ids),
                    relation_types=[relation.relation_type],
                )
            )
    return deduplicate_candidates_by_chunk(candidates)[:limit]


def _seed_score(candidate: Candidate) -> float:
    return candidate.rerank_score or candidate.fused_score or 1.0
