"""Explicit post-hit expansion from canonical Chunk seeds."""

from __future__ import annotations

from typing import TYPE_CHECKING

from paperos_core.domain.provenance import SEMANTIC_RELATION_TYPES
from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView
from paperos_core.retrieval.fusion import deduplicate_candidates_by_chunk

if TYPE_CHECKING:
    from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter


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


async def semantic_post_hit_expand(
    compat: CogneeCompatibilityAdapter,
    corpus: CorpusView,
    seeds: list[Candidate],
    *,
    dataset_name: str,
    document_ids: set[str],
    limit: int,
) -> list[Candidate]:
    """Expand only through one direct relation on seed-grounded semantic objects."""
    relations = await compat.semantic_relations_for_chunks(
        [seed.chunk_id for seed in seeds],
        dataset_name=dataset_name,
        relation_types={item.value for item in SEMANTIC_RELATION_TYPES},
        limit=limit,
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
                    channel="semantic_expansion",
                    score=relation.score,
                    object_id=relation.source_canonical_id,
                    object_type=f"semantic_relation:{relation.relation_type}",
                    knowledge_kind="structured_relation",
                    derived_from_ids=list(relation.derived_from_ids),
                    relation_types=[relation.relation_type],
                )
            )
    return deduplicate_candidates_by_chunk(candidates)[:limit]


def _seed_score(candidate: Candidate) -> float:
    return candidate.rerank_score or candidate.fused_score or 1.0
