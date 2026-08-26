"""Chunk-only semantic retrieval from the canonical Chunk collection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView

if TYPE_CHECKING:
    from paperos_core.adapters.cognee.search import CogneeSearchAdapter

_CHUNK_TYPE = "ChunkDataPoint"
_CHUNK_SEARCH_TYPE = "PAPEROS_CHUNKS"


async def semantic_retrieve(
    search: CogneeSearchAdapter,
    corpus: CorpusView,
    query: str,
    *,
    dataset_name: str,
    limit: int,
    document_ids: set[str],
) -> list[Candidate]:
    """Return only canonical Chunk candidates; derived nodes are never seeds."""
    hits = await search.graph_search(
        query,
        dataset=dataset_name,
        top_k=limit * 2,
        search_type=_CHUNK_SEARCH_TYPE,
    )
    candidates: dict[str, Candidate] = {}
    for hit in hits:
        if hit.object_type != _CHUNK_TYPE:
            continue
        chunk = corpus.chunks.get(hit.canonical_id)
        if chunk is None or chunk.document_id not in document_ids:
            continue
        candidate = corpus.candidate_for_chunk(
            chunk.id,
            channel="vector",
            score=hit.score,
        )
        existing = candidates.get(chunk.id)
        if existing is None or hit.score > existing.channel_scores["vector"]:
            candidates[chunk.id] = candidate
    return sorted(
        candidates.values(),
        key=lambda item: (-item.channel_scores["vector"], item.chunk_id),
    )[:limit]
