"""Document summary retrieval with source-chunk backtracking."""

from paperos_core.adapters.cognee.repository import (
    SUMMARY_VECTOR_COLLECTIONS,
    CogneeRepository,
)
from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView


async def global_context_retrieve(
    repository: CogneeRepository,
    corpus: CorpusView,
    queries: list[str],
    *,
    limit: int,
    document_ids: set[str],
) -> list[Candidate]:
    hits = await repository.search_vectors(
        queries[:4],
        collections=SUMMARY_VECTOR_COLLECTIONS,
        limit=limit,
    )
    candidates: list[Candidate] = []
    for hit in hits:
        for chunk_id in hit.source_chunk_ids[:2]:
            chunk = corpus.chunks.get(chunk_id)
            if chunk is None or chunk.document_id not in document_ids:
                continue
            candidates.append(
                corpus.candidate_for_chunk(
                    chunk_id,
                    channel="global_context",
                    score=hit.score,
                    object_id=hit.canonical_id,
                    object_type="summary",
                    knowledge_kind="system_inference",
                    derived_from_ids=[hit.canonical_id, *hit.derived_from_ids],
                )
            )
    return candidates[:limit]
