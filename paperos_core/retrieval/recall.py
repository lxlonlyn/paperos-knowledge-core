"""Cognee recall context channel with deterministic chunk matching."""

from __future__ import annotations

from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView

_PREFIX_CHARS = 200


def recall_retrieve(
    contexts: list[str],
    corpus: CorpusView,
    query: str,
    *,
    limit: int,
    document_ids: set[str],
) -> list[Candidate]:
    """Backtrack recall context passages to canonical chunks by text prefix."""
    candidates: dict[str, Candidate] = {}
    for chunk_id, chunk in corpus.chunks.items():
        if chunk.document_id not in document_ids:
            continue
        prefix = chunk.text[:_PREFIX_CHARS].strip()
        if not prefix:
            continue
        score = sum(1.0 for context in contexts if prefix in context)
        if score <= 0:
            continue
        candidates[chunk_id] = corpus.candidate_for_chunk(
            chunk_id,
            channel="recall",
            score=score,
            object_id=chunk_id,
            object_type="chunk",
            knowledge_kind="source_fact",
        )
    return sorted(
        candidates.values(),
        key=lambda item: (-item.channel_scores["recall"], item.id),
    )[:limit]
