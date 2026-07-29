"""User-confirmed derived knowledge with canonical evidence backtracking."""

from __future__ import annotations

import re

from paperos_core.feedback.service import FeedbackService
from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView

_TOKEN = re.compile(r"[\w-]{2,}", re.UNICODE)


def confirmed_knowledge_retrieve(
    service: FeedbackService,
    corpus: CorpusView,
    queries: list[str],
    *,
    limit: int,
    document_ids: set[str],
) -> list[Candidate]:
    query_tokens = {
        token.casefold() for query in queries for token in _TOKEN.findall(query)
    }
    results: dict[str, Candidate] = {}
    for improvement in service.confirmed_improvements():
        text = improvement.text or ""
        score = 1.0 + sum(
            token in text.casefold() for token in query_tokens
        )
        for chunk_id in improvement.source_chunk_ids:
            chunk = corpus.chunks.get(chunk_id)
            if chunk is None or chunk.document_id not in document_ids:
                continue
            candidate = corpus.candidate_for_chunk(
                chunk_id,
                channel="confirmed_knowledge",
                score=float(score),
                object_id=improvement.id,
                object_type="improvement",
                knowledge_kind="user_confirmed",
                derived_from_ids=[
                    improvement.id,
                    improvement.feedback_id,
                    *improvement.derived_from_ids,
                ],
            )
            if text.strip():
                candidate.text = text
            current = results.get(chunk_id)
            if (
                current is None
                or score > current.channel_scores["confirmed_knowledge"]
            ):
                results[chunk_id] = candidate
    return sorted(
        results.values(),
        key=lambda item: (-item.channel_scores["confirmed_knowledge"], item.id),
    )[:limit]
