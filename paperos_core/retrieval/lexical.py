"""SQLite FTS5 lexical retrieval over canonical records."""

from __future__ import annotations

import re

from paperos_core.errors import IndexStorageError
from paperos_core.indexes.lexical_store import LexicalStore
from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView

_TERM = re.compile(r"[\w-]{2,}", re.UNICODE)


def lexical_retrieve(
    store: LexicalStore,
    corpus: CorpusView,
    queries: list[str],
    *,
    limit: int,
    document_ids: set[str],
) -> list[Candidate]:
    results: dict[str, Candidate] = {}
    for query in queries[:8]:
        for query_rank, fts_query in enumerate(_fts_queries(query)):
            try:
                rows = store.search(
                    fts_query,
                    active_snapshot_ids=corpus.active_snapshot_ids,
                    limit=limit * 2,
                )
            except IndexStorageError:
                continue
            for result_rank, row in enumerate(rows, 1):
                object_id = str(row["object_id"])
                if object_id not in corpus.chunks:
                    continue
                chunk = corpus.chunks[object_id]
                if chunk.document_id not in document_ids:
                    continue
                # SQLite FTS5 BM25 is ordered ascending (usually negative),
                # so applying abs() and then an inverse reverses relevance.
                # Preserve the engine's real rank and keep the full combined
                # query ahead of its individual-term fallback queries.
                score = 1.0 / (1 + query_rank * limit * 2 + result_rank)
                existing = results.get(object_id)
                if existing is None or score > existing.channel_scores["lexical"]:
                    results[object_id] = corpus.candidate_for_chunk(
                        object_id, channel="lexical", score=score
                    )
    return sorted(
        results.values(),
        key=lambda item: (-item.channel_scores["lexical"], item.id),
    )[:limit]


def _fts_queries(query: str) -> list[str]:
    terms = list(dict.fromkeys(_TERM.findall(query)))
    ascii_terms = [term for term in terms if any(char.isascii() for char in term)]
    selected = ascii_terms or terms
    escaped = [term.replace('"', "") for term in selected[:20]]
    if not escaped:
        return []
    combined = " OR ".join(f'"{term}"' for term in escaped)
    individual = [f'"{term}"' for term in escaped if len(term) >= 4]
    return list(dict.fromkeys([combined, *individual]))
