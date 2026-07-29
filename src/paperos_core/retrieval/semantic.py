"""Local-embedding chunk plus entity/claim retrieval."""

from __future__ import annotations

import re

from paperos_core.domain.knowledge import Claim, Entity
from paperos_core.indexes.vector_store import VectorStore
from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView

_TOKEN = re.compile(r"[\w-]{2,}", re.UNICODE)


async def semantic_retrieve(
    store: VectorStore,
    corpus: CorpusView,
    queries: list[str],
    *,
    limit: int,
    document_ids: set[str],
) -> list[Candidate]:
    results: dict[str, Candidate] = {}
    for query in queries[:4]:
        for row in await store.search(query, limit=limit * 2):
            chunk_id = str(row["object_id"])
            if corpus.chunks[chunk_id].document_id not in document_ids:
                continue
            raw_score = row["score"]
            score = (
                float(raw_score) if isinstance(raw_score, (str, int, float)) else 0.0
            )
            existing = results.get(chunk_id)
            if existing is None or score > existing.channel_scores["semantic"]:
                results[chunk_id] = corpus.candidate_for_chunk(
                    chunk_id, channel="semantic", score=score
                )
    return sorted(
        results.values(),
        key=lambda item: (-item.channel_scores["semantic"], item.id),
    )[:limit]


def entity_claim_retrieve(
    corpus: CorpusView,
    queries: list[str],
    *,
    limit: int,
    document_ids: set[str],
) -> list[Candidate]:
    query_tokens = {token.casefold() for query in queries for token in _TOKEN.findall(query)}
    candidates: dict[str, Candidate] = {}
    for document_id, enrichment in corpus.enrichments.items():
        if document_id not in document_ids:
            continue
        semantic_items: list[Entity | Claim] = [
            *enrichment.entities,
            *enrichment.claims,
        ]
        for item in semantic_items:
            searchable = (
                f"{getattr(item, 'name', '')} {getattr(item, 'text', '')} "
                f"{getattr(item, 'description', '')}"
            ).casefold()
            overlap = sum(token in searchable for token in query_tokens)
            score = float(overlap) + (item.confidence or 0.5)
            for chunk_id in item.source_chunk_ids:
                candidate = corpus.candidate_for_chunk(
                    chunk_id,
                    channel="entity_claim",
                    score=score,
                    object_id=item.id,
                    object_type="entity" if isinstance(item, Entity) else "claim",
                    knowledge_kind="system_inference",
                    derived_from_ids=[item.id, *item.derived_from_ids],
                )
                existing = candidates.get(chunk_id)
                if (
                    existing is None
                    or score > existing.channel_scores["entity_claim"]
                ):
                    candidates[chunk_id] = candidate
    return sorted(
        candidates.values(),
        key=lambda item: (-item.channel_scores["entity_claim"], item.id),
    )[:limit]
