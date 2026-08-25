"""Chunk-ID deduplication and reciprocal-rank fusion."""

from __future__ import annotations

from paperos_core.retrieval.candidates import Candidate


def deduplicate_candidates_by_chunk(
    candidates: list[Candidate],
) -> list[Candidate]:
    """Merge all provenance for one canonical chunk into one candidate."""
    merged: dict[str, Candidate] = {}
    order: list[str] = []
    for incoming in candidates:
        existing = merged.get(incoming.chunk_id)
        if existing is None:
            existing = incoming.model_copy(deep=True)
            existing.id = incoming.chunk_id
            merged[incoming.chunk_id] = existing
            order.append(incoming.chunk_id)
            continue
        _merge(existing, incoming)
    return [merged[chunk_id] for chunk_id in order]


def weighted_rrf(
    channels: dict[str, list[Candidate]],
    weights: dict[str, float] | None = None,
    *,
    rank_constant: int = 60,
) -> list[Candidate]:
    """Fuse lexical and vector ranks, keyed solely by canonical chunk_id."""
    merged: dict[str, Candidate] = {}
    weights = weights or {}
    for channel, candidates in channels.items():
        for rank, incoming in enumerate(candidates, start=1):
            candidate = merged.get(incoming.chunk_id)
            if candidate is None:
                candidate = incoming.model_copy(deep=True)
                candidate.id = incoming.chunk_id
                candidate.channels = []
                candidate.channel_ranks = {}
                candidate.channel_scores = {}
                candidate.fused_score = 0.0
                merged[incoming.chunk_id] = candidate
            _merge(candidate, incoming)
            candidate.channel_ranks[channel] = rank
            candidate.channel_scores[channel] = incoming.channel_scores.get(channel, 0.0)
            candidate.fused_score += weights.get(channel, 1.0) / (rank_constant + rank)
    return sorted(
        merged.values(),
        key=lambda item: (-item.fused_score, item.chunk_id),
    )


def _merge(target: Candidate, incoming: Candidate) -> None:
    target.channels = list(dict.fromkeys([*target.channels, *incoming.channels]))
    target.channel_ranks.update(incoming.channel_ranks)
    for channel, score in incoming.channel_scores.items():
        target.channel_scores[channel] = max(
            target.channel_scores.get(channel, float("-inf")), score
        )
    target.derived_from_ids = list(
        dict.fromkeys([*target.derived_from_ids, *incoming.derived_from_ids])
    )
    target.relation_types = list(dict.fromkeys([*target.relation_types, *incoming.relation_types]))
    if incoming.source_work_id and not target.source_work_id:
        target.source_work_id = incoming.source_work_id
    if incoming.knowledge_kind != "source_fact":
        target.knowledge_kind = incoming.knowledge_kind
