"""Stable-ID deduplication and weighted reciprocal-rank fusion."""

from __future__ import annotations

from paperos_core.retrieval.candidates import Candidate


def weighted_rrf(
    channels: dict[str, list[Candidate]],
    weights: dict[str, float],
    *,
    rank_constant: int = 60,
) -> list[Candidate]:
    merged: dict[str, Candidate] = {}
    for channel, candidates in channels.items():
        for rank, incoming in enumerate(candidates, start=1):
            if incoming.id not in merged:
                merged[incoming.id] = incoming.model_copy(deep=True)
            candidate = merged[incoming.id]
            if channel not in candidate.channels:
                candidate.channels.append(channel)
            candidate.channel_ranks[channel] = rank
            candidate.channel_scores[channel] = incoming.channel_scores.get(
                channel, 0
            )
            candidate.fused_score += weights.get(channel, 1.0) / (
                rank_constant + rank
            )
            candidate.derived_from_ids = list(
                dict.fromkeys(
                    [*candidate.derived_from_ids, *incoming.derived_from_ids]
                )
            )
            if incoming.knowledge_kind != "source_fact":
                candidate.knowledge_kind = incoming.knowledge_kind
            if incoming.knowledge_kind == "user_confirmed":
                candidate.object_id = incoming.object_id
                candidate.object_type = incoming.object_type
                candidate.text = incoming.text
    return sorted(
        merged.values(),
        key=lambda item: (-item.fused_score, item.id),
    )
