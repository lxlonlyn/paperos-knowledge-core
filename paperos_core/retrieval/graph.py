"""Typed graph retrieval through the Cognee 1.4.0 compatibility boundary."""

from __future__ import annotations

from paperos_core.adapters.cognee.compat import (
    CogneeCompatibilityAdapter,
    CogneeVectorHit,
)
from paperos_core.adapters.cognee.search import CogneeSearchAdapter
from paperos_core.domain.provenance import RelationType
from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView

_SEED_TYPES = {
    "EntityDataPoint",
    "ClaimDataPoint",
    "SummaryDataPoint",
    "TripletDataPoint",
    "ConceptRelationDataPoint",
}


async def graph_retrieve(
    search: CogneeSearchAdapter,
    compat: CogneeCompatibilityAdapter,
    corpus: CorpusView,
    query: str,
    *,
    dataset_name: str,
    search_type: str,
    limit: int,
    depth: int,
    document_ids: set[str],
) -> list[Candidate]:
    from paperos_core.retrieval.ablation import (
        current_ablation_policy,
        current_ablation_trace,
        is_claim_object_type,
        record_claim_leak,
    )

    policy = current_ablation_policy()
    trace = current_ablation_trace()
    seed_types = set(_SEED_TYPES)
    if policy is not None and (
        not policy.claim_seeds_allowed_in_graph or policy.claim_blind
    ):
        seed_types.discard("ClaimDataPoint")
    edge_types = {relation.value for relation in RelationType}
    if policy is not None and (
        not policy.about_edges_visible or policy.claim_blind
    ):
        edge_types.discard(RelationType.ABOUT.value)
    exclude_node_types = (
        {"ClaimDataPoint"} if policy is not None and policy.claim_blind else None
    )
    hits = await search.graph_search(
        query,
        dataset=dataset_name,
        top_k=limit * 2,
        search_type=search_type,
    )
    if trace is not None:
        trace.raw_hits["graph"] = [
            {
                "object_id": hit.canonical_id,
                "object_type": hit.object_type,
                "score": hit.score,
            }
            for hit in hits
        ]
    for hit in hits:
        if (
            policy is not None
            and policy.claim_blind
            and is_claim_object_type(hit.object_type)
        ):
            record_claim_leak(
                stage="graph_raw_hits_filtered",
                object_id=hit.canonical_id,
                object_type=hit.object_type,
            )
    resolved = await compat.resolve_graph_nodes(
        [hit.cognee_id for hit in hits if hit.object_type in seed_types]
    )
    seeds = [
        CogneeVectorHit(
            cognee_id=hit.cognee_id,
            canonical_id=hit.canonical_id,
            object_type=hit.object_type,
            text=hit.text,
            score=hit.score,
            source_chunk_ids=tuple(
                _string_list(
                    resolved.get(hit.cognee_id, {}).get("source_chunk_ids")
                )
            ),
            derived_from_ids=tuple(
                _string_list(
                    resolved.get(hit.cognee_id, {}).get("derived_from_ids")
                )
            ),
            canonical_snapshot_id=None,
        )
        for hit in hits
        if hit.object_type in seed_types
    ]
    seeds = [seed for seed in seeds if seed.source_chunk_ids]
    allowed_seeds = [
        seed
        for seed in seeds
        if any(
            chunk_id in corpus.chunks
            and corpus.chunks[chunk_id].document_id in document_ids
            for chunk_id in seed.source_chunk_ids
        )
    ]
    if trace is not None:
        trace.graph_seeds = [
            {
                "object_id": seed.canonical_id,
                "object_type": seed.object_type,
                "score": seed.score,
            }
            for seed in allowed_seeds
        ]
    traversed = await compat.typed_traverse(
        allowed_seeds,
        depth=depth,
        edge_types=edge_types,
        exclude_node_types=exclude_node_types,
    )
    if trace is not None:
        trace.graph_traversal = [
            {
                "source_canonical_id": relation.source_canonical_id,
                "target_canonical_id": relation.target_canonical_id,
                "relation_type": relation.relation_type,
                "score": relation.score,
            }
            for relation in traversed
        ]
        for relation in traversed:
            if relation.relation_type == RelationType.ABOUT.value:
                record_claim_leak(
                    stage="graph_traversal",
                    object_id=relation.source_canonical_id,
                    relation_type=relation.relation_type,
                )
    candidates: dict[str, Candidate] = {}
    for seed in allowed_seeds:
        if policy is not None and policy.claim_blind and is_claim_object_type(
            seed.object_type
        ):
            record_claim_leak(
                stage="graph_seed",
                object_id=seed.canonical_id,
                object_type=seed.object_type,
            )
            continue
        for chunk_id in seed.source_chunk_ids:
            chunk = corpus.chunks.get(chunk_id)
            if chunk is None or chunk.document_id not in document_ids:
                continue
            candidate = corpus.candidate_for_chunk(
                chunk_id,
                channel="graph",
                score=seed.score,
                object_id=seed.canonical_id,
                object_type=seed.object_type.removesuffix("DataPoint").casefold(),
                knowledge_kind="system_inference",
                derived_from_ids=[seed.canonical_id, *seed.derived_from_ids],
            )
            existing = candidates.get(chunk_id)
            if existing is None or seed.score > existing.channel_scores["graph"]:
                candidates[chunk_id] = candidate
    for relation in traversed:
        if policy is not None and policy.claim_blind:
            if relation.relation_type == RelationType.ABOUT.value:
                continue
        for chunk_id in relation.source_chunk_ids:
            chunk = corpus.chunks.get(chunk_id)
            if chunk is None or chunk.document_id not in document_ids:
                continue
            candidate = corpus.candidate_for_chunk(
                chunk_id,
                channel="graph",
                score=relation.score,
                object_id=relation.source_canonical_id,
                object_type=f"graph_relation:{relation.relation_type}",
                knowledge_kind="structured_relation",
                derived_from_ids=[
                    relation.source_canonical_id,
                    relation.target_canonical_id,
                    *relation.derived_from_ids,
                ],
            )
            existing = candidates.get(chunk_id)
            if existing is None or relation.score > existing.channel_scores["graph"]:
                candidates[chunk_id] = candidate
    return sorted(
        candidates.values(),
        key=lambda item: (-item.channel_scores["graph"], item.id),
    )[:limit]


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []
