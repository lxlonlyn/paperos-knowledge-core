"""Documented retrieval profile planning."""

from __future__ import annotations

from paperos_core.config import RuntimeSettings
from paperos_core.retrieval.candidates import QueryPlan, QueryRequest, RetrievalProfile


def build_query_plan(request: QueryRequest, config: RuntimeSettings) -> QueryPlan:
    profile_config = getattr(config.retrieval.profiles, request.profile.value)
    channels = {
        RetrievalProfile.TRUTH: [
            "lexical",
            "semantic",
            "confirmed_knowledge",
        ],
        RetrievalProfile.ASSOCIATIVE: [
            "semantic",
            "entity_claim",
            "graph",
            "global_context",
            "confirmed_knowledge",
        ],
        RetrievalProfile.COMPREHENSIVE: [
            "lexical",
            "semantic",
            "entity_claim",
            "graph",
            "global_context",
            "confirmed_knowledge",
        ],
    }[request.profile]
    search_types = {
        RetrievalProfile.TRUTH: {
            "semantic": "PAPEROS_CHUNKS",
        },
        RetrievalProfile.ASSOCIATIVE: {
            "semantic": "PAPEROS_ASSOCIATIVE_SEEDS",
            "entity_claim": "PAPEROS_ENTITY_CLAIM",
            "graph": "PAPEROS_GRAPH_SEEDS",
            "global_context": "PAPEROS_SUMMARIES",
        },
        RetrievalProfile.COMPREHENSIVE: {
            "semantic": "PAPEROS_CHUNKS",
            "entity_claim": "PAPEROS_ENTITY_CLAIM",
            "graph": "PAPEROS_GRAPH_SEEDS",
            "global_context": "PAPEROS_SUMMARIES",
            "recall": "PAPEROS_GRAPH_SEEDS",
        },
    }[request.profile]
    return QueryPlan(
        profile=request.profile,
        channels=channels,
        search_types=search_types,
        top_k=request.top_k or config.retrieval.top_k,
        candidate_pool_size=config.retrieval.candidate_pool_size,
        graph_depth=config.retrieval.graph_depth,
        weights={
            "lexical": profile_config.lexical_weight,
            "semantic": profile_config.semantic_weight,
            "entity_claim": profile_config.semantic_weight,
            "graph": profile_config.graph_weight,
            "global_context": profile_config.global_context_weight,
            "confirmed_knowledge": profile_config.confirmed_knowledge_weight,
            "subject_claim": profile_config.graph_weight,
        },
    )
