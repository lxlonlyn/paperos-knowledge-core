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
    search_type = {
        RetrievalProfile.TRUTH: "GRAPH_COMPLETION",
        RetrievalProfile.ASSOCIATIVE: "GRAPH_COMPLETION_DECOMPOSITION",
        RetrievalProfile.COMPREHENSIVE: "GRAPH_COMPLETION",
    }[request.profile]
    return QueryPlan(
        profile=request.profile,
        channels=channels,
        search_type=search_type,
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
        },
    )
