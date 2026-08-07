"""Query orchestration: raw query -> profile mapping -> Cognee search/recall."""

from __future__ import annotations

import math

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.adapters.cognee.llm import LLMClient
from paperos_core.adapters.cognee.search import CogneeSearchAdapter
from paperos_core.config import RuntimeSettings
from paperos_core.feedback.service import FeedbackService
from paperos_core.indexes.manager import IndexManager
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.paths import DataPaths
from paperos_core.retrieval.cache import QueryCache
from paperos_core.retrieval.candidates import (
    Candidate,
    QueryRequest,
    QueryResponse,
    RetrievalProfile,
)
from paperos_core.retrieval.confirmed import confirmed_knowledge_retrieve
from paperos_core.retrieval.corpus import CorpusView
from paperos_core.retrieval.diversify import diversify
from paperos_core.retrieval.evidence import format_evidence
from paperos_core.retrieval.fusion import weighted_rrf
from paperos_core.retrieval.global_context import global_context_retrieve
from paperos_core.retrieval.graph import graph_retrieve
from paperos_core.retrieval.lexical import lexical_retrieve
from paperos_core.retrieval.profiles import build_query_plan
from paperos_core.retrieval.rerank import rerank_candidates
from paperos_core.retrieval.semantic import (
    entity_claim_retrieve,
    semantic_retrieve,
)
from paperos_core.retrieval.synthesis import synthesize_answer
from paperos_core.runtime.local_inference.client import LocalInferenceClient


class RetrievalService:
    """Run profile mapping through answer synthesis without mutating evidence."""

    def __init__(
        self,
        config: RuntimeSettings,
        paths: DataPaths,
        canonical_repository: CanonicalRepository,
        registry: SourceRegistry,
        search: CogneeSearchAdapter,
        compat: CogneeCompatibilityAdapter,
        index_manager: IndexManager,
        model_client: LocalInferenceClient,
        llm: LLMClient,
        feedback: FeedbackService,
    ) -> None:
        self.config = config
        self.paths = paths
        self.canonical_repository = canonical_repository
        self.registry = registry
        self.search = search
        self.compat = compat
        self.index_manager = index_manager
        self.model_client = model_client
        self.llm = llm
        self.feedback = feedback
        self.cache = QueryCache(paths, feedback)

    async def query(self, request: QueryRequest) -> QueryResponse:
        dataset_name = (request.dataset or self.config.dataset).strip()
        request = request.model_copy(update={"dataset": dataset_name})
        corpus = CorpusView.load(
            self.paths, self.canonical_repository, self.registry
        )
        cache_key = self.cache.key(request, corpus)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        document_ids = corpus.filtered_document_ids(
            request.document_ids, dataset_name
        )
        plan = build_query_plan(request, self.config)
        pool = plan.candidate_pool_size
        query = request.query
        stages = ["profile_mapping"]
        explicit_document_ids = corpus.explicitly_mentioned_document_ids(query)
        comparative_query = (
            request.profile is not RetrievalProfile.TRUTH
            and _has_multi_document_cue(query)
        )
        apply_explicit_scope = bool(explicit_document_ids) and not (
            len(explicit_document_ids) == 1 and comparative_query
        )
        if request.document_ids is None and apply_explicit_scope:
            document_ids.intersection_update(explicit_document_ids)
            stages.append("explicit_document_scope")
        channels: dict[str, list[Candidate]] = {}

        if "lexical" in plan.channels:
            channels["lexical"] = lexical_retrieve(
                self.index_manager.lexical,
                corpus,
                [query],
                limit=pool,
                document_ids=document_ids,
            )
            stages.append("lexical_retrieval")
        if "semantic" in plan.channels:
            channels["semantic"] = await semantic_retrieve(
                self.search,
                self.compat,
                corpus,
                query,
                dataset_name=dataset_name,
                search_type=plan.search_types["semantic"],
                limit=pool,
                document_ids=document_ids,
                chunk_only=request.profile is RetrievalProfile.TRUTH,
            )
            stages.append("cognee_search")
        if "entity_claim" in plan.channels:
            channels["entity_claim"] = await entity_claim_retrieve(
                self.search,
                self.compat,
                corpus,
                query,
                dataset_name=dataset_name,
                search_type=plan.search_types["entity_claim"],
                limit=pool,
                document_ids=document_ids,
            )
            stages.append("entity_claim_search")
        if "graph" in plan.channels:
            channels["graph"] = await graph_retrieve(
                self.search,
                self.compat,
                corpus,
                query,
                dataset_name=dataset_name,
                search_type=plan.search_types["graph"],
                limit=pool,
                depth=plan.graph_depth,
                document_ids=document_ids,
            )
            stages.append("typed_traversal")
        if "global_context" in plan.channels:
            channels["global_context"] = await global_context_retrieve(
                self.search,
                self.compat,
                corpus,
                query,
                dataset_name=dataset_name,
                search_type=plan.search_types["global_context"],
                limit=pool,
                document_ids=document_ids,
            )
            stages.append("global_context")
        if "confirmed_knowledge" in plan.channels:
            channels["confirmed_knowledge"] = confirmed_knowledge_retrieve(
                self.feedback,
                corpus,
                [query],
                limit=pool,
                document_ids=document_ids,
            )
            stages.append("confirmed_knowledge_retrieval")

        fused = weighted_rrf(channels, plan.weights)[:pool]
        stages.append("fusion")
        if self.config.retrieval.rerank_enabled:
            reranked = await rerank_candidates(
                self.model_client,
                query,
                fused,
                limit=pool,
            )
            stages.append("rerank")
        else:
            reranked = fused
        truth_profile = request.profile is RetrievalProfile.TRUTH
        if truth_profile:
            max_per_document = plan.top_k
        elif apply_explicit_scope:
            max_per_document = math.ceil(
                plan.top_k / max(len(explicit_document_ids), 1)
            )
        else:
            max_per_document = self.config.retrieval.max_chunks_per_document
        selected = diversify(
            reranked,
            limit=plan.top_k,
            max_per_document=max_per_document,
            max_per_section=self.config.retrieval.max_chunks_per_section,
            seed_each_document=not truth_profile,
            aspect_queries=[query],
        )
        stages.append("diversification")
        evidence = format_evidence(selected, corpus.bundles)
        recall_context: list[str] | None = None
        if request.profile is RetrievalProfile.COMPREHENSIVE:
            recall_hits = await self.search.recall_context(
                query,
                dataset=dataset_name,
                top_k=pool,
                search_type=plan.search_types["recall"],
            )
            recall_context = [hit.text for hit in recall_hits]
            stages.append("cognee_recall")
        answer = await synthesize_answer(
            self.llm,
            query=request.query,
            profile=request.profile,
            evidence=evidence,
            recall_context=recall_context,
        )
        stages.append("synthesis")
        response = QueryResponse(
            id=cache_key,
            query=request.query,
            profile=request.profile,
            dataset=dataset_name,
            answer=answer,
            answer_model=self.llm.model,
            stages=stages,
            channels_used=list(channels),
            evidence=evidence,
            candidates=selected,
            distinct_documents=len({item.document_id for item in evidence}),
            provenance_complete=all(
                item.chunk_id in corpus.chunks
                and item.source_file_id
                and item.document_id
                and item.evidence_id
                for item in evidence
            ),
        )
        self.cache.put(response)
        return response


def _has_multi_document_cue(query: str) -> bool:
    normalized = query.casefold()
    return any(
        cue in normalized
        for cue in (
            "分别",
            "比较",
            "连续谱",
            "这四篇",
            "三篇",
            "compare",
            "versus",
            " vs ",
        )
    )
