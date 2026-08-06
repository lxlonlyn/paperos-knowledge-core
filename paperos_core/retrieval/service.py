"""Query orchestration: raw query -> profile mapping -> Cognee search/recall."""

from __future__ import annotations

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.adapters.cognee.search import CogneeSearchAdapter
from paperos_core.adapters.llm import LLMClient
from paperos_core.config import RuntimeSettings
from paperos_core.feedback.service import FeedbackService
from paperos_core.indexes.manager import IndexManager
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.paths import DataPaths
from paperos_core.retrieval.cache import QueryCache
from paperos_core.retrieval.candidates import (
    Candidate,
    ExpansionTrace,
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
from paperos_core.retrieval.planner import QueryPlanner
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
        self.planner = QueryPlanner(config)

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
        plan = self.planner.plan(request)
        pool = plan.candidate_pool_size
        query = request.query
        stages = ["profile_mapping"]
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
                search_type=plan.search_type,
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
                search_type=plan.search_type,
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
                search_type=plan.search_type,
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
                search_type=plan.search_type,
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
        selected = diversify(
            reranked,
            limit=min(plan.top_k, 6) if truth_profile else plan.top_k,
            max_per_document=(
                plan.top_k
                if truth_profile
                else self.config.retrieval.max_chunks_per_document
            ),
            max_per_section=self.config.retrieval.max_chunks_per_section,
            seed_each_document=not truth_profile,
            aspect_queries=[query],
        )
        stages.append("diversification")
        evidence = format_evidence(selected, corpus.bundles)
        recall_context: list[str] | None = None
        if request.profile is RetrievalProfile.COMPREHENSIVE:
            recall_context = await self.search.recall_context(
                query,
                dataset=dataset_name,
                top_k=pool,
            )
            stages.append("cognee_recall")
        answer = await synthesize_answer(
            self.llm,
            query=request.query,
            profile=request.profile,
            evidence=evidence,
            recall_context=recall_context,
        )
        stages.append("synthesis")
        expansion = ExpansionTrace(
            model="",
            lexical_queries=[],
            semantic_queries=[],
            entity_queries=[],
            relation_queries=[],
            hyde_text="",
            raw_output="",
        )
        response = QueryResponse(
            id=cache_key,
            query=request.query,
            profile=request.profile,
            dataset=dataset_name,
            answer=answer,
            answer_model=self.llm.config.model,
            stages=stages,
            channels_used=list(channels),
            expansion=expansion,
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
