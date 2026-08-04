"""Cumulative Gate 5 query orchestration over real derived stores."""

from __future__ import annotations

from paperos_core.adapters.cognee.repository import CogneeRepository
from paperos_core.adapters.llm import DeepSeekClient
from paperos_core.adapters.models.client import (
    LocalModelGatewayClient,
    LocalModelGatewayProcess,
)
from paperos_core.config import PaperOSConfig
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


class RetrievalService:
    """Run planning through answer synthesis without mutating source evidence."""

    def __init__(
        self,
        config: PaperOSConfig,
        paths: DataPaths,
        canonical_repository: CanonicalRepository,
        registry: SourceRegistry,
        cognee_repository: CogneeRepository,
        index_manager: IndexManager,
        model_client: LocalModelGatewayClient,
        model_process: LocalModelGatewayProcess,
        deepseek: DeepSeekClient,
        feedback: FeedbackService,
    ) -> None:
        self.config = config
        self.paths = paths
        self.canonical_repository = canonical_repository
        self.registry = registry
        self.cognee_repository = cognee_repository
        self.index_manager = index_manager
        self.model_client = model_client
        self.model_process = model_process
        self.deepseek = deepseek
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
        await self.model_process.start()
        document_ids = corpus.filtered_document_ids(
            request.document_ids, dataset_name
        )
        plan = self.planner.plan(request)
        expansion_result = await self.model_client.expand_query(
            request.query, profile=request.profile.value
        )
        llm_plan, planner_raw = await self.deepseek.plan_query(
            query=request.query, profile=request.profile.value
        )
        expansion = ExpansionTrace(
            model=expansion_result.model,
            lexical_queries=list(
                dict.fromkeys(
                    [
                        *expansion_result.lexical_queries,
                        *llm_plan.lexical_queries,
                    ]
                )
            ),
            semantic_queries=list(
                dict.fromkeys(
                    [
                        *expansion_result.semantic_queries,
                        *llm_plan.semantic_queries,
                    ]
                )
            ),
            entity_queries=list(
                dict.fromkeys(
                    [
                        *expansion_result.entity_queries,
                        *llm_plan.entity_queries,
                    ]
                )
            ),
            relation_queries=list(
                dict.fromkeys(
                    [
                        *expansion_result.relation_queries,
                        *llm_plan.relation_queries,
                    ]
                )
            ),
            hyde_text=llm_plan.hyde_text or expansion_result.hyde_text,
            raw_output=expansion_result.raw_output,
            planner_model=self.deepseek.config.llm_model,
            planner_raw_output=planner_raw,
        )
        pool = plan.candidate_pool_size
        queries = list(
            dict.fromkeys(
                [
                    request.query,
                    *expansion.semantic_queries,
                    expansion.hyde_text,
                ]
            )
        )
        stages = ["query_planning", "query_expansion"]
        channels: dict[str, list[Candidate]] = {}

        if "lexical" in plan.channels:
            channels["lexical"] = lexical_retrieve(
                self.index_manager.lexical,
                corpus,
                [request.query, *expansion.lexical_queries],
                limit=pool,
                document_ids=document_ids,
            )
            stages.append("lexical_retrieval")
        if "semantic" in plan.channels:
            channels["semantic"] = await semantic_retrieve(
                self.cognee_repository,
                corpus,
                queries,
                limit=pool,
                document_ids=document_ids,
            )
            stages.append("semantic_retrieval")
        if "entity_claim" in plan.channels:
            channels["entity_claim"] = await entity_claim_retrieve(
                self.cognee_repository,
                corpus,
                [
                    request.query,
                    *expansion.entity_queries,
                    *expansion.relation_queries,
                ],
                limit=pool,
                document_ids=document_ids,
            )
            stages.append("entity_claim_retrieval")
        if "graph" in plan.channels:
            channels["graph"] = await graph_retrieve(
                self.cognee_repository,
                corpus,
                [
                    request.query,
                    *expansion.entity_queries,
                    *expansion.relation_queries,
                ],
                limit=pool,
                depth=plan.graph_depth,
                document_ids=document_ids,
            )
            stages.append("graph_traversal")
        if "global_context" in plan.channels:
            channels["global_context"] = await global_context_retrieve(
                self.cognee_repository,
                corpus,
                queries,
                limit=pool,
                document_ids=document_ids,
            )
            stages.append("global_context")
        if "confirmed_knowledge" in plan.channels:
            channels["confirmed_knowledge"] = confirmed_knowledge_retrieve(
                self.feedback,
                corpus,
                [request.query, *expansion.semantic_queries],
                limit=pool,
                document_ids=document_ids,
            )
            stages.append("confirmed_knowledge_retrieval")

        fused = weighted_rrf(channels, plan.weights)[:pool]
        stages.extend(["fusion", "evidence_backtracking"])
        reranked = await rerank_candidates(
            self.model_client,
            (
                f"{request.query}\nRetrieval intent: "
                + " ; ".join(llm_plan.semantic_queries[:3])
            ),
            fused,
            limit=pool,
        )
        stages.append("rerank")
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
            aspect_queries=[
                *llm_plan.lexical_queries,
                *llm_plan.entity_queries,
            ],
        )
        stages.append("diversification")
        evidence = format_evidence(selected, corpus.bundles)
        answer = await synthesize_answer(
            self.deepseek,
            query=request.query,
            profile=request.profile,
            evidence=evidence,
        )
        stages.append("synthesis")
        response = QueryResponse(
            id=cache_key,
            query=request.query,
            profile=request.profile,
            dataset=dataset_name,
            answer=answer,
            answer_model=self.deepseek.config.llm_model,
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
