"""Single chunk-first production retrieval pipeline."""

from __future__ import annotations

import asyncio

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.adapters.cognee.llm import LLMClient
from paperos_core.adapters.cognee.search import CogneeSearchAdapter
from paperos_core.config import RuntimeSettings
from paperos_core.feedback.service import FeedbackService
from paperos_core.indexes.manager import IndexManager
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry
from paperos_core.paths import DataPaths
from paperos_core.retrieval.cache import QueryCache
from paperos_core.retrieval.candidates import (
    Candidate,
    QueryRequest,
    QueryResponse,
    RetrievalTrace,
)
from paperos_core.retrieval.corpus import CorpusView
from paperos_core.retrieval.diversify import diversify
from paperos_core.retrieval.evidence import format_evidence
from paperos_core.retrieval.expansion import (
    citation_post_hit_expand,
    graph_post_hit_expand,
    local_neighbor_expand,
)
from paperos_core.retrieval.fusion import (
    deduplicate_candidates_by_chunk,
    weighted_rrf,
)
from paperos_core.retrieval.lexical import lexical_retrieve
from paperos_core.retrieval.rerank import rerank_candidates
from paperos_core.retrieval.semantic import semantic_retrieve
from paperos_core.retrieval.synthesis import synthesize_answer
from paperos_core.runtime.local_inference.client import LocalInferenceClient


class RetrievalService:
    """Retrieve and synthesize exclusively from canonical source Chunks."""

    def __init__(
        self,
        config: RuntimeSettings,
        paths: DataPaths,
        canonical_repository: CanonicalRepository,
        registry: SourceRegistry,
        scholarly_registry: ScholarlyRegistry,
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
        self.scholarly_registry = scholarly_registry
        self.search = search
        self.compat = compat
        self.index_manager = index_manager
        self.model_client = model_client
        self.llm = llm
        self.cache = QueryCache(paths, feedback)

    async def query(self, request: QueryRequest) -> QueryResponse:
        dataset_name = (request.dataset or self.config.dataset).strip()
        request = request.model_copy(update={"dataset": dataset_name})
        corpus = CorpusView.load(
            self.paths,
            self.canonical_repository,
            self.registry,
            self.scholarly_registry,
        )
        cache_key = self.cache.key(request, corpus)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        document_ids = corpus.filtered_document_ids(
            request.document_ids, dataset_name
        )
        explicit_work_ids = set(
            request.source_work_ids or []
        ) | set(request.subject_work_ids or []) | set(request.work_ids or [])
        if explicit_work_ids:
            document_ids.intersection_update(
                corpus.document_ids_for_works(explicit_work_ids)
            )

        top_k = request.top_k or self.config.retrieval.top_k
        pool_size = self.config.retrieval.candidate_pool_size
        stages = ["explicit_filters", "lexical_chunk_retrieval"]
        channels: dict[str, list[Candidate]] = {
            "lexical": lexical_retrieve(
                self.index_manager.lexical,
                corpus,
                [request.query],
                limit=pool_size,
                document_ids=document_ids,
            )
        }
        channels["vector"] = await semantic_retrieve(
            self.search,
            corpus,
            request.query,
            dataset_name=dataset_name,
            limit=pool_size,
            document_ids=document_ids,
        )
        stages.append("vector_chunk_retrieval")

        fused = weighted_rrf(channels, {"lexical": 1.0, "vector": 1.0})
        fused = deduplicate_candidates_by_chunk(fused)[:pool_size]
        stages.extend(["rrf", "chunk_id_dedup"])
        first_stage_chunk_ids = [item.chunk_id for item in fused]

        first_reranked = await self._rerank(
            request.query, fused, limit=pool_size
        )
        if self.config.retrieval.rerank_enabled:
            stages.append("first_rerank")
        seeds = first_reranked[:top_k]
        first_stage_ids = {item.chunk_id for item in first_reranked}

        local_expanded: list[Candidate] = []
        citation_expanded: list[Candidate] = []
        graph_expanded: list[Candidate] = []
        if request.expand_context:
            local_expanded = local_neighbor_expand(
                corpus, seeds, document_ids=document_ids
            )
            stages.append("local_post_hit_expansion")
        if request.expand_graph:
            citation_expanded, graph_expanded = await asyncio.gather(
                citation_post_hit_expand(
                    self.compat,
                    corpus,
                    seeds,
                    dataset_name=dataset_name,
                    document_ids=document_ids,
                    limit=pool_size,
                ),
                graph_post_hit_expand(
                    self.compat,
                    corpus,
                    seeds,
                    depth=self.config.retrieval.graph_depth,
                    document_ids=document_ids,
                    limit=pool_size,
                    claim_enrichment_enabled=(
                        self.config.ingestion.claim_enrichment_enabled
                    ),
                ),
            )
            stages.extend(
                ["citation_post_hit_expansion", "graph_post_hit_expansion"]
            )

        expanded = deduplicate_candidates_by_chunk(
            [*local_expanded, *citation_expanded, *graph_expanded]
        )
        genuinely_new = [
            item for item in expanded if item.chunk_id not in first_stage_ids
        ]
        if genuinely_new:
            merged = deduplicate_candidates_by_chunk(
                [*first_reranked, *genuinely_new]
            )
            reranked = await self._rerank(
                request.query, merged, limit=pool_size
            )
            if self.config.retrieval.rerank_enabled:
                stages.append("second_rerank")
        else:
            reranked = first_reranked

        selected = diversify(
            reranked,
            limit=top_k,
            max_per_document=self.config.retrieval.max_chunks_per_document,
            max_per_section=self.config.retrieval.max_chunks_per_section,
        )
        selected = deduplicate_candidates_by_chunk(selected)
        stages.extend(["final_selection", "source_grounded_evidence"])
        evidence = format_evidence(selected, corpus)
        answer = await synthesize_answer(
            self.llm,
            query=request.query,
            evidence=evidence,
        )
        stages.append("synthesis")

        trace = RetrievalTrace(
            applied_document_ids=sorted(document_ids),
            first_stage_chunk_ids=first_stage_chunk_ids,
            first_reranked_chunk_ids=[
                item.chunk_id for item in first_reranked
            ],
            local_expanded_chunk_ids=[
                item.chunk_id for item in local_expanded
            ],
            citation_expanded_chunk_ids=[
                item.chunk_id for item in citation_expanded
            ],
            graph_expanded_chunk_ids=[
                item.chunk_id for item in graph_expanded
            ],
            seed_chunk_ids=[item.chunk_id for item in seeds],
            relation_types=list(
                dict.fromkeys(
                    relation
                    for item in [*citation_expanded, *graph_expanded]
                    for relation in item.relation_types
                )
            ),
            derived_from_ids=list(
                dict.fromkeys(
                    derived_id
                    for item in expanded
                    for derived_id in item.derived_from_ids
                )
            ),
            second_reranked_chunk_ids=(
                [item.chunk_id for item in reranked] if genuinely_new else []
            ),
            final_selected_chunk_ids=[item.chunk_id for item in selected],
        )
        response = QueryResponse(
            id=cache_key,
            query=request.query,
            dataset=dataset_name,
            answer=answer,
            answer_model=self.llm.model,
            stages=stages,
            channels_used=list(
                dict.fromkeys(
                    channel
                    for item in selected
                    for channel in item.channels
                )
            ),
            evidence=evidence,
            candidates=selected,
            distinct_documents=len({item.document_id for item in evidence}),
            provenance_complete=all(
                item.chunk_id in corpus.chunks
                and item.document_id == corpus.chunks[item.chunk_id].document_id
                and item.text == corpus.chunks[item.chunk_id].text
                for item in evidence
            ),
            trace=trace,
        )
        self.cache.put(response)
        return response

    async def _rerank(
        self, query: str, candidates: list[Candidate], *, limit: int
    ) -> list[Candidate]:
        if not self.config.retrieval.rerank_enabled:
            return candidates[:limit]
        return await rerank_candidates(
            self.model_client, query, candidates, limit=limit
        )
