"""Single chunk-first production retrieval pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from paperos_core.domain.ids import QUERY_RESPONSE_ID_VERSION, stable_id
from paperos_core.errors import ConfigurationError
from paperos_core.retrieval.candidates import (
    Candidate,
    QueryReplay,
    QueryRequest,
    QueryResponse,
    RetrievalTrace,
    VectorSearchDiagnostics,
)
from paperos_core.retrieval.corpus import CorpusView
from paperos_core.retrieval.evidence import format_evidence
from paperos_core.retrieval.expansion import (
    local_neighbor_expand,
    semantic_post_hit_expand,
)
from paperos_core.retrieval.fusion import (
    deduplicate_candidates_by_chunk,
    weighted_rrf,
)
from paperos_core.retrieval.lexical import lexical_retrieve
from paperos_core.retrieval.rerank import rerank_candidates
from paperos_core.retrieval.semantic import semantic_retrieve
from paperos_core.retrieval.synthesis import (
    FinalSynthesisContext,
    render_synthesis_prompt,
    select_synthesis_evidence,
    synthesize_answer,
)

if TYPE_CHECKING:
    from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
    from paperos_core.adapters.cognee.llm import LLMClient
    from paperos_core.adapters.cognee.search import CogneeSearchAdapter
    from paperos_core.config import RuntimeSettings
    from paperos_core.indexes.manager import IndexManager
    from paperos_core.ingestion.canonical_repository import CanonicalRepository
    from paperos_core.ingestion.registry import SourceRegistry
    from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry
    from paperos_core.paths import DataPaths
    from paperos_core.runtime.local_inference.client import LocalInferenceClient

NO_EVIDENCE_ANSWER = "未检索到可用于回答的论文证据"
NO_EVIDENCE_MODEL = "paperos/no-evidence"


def effective_candidate_pool_size(candidate_pool_size: int, top_k: int) -> int:
    """Return the real first-stage pool without silently truncating top_k."""

    return max(candidate_pool_size, top_k)


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

    async def query(self, request: QueryRequest) -> QueryResponse:
        if (request.expand_context or request.expand_graph) and not (
            self.config.retrieval.rerank_enabled
        ):
            raise ConfigurationError(
                "Post-hit expansion requires retrieval.rerank_enabled=true.",
                details={
                    "expand_context": request.expand_context,
                    "expand_graph": request.expand_graph,
                    "rerank_enabled": False,
                },
            )
        dataset_name = (request.dataset or self.config.dataset).strip()
        request = request.model_copy(update={"dataset": dataset_name})
        corpus = CorpusView.load(
            self.paths,
            self.canonical_repository,
            self.registry,
            self.scholarly_registry,
        )

        requested_document_ids = sorted(set(request.document_ids or []))
        requested_work_ids = sorted(set(request.work_ids or []))
        document_ids = corpus.filtered_document_ids(request.document_ids, dataset_name)
        resolved_work_document_ids: set[str] = set()
        if request.work_ids is not None:
            resolved_work_document_ids = corpus.document_ids_for_works(
                set(request.work_ids)
            )
            document_ids.intersection_update(resolved_work_document_ids)
        snapshot_resolver = getattr(corpus, "snapshot_ids_for_documents", None)
        allowed_snapshot_ids = (
            snapshot_resolver(document_ids)
            if callable(snapshot_resolver)
            else set(getattr(corpus, "active_snapshot_ids", set()))
        )

        top_k = request.top_k or self.config.retrieval.top_k
        pool_size = effective_candidate_pool_size(
            self.config.retrieval.candidate_pool_size,
            top_k,
        )
        filter_trace = RetrievalTrace(
            requested_document_ids=requested_document_ids,
            requested_work_ids=requested_work_ids,
            resolved_work_document_ids=sorted(resolved_work_document_ids),
            applied_document_ids=sorted(document_ids),
            applied_snapshot_ids=sorted(allowed_snapshot_ids),
        )
        if not document_ids or not allowed_snapshot_ids:
            return self._no_evidence_response(
                request,
                dataset_name=dataset_name,
                stages=["explicit_filters", "no_evidence"],
                trace=filter_trace,
            )

        lexical_diagnostics: dict[str, list[int]] = {}
        vector_diagnostics = VectorSearchDiagnostics()
        stages = ["explicit_filters", "lexical_chunk_retrieval"]
        channels: dict[str, list[Candidate]] = {
            "lexical": lexical_retrieve(
                self.index_manager.lexical,
                corpus,
                [request.query],
                limit=pool_size,
                document_ids=document_ids,
                active_snapshot_ids=allowed_snapshot_ids,
                diagnostics=lexical_diagnostics,
            )
        }
        channels["vector"] = await semantic_retrieve(
            self.search,
            corpus,
            request.query,
            dataset_name=dataset_name,
            limit=pool_size,
            document_ids=document_ids,
            active_snapshot_ids=allowed_snapshot_ids,
            diagnostics=vector_diagnostics,
        )
        stages.append("vector_chunk_retrieval")

        fused = weighted_rrf(channels, {"lexical": 1.0, "vector": 1.0})
        fused = deduplicate_candidates_by_chunk(fused)[:pool_size]
        stages.extend(["rrf", "chunk_id_dedup"])
        first_stage_chunk_ids = [item.chunk_id for item in fused]

        first_reranked = (
            await self._rerank(request.query, fused, limit=pool_size) if fused else []
        )
        if self.config.retrieval.rerank_enabled and fused:
            stages.append("first_rerank")
        seeds = first_reranked[:top_k]
        first_stage_ids = {item.chunk_id for item in first_reranked}

        local_expanded: list[Candidate] = []
        semantic_expanded: list[Candidate] = []
        if request.expand_context and seeds:
            local_expanded = local_neighbor_expand(corpus, seeds, document_ids=document_ids)
            stages.append("local_post_hit_expansion")
        if request.expand_graph and seeds:
            semantic_expanded = await semantic_post_hit_expand(
                self.compat,
                corpus,
                seeds,
                dataset_name=dataset_name,
                document_ids=document_ids,
                limit=pool_size,
            )
            stages.append("semantic_relation_expansion")

        expanded = deduplicate_candidates_by_chunk([*local_expanded, *semantic_expanded])
        local_new = [item for item in local_expanded if item.chunk_id not in first_stage_ids]
        semantic_new = [
            item for item in semantic_expanded if item.chunk_id not in first_stage_ids
        ]
        genuinely_new = [item for item in expanded if item.chunk_id not in first_stage_ids]
        second_rerank_candidates: list[Candidate] = []
        if genuinely_new:
            second_rerank_candidates = deduplicate_candidates_by_chunk(
                [*first_reranked, *genuinely_new]
            )
            reranked = await self._rerank(
                request.query, second_rerank_candidates, limit=pool_size
            )
            stages.append("second_rerank")
        else:
            reranked = first_reranked

        selected = deduplicate_candidates_by_chunk(reranked)[:top_k]
        stages.extend(["final_selection", "source_grounded_evidence"])
        ranked_evidence = format_evidence(selected, corpus)
        evidence = (
            select_synthesis_evidence(
                original_query=request.query,
                ranked_evidence=ranked_evidence,
                max_input_tokens=self.config.retrieval.synthesis_max_input_tokens,
            )
            if ranked_evidence
            else []
        )
        selected = selected[: len(evidence)]
        if evidence:
            synthesis_context = FinalSynthesisContext(
                original_query=request.query,
                evidence=evidence,
            )
            synthesis_prompt = render_synthesis_prompt(synthesis_context)
            answer = await synthesize_answer(
                self.llm,
                prompt=synthesis_prompt,
                evidence=evidence,
            )
            answer_model = self.llm.model
            stages.append("synthesis")
        else:
            selected = []
            synthesis_prompt = ""
            answer = NO_EVIDENCE_ANSWER
            answer_model = NO_EVIDENCE_MODEL
            stages.append("no_evidence")

        trace = RetrievalTrace(
            requested_document_ids=requested_document_ids,
            requested_work_ids=requested_work_ids,
            resolved_work_document_ids=sorted(resolved_work_document_ids),
            applied_document_ids=sorted(document_ids),
            applied_snapshot_ids=sorted(allowed_snapshot_ids),
            candidate_pool_sizes=[pool_size],
            lexical_request_limits=lexical_diagnostics.get("request_limits", []),
            lexical_filtered_counts=lexical_diagnostics.get("filtered_counts", []),
            vector_request_limits=vector_diagnostics.request_limits,
            vector_raw_hit_counts=vector_diagnostics.raw_hit_counts,
            vector_filtered_counts=vector_diagnostics.filtered_hit_counts,
            vector_backend_exhausted=[vector_diagnostics.backend_exhausted],
            vector_safety_limit_reached=[
                vector_diagnostics.safety_limit_reached
            ],
            first_stage_chunk_ids=first_stage_chunk_ids,
            first_reranked_chunk_ids=[item.chunk_id for item in first_reranked],
            local_expanded_chunk_ids=[item.chunk_id for item in local_expanded],
            local_new_chunk_ids=[item.chunk_id for item in local_new],
            semantic_expanded_chunk_ids=[item.chunk_id for item in semantic_expanded],
            semantic_new_chunk_ids=[item.chunk_id for item in semantic_new],
            seed_chunk_ids=[item.chunk_id for item in seeds],
            relation_types=list(
                dict.fromkeys(
                    relation for item in semantic_expanded for relation in item.relation_types
                )
            ),
            derived_from_ids=list(
                dict.fromkeys(
                    derived_id for item in expanded for derived_id in item.derived_from_ids
                )
            ),
            second_reranked_chunk_ids=(
                [item.chunk_id for item in reranked] if genuinely_new else []
            ),
            second_rerank_candidate_ids=[
                item.chunk_id for item in second_rerank_candidates
            ],
            final_selected_chunk_ids=[item.chunk_id for item in selected],
        )
        response = QueryResponse(
            id=stable_id(
                "query_response",
                request.model_dump_json(),
                id_version=QUERY_RESPONSE_ID_VERSION,
            ),
            query=request.query,
            dataset=dataset_name,
            answer=answer,
            answer_model=answer_model,
            stages=stages,
            channels_used=list(
                dict.fromkeys(channel for item in selected for channel in item.channels)
            ),
            evidence=evidence,
            replay=QueryReplay(
                original_query=request.query,
                replay_text=synthesis_prompt,
            ),
            candidates=selected,
            distinct_documents=len({item.document_id for item in evidence}),
            provenance_complete=bool(evidence)
            and all(
                item.chunk_id in corpus.chunks
                and item.document_id == corpus.chunks[item.chunk_id].document_id
                and item.text == corpus.chunks[item.chunk_id].text
                for item in evidence
            ),
            trace=trace,
        )
        return response

    def _no_evidence_response(
        self,
        request: QueryRequest,
        *,
        dataset_name: str,
        stages: list[str],
        trace: RetrievalTrace,
    ) -> QueryResponse:
        return QueryResponse(
            id=stable_id(
                "query_response",
                request.model_dump_json(),
                id_version=QUERY_RESPONSE_ID_VERSION,
            ),
            query=request.query,
            dataset=dataset_name,
            answer=NO_EVIDENCE_ANSWER,
            answer_model=NO_EVIDENCE_MODEL,
            stages=stages,
            channels_used=[],
            evidence=[],
            replay=QueryReplay(original_query=request.query, replay_text=""),
            candidates=[],
            distinct_documents=0,
            provenance_complete=False,
            trace=trace,
        )

    async def _rerank(
        self, query: str, candidates: list[Candidate], *, limit: int
    ) -> list[Candidate]:
        if not self.config.retrieval.rerank_enabled:
            return candidates[:limit]
        return await rerank_candidates(self.model_client, query, candidates, limit=limit)
