"""Structured local reranking with MaxP aggregation to parent Chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from paperos_core.errors import CanonicalValidationError, LocalInferenceResponseError
from paperos_core.retrieval.candidates import Candidate, RerankDiagnostics

if TYPE_CHECKING:
    from paperos_core.domain.canonical import RerankSpan
    from paperos_core.retrieval.corpus import CorpusView
    from paperos_core.runtime.local_inference.client import LocalInferenceClient
    from paperos_core.runtime.local_inference.schemas import RerankResult


@dataclass(slots=True)
class RerankPass:
    candidates: list[Candidate]
    projection_version: str | None
    span_count: int


async def rerank_candidates(
    client: LocalInferenceClient,
    query: str,
    candidates: list[Candidate],
    *,
    corpus: CorpusView,
    limit: int,
) -> RerankPass:
    """Score structured spans once, then rank canonical parents by MaxP."""

    if not candidates:
        return RerankPass(candidates=[], projection_version=None, span_count=0)

    scoring_ids: list[str] = []
    scoring_texts: list[str] = []
    span_mapping: dict[str, tuple[int, RerankSpan]] = {}
    versions: set[str] = set()
    for parent_index, candidate in enumerate(candidates):
        chunk = corpus.chunks.get(candidate.chunk_id)
        spans = corpus.rerank_spans_by_chunk.get(candidate.chunk_id, [])
        if chunk is None or not spans:
            raise CanonicalValidationError(
                "Canonical parent Chunk is missing its structured rerank projection.",
                affected=candidate.chunk_id,
            )
        for span in spans:
            if span.id in span_mapping:
                raise CanonicalValidationError(
                    "RerankSpan scoring IDs must be globally unique.",
                    affected=span.id,
                )
            scoring_ids.append(span.id)
            scoring_texts.append(span.scoring_text(chunk))
            span_mapping[span.id] = (parent_index, span)
            versions.add(span.projection_version)
    if len(versions) != 1:
        raise CanonicalValidationError(
            "One rerank pass cannot mix projection versions.",
            affected=",".join(sorted(versions)),
        )

    results = await client.rerank(
        query,
        scoring_ids,
        scoring_texts,
        limit=len(scoring_ids),
    )
    if len(results) != len(scoring_ids):
        raise LocalInferenceResponseError(
            "Local reranker did not return every structured scoring span.",
            details={
                "expected_count": len(scoring_ids),
                "returned_count": len(results),
            },
        )

    best_by_parent: dict[int, tuple[RerankResult, RerankSpan]] = {}
    for result in results:
        parent_index, span = span_mapping[result.candidate_id]
        previous = best_by_parent.get(parent_index)
        if previous is None or _span_result_better(result, span, *previous):
            best_by_parent[parent_index] = (result, span)

    ranked: list[tuple[int, Candidate]] = []
    for parent_index, candidate in enumerate(candidates):
        result, span = best_by_parent[parent_index]
        parent = candidate.model_copy(deep=True)
        parent.rerank_score = result.relevance_score
        parent.rerank_diagnostics = RerankDiagnostics(
            document_token_count=result.document_token_count,
            input_token_count=result.input_token_count,
            effective_input_token_count=result.effective_input_token_count,
            model_max_input_tokens=result.model_max_input_tokens,
            query_token_count=result.query_token_count,
            special_prompt_token_count=result.special_prompt_token_count,
            truncated=result.truncated,
            window_count=result.window_count,
            winning_window_document_token_count=(
                result.winning_window_document_token_count
            ),
            winning_window_index=span.ordinal,
            winning_window_score=result.relevance_score,
            winning_window_text=result.winning_window_text,
        )
        ranked.append((parent_index, parent))
    ranked.sort(
        key=lambda item: (
            -(item[1].rerank_score or 0.0),
            item[0],
        )
    )
    selected = [item[1] for item in ranked[: min(limit, len(ranked))]]
    for final_rank, candidate in enumerate(selected, start=1):
        candidate.final_rank = final_rank
    return RerankPass(
        candidates=selected,
        projection_version=next(iter(versions)),
        span_count=len(scoring_ids),
    )


def _span_result_better(
    result: RerankResult,
    span: RerankSpan,
    previous_result: RerankResult,
    previous_span: RerankSpan,
) -> bool:
    if result.relevance_score != previous_result.relevance_score:
        return result.relevance_score > previous_result.relevance_score
    return span.ordinal < previous_span.ordinal


__all__ = ["RerankPass", "rerank_candidates"]
