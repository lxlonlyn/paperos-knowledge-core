"""Hybrid full-Chunk and structured-span reranking of canonical parents."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from paperos_core.errors import CanonicalValidationError, LocalInferenceResponseError
from paperos_core.retrieval.candidates import Candidate, RerankDiagnostics

_RRF_RANK_CONSTANT = 60
_FULL_SCORING_ID_PREFIX = "full:"

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
    """Fuse full-Chunk and structured MaxP ranks, returning canonical parents."""

    if not candidates:
        return RerankPass(candidates=[], projection_version=None, span_count=0)

    scoring_ids: list[str] = []
    scoring_texts: list[str] = []
    full_mapping: dict[str, int] = {}
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
        full_scoring_id = f"{_FULL_SCORING_ID_PREFIX}{candidate.chunk_id}"
        if full_scoring_id in full_mapping or full_scoring_id in span_mapping:
            raise CanonicalValidationError(
                "Reranker scoring IDs must be globally unique.",
                affected=full_scoring_id,
            )
        scoring_ids.append(full_scoring_id)
        scoring_texts.append(chunk.text)
        full_mapping[full_scoring_id] = parent_index
        for span in spans:
            if span.id in full_mapping or span.id in span_mapping:
                raise CanonicalValidationError(
                    "Reranker scoring IDs must be globally unique.",
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
            "Local reranker did not return every requested scoring document.",
            details={
                "expected_count": len(scoring_ids),
                "returned_count": len(results),
            },
        )

    returned_ids = [result.candidate_id for result in results]
    returned_id_counts = Counter(returned_ids)
    returned_id_set = set(returned_ids)
    expected_id_set = set(scoring_ids)
    duplicate_ids = sorted(
        candidate_id
        for candidate_id, count in returned_id_counts.items()
        if count > 1
    )
    if duplicate_ids or returned_id_set != expected_id_set:
        raise LocalInferenceResponseError(
            "Local reranker returned an invalid scoring document ID set.",
            details={
                "reason": "rerank_candidate_id_mismatch",
                "expected_count": len(scoring_ids),
                "returned_count": len(results),
                "duplicate_candidate_ids": duplicate_ids,
                "missing_candidate_ids": sorted(expected_id_set - returned_id_set),
                "unknown_candidate_ids": sorted(returned_id_set - expected_id_set),
            },
        )

    full_by_parent: dict[int, RerankResult] = {}
    best_span_by_parent: dict[int, tuple[RerankResult, RerankSpan]] = {}
    for result in results:
        full_parent_index = full_mapping.get(result.candidate_id)
        if full_parent_index is not None:
            full_by_parent[full_parent_index] = result
            continue
        parent_index, span = span_mapping[result.candidate_id]
        previous = best_span_by_parent.get(parent_index)
        if previous is None or _span_result_better(result, span, *previous):
            best_span_by_parent[parent_index] = (result, span)

    full_ranks = _parent_ranks(
        len(candidates),
        {index: result.relevance_score for index, result in full_by_parent.items()},
    )
    structured_ranks = _parent_ranks(
        len(candidates),
        {
            index: result.relevance_score
            for index, (result, _span) in best_span_by_parent.items()
        },
    )

    ranked: list[tuple[int, Candidate]] = []
    for parent_index, candidate in enumerate(candidates):
        result, span = best_span_by_parent[parent_index]
        parent = candidate.model_copy(deep=True)
        parent.rerank_score = _reciprocal_rank(full_ranks[parent_index]) + (
            _reciprocal_rank(structured_ranks[parent_index])
        )
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
        span_count=len(span_mapping),
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


def _parent_ranks(
    parent_count: int,
    scores: dict[int, float],
) -> dict[int, int]:
    if set(scores) != set(range(parent_count)):
        raise LocalInferenceResponseError(
            "Local reranker did not score every canonical parent.",
            details={
                "expected_count": parent_count,
                "returned_count": len(scores),
            },
        )
    ranked = sorted(scores, key=lambda index: (-scores[index], index))
    return {parent_index: rank for rank, parent_index in enumerate(ranked, start=1)}


def _reciprocal_rank(rank: int) -> float:
    return 1.0 / (_RRF_RANK_CONSTANT + rank)


__all__ = ["RerankPass", "rerank_candidates"]
