"""Real local Qwen3 reranking over fused candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING

from paperos_core.retrieval.candidates import Candidate, RerankDiagnostics

if TYPE_CHECKING:
    from paperos_core.runtime.local_inference.client import LocalInferenceClient


async def rerank_candidates(
    client: LocalInferenceClient,
    query: str,
    candidates: list[Candidate],
    *,
    limit: int,
) -> list[Candidate]:
    if not candidates:
        return []
    results = await client.rerank(
        query,
        [candidate.id for candidate in candidates],
        [candidate.text for candidate in candidates],
        limit=min(limit, len(candidates)),
    )
    by_id = {candidate.id: candidate for candidate in candidates}
    reranked: list[Candidate] = []
    for result in results:
        candidate = by_id[result.candidate_id].model_copy(deep=True)
        candidate.rerank_score = result.relevance_score
        candidate.rerank_diagnostics = RerankDiagnostics(
            document_token_count=result.document_token_count,
            input_token_count=result.input_token_count,
            effective_input_token_count=result.effective_input_token_count,
            model_max_input_tokens=result.model_max_input_tokens,
            query_token_count=result.query_token_count,
            special_prompt_token_count=result.special_prompt_token_count,
            truncated=result.truncated,
            window_count=result.window_count,
            winning_window_document_token_count=result.winning_window_document_token_count,
            winning_window_index=result.winning_window_index,
            winning_window_score=result.relevance_score,
            winning_window_text=result.winning_window_text,
        )
        candidate.final_rank = result.final_rank
        reranked.append(candidate)
    return reranked
