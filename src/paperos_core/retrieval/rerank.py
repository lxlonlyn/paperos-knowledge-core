"""Real local Qwen3 reranking over fused candidates."""

from paperos_core.adapters.models.client import LocalModelGatewayClient
from paperos_core.retrieval.candidates import Candidate


async def rerank_candidates(
    client: LocalModelGatewayClient,
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
        candidate.final_rank = result.final_rank
        reranked.append(candidate)
    return reranked
