"""Provider-neutral evidence-grounded answer synthesis."""

import re

from paperos_core.adapters.cognee.llm import LLMClient
from paperos_core.retrieval.candidates import Evidence


async def synthesize_answer(
    client: LLMClient,
    *,
    query: str,
    evidence: list[Evidence],
) -> str:
    answer = await client.synthesize_answer(
        query=query,
        evidence=[item.model_dump(mode="json") for item in evidence],
    )
    for item in evidence:
        canonical = f"[{item.evidence_id}]"
        for rendered in (
            f"[{item.chunk_id}]",
            f"［{item.chunk_id}］",
            f"【{item.chunk_id}】",
        ):
            answer = answer.replace(rendered, canonical)
    known_chunk_ids = [item.chunk_id for item in evidence]

    def expand_short_id(match: re.Match[str]) -> str:
        prefix = match.group(0)
        matches = [chunk_id for chunk_id in known_chunk_ids if chunk_id.startswith(prefix)]
        return f"evidence:{matches[0]}" if len(matches) == 1 else prefix

    answer = re.sub(r"(?<!evidence:)chunk_[0-9a-f]{6,31}", expand_short_id, answer)
    return answer
