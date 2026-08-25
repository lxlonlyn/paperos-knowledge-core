"""Provider-neutral evidence-grounded answer synthesis."""

import re
from dataclasses import dataclass

from paperos_core.adapters.cognee.llm import LLMClient
from paperos_core.retrieval.candidates import Evidence


@dataclass(frozen=True, slots=True)
class FinalSynthesisContext:
    """All query-dependent input supplied to final answer synthesis."""

    original_query: str
    evidence: list[Evidence]


def render_synthesis_prompt(context: FinalSynthesisContext) -> str:
    """Render the sole portable user prompt used for final synthesis."""
    sections = [
        "# Task",
        "",
        "Answer the research question below using the supplied paper evidence.",
        "",
        "Base factual claims on the supplied evidence. Distinguish clearly between:",
        "- statements directly supported by the evidence;",
        "- comparisons or conclusions synthesized across multiple sources.",
        "",
        "When sources disagree, describe the disagreement rather than silently choosing one.",
        "Do not invent details that are not supported by the supplied evidence.",
        "Cite supporting claims inline using the exact Evidence ID in square brackets.",
        "Attribute each source's statements to that source; do not transfer a later paper's",
        "critique or claim to the paper it discusses.",
        "",
        "# Research Question",
        "",
        context.original_query,
        "",
        "# Evidence",
    ]
    for index, item in enumerate(context.evidence, start=1):
        sections.extend(
            [
                "",
                f"## Evidence {index}",
                "",
                f"Evidence ID: {item.evidence_id}",
                f"Paper: {item.title}",
            ]
        )
        if item.authors:
            sections.append(f"Authors: {'; '.join(item.authors)}")
        if item.year is not None:
            sections.append(f"Year: {item.year}")
        if item.section_path:
            sections.append(f"Section: {item.section_path}")
        pages = _render_pages(item.page_start, item.page_end)
        if pages is not None:
            sections.append(f"Pages: {pages}")
        sections.extend(
            [
                f"Chunk ID: {item.chunk_id}",
                "",
                item.text,
            ]
        )
    sections.extend(
        [
            "",
            "# Requested Output",
            "",
            "Provide a direct, evidence-grounded answer to the research question.",
            "Synthesize information across the supplied papers when useful.",
            "Use the language of the research question.",
            "State clearly when the supplied evidence is insufficient.",
        ]
    )
    return "\n".join(sections)


def _render_pages(page_start: int | None, page_end: int | None) -> str | None:
    if page_start is None:
        return str(page_end) if page_end is not None else None
    if page_end is None or page_end == page_start:
        return str(page_start)
    return f"{page_start}-{page_end}"


async def synthesize_answer(
    client: LLMClient,
    *,
    prompt: str,
    evidence: list[Evidence],
) -> str:
    answer = await client.synthesize_answer(
        prompt=prompt,
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
