"""Human-readable Markdown projection of final Chunk objects."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from paperos_core.domain.canonical import (
    CanonicalBundle,
    Chunk,
    CitationMention,
    ReferenceEntry,
)
from paperos_core.ingestion.retrieval_text import effective_index_text
from paperos_core.ingestion.chunk_dp import TINY_TOKEN_THRESHOLD


def render_chunk_review_markdown(
    *,
    bundle: CanonicalBundle,
    chunks: list[Chunk],
    mentions: list[CitationMention],
    source_pdf: Path,
    target_tokens: int,
    hard_max_tokens: int,
    overlap_tokens: int,
    invariants: dict[str, Any] | None = None,
    citation_stats: dict[str, Any] | None = None,
) -> str:
    references_by_id = {reference.id: reference for reference in bundle.references}
    mentions_by_chunk: dict[str, list[CitationMention]] = {}
    for mention in mentions:
        if mention.chunk_id:
            mentions_by_chunk.setdefault(mention.chunk_id, []).append(mention)

    token_counts = [chunk.token_count or 0 for chunk in chunks]
    tiny = sum(
        1 for count in token_counts if count < TINY_TOKEN_THRESHOLD
    )
    emergency = sum(
        int(chunk.metadata.get("emergency_oversized_sentence_splits") or 0)
        for chunk in chunks
    )
    resolved_refs = sum(1 for mention in mentions if mention.reference_entry_id)
    resolved_works = sum(1 for mention in mentions if mention.resolved_work_id)
    span_count = len({mention.citation_span_id for mention in mentions})

    lines = [
        "# Chunk Review",
        "",
        f"Paper: {bundle.document.title}",
        f"Source PDF: {source_pdf}",
        "",
        f"Target tokens: {target_tokens}",
        f"Hard max: {hard_max_tokens}",
        f"Overlap: {overlap_tokens}",
        f"Chunk count: {len(chunks)}",
        "",
        "## Statistics",
        "",
        f"- Min tokens: {min(token_counts) if token_counts else 0}",
        f"- Median tokens: {statistics.median(token_counts) if token_counts else 0:.1f}",
        f"- Mean tokens: {statistics.mean(token_counts) if token_counts else 0:.1f}",
        f"- Max tokens: {max(token_counts) if token_counts else 0}",
        f"- Tiny chunks (<{TINY_TOKEN_THRESHOLD}): {tiny}",
        f"- Emergency oversized sentence splits: {emergency}",
        f"- Citation spans: {span_count}",
        f"- Atomic citation targets: {len(mentions)}",
        f"- ReferenceEntry resolved (atomic): {resolved_refs}",
        f"- Work resolved (atomic): {resolved_works}",
        "",
    ]
    if citation_stats:
        lines.extend(
            [
                f"- Fully resolved spans: {citation_stats.get('fully_resolved_span_count', 0)}",
                f"- Partially resolved spans: {citation_stats.get('partially_resolved_span_count', 0)}",
                f"- Unresolved spans: {citation_stats.get('unresolved_span_count', 0)}",
                "",
            ]
        )
    lines.extend(
        [
            "",
        ]
    )
    if invariants:
        lines.extend(["## Invariants", ""])
        for key, value in invariants.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.append("---")
    lines.append("")
    for index, chunk in enumerate(chunks, start=1):
        lines.extend(
            _render_chunk_block(
                index=index,
                chunk=chunk,
                mentions=mentions_by_chunk.get(chunk.id, []),
                references_by_id=references_by_id,
            )
        )
    return "\n".join(lines)


def _render_chunk_block(
    *,
    index: int,
    chunk: Chunk,
    mentions: list[CitationMention],
    references_by_id: dict[str, ReferenceEntry],
) -> list[str]:
    emergency = int(chunk.metadata.get("emergency_oversized_sentence_splits") or 0)
    end_boundary = chunk.metadata.get("end_boundary") or "sentence"
    lines = [
        f"## Chunk {index:04d}",
        "",
        f"**Chunk ID:** {chunk.id}",
        f"**Tokens:** {chunk.token_count}",
        f"**Major section:** {chunk.major_section_title or chunk.major_section_id or 'n/a'}",
        f"**Section path:** {chunk.section_path or 'n/a'}",
        f"**Pages:** {chunk.page_start or '?'}–{chunk.page_end or '?'}",
        f"**Elements:** {', '.join(chunk.element_ids)}",
        "",
        f"**Start boundary:** sentence",
        f"**End boundary:** {end_boundary}",
    ]
    if emergency:
        lines.append(f"**Emergency splits:** {emergency} (EMERGENCY_OVERSIZED_SENTENCE_SPLIT)")
    lines.extend(["", "**Citation mentions:**", ""])
    if not mentions:
        lines.append("- (none)")
    else:
        for mention in mentions:
            reference = (
                references_by_id.get(mention.reference_entry_id)
                if mention.reference_entry_id
                else None
            )
            lines.append(
                f"- `{mention.surface_text}` → atomic `{mention.atomic_key}` "
                f"({mention.resolution_status}, span={mention.span_resolution_status})"
            )
            lines.append(
                f"  - ReferenceEntry: {reference.raw_text[:120] + '...' if reference and len(reference.raw_text) > 120 else (reference.raw_text if reference else 'unresolved')}"
            )
            lines.append(
                f"  - Work: {mention.resolved_work_id or 'unresolved'}"
            )
    lines.extend(
        [
            "",
            "### Retrieval context",
            "",
            (chunk.retrieval_text or "").split(chunk.text, 1)[0].strip()
            if chunk.retrieval_text and chunk.text in (chunk.retrieval_text or "")
            else (chunk.retrieval_text or ""),
            "",
            "### Authoritative text",
            "",
            chunk.text,
            "",
            "### Effective retrieval text",
            "",
            effective_index_text(chunk),
            "",
            "---",
            "",
        ]
    )
    return lines
