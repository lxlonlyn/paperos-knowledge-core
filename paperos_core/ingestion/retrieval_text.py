"""Rebuildable retrieval projection separate from authoritative ``Chunk.text``."""

from __future__ import annotations

from paperos_core.domain.canonical import Chunk, CitationMention, Document, ReferenceEntry


def build_retrieval_text(
    *,
    document: Document,
    chunk: Chunk,
    mentions: list[CitationMention],
    references_by_id: dict[str, ReferenceEntry],
) -> str:
    lines: list[str] = [
        f"Paper:\n{document.title}",
    ]
    if chunk.section_path:
        lines.append(f"Section:\n{chunk.section_path}")
    elif chunk.major_section_title:
        lines.append(f"Section:\n{chunk.major_section_title}")

    resolved_lines: list[str] = []
    for mention in mentions:
        if mention.chunk_id and mention.chunk_id != chunk.id:
            continue
        reference = (
            references_by_id.get(mention.reference_entry_id)
            if mention.reference_entry_id
            else None
        )
        title = _reference_title(reference)
        key = mention.surface_text
        if title:
            resolved_lines.append(f"{key} = {title}")
        else:
            resolved_lines.append(f"{key} = unresolved")
    if resolved_lines:
        lines.append("Referenced works:\n" + "\n".join(resolved_lines))
    lines.append(chunk.text)
    return "\n\n".join(line for line in lines if line.strip())


def effective_index_text(chunk: Chunk) -> str:
    """Single retrieval string for embedding / lexical indexing."""
    return (chunk.retrieval_text or chunk.text).strip()


def _reference_title(reference: ReferenceEntry | None) -> str | None:
    if reference is None:
        return None
    if reference.title:
        return reference.title
    body = reference.raw_text
    if len(body) > 240:
        return body[:237] + "..."
    return body or None
