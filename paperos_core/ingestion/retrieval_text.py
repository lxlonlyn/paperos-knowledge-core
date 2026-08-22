"""Rebuildable retrieval projection separate from authoritative ``Chunk.text``."""

from __future__ import annotations

import re

from paperos_core.domain.canonical import Chunk, CitationMention, Document, ReferenceEntry
from paperos_core.ingestion.normalization import plain_text

_REF_MARKER_RE = re.compile(r"^\s*[\[\(]?\s*[^\]\)\s]{1,40}\s*[\]\)]?\s*")


def build_retrieval_text(
    *,
    document: Document,
    chunk: Chunk,
    mentions: list[CitationMention],
    references_by_id: dict[str, ReferenceEntry],
) -> str:
    lines: list[str] = [f"Paper:\n{document.title}"]
    section_header = chunk.section_path or chunk.major_section_title
    if section_header and not _section_header_redundant(section_header, chunk.text):
        lines.append(f"Section:\n{section_header}")

    resolved_lines: list[str] = []
    seen: set[str] = set()
    for mention in sorted(
        mentions,
        key=lambda item: (item.group_index, item.atomic_key, item.character_start),
    ):
        if mention.chunk_id and mention.chunk_id != chunk.id:
            continue
        dedupe_key = mention.resolved_work_id or mention.reference_entry_id
        if dedupe_key:
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
        reference = (
            references_by_id.get(mention.reference_entry_id)
            if mention.reference_entry_id
            else None
        )
        label = _atomic_display_label(mention.atomic_key)
        title = _compact_reference_line(reference, atomic_key=mention.atomic_key)
        if title:
            resolved_lines.append(f"{label} = {title}")
        elif not dedupe_key:
            resolved_lines.append(f"{label} = unresolved")
    if resolved_lines:
        lines.append("Referenced works:\n" + "\n".join(resolved_lines))
    lines.append(chunk.text)
    return "\n\n".join(line for line in lines if line.strip())


def effective_index_text(chunk: Chunk) -> str:
    """Single retrieval string for embedding / lexical indexing."""
    return (chunk.retrieval_text or chunk.text).strip()


def _atomic_display_label(atomic_key: str) -> str:
    if atomic_key.isdigit() or re.fullmatch(r"[A-Za-z0-9*+]+", atomic_key):
        return f"[{atomic_key}]"
    return atomic_key


def _compact_reference_line(
    reference: ReferenceEntry | None,
    *,
    atomic_key: str,
) -> str | None:
    if reference is None:
        return None
    if reference.title:
        return reference.title.strip()
    body = _strip_leading_marker(reference.raw_text, atomic_key=atomic_key)
    if not body:
        return None
    if len(body) > 180:
        return body[:177] + "..."
    return body


def _strip_leading_marker(raw_text: str, *, atomic_key: str) -> str:
    body = plain_text(raw_text)
    body = _REF_MARKER_RE.sub("", body, count=1).strip()
    marker = f"[{atomic_key}]"
    if body.casefold().startswith(marker.casefold()):
        body = body[len(marker) :].strip()
    return body


def _section_header_redundant(section_header: str, chunk_text: str) -> bool:
    header = plain_text(section_header).strip()
    if not header:
        return True
    prefix = plain_text(chunk_text).lstrip()
    if prefix.casefold().startswith(header.casefold()):
        return True
    first_line = prefix.splitlines()[0].strip() if prefix else ""
    return first_line.casefold() == header.casefold()
