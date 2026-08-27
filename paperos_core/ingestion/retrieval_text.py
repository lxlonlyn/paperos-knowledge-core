"""Rebuildable retrieval projection separate from authoritative ``Chunk.text``."""

from __future__ import annotations

import re
from collections.abc import Mapping

from paperos_core.domain.canonical import (
    Chunk,
    CitationMention,
    Document,
    ReferenceEntry,
)
from paperos_core.domain.scholarly import ScholarlyContext, ScholarlyWork
from paperos_core.ingestion.normalization import plain_text


def build_retrieval_text(
    *,
    document: Document,
    chunk: Chunk,
    mentions: list[CitationMention],
    references_by_id: dict[str, ReferenceEntry],
    works_by_id: Mapping[str, ScholarlyWork] | None = None,
) -> str:
    lines: list[str] = []
    if title := document.title.strip():
        lines.append(f"Paper:\n{title}")
    if document.year is not None:
        lines.append(f"Year:\n{document.year}")
    if region := (chunk.document_region or "").strip():
        lines.append(f"Region:\n{region}")
    if section_header := (chunk.section_path or chunk.major_section_title):
        lines.append(f"Section:\n{section_header}")

    resolved_lines: list[str] = []
    seen: set[str] = set()
    for mention in sorted(
        mentions,
        key=lambda item: (item.character_start, item.group_index, item.atomic_key),
    ):
        if mention.chunk_id and mention.chunk_id != chunk.id:
            continue
        work_id = mention.resolved_work_id
        if not work_id or work_id in seen:
            continue
        seen.add(work_id)
        reference = (
            references_by_id.get(mention.reference_entry_id)
            if mention.reference_entry_id
            else None
        )
        label = _atomic_display_label(mention.atomic_key)
        work = works_by_id.get(work_id) if works_by_id is not None else None
        cited_title = (
            work.title.strip()
            if work is not None and work.title.strip()
            else _compact_reference_line(reference, atomic_key=mention.atomic_key)
        )
        identity = f"Work ID: {work_id}"
        resolved_lines.append(
            f"{label} = {cited_title} ({identity})"
            if cited_title
            else f"{label} = {identity}"
        )
    if resolved_lines:
        lines.append("Referenced works:\n" + "\n".join(resolved_lines))
    retrieval_content = chunk.metadata.get("retrieval_content_text") or chunk.text
    lines.append(str(retrieval_content))
    return "\n\n".join(line for line in lines if line.strip())


def effective_index_text(chunk: Chunk) -> str:
    """Single retrieval string for embedding / lexical indexing."""
    return (chunk.retrieval_text or chunk.text).strip()


def bind_scholarly_citations(
    *,
    document: Document,
    chunks: list[Chunk],
    mentions: list[CitationMention],
    references: list[ReferenceEntry],
    scholarly: ScholarlyContext,
) -> tuple[list[Chunk], list[CitationMention]]:
    """Bind bibliography-first mentions to final Work IDs and rebuild retrieval text."""

    resolutions = scholarly.resolution_by_reference()
    works_by_id = {work.id: work for work in scholarly.works}
    bound_mentions: list[CitationMention] = []
    for mention in mentions:
        resolution = (
            resolutions.get(mention.reference_entry_id)
            if mention.reference_entry_id is not None
            else None
        )
        work_id = (
            resolution.work_id
            if resolution is not None
            and resolution.resolution_status == "resolved"
            else None
        )
        bound_mentions.append(
            mention.model_copy(update={"resolved_work_id": work_id})
        )

    mentions_by_chunk: dict[str, list[CitationMention]] = {}
    for mention in bound_mentions:
        if mention.chunk_id is not None:
            mentions_by_chunk.setdefault(mention.chunk_id, []).append(mention)
    references_by_id = {reference.id: reference for reference in references}
    finalized_chunks: list[Chunk] = []
    for chunk in chunks:
        chunk_mentions = mentions_by_chunk.get(chunk.id, [])
        finalized_chunks.append(
            chunk.model_copy(
                update={
                    "retrieval_text": build_retrieval_text(
                        document=document,
                        chunk=chunk,
                        mentions=chunk_mentions,
                        references_by_id=references_by_id,
                        works_by_id=works_by_id,
                    ),
                    "citation_mention_ids": [
                        mention.id for mention in chunk_mentions
                    ],
                    "citation_reference_entry_ids": list(
                        dict.fromkeys(
                            mention.reference_entry_id
                            for mention in chunk_mentions
                            if mention.reference_entry_id is not None
                        )
                    ),
                    "citation_work_ids": list(
                        dict.fromkeys(
                            mention.resolved_work_id
                            for mention in chunk_mentions
                            if mention.resolved_work_id is not None
                        )
                    ),
                }
            )
        )
    return finalized_chunks, bound_mentions


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
    escaped = re.escape(atomic_key.strip())
    marker = re.compile(
        rf"^\s*(?:\[\s*{escaped}\s*\]|\(\s*{escaped}\s*\)|"
        rf"{escaped}\s*[.)])\s*",
        flags=re.IGNORECASE,
    )
    return marker.sub("", body, count=1).strip()
