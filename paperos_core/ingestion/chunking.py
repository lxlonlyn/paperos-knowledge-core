"""Structure-aware canonical chunking."""

from __future__ import annotations

import math
from collections.abc import Iterable

from paperos_core.domain.canonical import Chunk, Element, Section
from paperos_core.domain.enums import ElementType
from paperos_core.domain.ids import CHUNKING_VERSION, chunk_id

_TEXT_TYPES = {
    ElementType.TITLE,
    ElementType.PARAGRAPH,
    ElementType.LIST,
    ElementType.LIST_ITEM,
    ElementType.FORMULA,
    ElementType.CAPTION,
    ElementType.FOOTNOTE,
}


def estimate_tokens(text: str) -> int:
    """Conservative deterministic approximation when no tokenizer is required."""
    return max(1, math.ceil(len(text) / 4))


def _element_text(element: Element) -> str:
    if element.element_type == ElementType.FORMULA:
        return element.latex or element.text or ""
    return element.text or element.markdown or ""


def build_chunks(
    *,
    document_id: str,
    snapshot_id: str,
    sections: list[Section],
    elements: Iterable[Element],
    target_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    section_by_id = {section.id: section for section in sections}
    eligible = [
        element
        for element in elements
        if element.element_type in _TEXT_TYPES and _element_text(element).strip()
    ]
    grouped: dict[str | None, list[Element]] = {}
    for element in eligible:
        grouped.setdefault(element.section_id, []).append(element)

    target_chars = max(200, target_tokens * 4)
    overlap_chars = max(0, overlap_tokens * 4)
    pending: list[tuple[list[Element], str | None]] = []
    section_order: list[str | None] = [None, *(section.id for section in sections)]
    for section_id in section_order:
        rows = grouped.get(section_id, [])
        if not rows:
            continue
        current: list[Element] = []
        current_size = 0
        for element in rows:
            value = _element_text(element).strip()
            extra = len(value) + (2 if current else 0)
            if current and current_size + extra > target_chars:
                pending.append((current, section_id))
                overlap: list[Element] = []
                overlap_size = 0
                for previous in reversed(current):
                    previous_text = _element_text(previous)
                    if overlap and overlap_size + len(previous_text) > overlap_chars:
                        break
                    if overlap_chars:
                        overlap.insert(0, previous)
                        overlap_size += len(previous_text)
                current = overlap
                current_size = sum(len(_element_text(item)) + 2 for item in current)
            current.append(element)
            current_size += extra
        if current:
            pending.append((current, section_id))

    chunks: list[Chunk] = []
    character_cursor = 0
    for order, (rows, section_id) in enumerate(pending):
        text = "\n\n".join(_element_text(item).strip() for item in rows).strip()
        pages = [item.page for item in rows if item.page is not None]
        section = section_by_id.get(section_id) if section_id else None
        ids = [item.id for item in rows]
        identifier = chunk_id(document_id, order, ids)
        chunks.append(
            Chunk(
                id=identifier,
                document_id=document_id,
                canonical_snapshot_id=snapshot_id,
                text=text,
                order=order,
                element_ids=ids,
                section_id=section_id,
                section_path=section.path if section else None,
                page_start=min(pages) if pages else None,
                page_end=max(pages) if pages else None,
                token_count=estimate_tokens(text),
                character_start=character_cursor,
                character_end=character_cursor + len(text),
                chunking_version=CHUNKING_VERSION,
            )
        )
        character_cursor += len(text) + 2

    return [
        chunk.model_copy(
            update={
                "previous_chunk_id": chunks[index - 1].id if index else None,
                "next_chunk_id": (chunks[index + 1].id if index + 1 < len(chunks) else None),
            }
        )
        for index, chunk in enumerate(chunks)
    ]
