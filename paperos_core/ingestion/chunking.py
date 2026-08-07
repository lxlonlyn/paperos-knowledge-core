"""Structure-aware canonical chunking with real tokenizer and span provenance.

PaperOS owns the academic chunking rules; Cognee's custom pipeline executes
them (``AcademicChunkTask``) and Cognee provides the tokenizer and token
limits. Chunks never cross sections, oversized elements are split into
element-internal spans, tables and formulas retain their canonical text, and
every chunk records the exact element spans it covers (including any spans
re-included as overlap). Presentation labels and section context belong to
metadata, not to the element-local coordinate system.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol

from paperos_core.domain.canonical import Chunk, ChunkSpan, Element, Section
from paperos_core.domain.enums import ElementType
from paperos_core.domain.ids import chunk_id

_TEXT_TYPES = {
    ElementType.TITLE,
    ElementType.PARAGRAPH,
    ElementType.LIST,
    ElementType.LIST_ITEM,
    ElementType.CAPTION,
    ElementType.FOOTNOTE,
    ElementType.TABLE,
    ElementType.FORMULA,
}

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？；;])\s+")


class Tokenizer(Protocol):
    """The minimal tokenizer contract used by the academic chunker."""

    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class ElementSpan:
    """One element-internal text span with exact provenance."""

    element_id: str
    span_key: str
    text: str
    tokens: int
    character_start_in_element: int
    character_end_in_element: int
    token_start: int
    token_end: int
    page: int | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    section_id: str | None = None
    section_path: str | None = None

    @property
    def span_id(self) -> str:
        return f"{self.element_id}:{self.span_key}"


@dataclass(frozen=True, slots=True)
class _TextRange:
    text: str
    start: int
    end: int


def build_chunks(
    *,
    document_id: str,
    snapshot_id: str,
    sections: list[Section],
    elements: Iterable[Element],
    target_tokens: int,
    overlap_tokens: int,
    tokenizer: Tokenizer,
) -> list[Chunk]:
    """Build section-local, span-identified chunks from canonical elements."""
    count = tokenizer.count_tokens
    elements = list(elements)
    section_by_id = {section.id: section for section in sections}
    eligible = [
        element
        for element in elements
        if element.element_type in _TEXT_TYPES
        and _element_text(element).strip()
    ]
    grouped: dict[str | None, list[Element]] = {}
    for element in eligible:
        grouped.setdefault(element.section_id, []).append(element)

    section_spans: dict[str | None, list[ElementSpan]] = {
        section_id: [] for section_id in [None, *(section.id for section in sections)]
    }
    section_order: list[str | None] = [None, *(section.id for section in sections)]
    for section_id in section_order:
        rows = sorted(
            grouped.get(section_id, []),
            key=lambda element: element.order,
        )
        for element in rows:
            section = section_by_id.get(section_id) if section_id else None
            text = _element_text(element)
            section_spans[section_id].extend(
                _element_spans(
                    element,
                    text,
                    count,
                    target_tokens,
                    section_id=section_id,
                    section_path=section.path if section else None,
                )
            )

    built: list[Chunk] = []
    for section_id in section_order:
        chunks = _pack_section(
            section_spans[section_id],
            document_id=document_id,
            snapshot_id=snapshot_id,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            count=count,
            start_order=len(built),
        )
        built.extend(chunks)
    return [
        chunk.model_copy(
            update={
                "previous_chunk_id": built[index - 1].id if index else None,
                "next_chunk_id": (
                    built[index + 1].id if index + 1 < len(built) else None
                ),
            }
        )
        for index, chunk in enumerate(built)
    ]


def _pack_section(
    spans: list[ElementSpan],
    *,
    document_id: str,
    snapshot_id: str,
    target_tokens: int,
    overlap_tokens: int,
    count: Any,
    start_order: int,
) -> list[Chunk]:
    """Pack one section's spans into chunks, never crossing the section."""
    built: list[Chunk] = []
    pending: list[ElementSpan] = []
    pending_overlap_spans: list[ElementSpan] = []
    pending_overlap_source: list[str] = []
    overlap_candidate: list[ElementSpan] = []
    overlap_source_candidate: list[str] = []
    def flush() -> None:
        if not pending:
            return
        built.append(
            _make_chunk(
                document_id=document_id,
                snapshot_id=snapshot_id,
                order=start_order + len(built),
                spans=list(pending),
                count=count,
                overlap_source_chunk_ids=pending_overlap_source,
                overlap_spans=pending_overlap_spans,
            )
        )

    for span in spans:
        if pending and pending[0].section_id != span.section_id:
            flush()
            pending = []
            pending_overlap_spans = []
            pending_overlap_source = []
            overlap_candidate = []
            overlap_source_candidate = []
        if pending and _tokens([*pending, span], count) > target_tokens:
            flushed = list(pending)
            flush()
            pending = []
            pending_overlap_spans = []
            pending_overlap_source = []
            overlap_candidate = _overlap_tail(
                flushed, count, overlap_tokens
            )
            overlap_source_candidate = (
                [built[-1].id] if overlap_candidate else []
            )
        if not pending and overlap_candidate:
            while overlap_candidate and _tokens([*overlap_candidate, span], count) > target_tokens:
                overlap_candidate.pop(0)
            if overlap_candidate:
                pending = list(overlap_candidate)
                pending_overlap_spans = list(overlap_candidate)
                pending_overlap_source = overlap_source_candidate
            overlap_candidate = []
            overlap_source_candidate = []
        pending.append(span)
    flush()
    return built


def _tokens(spans: list[ElementSpan], count: Any) -> int:
    return _count_tokens(count, _join_span_text(spans))


def _element_text(element: Element) -> str:
    """Select verbatim canonical text for element-local span coordinates."""
    if element.element_type == ElementType.TABLE:
        return (
            element.markdown
            or element.text
            or element.html
            or ""
        )
    if element.element_type == ElementType.FORMULA:
        return (
            element.latex
            or element.text
            or element.markdown
            or ""
        )
    return element.text if element.text is not None else (element.markdown or "")


def _element_spans(
    element: Element,
    text: str,
    count: Any,
    target_tokens: int,
    *,
    section_id: str | None,
    section_path: str | None,
) -> list[ElementSpan]:
    units = _split_units(text, count, target_tokens)
    return [
        ElementSpan(
            element_id=element.id,
            span_key=f"{unit.start}:{unit.end}",
            text=unit.text,
            tokens=_count_tokens(count, unit.text),
            character_start_in_element=unit.start,
            character_end_in_element=unit.end,
            token_start=_count_tokens(count, text[: unit.start]),
            token_end=_count_tokens(count, text[: unit.end]),
            page=element.page,
            bounding_box=element.bounding_box,
            section_id=section_id,
            section_path=section_path,
        )
        for unit in units
    ]


def _split_units(
    text: str,
    count: Any,
    target_tokens: int,
) -> list[_TextRange]:
    """Recursively split text until every unit fits the token budget."""
    return _split_range(text, 0, len(text), count, target_tokens)


def _split_range(
    source: str,
    start: int,
    end: int,
    count: Any,
    target_tokens: int,
) -> list[_TextRange]:
    value = source[start:end]
    if _count_tokens(count, value) <= target_tokens:
        return [_TextRange(text=value, start=start, end=end)]
    for pattern in (_PARAGRAPH_SPLIT, _SENTENCE_SPLIT, re.compile(r"\s+")):
        ranges = _partition_ranges(source, start, end, pattern)
        if len(ranges) > 1:
            result: list[_TextRange] = []
            for part_start, part_end in ranges:
                result.extend(_split_range(source, part_start, part_end, count, target_tokens))
            return result
    midpoint = start + max(1, (end - start) // 2)
    if midpoint >= end:
        return [_TextRange(text=value, start=start, end=end)]
    return [
        *_split_range(source, start, midpoint, count, target_tokens),
        *_split_range(source, midpoint, end, count, target_tokens),
    ]


def _partition_ranges(
    source: str, start: int, end: int, pattern: re.Pattern[str]
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = start
    for match in pattern.finditer(source, start, end):
        boundary = match.end()
        if boundary > cursor:
            ranges.append((cursor, boundary))
            cursor = boundary
    if cursor < end:
        ranges.append((cursor, end))
    return ranges if len(ranges) > 1 else [(start, end)]


def _overlap_tail(
    spans: list[ElementSpan],
    count: Any,
    overlap_tokens: int,
) -> list[ElementSpan]:
    """Return whole trailing spans that fit the overlap budget (exact sources)."""
    if overlap_tokens <= 0 or not spans:
        return []
    tail: list[ElementSpan] = []
    for span in reversed(spans):
        candidate = [span, *tail]
        if _tokens(candidate, count) > overlap_tokens:
            break
        tail.insert(0, span)
    return tail


def _make_chunk(
    *,
    document_id: str,
    snapshot_id: str,
    order: int,
    spans: list[ElementSpan],
    count: Any,
    overlap_source_chunk_ids: list[str],
    overlap_spans: list[ElementSpan],
) -> Chunk:
    text = _join_span_text(spans)
    pages = [span.page for span in spans if span.page is not None]
    sections = [span.section_id for span in spans if span.section_id is not None]
    section_paths = [
        span.section_path for span in spans if span.section_path is not None
    ]
    element_ids = list(dict.fromkeys(span.element_id for span in spans))
    span_ids = [span.span_id for span in spans]
    identifier = chunk_id(document_id, order, span_ids)
    return Chunk(
        id=identifier,
        document_id=document_id,
        canonical_snapshot_id=snapshot_id,
        text=text,
        order=order,
        element_ids=element_ids,
        element_span_ids=span_ids,
        spans=[
            ChunkSpan(
                id=span.span_id,
                element_id=span.element_id,
                text=span.text,
                character_start_in_element=span.character_start_in_element,
                character_end_in_element=span.character_end_in_element,
                token_start=span.token_start,
                token_end=span.token_end,
            )
            for span in spans
        ],
        section_id=sections[0] if sections else None,
        section_path=section_paths[0] if section_paths else None,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        bounding_box=_merge_boxes(
            [span.bounding_box for span in spans if span.bounding_box is not None]
        ),
        token_count=_count_tokens(count, text),
        overlap_source_chunk_ids=overlap_source_chunk_ids,
        overlap_element_span_ids=[span.span_id for span in overlap_spans],
    )


def _merge_boxes(
    boxes: list[tuple[float, float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _join_span_text(spans: list[ElementSpan]) -> str:
    if not spans:
        return ""
    parts = [spans[0].text]
    for previous, current in pairwise(spans):
        contiguous = (
            previous.element_id == current.element_id
            and previous.character_end_in_element == current.character_start_in_element
        )
        if not contiguous:
            parts.append("\n\n")
        parts.append(current.text)
    return "".join(parts)


def _count_tokens(count: Any, text: str) -> int:
    return count(text) if text else 0
