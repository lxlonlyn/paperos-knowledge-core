"""Structure-aware canonical chunking with real tokenizer and span provenance.

PaperOS owns the academic chunking rules; Cognee's custom pipeline executes
them (``AcademicChunkTask``) and Cognee provides the tokenizer and token
limits. Chunks never cross sections, oversized elements are split into
element-internal spans, tables produce searchable text, formulas carry their
caption/section context, and every chunk records the exact element spans it
covers (including any spans re-included as overlap).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from paperos_core.domain.canonical import Chunk, Element, Section
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
    page: int | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    section_id: str | None = None
    section_path: str | None = None

    @property
    def span_id(self) -> str:
        return f"{self.element_id}:{self.span_key}"


def resolve_cognee_tokenizer() -> Any:
    """Resolve the tokenizer Cognee would use for the configured embedding model."""
    from cognee.infrastructure.databases.vector.embeddings.config import (  # type: ignore[import-untyped]
        get_embedding_config,
    )
    from cognee.infrastructure.llm.tokenizer.resolver import (  # type: ignore[import-untyped]
        resolve_embedding_tokenizer,
    )

    config = get_embedding_config()
    return resolve_embedding_tokenizer(
        provider=config.embedding_provider,
        model=config.embedding_model,
        max_completion_tokens=config.embedding_max_completion_tokens,
        huggingface_tokenizer=config.huggingface_tokenizer,
    )


def build_chunks(
    *,
    document_id: str,
    snapshot_id: str,
    sections: list[Section],
    elements: Iterable[Element],
    target_tokens: int,
    overlap_tokens: int,
    tokenizer: Tokenizer | None = None,
) -> list[Chunk]:
    """Build section-local, span-identified chunks from canonical elements."""
    count = tokenizer.count_tokens if tokenizer is not None else resolve_cognee_tokenizer().count_tokens
    elements = list(elements)
    section_by_id = {section.id: section for section in sections}
    elements_by_id = {element.id: element for element in elements}
    captured_captions = {
        caption_id
        for element in elements
        if element.element_type == ElementType.FORMULA
        for caption_id in element.caption_element_ids
    }
    eligible = [
        element
        for element in elements
        if element.element_type in _TEXT_TYPES
        and not (
            element.element_type == ElementType.CAPTION
            and element.id in captured_captions
        )
        and _element_text(element, elements_by_id, section_by_id.get(element.section_id)).strip()
    ]
    grouped: dict[str | None, list[Element]] = {}
    for element in eligible:
        grouped.setdefault(element.section_id, []).append(element)

    separator_tokens = max(1, count("\n\n"))
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
            text = _element_text(
                element,
                elements_by_id,
                section.title if section else None,
            )
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
    character_cursor = 0
    for section_id in section_order:
        chunks, character_cursor = _pack_section(
            section_spans[section_id],
            document_id=document_id,
            snapshot_id=snapshot_id,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
            count=count,
            separator_tokens=separator_tokens,
            start_order=len(built),
            character_cursor=character_cursor,
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
    separator_tokens: int,
    start_order: int,
    character_cursor: int,
) -> tuple[list[Chunk], int]:
    """Pack one section's spans into chunks, never crossing the section."""
    built: list[Chunk] = []
    pending: list[ElementSpan] = []
    pending_overlap_spans: list[ElementSpan] = []
    pending_overlap_source: list[str] = []
    overlap_candidate: list[ElementSpan] = []
    overlap_source_candidate: list[str] = []
    cursor = character_cursor

    def flush() -> None:
        nonlocal cursor
        if not pending:
            return
        built.append(
            _make_chunk(
                document_id=document_id,
                snapshot_id=snapshot_id,
                order=start_order + len(built),
                spans=list(pending),
                count=count,
                separator_tokens=separator_tokens,
                character_cursor=cursor,
                overlap_source_chunk_ids=pending_overlap_source,
                overlap_spans=pending_overlap_spans,
            )
        )
        cursor += _text_length(pending) + 2

    for span in spans:
        if pending and _tokens([*pending, span], separator_tokens) > target_tokens:
            flushed = list(pending)
            flush()
            pending = []
            pending_overlap_spans = []
            pending_overlap_source = []
            overlap_candidate = _overlap_tail(
                flushed, count, overlap_tokens, separator_tokens
            )
            overlap_source_candidate = (
                [built[-1].id] if overlap_candidate else []
            )
        if not pending and overlap_candidate:
            # Carry overlap only when the incoming span still fits the limit;
            # otherwise drop it instead of emitting a duplicated chunk or
            # crossing the token budget.
            if _tokens([*overlap_candidate, span], separator_tokens) <= target_tokens:
                pending = list(overlap_candidate)
                pending_overlap_spans = list(overlap_candidate)
                pending_overlap_source = overlap_source_candidate
            overlap_candidate = []
            overlap_source_candidate = []
        pending.append(span)
    flush()
    return built, cursor


def _tokens(spans: list[ElementSpan], separator_tokens: int) -> int:
    if not spans:
        return 0
    return sum(span.tokens for span in spans) + (
        separator_tokens * (len(spans) - 1)
    )


def _element_text(
    element: Element,
    elements_by_id: dict[str, Element],
    section_title: str | None,
) -> str:
    if element.element_type == ElementType.TABLE:
        body = (
            element.markdown
            or element.text
            or element.html
            or ""
        ).strip()
        return f"[Table]\n{body}" if body else ""
    if element.element_type == ElementType.FORMULA:
        body = (
            element.latex
            or element.text
            or element.markdown
            or ""
        ).strip()
        if not body:
            return ""
        parts: list[str] = []
        if section_title:
            parts.append(f"Section: {section_title}")
        parts.append(f"[Formula]\n{body}")
        for caption_id in element.caption_element_ids:
            caption = elements_by_id.get(caption_id)
            caption_text = (
                (caption.text or caption.markdown or "").strip()
                if caption is not None
                else ""
            )
            if caption_text:
                parts.append(f"Caption: {caption_text}")
        return "\n".join(parts)
    return (element.text or element.markdown or "").strip()


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
            span_key=str(index),
            text=unit,
            tokens=count(unit),
            page=element.page,
            bounding_box=element.bounding_box,
            section_id=section_id,
            section_path=section_path,
        )
        for index, unit in enumerate(units)
    ]


def _split_units(
    text: str,
    count: Any,
    target_tokens: int,
) -> list[str]:
    """Recursively split text until every unit fits the token budget."""
    if count(text) <= target_tokens:
        return [text]
    for splitter in (_split_paragraphs, _split_sentences, _split_words, _split_chars):
        parts = splitter(text)
        if len(parts) > 1:
            units: list[str] = []
            for part in parts:
                units.extend(_split_units(part, count, target_tokens))
            return units
    return [text]


def _split_paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in _PARAGRAPH_SPLIT.split(text) if part.strip()]
    return parts if len(parts) > 1 else [text]


def _split_sentences(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    return parts if len(parts) > 1 else [text]


def _split_words(text: str) -> list[str]:
    parts = [part for part in re.split(r"\s+", text) if part]
    return parts if len(parts) > 1 else [text]


def _split_chars(text: str) -> list[str]:
    width = max(200, len(text) // 2)
    parts = [text[index : index + width] for index in range(0, len(text), width)]
    return parts if len(parts) > 1 else [text]


def _overlap_tail(
    spans: list[ElementSpan],
    count: Any,
    overlap_tokens: int,
    separator_tokens: int,
) -> list[ElementSpan]:
    """Return whole trailing spans that fit the overlap budget (exact sources)."""
    if overlap_tokens <= 0 or not spans:
        return []
    tail: list[ElementSpan] = []
    total = 0
    for span in reversed(spans):
        candidate = total + span.tokens + (separator_tokens if tail else 0)
        if candidate > overlap_tokens:
            break
        tail.insert(0, span)
        total = candidate
    return tail


def _make_chunk(
    *,
    document_id: str,
    snapshot_id: str,
    order: int,
    spans: list[ElementSpan],
    count: Any,
    separator_tokens: int,
    character_cursor: int,
    overlap_source_chunk_ids: list[str],
    overlap_spans: list[ElementSpan],
) -> Chunk:
    text = "\n\n".join(span.text for span in spans)
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
        section_id=sections[0] if sections else None,
        section_path=section_paths[0] if section_paths else None,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        bounding_box=_merge_boxes(
            [span.bounding_box for span in spans if span.bounding_box is not None]
        ),
        token_count=count(text),
        character_start=character_cursor,
        character_end=character_cursor + len(text),
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


def _text_length(spans: list[ElementSpan]) -> int:
    return sum(len(span.text) for span in spans) + (2 * max(0, len(spans) - 1))
