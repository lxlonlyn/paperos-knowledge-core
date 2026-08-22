"""Sentence-level units for deterministic section-local DP chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from paperos_core.domain.canonical import Element, Section
from paperos_core.domain.enums import ElementType

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？；;])\s+")


class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class SentenceUnit:
    text: str
    tokens: int
    element_id: str
    span_key: str
    character_start_in_element: int
    character_end_in_element: int
    token_start: int
    token_end: int
    section_id: str | None
    section_path: str | None
    page: int | None
    bounding_box: tuple[float, float, float, float] | None
    paragraph_end: bool
    subsection_end: bool
    unit_kind: str = "sentence"
    emergency_split: bool = False

    @property
    def span_id(self) -> str:
        return f"{self.element_id}:{self.span_key}"


@dataclass(frozen=True, slots=True)
class _TextRange:
    text: str
    start: int
    end: int


_PROSE_TYPES = {
    ElementType.TITLE,
    ElementType.PARAGRAPH,
    ElementType.LIST,
    ElementType.LIST_ITEM,
    ElementType.CAPTION,
    ElementType.FOOTNOTE,
}


def element_text(element: Element) -> str:
    if element.element_type == ElementType.TABLE:
        return element.markdown or element.text or element.html or ""
    if element.element_type == ElementType.FORMULA:
        return element.latex or element.text or element.markdown or ""
    return element.text if element.text is not None else (element.markdown or "")


def units_for_element(
    element: Element,
    *,
    count: Any,
    hard_max_tokens: int,
    section_id: str | None,
    section_path: str | None,
    subsection_end: bool,
) -> list[SentenceUnit]:
    text = element_text(element)
    if not text.strip():
        return []
    if element.element_type == ElementType.TABLE:
        return _table_units(
            element,
            text,
            count=count,
            hard_max_tokens=hard_max_tokens,
            section_id=section_id,
            section_path=section_path,
            subsection_end=subsection_end,
        )
    if element.element_type == ElementType.FORMULA:
        tokens = _count(count, text)
        return [
            _unit_from_range(
                element,
                _TextRange(text=text, start=0, end=len(text)),
                count=count,
                section_id=section_id,
                section_path=section_path,
                paragraph_end=True,
                subsection_end=subsection_end,
                unit_kind="formula",
            )
        ]
    ranges = _sentence_ranges(text, count, hard_max_tokens)
    units: list[SentenceUnit] = []
    for index, unit_range in enumerate(ranges):
        paragraph_end = _ends_paragraph(text, unit_range.end)
        units.append(
            _unit_from_range(
                element,
                unit_range,
                count=count,
                section_id=section_id,
                section_path=section_path,
                paragraph_end=paragraph_end,
                subsection_end=subsection_end and index == len(ranges) - 1,
                unit_kind="sentence",
                emergency_split=_count(count, unit_range.text) > hard_max_tokens,
            )
        )
    return units


def _sentence_ranges(text: str, count: Any, hard_max_tokens: int) -> list[_TextRange]:
    paragraphs = _split_paragraphs(text)
    ranges: list[_TextRange] = []
    for paragraph_start, paragraph_end in paragraphs:
        paragraph = text[paragraph_start:paragraph_end]
        if _count(count, paragraph) <= hard_max_tokens:
            ranges.append(_TextRange(text=paragraph, start=paragraph_start, end=paragraph_end))
            continue
        cursor = paragraph_start
        for match in _SENTENCE_SPLIT.finditer(paragraph):
            boundary = paragraph_start + match.end()
            if boundary > cursor:
                piece = text[cursor:boundary]
                ranges.extend(
                    _emergency_split(text, cursor, boundary, count, hard_max_tokens)
                    if _count(count, piece) > hard_max_tokens
                    else [_TextRange(text=piece, start=cursor, end=boundary)]
                )
                cursor = boundary
        if cursor < paragraph_end:
            piece = text[cursor:paragraph_end]
            ranges.extend(
                _emergency_split(text, cursor, paragraph_end, count, hard_max_tokens)
                if _count(count, piece) > hard_max_tokens
                else [_TextRange(text=piece, start=cursor, end=paragraph_end)]
            )
    return ranges


def _emergency_split(
    source: str,
    start: int,
    end: int,
    count: Any,
    hard_max_tokens: int,
) -> list[_TextRange]:
    value = source[start:end]
    if _count(count, value) <= hard_max_tokens:
        return [_TextRange(text=value, start=start, end=end)]
    midpoint = start + max(1, (end - start) // 2)
    return [
        *_emergency_split(source, start, midpoint, count, hard_max_tokens),
        *_emergency_split(source, midpoint, end, count, hard_max_tokens),
    ]


def _table_units(
    element: Element,
    text: str,
    *,
    count: Any,
    hard_max_tokens: int,
    section_id: str | None,
    section_path: str | None,
    subsection_end: bool,
) -> list[SentenceUnit]:
    if _count(count, text) <= hard_max_tokens:
        return [
            _unit_from_range(
                element,
                _TextRange(text=text, start=0, end=len(text)),
                count=count,
                section_id=section_id,
                section_path=section_path,
                paragraph_end=True,
                subsection_end=subsection_end,
                unit_kind="table",
            )
        ]
    lines = text.splitlines()
    if len(lines) <= 2:
        return _sentence_ranges(text, count, hard_max_tokens)  # fallback
    header = lines[:2]
    body = lines[2:]
    units: list[SentenceUnit] = []
    batch: list[str] = list(header)
    cursor = 0
    for row in body:
        candidate = "\n".join([*batch, row])
        if batch and _count(count, candidate) > hard_max_tokens:
            block = "\n".join(batch)
            units.append(
                _unit_from_range(
                    element,
                    _TextRange(text=block, start=cursor, end=cursor + len(block)),
                    count=count,
                    section_id=section_id,
                    section_path=section_path,
                    paragraph_end=False,
                    subsection_end=False,
                    unit_kind="table_part",
                )
            )
            cursor += len(block) + 1
            batch = [*header, row]
        else:
            batch.append(row)
    if batch:
        block = "\n".join(batch)
        units.append(
            _unit_from_range(
                element,
                _TextRange(text=block, start=cursor, end=cursor + len(block)),
                count=count,
                section_id=section_id,
                section_path=section_path,
                paragraph_end=True,
                subsection_end=subsection_end,
                unit_kind="table_part",
            )
        )
    return units


def _unit_from_range(
    element: Element,
    unit_range: _TextRange,
    *,
    count: Any,
    section_id: str | None,
    section_path: str | None,
    paragraph_end: bool,
    subsection_end: bool,
    unit_kind: str,
    emergency_split: bool = False,
) -> SentenceUnit:
    full_text = element_text(element)
    return SentenceUnit(
        text=unit_range.text,
        tokens=_count(count, unit_range.text),
        element_id=element.id,
        span_key=f"{unit_range.start}:{unit_range.end}",
        character_start_in_element=unit_range.start,
        character_end_in_element=unit_range.end,
        token_start=_count(count, full_text[: unit_range.start]),
        token_end=_count(count, full_text[: unit_range.end]),
        page=element.page,
        bounding_box=element.bounding_box,
        section_id=section_id,
        section_path=section_path,
        paragraph_end=paragraph_end,
        subsection_end=subsection_end,
        unit_kind=unit_kind,
        emergency_split=emergency_split,
    )


def _split_paragraphs(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for match in _PARAGRAPH_SPLIT.finditer(text):
        boundary = match.start()
        if boundary > cursor:
            ranges.append((cursor, boundary))
        cursor = match.end()
    if cursor < len(text):
        ranges.append((cursor, len(text)))
    return ranges or [(0, len(text))]


def _ends_paragraph(text: str, end: int) -> bool:
    tail = text[end:].lstrip()
    return not tail or tail.startswith("\n\n")


def _count(count: Any, text: str) -> int:
    return count(text) if text else 0


def resolve_major_section_id(
    section_id: str | None,
    section_by_id: dict[str, Section],
) -> str | None:
    if section_id is None:
        return "__unsectioned__"
    section = section_by_id.get(section_id)
    if section is None:
        return "__unsectioned__"
    if section.section_type == "references":
        return None
    chain: list[Section] = []
    current: Section | None = section
    while current is not None:
        chain.append(current)
        current = (
            section_by_id.get(current.parent_section_id)
            if current.parent_section_id
            else None
        )
    for candidate in reversed(chain):
        if candidate.level == 1 or candidate.section_type == "abstract":
            return candidate.id
    return chain[-1].id if chain else None
