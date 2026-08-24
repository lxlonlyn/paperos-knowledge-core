"""Sentence-level units for deterministic section-local DP chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol

from paperos_core.domain.canonical import Element, Section
from paperos_core.domain.enums import ElementType

from paperos_core.ingestion.inline_domains import (
    scan_inline_domains,
    sentence_boundary_allowed,
)

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?。！？；;])\s+")
_ET_AL_BOUNDARY = re.compile(r"\bet\s+al\.\s+", re.IGNORECASE)
_FORMULA_LEAD = re.compile(
    r"(?:"
    r"as\s+follows|"
    r"(?:is|are|was|were)\s+(?:defined|given|expressed)(?:\s+as)?|"
    r"can\s+be\s+written(?:\s+as)?|"
    r"we\s+(?:obtain|have|define)"
    r")\s*[.:]?\s*$",
    re.IGNORECASE,
)
_FORMULA_CONTINUATION = re.compile(
    r"^(?:where|with|in\s+which|here|such\s+that)\b",
    re.IGNORECASE,
)
_REAL_EMERGENCY_SPLIT_TYPES = frozenset(
    {"EMERGENCY_WHITESPACE", "EMERGENCY_TOKEN_SAFE", "EMERGENCY_FORCED"}
)

SplitType = Literal[
    "NORMAL",
    "EMERGENCY_WHITESPACE",
    "EMERGENCY_TOKEN_SAFE",
    "EMERGENCY_FORCED",
    "TABLE_PART",
]


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
    split_type: SplitType = "NORMAL"
    display_text: str | None = None
    paragraph_id: str | None = None

    @property
    def span_id(self) -> str:
        return f"{self.element_id}:{self.span_key}"

    @property
    def emergency_split(self) -> bool:
        return self.split_type in _REAL_EMERGENCY_SPLIT_TYPES


def formula_cohesion_boundary(left: SentenceUnit, right: SentenceUnit) -> bool:
    """Whether a DP break would separate formula-dependent prose."""
    if left.unit_kind != "formula" and right.unit_kind != "formula":
        return False
    if right.unit_kind == "formula" and left.unit_kind == "sentence":
        lead = left.text.rstrip()
        return lead.endswith(":") or bool(_FORMULA_LEAD.search(lead))
    if left.unit_kind == "formula" and right.unit_kind == "sentence":
        continuation = right.text.lstrip()
        if _FORMULA_CONTINUATION.match(continuation):
            return True
        first_alpha = next((char for char in continuation if char.isalpha()), None)
        return first_alpha is not None and first_alpha.islower()
    return False


@dataclass(frozen=True, slots=True)
class _TextRange:
    text: str
    start: int
    end: int
    split_type: SplitType = "NORMAL"


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
                paragraph_id=f"{element.id}:0",
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
                paragraph_id=_paragraph_id_for_offset(text, unit_range.start),
            )
        )
    return units


def _sentence_ranges(text: str, count: Any, hard_max_tokens: int) -> list[_TextRange]:
    domains = scan_inline_domains(text)
    paragraphs = _split_paragraphs(text)
    ranges: list[_TextRange] = []
    for paragraph_start, paragraph_end in paragraphs:
        paragraph = text[paragraph_start:paragraph_end]
        cursor = paragraph_start
        boundaries = [paragraph_start]
        for match in _SENTENCE_SPLIT.finditer(paragraph):
            boundary = paragraph_start + match.end()
            if sentence_boundary_allowed(boundary, domains) and not _is_et_al_boundary(
                text, boundary
            ):
                boundaries.append(boundary)
        if boundaries[-1] < paragraph_end:
            boundaries.append(paragraph_end)
        if len(boundaries) <= 2 and paragraph.strip():
            piece = text[paragraph_start:paragraph_end]
            ranges.extend(
                _emergency_split(text, paragraph_start, paragraph_end, count, hard_max_tokens)
                if _count(count, piece) > hard_max_tokens
                else [
                    _TextRange(
                        text=piece,
                        start=paragraph_start,
                        end=paragraph_end,
                    )
                ]
            )
            continue
        for start_offset, end_offset in zip(boundaries, boundaries[1:], strict=False):
            if end_offset <= start_offset:
                continue
            piece = text[start_offset:end_offset]
            ranges.extend(
                _emergency_split(text, start_offset, end_offset, count, hard_max_tokens)
                if _count(count, piece) > hard_max_tokens
                else [_TextRange(text=piece, start=start_offset, end=end_offset)]
            )
    return ranges


def _is_et_al_boundary(text: str, boundary: int) -> bool:
    window = text[max(0, boundary - 12) : boundary + 1]
    return bool(_ET_AL_BOUNDARY.search(window))


def _emergency_split(
    source: str,
    start: int,
    end: int,
    count: Any,
    hard_max_tokens: int,
) -> list[_TextRange]:
    value = source[start:end]
    if _count(count, value) <= hard_max_tokens:
        return [_TextRange(text=value, start=start, end=end, split_type="EMERGENCY_FORCED")]

    midpoint = _nearest_whitespace_split(source, start, end, count, hard_max_tokens)
    if midpoint is not None and midpoint > start and midpoint < end:
        left = _emergency_split(source, start, midpoint, count, hard_max_tokens)
        right = _emergency_split(source, midpoint, end, count, hard_max_tokens)
        if left:
            last = left[-1]
            left[-1] = _TextRange(
                text=last.text,
                start=last.start,
                end=last.end,
                split_type="EMERGENCY_WHITESPACE",
            )
        if right:
            first = right[0]
            right[0] = _TextRange(
                text=first.text,
                start=first.start,
                end=first.end,
                split_type="EMERGENCY_WHITESPACE",
            )
        return [*left, *right]

    forced_mid = start + max(1, (end - start) // 2)
    left = _emergency_split(source, start, forced_mid, count, hard_max_tokens)
    right = _emergency_split(source, forced_mid, end, count, hard_max_tokens)
    if left:
        last = left[-1]
        left[-1] = _TextRange(
            text=last.text,
            start=last.start,
            end=last.end,
            split_type="EMERGENCY_FORCED",
        )
    if right:
        first = right[0]
        right[0] = _TextRange(
            text=first.text,
            start=first.start,
            end=first.end,
            split_type="EMERGENCY_FORCED",
        )
    return [*left, *right]


def _nearest_whitespace_split(
    source: str,
    start: int,
    end: int,
    count: Any,
    hard_max_tokens: int,
) -> int | None:
    target = start + (end - start) // 2
    best: int | None = None
    for index in range(target, end):
        if source[index].isspace() and _count(count, source[start:index]) <= hard_max_tokens:
            best = index
            break
    for index in range(target, start, -1):
        if source[index - 1].isspace() and _count(count, source[start:index]) <= hard_max_tokens:
            return index
    return best


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
                _TextRange(text=text, start=0, end=len(text), split_type="TABLE_PART"),
                count=count,
                section_id=section_id,
                section_path=section_path,
                paragraph_end=True,
                subsection_end=subsection_end,
                unit_kind="table",
            )
        ]
    lines = _line_ranges(text)
    if len(lines) <= 2:
        ranges = _sentence_ranges(text, count, hard_max_tokens)
        return [
            _unit_from_range(
                element,
                item,
                count=count,
                section_id=section_id,
                section_path=section_path,
                paragraph_end=index == len(ranges) - 1,
                subsection_end=subsection_end and index == len(ranges) - 1,
                unit_kind="table_part",
            )
            for index, item in enumerate(ranges)
        ]

    header_end = lines[1][1]
    header_text = text[:header_end].rstrip("\r\n")
    source_ranges: list[tuple[int, int]] = []
    batch_start = 0
    batch_end = header_end
    for row_start, row_end in lines[2:]:
        candidate_display = _table_display_text(
            text, start=batch_start, end=row_end, header_text=header_text
        )
        if batch_end > batch_start and _count(count, candidate_display) > hard_max_tokens:
            source_ranges.append((batch_start, batch_end))
            batch_start = row_start
        batch_end = row_end
    if batch_end > batch_start:
        source_ranges.append((batch_start, batch_end))

    units: list[SentenceUnit] = []
    for start, end in source_ranges:
        display_text = _table_display_text(
            text, start=start, end=end, header_text=header_text
        )
        units.append(
            _unit_from_range(
                element,
                _TextRange(
                    text=text[start:end],
                    start=start,
                    end=end,
                    split_type="TABLE_PART",
                ),
                count=count,
                section_id=section_id,
                section_path=section_path,
                paragraph_end=False,
                subsection_end=False,
                unit_kind="table_part",
                display_text=display_text if start > 0 else None,
            )
        )
    if units:
        units[-1] = replace(
            units[-1], paragraph_end=True, subsection_end=subsection_end
        )
    return units


def _line_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for line in text.splitlines(keepends=True):
        end = cursor + len(line)
        ranges.append((cursor, end))
        cursor = end
    if cursor < len(text):
        ranges.append((cursor, len(text)))
    return ranges


def _table_display_text(
    source: str,
    *,
    start: int,
    end: int,
    header_text: str,
) -> str:
    authoritative = source[start:end]
    if start == 0:
        return authoritative
    return f"{header_text}\n{authoritative}"


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
    paragraph_id: str | None = None,
    display_text: str | None = None,
) -> SentenceUnit:
    full_text = element_text(element)
    return SentenceUnit(
        text=unit_range.text,
        tokens=_count(count, display_text or unit_range.text),
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
        split_type=unit_range.split_type,
        display_text=display_text,
        paragraph_id=paragraph_id,
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


def _paragraph_id_for_offset(text: str, offset: int) -> str:
    paragraph_index = 0
    for paragraph_start, paragraph_end in _split_paragraphs(text):
        if paragraph_start <= offset < paragraph_end:
            return f"p{paragraph_index}"
        paragraph_index += 1
    return f"p{paragraph_index}"


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
