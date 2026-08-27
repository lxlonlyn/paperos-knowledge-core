"""Sentence-level units for deterministic section-local DP chunking."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import pairwise
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
    {
        "EMERGENCY_PUNCTUATION",
        "EMERGENCY_WHITESPACE",
        "EMERGENCY_TOKEN_SAFE",
        "EMERGENCY_PROTECTED_DOMAIN",
        "EMERGENCY_FORCED",
    }
)

SplitType = Literal[
    "NORMAL",
    "EMERGENCY_PUNCTUATION",
    "EMERGENCY_WHITESPACE",
    "EMERGENCY_TOKEN_SAFE",
    "EMERGENCY_PROTECTED_DOMAIN",
    "EMERGENCY_FORCED",
    "TABLE_PART",
    "FIGURE_PLACEHOLDER",
    "FIGURE_PART",
]

_FALLBACK_PUNCTUATION = frozenset(".!?;:。！？；：,，、")
_FIGURE_ALT_KEYS = ("alt", "alt_text", "description")


class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class SupplementalSpan:
    text: str
    element_id: str
    span_key: str
    character_start_in_element: int
    character_end_in_element: int
    token_start: int
    token_end: int
    source_field: str

    @property
    def span_id(self) -> str:
        return f"{self.element_id}:{self.span_key}"


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
    provenance_kind: Literal["source", "projection"] = "source"
    source_field: str | None = None
    unit_kind: str = "sentence"
    split_type: SplitType = "NORMAL"
    fallback_reason: str | None = None
    display_text: str | None = None
    paragraph_id: str | None = None
    supplemental_spans: tuple[SupplementalSpan, ...] = ()

    @property
    def span_id(self) -> str:
        return f"{self.element_id}:{self.span_key}"

    @property
    def emergency_split(self) -> bool:
        return (
            self.split_type in _REAL_EMERGENCY_SPLIT_TYPES
            or self.fallback_reason in _REAL_EMERGENCY_SPLIT_TYPES
        )


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
    fallback_reason: str | None = None


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


def _element_text_source_field(element: Element) -> str:
    if element.element_type == ElementType.TABLE:
        if element.markdown is not None:
            return "markdown"
        if element.text is not None:
            return "text"
        return "html"
    if element.element_type == ElementType.FORMULA:
        if element.latex is not None:
            return "latex"
        if element.text is not None:
            return "text"
        return "markdown"
    return "text" if element.text is not None else "markdown"


@dataclass(frozen=True, slots=True)
class _FigureTextSource:
    element: Element
    source_field: str
    source_text: str
    source_start: int
    text: str


@dataclass(frozen=True, slots=True)
class _FigureTextRange:
    source: _FigureTextSource
    description_start: int
    description_end: int


def figure_caption_element_ids(elements: list[Element]) -> set[str]:
    """Return captions consumed by Figure placeholders, never table captions."""
    figures = {
        element.id: element
        for element in elements
        if element.element_type == ElementType.FIGURE
    }
    bound = {
        caption_id
        for figure in figures.values()
        for caption_id in figure.caption_element_ids
    }
    bound.update(
        element.id
        for element in elements
        if element.element_type == ElementType.CAPTION
        and element.parent_element_id in figures
    )
    return bound


def figure_description(
    figure: Element, *, elements_by_id: Mapping[str, Element]
) -> str:
    """Use only canonical caption/alt text; never synthesize a description."""
    return "\n".join(
        source.text
        for source in _figure_text_sources(
            figure, elements_by_id=elements_by_id
        )
    )


def _figure_text_sources(
    figure: Element, *, elements_by_id: Mapping[str, Element]
) -> list[_FigureTextSource]:
    captions: list[_FigureTextSource] = []
    for caption_id in figure.caption_element_ids:
        caption = elements_by_id.get(caption_id)
        if caption is None or caption.element_type != ElementType.CAPTION:
            continue
        value = element_text(caption)
        if not value.strip():
            continue
        captions.append(
            _FigureTextSource(
                element=caption,
                source_field=_element_text_source_field(caption),
                source_text=value,
                source_start=0,
                text=value,
            )
        )
    if captions:
        return captions
    for key in _FIGURE_ALT_KEYS:
        alt_value = figure.metadata.get(key)
        if isinstance(alt_value, str) and alt_value.strip():
            stripped = alt_value.strip()
            return [
                _FigureTextSource(
                    element=figure,
                    source_field=f"metadata.{key}",
                    source_text=alt_value,
                    source_start=alt_value.index(stripped),
                    text=stripped,
                )
            ]
    return []


def figure_placeholder(
    figure: Element,
    *,
    elements_by_id: Mapping[str, Element],
    description: str | None = None,
    part: tuple[int, int] | None = None,
) -> str:
    """Render stable, path-free semantic Figure evidence."""
    attributes = [f"id={figure.id}"]
    if figure.page is not None:
        attributes.append(f"page={figure.page}")
    if part is not None:
        attributes.append(f"part={part[0]}/{part[1]}")
    value = (
        figure_description(figure, elements_by_id=elements_by_id)
        if description is None
        else description
    )
    return f"[FIGURE {' '.join(attributes)}]\nDescription: {value}\n[/FIGURE]"


def units_for_element(
    element: Element,
    *,
    count: Any,
    hard_max_tokens: int,
    section_id: str | None,
    section_path: str | None,
    subsection_end: bool,
    elements_by_id: Mapping[str, Element] | None = None,
) -> list[SentenceUnit]:
    if element.element_type == ElementType.FIGURE:
        return _figure_units(
            element,
            elements_by_id=elements_by_id or {element.id: element},
            count=count,
            hard_max_tokens=hard_max_tokens,
            section_id=section_id,
            section_path=section_path,
            subsection_end=subsection_end,
        )
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
        ranges = (
            _emergency_split(text, 0, len(text), count, hard_max_tokens)
            if _count(count, text) > hard_max_tokens
            else [_TextRange(text=text, start=0, end=len(text))]
        )
        return [
            _unit_from_range(
                element,
                item,
                count=count,
                section_id=section_id,
                section_path=section_path,
                paragraph_end=index == len(ranges) - 1,
                subsection_end=subsection_end and index == len(ranges) - 1,
                unit_kind="formula",
                paragraph_id=f"{element.id}:0",
            )
            for index, item in enumerate(ranges)
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
        for start_offset, end_offset in pairwise(boundaries):
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
    return _fallback_ranges(
        source,
        start=start,
        end=end,
        count=count,
        hard_max_tokens=hard_max_tokens,
        render=lambda value: value,
    )


def _fallback_ranges(
    source: str,
    *,
    start: int,
    end: int,
    count: Any,
    hard_max_tokens: int,
    render: Any,
) -> list[_TextRange]:
    """Split exact source ranges, preferring safe punctuation then whitespace."""
    if start >= end:
        return []
    cursor = start
    ranges: list[_TextRange] = []
    last_reason: SplitType = "EMERGENCY_TOKEN_SAFE"
    domains = scan_inline_domains(source)
    while _count(count, render(source[cursor:end])) > hard_max_tokens:
        boundary, reason = _preferred_fallback_boundary(
            source,
            start=cursor,
            end=end,
            domains=domains,
            fits=lambda candidate, current_cursor=cursor: (
                _count(count, render(source[current_cursor:candidate]))
                <= hard_max_tokens
            ),
            protected_domain_oversized=lambda domain: (
                _count(count, source[domain.start : domain.end])
                > hard_max_tokens
            ),
        )
        if boundary <= cursor or boundary >= end:
            raise ValueError("Unable to make progress while enforcing chunk hard max")
        ranges.append(
            _TextRange(
                text=source[cursor:boundary],
                start=cursor,
                end=boundary,
                split_type=reason,
            )
        )
        cursor = boundary
        last_reason = reason
    ranges.append(
        _TextRange(
            text=source[cursor:end],
            start=cursor,
            end=end,
            split_type=last_reason,
        )
    )
    return ranges


def _preferred_fallback_boundary(
    source: str,
    *,
    start: int,
    end: int,
    domains: list[Any],
    fits: Any,
    protected_domain_oversized: Any,
) -> tuple[int, SplitType]:
    def safe(position: int) -> bool:
        return not any(domain.start < position < domain.end for domain in domains)

    punctuation = [
        index + 1
        for index in range(start, end - 1)
        if source[index] in _FALLBACK_PUNCTUATION
        and safe(index + 1)
        and fits(index + 1)
    ]
    if punctuation:
        return max(punctuation), "EMERGENCY_PUNCTUATION"
    whitespace = [
        index + 1
        for index in range(start, end - 1)
        if source[index].isspace() and safe(index + 1) and fits(index + 1)
    ]
    if whitespace:
        return max(whitespace), "EMERGENCY_WHITESPACE"

    low = start + 1
    high = end - 1
    token_safe_limit = start
    while low <= high:
        middle = (low + high) // 2
        if fits(middle):
            token_safe_limit = middle
            low = middle + 1
        else:
            high = middle - 1
    if token_safe_limit == start:
        for candidate in range(start + 1, end):
            if fits(candidate):
                token_safe_limit = candidate
                break
    if token_safe_limit == start:
        raise ValueError("chunk_hard_max_tokens cannot hold a single text unit")
    for candidate in range(token_safe_limit, start, -1):
        if safe(candidate) and fits(candidate):
            return candidate, "EMERGENCY_TOKEN_SAFE"

    containing = next(
        (
            domain
            for domain in domains
            if domain.start <= token_safe_limit < domain.end
            and domain.start <= start
        ),
        None,
    )
    if containing is None:
        raise ValueError("No protected-domain-safe hard-max split boundary exists")
    if not protected_domain_oversized(containing):
        raise ValueError(
            "A protected inline domain that fits chunk_hard_max_tokens "
            "cannot be emitted intact with its structural wrapper"
        )
    return token_safe_limit, "EMERGENCY_PROTECTED_DOMAIN"


def _figure_units(
    element: Element,
    *,
    elements_by_id: Mapping[str, Element],
    count: Any,
    hard_max_tokens: int,
    section_id: str | None,
    section_path: str | None,
    subsection_end: bool,
) -> list[SentenceUnit]:
    description = figure_description(element, elements_by_id=elements_by_id)
    text_ranges = _figure_text_ranges(
        element, elements_by_id=elements_by_id
    )
    whole = figure_placeholder(
        element, elements_by_id=elements_by_id, description=description
    )
    if _count(count, whole) <= hard_max_tokens:
        return [
            _figure_unit(
                element,
                text=whole,
                description=description,
                start=0,
                end=len(description),
                count=count,
                section_id=section_id,
                section_path=section_path,
                subsection_end=subsection_end,
                split_type="FIGURE_PLACEHOLDER",
                part_index=None,
                text_ranges=text_ranges,
            )
        ]
    if not description:
        raise ValueError(
            f"Figure placeholder {element.id} exceeds chunk_hard_max_tokens"
        )

    total = 1
    ranges: list[_TextRange] = []
    for _attempt in range(4):
        ranges = _fallback_ranges(
            description,
            start=0,
            end=len(description),
            count=count,
            hard_max_tokens=hard_max_tokens,
            render=lambda value, expected_total=total: figure_placeholder(
                element,
                elements_by_id=elements_by_id,
                description=value,
                part=(expected_total, expected_total),
            ),
        )
        if len(ranges) == total:
            break
        total = len(ranges)
    total = len(ranges)
    units: list[SentenceUnit] = []
    for index, item in enumerate(ranges, start=1):
        placeholder = figure_placeholder(
            element,
            elements_by_id=elements_by_id,
            description=item.text,
            part=(index, total),
        )
        if _count(count, placeholder) > hard_max_tokens:
            raise ValueError(
                f"Figure part {element.id}:{index} exceeds chunk_hard_max_tokens"
            )
        units.append(
            _figure_unit(
                element,
                text=placeholder,
                description=description,
                start=item.start,
                end=item.end,
                count=count,
                section_id=section_id,
                section_path=section_path,
                subsection_end=subsection_end and index == total,
                split_type=item.split_type,
                part_index=index,
                text_ranges=text_ranges,
            )
        )
    return units


def _figure_unit(
    element: Element,
    *,
    text: str,
    description: str,
    start: int,
    end: int,
    count: Any,
    section_id: str | None,
    section_path: str | None,
    subsection_end: bool,
    split_type: SplitType,
    part_index: int | None,
    text_ranges: list[_FigureTextRange],
) -> SentenceUnit:
    return SentenceUnit(
        text=text,
        tokens=_count(count, text),
        element_id=element.id,
        span_key=(
            f"figure:{start}:{end}"
            if part_index is None
            else f"figure-part-{part_index}:{start}:{end}"
        ),
        character_start_in_element=0,
        character_end_in_element=0,
        token_start=0,
        token_end=0,
        provenance_kind="projection",
        source_field=None,
        page=element.page,
        bounding_box=element.bounding_box,
        section_id=section_id,
        section_path=section_path,
        paragraph_end=True,
        subsection_end=subsection_end,
        unit_kind="figure" if part_index is None else "figure_part",
        split_type=split_type,
        fallback_reason=(
            split_type if split_type in _REAL_EMERGENCY_SPLIT_TYPES else None
        ),
        paragraph_id=f"{element.id}:figure",
        supplemental_spans=tuple(
            _figure_text_supplemental_span(
                text_range.source,
                description=description,
                description_start=text_range.description_start,
                unit_start=start,
                unit_end=end,
                count=count,
            )
            for text_range in text_ranges
            if text_range.description_start < end
            and text_range.description_end > start
        ),
    )


def _figure_text_ranges(
    figure: Element, *, elements_by_id: Mapping[str, Element]
) -> list[_FigureTextRange]:
    ranges: list[_FigureTextRange] = []
    cursor = 0
    for source in _figure_text_sources(
        figure, elements_by_id=elements_by_id
    ):
        if ranges:
            cursor += 1
        start = cursor
        cursor += len(source.text)
        ranges.append(
            _FigureTextRange(
                source=source,
                description_start=start,
                description_end=cursor,
            )
        )
    return ranges


def _figure_text_supplemental_span(
    source: _FigureTextSource,
    *,
    description: str,
    description_start: int,
    unit_start: int,
    unit_end: int,
    count: Any,
) -> SupplementalSpan:
    overlap_start = max(description_start, unit_start)
    overlap_end = min(description_start + len(source.text), unit_end)
    relative_start = overlap_start - description_start
    relative_end = overlap_end - description_start
    source_start = source.source_start + relative_start
    source_end = source.source_start + relative_end
    return SupplementalSpan(
        text=description[overlap_start:overlap_end],
        element_id=source.element.id,
        span_key=(
            f"figure-text:{source.source_field}:{source_start}:{source_end}"
        ),
        character_start_in_element=source_start,
        character_end_in_element=source_end,
        token_start=_count(count, source.source_text[:source_start]),
        token_end=_count(count, source.source_text[:source_end]),
        source_field=source.source_field,
    )


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
    source_ranges: list[tuple[int, int]] = [(0, header_end)]
    batch_start = lines[2][0]
    batch_end = batch_start
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

    hard_safe_ranges: list[_TextRange] = []
    header_prefix = f"{header_text}\n"
    for start, end in source_ranges:
        display_prefix = (
            header_prefix
            if start > 0 and _count(count, header_prefix) < hard_max_tokens
            else ""
        )
        if _count(count, f"{display_prefix}{text[start:end]}") <= hard_max_tokens:
            hard_safe_ranges.append(
                _TextRange(
                    text=text[start:end],
                    start=start,
                    end=end,
                    split_type="TABLE_PART",
                )
            )
            continue
        hard_safe_ranges.extend(
            replace(
                item,
                split_type="TABLE_PART",
                fallback_reason=item.split_type,
            )
            for item in _fallback_ranges(
                text,
                start=start,
                end=end,
                count=count,
                hard_max_tokens=hard_max_tokens,
                render=lambda value, prefix=display_prefix: f"{prefix}{value}",
            )
        )

    units: list[SentenceUnit] = []
    for item in hard_safe_ranges:
        start, end = item.start, item.end
        display_text = _table_display_text(
            text, start=start, end=end, header_text=header_text
        )
        if _count(count, display_text) > hard_max_tokens:
            display_text = text[start:end]
        units.append(
            _unit_from_range(
                element,
                item,
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
        provenance_kind="source",
        source_field=_element_text_source_field(element),
        page=element.page,
        bounding_box=element.bounding_box,
        section_id=section_id,
        section_path=section_path,
        paragraph_end=paragraph_end,
        subsection_end=subsection_end,
        unit_kind=unit_kind,
        split_type=unit_range.split_type,
        fallback_reason=unit_range.fallback_reason,
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
