"""Structure-aware canonical chunking with major-section DP and span provenance.

PaperOS owns academic chunking rules. Chunks never cross major sections,
sentence-level DP chooses boundaries inside each section, and authoritative
``Chunk.text`` stays separate from rebuildable ``retrieval_text``.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise
from typing import Any, Protocol

from paperos_core.domain.canonical import (
    Chunk,
    ChunkSpan,
    CitationMention,
    Document,
    Element,
    ReferenceEntry,
    Section,
)
from paperos_core.domain.enums import ElementType
from paperos_core.domain.ids import chunk_id
from paperos_core.ingestion.chunk_dp import partition_units
from paperos_core.ingestion.citations import (
    attach_mentions_to_chunks,
    build_scoped_reference_indexes,
    extract_citation_mentions_from_text,
)
from paperos_core.ingestion.bibliography_scope import scope_for_element
from paperos_core.ingestion.document_regions import (
    build_document_regions,
    region_for_element,
    region_id_for_element,
)
from paperos_core.ingestion.retrieval_text import build_retrieval_text
from paperos_core.ingestion.chunk_eligibility import classify_chunk_eligibility
from paperos_core.ingestion.sentence_units import (
    SentenceUnit,
    _PROSE_TYPES,
    element_text,
    resolve_major_section_id,
    units_for_element,
)


class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...


def build_chunks(
    *,
    document: Document,
    snapshot_id: str,
    sections: list[Section],
    elements: Iterable[Element],
    references: list[ReferenceEntry],
    target_tokens: int,
    hard_max_tokens: int,
    overlap_tokens: int,
    tokenizer: Tokenizer,
) -> tuple[list[Chunk], list[CitationMention]]:
    """Build section-local, span-identified chunks from canonical elements."""
    count = tokenizer.count_tokens
    elements = list(elements)
    section_by_id = {section.id: section for section in sections}
    document_regions, element_regions = build_document_regions(
        elements=elements, sections=sections
    )
    reference_indexes = build_scoped_reference_indexes(
        references=references,
        elements=list(elements),
        sections=sections,
    )
    references_by_id = {reference.id: reference for reference in references}

    eligible: list[Element] = []
    for element in elements:
        eligibility = classify_chunk_eligibility(element, section_by_id=section_by_id)
        if not eligibility.eligible:
            continue
        eligible.append(element)

    grouped: dict[tuple[str, str], list[Element]] = {}
    for element in sorted(eligible, key=lambda item: item.order):
        major_id = resolve_major_section_id(element.section_id, section_by_id)
        if major_id is None:
            continue
        region_id = region_id_for_element(element.id, element_regions) or "region_main_1"
        grouped.setdefault((major_id, region_id), []).append(element)

    built: list[Chunk] = []
    all_mentions: list[CitationMention] = []
    order = 0
    group_keys = _major_region_group_order(sections, grouped)
    for major_id, region_id in group_keys:
        elements_in_group = grouped.get((major_id, region_id), [])
        units: list[SentenceUnit] = []
        for index, element in enumerate(elements_in_group):
            section = (
                section_by_id.get(element.section_id) if element.section_id else None
            )
            subsection_end = _is_subsection_boundary(
                elements_in_group, index, section_by_id
            )
            units.extend(
                units_for_element(
                    element,
                    count=count,
                    hard_max_tokens=hard_max_tokens,
                    section_id=element.section_id,
                    section_path=section.path if section else None,
                    subsection_end=subsection_end,
                )
            )
            prose = element_text(element)
            if prose and (
                element.element_type in _PROSE_TYPES
                or element.element_type == ElementType.TABLE
            ):
                region = region_for_element(element.id, element_regions)
                region_info = element_regions.get(element.id)
                region_id = region_info.region_id if region_info else None
                scope_id, scope_diag = scope_for_element(
                    element.id,
                    element_regions,
                    reference_indexes,
                    document_regions,
                )
                extracted = extract_citation_mentions_from_text(
                    document_id=document.id,
                    snapshot_id=snapshot_id,
                    element_id=element.id,
                    text=prose,
                    reference_index=reference_indexes,
                    document_region=region,
                    bibliography_scope_id=scope_id,
                    region_instance_id=region_id,
                )
                if scope_diag:
                    extracted = [
                        mention.model_copy(
                            update={
                                "metadata": {
                                    **mention.metadata,
                                    "bibliography_scope_diagnostic": scope_diag,
                                }
                            }
                        )
                        for mention in extracted
                    ]
                all_mentions.extend(extracted)
        ranges = partition_units(
            units,
            target_tokens=target_tokens,
            hard_max_tokens=hard_max_tokens,
            count=count,
        )
        major_section = section_by_id.get(major_id)
        major_title = major_section.title if major_section else None
        section_chunks = _chunks_from_ranges(
            ranges,
            units=units,
            document_id=document.id,
            snapshot_id=snapshot_id,
            major_section_id=None if major_id == "__unsectioned__" else major_id,
            major_section_title=major_title,
            count=count,
            overlap_tokens=overlap_tokens,
            start_order=order,
            section_by_id=section_by_id,
            element_regions=element_regions,
            region_instance_id=region_id,
        )
        built.extend(section_chunks)
        order += len(section_chunks)

    all_mentions = attach_mentions_to_chunks(all_mentions, chunks=built)

    mentions_by_chunk: dict[str, list[CitationMention]] = {}
    for mention in all_mentions:
        if mention.chunk_id:
            mentions_by_chunk.setdefault(mention.chunk_id, []).append(mention)

    finalized: list[Chunk] = []
    for index, chunk in enumerate(built):
        mentions = mentions_by_chunk.get(chunk.id, [])
        retrieval = build_retrieval_text(
            document=document,
            chunk=chunk,
            mentions=mentions,
            references_by_id=references_by_id,
        )
        finalized.append(
            chunk.model_copy(
                update={
                    "retrieval_text": retrieval,
                    "citation_mention_ids": [mention.id for mention in mentions],
                    "previous_chunk_id": built[index - 1].id if index else None,
                    "next_chunk_id": (
                        built[index + 1].id if index + 1 < len(built) else None
                    ),
                }
            )
        )
    return finalized, all_mentions


def _major_region_group_order(
    sections: list[Section],
    grouped: dict[tuple[str, str], list[Element]],
) -> list[tuple[str, str]]:
    majors = _major_section_order(
        sections,
        {major_id: elements for (major_id, _), elements in grouped.items()},
    )
    ordered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for major_id in majors:
        region_ids = sorted(
            {
                region_id
                for key, elements in grouped.items()
                if key[0] == major_id and elements
                for region_id in [key[1]]
            }
        )
        for region_id in region_ids:
            key = (major_id, region_id)
            if key not in seen:
                ordered.append(key)
                seen.add(key)
    for key in sorted(grouped):
        if key not in seen and grouped[key]:
            ordered.append(key)
    return ordered


def _major_section_order(
    sections: list[Section],
    grouped: dict[str, list[Element]],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for section in sorted(sections, key=lambda item: item.order):
        major_id = resolve_major_section_id(section.id, {s.id: s for s in sections})
        if major_id and major_id in grouped and major_id not in seen:
            ordered.append(major_id)
            seen.add(major_id)
    if "__unsectioned__" in grouped and "__unsectioned__" not in seen:
        ordered.append("__unsectioned__")
    return ordered


def _is_subsection_boundary(
    elements: list[Element],
    index: int,
    section_by_id: dict[str, Section],
) -> bool:
    current = elements[index]
    if current.element_type == ElementType.TITLE:
        return True
    if index + 1 >= len(elements):
        return True
    next_element = elements[index + 1]
    if next_element.element_type == ElementType.TITLE:
        return True
    current_section = section_by_id.get(current.section_id or "")
    next_section = section_by_id.get(next_element.section_id or "")
    if current_section and next_section and current_section.id != next_section.id:
        return True
    return False


def _chunks_from_ranges(
    ranges: list[tuple[int, int]],
    *,
    units: list[SentenceUnit],
    document_id: str,
    snapshot_id: str,
    major_section_id: str | None,
    major_section_title: str | None,
    count: Any,
    overlap_tokens: int,
    start_order: int,
    section_by_id: dict[str, Section],
    element_regions: dict,
    region_instance_id: str | None = None,
) -> list[Chunk]:
    built: list[Chunk] = []
    overlap_tail: list[SentenceUnit] = []
    for range_index, (start, end) in enumerate(ranges):
        chunk_units = list(units[start:end])
        if overlap_tokens > 0 and overlap_tail:
            merged = [*overlap_tail, *chunk_units]
            while (
                len(overlap_tail) > 1
                and _unit_tokens(merged, count) > sum(unit.tokens for unit in chunk_units)
            ):
                overlap_tail = overlap_tail[1:]
                merged = [*overlap_tail, *chunk_units]
            if overlap_tail:
                chunk_units = merged
        chunk = _make_chunk(
            units=chunk_units,
            document_id=document_id,
            snapshot_id=snapshot_id,
            order=start_order + len(built),
            major_section_id=major_section_id,
            major_section_title=major_section_title,
            count=count,
            overlap_source_chunk_ids=[built[-1].id] if overlap_tail and built else [],
            overlap_spans=overlap_tail,
            section_by_id=section_by_id,
            element_regions=element_regions,
            region_instance_id=region_instance_id,
        )
        built.append(chunk)
        overlap_tail = _overlap_tail_units(chunk_units, count, overlap_tokens)
    return built


def _overlap_tail_units(
    units: list[SentenceUnit],
    count: Any,
    overlap_tokens: int,
) -> list[SentenceUnit]:
    if overlap_tokens <= 0:
        return []
    tail: list[SentenceUnit] = []
    for unit in reversed(units):
        candidate = [unit, *tail]
        if _unit_tokens(candidate, count) > overlap_tokens:
            break
        tail.insert(0, unit)
    return tail


def _unit_tokens(units: list[SentenceUnit], count: Any) -> int:
    return count(_join_unit_text(units)) if units else 0


def _make_chunk(
    *,
    units: list[SentenceUnit],
    document_id: str,
    snapshot_id: str,
    order: int,
    major_section_id: str | None,
    major_section_title: str | None,
    count: Any,
    overlap_source_chunk_ids: list[str],
    overlap_spans: list[SentenceUnit],
    section_by_id: dict[str, Section],
    element_regions: dict,
    region_instance_id: str | None = None,
) -> Chunk:
    text = _join_unit_text(units)
    pages = [unit.page for unit in units if unit.page is not None]
    section_paths = [unit.section_path for unit in units if unit.section_path]
    section_ids = [unit.section_id for unit in units if unit.section_id]
    element_ids = list(dict.fromkeys(unit.element_id for unit in units))
    span_ids = [unit.span_id for unit in units]
    identifier = chunk_id(document_id, order, span_ids)
    emergency_splits = sum(1 for unit in units if unit.emergency_split)
    first_element_id = units[0].element_id if units else None
    document_region = (
        region_for_element(first_element_id, element_regions)
        if first_element_id
        else None
    )
    chunk_region_ids = {
        region_id_for_element(unit.element_id, element_regions) for unit in units
    }
    chunk_region_ids.discard(None)
    mixed_region = len(chunk_region_ids) > 1
    end_boundary = "sentence"
    if units and units[-1].subsection_end:
        end_boundary = "subsection"
    elif units and units[-1].paragraph_end:
        end_boundary = "paragraph"
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
                id=unit.span_id,
                element_id=unit.element_id,
                text=unit.text,
                character_start_in_element=unit.character_start_in_element,
                character_end_in_element=unit.character_end_in_element,
                token_start=unit.token_start,
                token_end=unit.token_end,
            )
            for unit in units
        ],
        section_id=section_ids[0] if section_ids else None,
        section_path=section_paths[0] if section_paths else None,
        major_section_id=major_section_id,
        major_section_title=major_section_title,
        page_start=min(pages) if pages else None,
        page_end=max(pages) if pages else None,
        bounding_box=_merge_boxes(
            [unit.bounding_box for unit in units if unit.bounding_box is not None]
        ),
        token_count=_unit_tokens(units, count),
        document_region=document_region,
        overlap_source_chunk_ids=overlap_source_chunk_ids,
        overlap_element_span_ids=[unit.span_id for unit in overlap_spans],
        metadata={
            "end_boundary": end_boundary,
            "emergency_oversized_sentence_splits": emergency_splits,
            "region_instance_id": region_instance_id
            or (next(iter(chunk_region_ids)) if len(chunk_region_ids) == 1 else None),
            "mixed_region_chunk": mixed_region,
        },
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


def _join_unit_text(units: list[SentenceUnit]) -> str:
    if not units:
        return ""
    parts = [units[0].text]
    for previous, current in pairwise(units):
        contiguous = (
            previous.element_id == current.element_id
            and previous.character_end_in_element == current.character_start_in_element
        )
        if not contiguous:
            parts.append("\n\n")
        parts.append(current.text)
    return "".join(parts)
