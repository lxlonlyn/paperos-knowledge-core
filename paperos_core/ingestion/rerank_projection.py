"""Rebuildable structured scoring spans over canonical parent Chunks."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from paperos_core.domain.canonical import Chunk, RerankProjection, RerankSpan
from paperos_core.domain.ids import RERANK_PROJECTION_VERSION, rerank_span_id
from paperos_core.ingestion.chunk_dp import partition_units
from paperos_core.ingestion.sentence_units import (
    SentenceUnit,
    fallback_text_ranges,
)

RERANK_TARGET_TOKENS = 256
RERANK_HARD_MAX_TOKENS = 384
RERANK_OVERLAP_TOKENS = 0


@dataclass(frozen=True, slots=True)
class _ParentRange:
    start: int
    end: int
    fallback_reason: str | None = None


def build_rerank_spans(
    chunk: Chunk,
    units: list[SentenceUnit],
    *,
    count: Any,
    target_tokens: int = RERANK_TARGET_TOKENS,
    hard_max_tokens: int = RERANK_HARD_MAX_TOKENS,
) -> list[RerankSpan]:
    """Project the exact units used by one Chunk into smaller scoring ranges."""

    if not units:
        return []
    unit_ranges = _parent_unit_ranges(chunk, units)
    projection_units: list[SentenceUnit] = []
    parent_ranges: list[_ParentRange] = []
    for unit, parent_range in zip(units, unit_ranges, strict=True):
        if count(unit.text) <= hard_max_tokens:
            projection_units.append(replace(unit, tokens=count(unit.text)))
            parent_ranges.append(parent_range)
            continue
        fallback_ranges = fallback_text_ranges(
            unit.text,
            count=count,
            hard_max_tokens=hard_max_tokens,
        )
        for index, (start, end, reason) in enumerate(fallback_ranges):
            text = unit.text[start:end]
            projection_units.append(
                replace(
                    unit,
                    text=text,
                    tokens=count(text),
                    span_key=f"{unit.span_key}:rerank:{start}:{end}",
                    paragraph_end=unit.paragraph_end
                    and index == len(fallback_ranges) - 1,
                    subsection_end=unit.subsection_end
                    and index == len(fallback_ranges) - 1,
                    fallback_reason=reason,
                )
            )
            parent_ranges.append(
                _ParentRange(
                    start=parent_range.start + start,
                    end=parent_range.start + end,
                    fallback_reason=reason,
                )
            )

    ranges = partition_units(
        projection_units,
        target_tokens=target_tokens,
        hard_max_tokens=hard_max_tokens,
        count=count,
    )
    ranges = _hard_safe_ranges(
        chunk,
        ranges,
        parent_ranges=parent_ranges,
        count=count,
        hard_max_tokens=hard_max_tokens,
    )
    character_ranges = _continuous_character_ranges(
        chunk,
        ranges,
        parent_ranges=parent_ranges,
        count=count,
        hard_max_tokens=hard_max_tokens,
    )
    spans: list[RerankSpan] = []
    for ordinal, ((start, end), (character_start, character_end)) in enumerate(
        zip(ranges, character_ranges, strict=True)
    ):
        scoring_text = chunk.text[character_start:character_end]
        token_count = count(scoring_text)
        if not scoring_text or token_count > hard_max_tokens:
            raise ValueError("RerankSpan violates its deterministic hard maximum")
        fallback_reasons = list(
            dict.fromkeys(
                item.fallback_reason
                for item in parent_ranges[start:end]
                if item.fallback_reason is not None
            )
        )
        spans.append(
            RerankSpan(
                id=rerank_span_id(
                    chunk.id,
                    ordinal,
                    character_start,
                    character_end,
                ),
                parent_chunk_id=chunk.id,
                canonical_snapshot_id=chunk.canonical_snapshot_id,
                ordinal=ordinal,
                character_start_in_chunk=character_start,
                character_end_in_chunk=character_end,
                unit_start=start,
                unit_end=end,
                token_count=token_count,
                fallback_reasons=fallback_reasons,
            )
        )
    return spans


def build_rerank_projection(
    snapshot_id: str,
    spans: list[RerankSpan],
    *,
    target_tokens: int = RERANK_TARGET_TOKENS,
    hard_max_tokens: int = RERANK_HARD_MAX_TOKENS,
) -> RerankProjection:
    return RerankProjection(
        snapshot_id=snapshot_id,
        projection_version=RERANK_PROJECTION_VERSION,
        target_tokens=target_tokens,
        hard_max_tokens=hard_max_tokens,
        overlap_tokens=RERANK_OVERLAP_TOKENS,
        spans=spans,
    )


def _parent_unit_ranges(
    chunk: Chunk,
    units: list[SentenceUnit],
) -> list[_ParentRange]:
    cursor = 0
    ranges: list[_ParentRange] = []
    for index, unit in enumerate(units):
        if index:
            previous = units[index - 1]
            contiguous = (
                previous.element_id == unit.element_id
                and previous.character_end_in_element
                == unit.character_start_in_element
            )
            if not contiguous:
                cursor += 2
        start = cursor
        end = start + len(unit.text)
        if chunk.text[start:end] != unit.text:
            raise ValueError("RerankProjection units do not reconstruct parent Chunk.text")
        ranges.append(_ParentRange(start=start, end=end))
        cursor = end
    if cursor != len(chunk.text):
        raise ValueError("RerankProjection units do not cover parent Chunk.text")
    return ranges


def _hard_safe_ranges(
    chunk: Chunk,
    ranges: list[tuple[int, int]],
    *,
    parent_ranges: list[_ParentRange],
    count: Any,
    hard_max_tokens: int,
) -> list[tuple[int, int]]:
    """Preserve DP output unless exact parent slicing exceeds the hard maximum."""

    safe: list[tuple[int, int]] = []
    for start, end in ranges:
        cursor = start
        while cursor < end:
            next_end = cursor + 1
            if _range_tokens(chunk, parent_ranges, cursor, next_end, count) > (
                hard_max_tokens
            ):
                raise ValueError("A rerank projection unit exceeds the hard maximum")
            while next_end < end and _range_tokens(
                chunk, parent_ranges, cursor, next_end + 1, count
            ) <= hard_max_tokens:
                next_end += 1
            safe.append((cursor, next_end))
            cursor = next_end
    return safe


def _continuous_character_ranges(
    chunk: Chunk,
    ranges: list[tuple[int, int]],
    *,
    parent_ranges: list[_ParentRange],
    count: Any,
    hard_max_tokens: int,
) -> list[tuple[int, int]]:
    """Assign structural separators to adjacent spans without changing DP ranges."""

    character_ranges = [
        [parent_ranges[start].start, parent_ranges[end - 1].end]
        for start, end in ranges
    ]
    for index in range(len(character_ranges) - 1):
        left = character_ranges[index]
        right = character_ranges[index + 1]
        gap_start = left[1]
        gap_end = right[0]
        if gap_start > gap_end:
            raise ValueError("RerankProjection parent ranges overlap")
        if gap_start == gap_end:
            continue
        boundary = next(
            (
                candidate
                for candidate in range(gap_end, gap_start - 1, -1)
                if count(chunk.text[left[0] : candidate]) <= hard_max_tokens
                and count(chunk.text[candidate : right[1]]) <= hard_max_tokens
            ),
            None,
        )
        if boundary is None:
            raise ValueError(
                "RerankProjection cannot assign a structural separator within "
                "its hard maximum"
            )
        left[1] = boundary
        right[0] = boundary
    return [(start, end) for start, end in character_ranges]


def _range_tokens(
    chunk: Chunk,
    parent_ranges: list[_ParentRange],
    start: int,
    end: int,
    count: Any,
) -> int:
    return int(
        count(chunk.text[parent_ranges[start].start : parent_ranges[end - 1].end])
    )


__all__ = [
    "RERANK_HARD_MAX_TOKENS",
    "RERANK_OVERLAP_TOKENS",
    "RERANK_TARGET_TOKENS",
    "build_rerank_projection",
    "build_rerank_spans",
]
