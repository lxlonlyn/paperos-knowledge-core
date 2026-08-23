"""Shared chunk eligibility classification for production and validation."""

from __future__ import annotations

from dataclasses import dataclass

from paperos_core.domain.canonical import Element, Section
from paperos_core.domain.enums import ElementType

from paperos_core.ingestion.sentence_units import (
    _PROSE_TYPES,
    element_text,
    resolve_major_section_id,
)

CONTAINER_HEADING_MAX_LEN = 120

ELIGIBLE_PROSE = "ELIGIBLE_PROSE"
ELIGIBLE_TABLE = "ELIGIBLE_TABLE"
ELIGIBLE_FORMULA = "ELIGIBLE_FORMULA"
EXCLUDE_REFERENCE = "EXCLUDE_REFERENCE"
EXCLUDE_REFERENCE_REGION = "EXCLUDE_REFERENCE_REGION"
EXCLUDE_HEADER = "EXCLUDE_HEADER"
EXCLUDE_FOOTER = "EXCLUDE_FOOTER"
EXCLUDE_PAGE_NUMBER = "EXCLUDE_PAGE_NUMBER"
EXCLUDE_PUBLICATION_METADATA = "EXCLUDE_PUBLICATION_METADATA"
EXCLUDE_CONTAINER_ONLY_HEADING = "EXCLUDE_CONTAINER_ONLY_HEADING"
EXCLUDE_NO_MAJOR_SECTION = "EXCLUDE_NO_MAJOR_SECTION"
EXCLUDE_EMPTY = "EXCLUDE_EMPTY"
EXCLUDE_UNSUPPORTED_TYPE = "EXCLUDE_UNSUPPORTED_TYPE"


@dataclass(frozen=True, slots=True)
class ChunkEligibility:
    eligible: bool
    reason: str


def _element_text(element: Element) -> str:
    return element_text(element).strip()


def _is_publication_metadata(element: Element) -> bool:
    text = (element.text or element.markdown or "").casefold()
    markers = (
        "received ",
        "revised ",
        "accepted ",
        "acm reference format",
        "copyright",
        "to cite this version",
    )
    return any(marker in text for marker in markers) and len(text) < 400


def _is_container_only_heading(element: Element) -> bool:
    if element.element_type != ElementType.TITLE:
        return False
    text = _element_text(element)
    return bool(text) and len(text) < CONTAINER_HEADING_MAX_LEN


def classify_chunk_eligibility(
    element: Element,
    *,
    section_by_id: dict[str, Section],
    region_type: str | None = None,
) -> ChunkEligibility:
    """Single production/validation entry point for authoritative chunk coverage."""
    if element.element_type == ElementType.REFERENCE:
        return ChunkEligibility(False, EXCLUDE_REFERENCE)
    if region_type == "references":
        return ChunkEligibility(False, EXCLUDE_REFERENCE_REGION)
    if element.element_type == ElementType.HEADER:
        return ChunkEligibility(False, EXCLUDE_HEADER)
    if element.element_type == ElementType.FOOTER:
        return ChunkEligibility(False, EXCLUDE_FOOTER)
    if element.element_type == ElementType.PAGE_NUMBER:
        return ChunkEligibility(False, EXCLUDE_PAGE_NUMBER)

    allowed = _PROSE_TYPES | {ElementType.TABLE, ElementType.FORMULA}
    if element.element_type not in allowed:
        return ChunkEligibility(False, EXCLUDE_UNSUPPORTED_TYPE)

    if _is_publication_metadata(element):
        return ChunkEligibility(False, EXCLUDE_PUBLICATION_METADATA)

    if _is_container_only_heading(element):
        return ChunkEligibility(False, EXCLUDE_CONTAINER_ONLY_HEADING)

    if resolve_major_section_id(element.section_id, section_by_id) is None:
        return ChunkEligibility(False, EXCLUDE_NO_MAJOR_SECTION)

    if not _element_text(element):
        return ChunkEligibility(False, EXCLUDE_EMPTY)

    if element.element_type == ElementType.TABLE:
        return ChunkEligibility(True, ELIGIBLE_TABLE)
    if element.element_type == ElementType.FORMULA:
        return ChunkEligibility(True, ELIGIBLE_FORMULA)
    return ChunkEligibility(True, ELIGIBLE_PROSE)
