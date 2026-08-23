"""Document-region segmentation over the ordered canonical element stream."""

from __future__ import annotations

import re
from dataclasses import dataclass

from paperos_core.domain.canonical import Element, Section
from paperos_core.domain.enums import ElementType
from paperos_core.ingestion.bibliography_scope import (
    REGION_ABSTRACT,
    REGION_MAIN,
    REGION_REFERENCES,
    REGION_SUPPLEMENT,
)
from paperos_core.ingestion.normalization import plain_text

REGION_TYPES = frozenset(
    {REGION_MAIN, REGION_ABSTRACT, REGION_REFERENCES, REGION_SUPPLEMENT}
)

_SUPPLEMENT_MARKER = re.compile(
    r"\b(?:supplement(?:ary)?(?:\s+material)?|appendix|appendices)\b",
    re.IGNORECASE,
)
_REFERENCES_MARKER = re.compile(
    r"\b(?:references|bibliography)\b",
    re.IGNORECASE,
)
_ABSTRACT_MARKER = re.compile(r"\babstract\b", re.IGNORECASE)


@dataclass(slots=True)
class DocumentRegion:
    region_id: str
    region_type: str
    start_order: int
    end_order: int | None = None
    bibliography_scope_id: str | None = None


def build_document_regions(
    *,
    elements: list[Element],
    sections: list[Section],
) -> tuple[list[DocumentRegion], dict[str, str]]:
    """Return regions and element_id -> region_type for citation/chunk routing."""
    section_by_id = {section.id: section for section in sections}
    ordered = sorted(elements, key=lambda item: item.order)
    regions: list[DocumentRegion] = []
    element_region: dict[str, str] = {}
    element_region_id: dict[str, str] = {}

    current_type = REGION_MAIN
    current_id = _open_region(regions, REGION_MAIN, start_order=0)
    in_references = False

    for element in ordered:
        marker_type = _marker_region_from_element(element, section_by_id)
        if marker_type is not None:
            _close_region(regions, end_order=element.order - 1)
            current_type = marker_type
            current_id = _open_region(regions, marker_type, start_order=element.order)
            in_references = marker_type == REGION_REFERENCES
            continue

        if element.element_type == ElementType.REFERENCE:
            if not in_references:
                _close_region(regions, end_order=element.order - 1)
                parent = REGION_SUPPLEMENT if current_type == REGION_SUPPLEMENT else REGION_MAIN
                current_type = REGION_REFERENCES
                current_id = _open_region(
                    regions, REGION_REFERENCES, start_order=element.order, suffix=parent
                )
                in_references = True
        elif in_references and element.element_type in {
            ElementType.PARAGRAPH,
            ElementType.TITLE,
            ElementType.LIST,
            ElementType.LIST_ITEM,
        }:
            _close_region(regions, end_order=element.order - 1)
            current_type = REGION_SUPPLEMENT if _is_supplement_context(element, section_by_id) else REGION_MAIN
            current_id = _open_region(regions, current_type, start_order=element.order)
            in_references = False

        if element.element_type == ElementType.REFERENCE:
            region_type = REGION_REFERENCES
        elif in_references:
            region_type = REGION_REFERENCES
        else:
            region_type = _body_region_type(element, section_by_id, default=current_type)

        element_region[element.id] = region_type
        element_region_id[element.id] = current_id

    _close_region(regions, end_order=ordered[-1].order if ordered else None)
    return regions, element_region


def region_for_element(
    element_id: str,
    element_region: dict[str, str],
    *,
    default: str = REGION_MAIN,
) -> str:
    region = element_region.get(element_id, default)
    if region == REGION_REFERENCES:
        return REGION_MAIN
    if region == REGION_ABSTRACT:
        return REGION_MAIN
    return region


def _body_region_type(
    element: Element,
    section_by_id: dict[str, Section],
    *,
    default: str,
) -> str:
    if element.section_id:
        section = section_by_id.get(element.section_id)
        if section and section.section_type == "abstract":
            return REGION_ABSTRACT
    if _is_supplement_context(element, section_by_id):
        return REGION_SUPPLEMENT
    return default


def _is_supplement_context(
    element: Element,
    section_by_id: dict[str, Section],
) -> bool:
    section_id = element.section_id
    while section_id:
        section = section_by_id.get(section_id)
        if section is None:
            break
        if _SUPPLEMENT_MARKER.search(section.title or ""):
            return True
        section_id = section.parent_section_id
    return False


def _marker_region_from_element(
    element: Element,
    section_by_id: dict[str, Section],
) -> str | None:
    if element.element_type != ElementType.TITLE:
        return None
    title = plain_text(element.text or element.markdown or "")
    if _SUPPLEMENT_MARKER.search(title):
        return REGION_SUPPLEMENT
    if _REFERENCES_MARKER.search(title) and "format" not in title.casefold():
        return REGION_REFERENCES
    if _ABSTRACT_MARKER.fullmatch(title.strip().casefold()) or _ABSTRACT_MARKER.search(
        title[:40]
    ):
        return REGION_ABSTRACT
    if element.section_id:
        section = section_by_id.get(element.section_id)
        if section and section.section_type == "abstract":
            return REGION_ABSTRACT
    return None


def _open_region(
    regions: list[DocumentRegion],
    region_type: str,
    *,
    start_order: int,
    suffix: str | None = None,
) -> str:
    index = sum(1 for region in regions if region.region_type == region_type) + 1
    label = region_type if suffix is None else f"{region_type}_{suffix}"
    region_id = f"region_{label}_{index}"
    regions.append(
        DocumentRegion(
            region_id=region_id,
            region_type=region_type,
            start_order=start_order,
        )
    )
    return region_id


def _close_region(regions: list[DocumentRegion], *, end_order: int | None) -> None:
    if not regions or regions[-1].end_order is not None:
        return
    regions[-1].end_order = end_order
