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
    r"\b(?:supplement(?:ary)?(?:\s+material)?)\b",
    re.IGNORECASE,
)
_APPENDIX_MARKER = re.compile(r"\b(?:appendix|appendices)\b", re.IGNORECASE)
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
    owner_body_region_id: str | None = None
    region_subtype: str | None = None


@dataclass(frozen=True, slots=True)
class ElementRegionInfo:
    region_type: str
    region_id: str
    region_subtype: str | None = None


def build_document_regions(
    *,
    elements: list[Element],
    sections: list[Section],
) -> tuple[list[DocumentRegion], dict[str, ElementRegionInfo]]:
    """Return regions and element_id -> region binding for citation/chunk routing."""
    section_by_id = {section.id: section for section in sections}
    ordered = sorted(elements, key=lambda item: item.order)
    regions: list[DocumentRegion] = []
    element_regions: dict[str, ElementRegionInfo] = {}

    current_type = REGION_MAIN
    current_subtype: str | None = None
    current_body_region_id = _open_region(
        regions, REGION_MAIN, start_order=0, region_subtype=None
    )
    current_id = current_body_region_id
    in_references = False

    for element in ordered:
        marker_type, marker_subtype = _marker_region_from_element(element, section_by_id)
        if marker_type is not None:
            _close_region(regions, end_order=element.order - 1)
            current_type = marker_type
            current_subtype = marker_subtype
            current_id = _open_region(
                regions,
                marker_type,
                start_order=element.order,
                region_subtype=marker_subtype,
                owner_body_region_id=(
                    current_body_region_id if marker_type == REGION_REFERENCES else None
                ),
            )
            if marker_type != REGION_REFERENCES:
                current_body_region_id = current_id
            in_references = marker_type == REGION_REFERENCES
            _bind_element(element_regions, element, current_type, current_id, current_subtype)
            continue

        if element.element_type == ElementType.REFERENCE:
            if not in_references:
                _close_region(regions, end_order=element.order - 1)
                parent_type = current_type if current_type != REGION_REFERENCES else REGION_MAIN
                current_type = REGION_REFERENCES
                current_subtype = None
                current_id = _open_region(
                    regions,
                    REGION_REFERENCES,
                    start_order=element.order,
                    owner_body_region_id=current_body_region_id,
                )
                in_references = True
        elif in_references and element.element_type in {
            ElementType.PARAGRAPH,
            ElementType.TITLE,
            ElementType.LIST,
            ElementType.LIST_ITEM,
        }:
            _close_region(regions, end_order=element.order - 1)
            current_type, current_subtype = _resume_body_region_type(
                element, section_by_id, default=REGION_MAIN
            )
            current_id = _open_region(
                regions,
                current_type,
                start_order=element.order,
                region_subtype=current_subtype,
            )
            current_body_region_id = current_id
            in_references = False

        if element.element_type == ElementType.REFERENCE:
            region_type = REGION_REFERENCES
        elif in_references:
            region_type = REGION_REFERENCES
        else:
            region_type = _body_region_type(
                element, section_by_id, default=current_type, default_subtype=current_subtype
            )
            if region_type != REGION_REFERENCES:
                current_subtype = _element_region_subtype(element, section_by_id, current_subtype)

        _bind_element(element_regions, element, region_type, current_id, current_subtype)

    _close_region(regions, end_order=ordered[-1].order if ordered else None)
    return regions, element_regions


def region_for_element(
    element_id: str,
    element_regions: dict[str, ElementRegionInfo],
    *,
    default: str = REGION_MAIN,
) -> str:
    info = element_regions.get(element_id)
    if info is None:
        return default
    region = info.region_type
    if region == REGION_REFERENCES:
        return REGION_MAIN
    if region == REGION_ABSTRACT:
        return REGION_MAIN
    return region


def region_id_for_element(
    element_id: str,
    element_regions: dict[str, ElementRegionInfo],
    *,
    default: str | None = None,
) -> str | None:
    info = element_regions.get(element_id)
    if info is None:
        return default
    return info.region_id


def region_subtype_for_element(
    element_id: str,
    element_regions: dict[str, ElementRegionInfo],
) -> str | None:
    info = element_regions.get(element_id)
    return info.region_subtype if info else None


def bibliography_owner_region_id(region: DocumentRegion) -> str | None:
    if region.region_type != REGION_REFERENCES:
        return region.region_id
    return region.owner_body_region_id


def _bind_element(
    element_regions: dict[str, ElementRegionInfo],
    element: Element,
    region_type: str,
    region_id: str,
    region_subtype: str | None,
) -> None:
    element_regions[element.id] = ElementRegionInfo(
        region_type=region_type,
        region_id=region_id,
        region_subtype=region_subtype,
    )


def _resume_body_region_type(
    element: Element,
    section_by_id: dict[str, Section],
    *,
    default: str,
) -> tuple[str, str | None]:
    if _is_appendix_context(element, section_by_id):
        return REGION_SUPPLEMENT, "appendix"
    if _is_supplement_context(element, section_by_id):
        return REGION_SUPPLEMENT, None
    return default, None


def _body_region_type(
    element: Element,
    section_by_id: dict[str, Section],
    *,
    default: str,
    default_subtype: str | None,
) -> str:
    if element.section_id:
        section = section_by_id.get(element.section_id)
        if section and section.section_type == "abstract":
            return REGION_ABSTRACT
    if _is_appendix_context(element, section_by_id):
        return REGION_SUPPLEMENT
    if _is_supplement_context(element, section_by_id):
        return REGION_SUPPLEMENT
    return default


def _element_region_subtype(
    element: Element,
    section_by_id: dict[str, Section],
    current_subtype: str | None,
) -> str | None:
    if _is_appendix_context(element, section_by_id):
        return "appendix"
    return current_subtype


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


def _is_appendix_context(
    element: Element,
    section_by_id: dict[str, Section],
) -> bool:
    section_id = element.section_id
    while section_id:
        section = section_by_id.get(section_id)
        if section is None:
            break
        if _APPENDIX_MARKER.search(section.title or ""):
            return True
        section_id = section.parent_section_id
    return False


def _marker_region_from_element(
    element: Element,
    section_by_id: dict[str, Section],
) -> tuple[str | None, str | None]:
    if element.element_type != ElementType.TITLE:
        return None, None
    title = plain_text(element.text or element.markdown or "")
    if _APPENDIX_MARKER.search(title):
        return REGION_SUPPLEMENT, "appendix"
    if _SUPPLEMENT_MARKER.search(title):
        return REGION_SUPPLEMENT, None
    if _REFERENCES_MARKER.search(title) and "format" not in title.casefold():
        return REGION_REFERENCES, None
    if _ABSTRACT_MARKER.fullmatch(title.strip().casefold()) or _ABSTRACT_MARKER.search(
        title[:40]
    ):
        return REGION_ABSTRACT, None
    if element.section_id:
        section = section_by_id.get(element.section_id)
        if section and section.section_type == "abstract":
            return REGION_ABSTRACT, None
    return None, None


def _open_region(
    regions: list[DocumentRegion],
    region_type: str,
    *,
    start_order: int,
    owner_body_region_id: str | None = None,
    region_subtype: str | None = None,
) -> str:
    index = sum(1 for region in regions if region.region_type == region_type) + 1
    region_id = f"region_{region_type}_{index}"
    if region_subtype:
        region_id = f"{region_id}_{region_subtype}"
    regions.append(
        DocumentRegion(
            region_id=region_id,
            region_type=region_type,
            start_order=start_order,
            owner_body_region_id=owner_body_region_id,
            region_subtype=region_subtype,
        )
    )
    return region_id


def _close_region(regions: list[DocumentRegion], *, end_order: int | None) -> None:
    if not regions or regions[-1].end_order is not None:
        return
    regions[-1].end_order = end_order
