"""Deterministic document regions and citation-namespace routing.

The ordered canonical element tape is the only input to region segmentation.
Citation resolution never participates in this phase. After all regions have
been built, each body region is assigned to one bibliography-region namespace
in a second pass.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from paperos_core.domain.canonical import Element, Section
from paperos_core.domain.enums import ElementType
from paperos_core.ingestion.bibliography_scope import (
    REGION_ABSTRACT,
    REGION_MAIN,
    REGION_REFERENCES,
    REGION_SUPPLEMENT,
)
from paperos_core.ingestion.normalization import plain_text, strip_heading_number

REGION_TYPES = frozenset(
    {REGION_MAIN, REGION_ABSTRACT, REGION_REFERENCES, REGION_SUPPLEMENT}
)

_SUPPLEMENT_MARKER = re.compile(
    r"\b(?:supplement(?:ary)?(?:\s+(?:material|information))?|supporting\s+information)\b",
    re.IGNORECASE,
)
_APPENDIX_MARKER = re.compile(
    r"^(?:appendix|appendices)(?:\s+[A-Z0-9]+)?\b", re.IGNORECASE
)
_REFERENCES_MARKER = re.compile(
    r"^(?:references|bibliography|literature\s+cited)\s*$", re.IGNORECASE
)
_ABSTRACT_MARKER = re.compile(r"^abstract\s*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class DocumentRegion:
    region_id: str
    region_type: str
    start_order: int
    end_order: int | None = None
    citation_namespace_id: str | None = None
    bibliography_scope_id: str | None = None
    owner_body_region_id: str | None = None
    region_subtype: str | None = None


@dataclass(frozen=True, slots=True)
class ElementRegionInfo:
    element_id: str
    order: int
    region_type: str
    region_id: str
    citation_namespace_id: str | None = None
    region_subtype: str | None = None


def build_document_regions(
    *,
    elements: list[Element],
    sections: list[Section],
) -> tuple[list[DocumentRegion], dict[str, ElementRegionInfo]]:
    """Build regions, then bind body regions to bibliography namespaces."""
    ordered = sorted(elements, key=lambda item: (item.order, item.id))
    if not ordered:
        return [], {}

    section_by_id = {section.id: section for section in sections}
    regions: list[DocumentRegion] = []
    element_region_ids: dict[str, str] = {}
    current: DocumentRegion | None = None
    references_seen = False

    for element in ordered:
        explicit_type, subtype = _explicit_region(element, section_by_id)
        target_type, target_subtype = _next_state(
            current=current,
            element=element,
            explicit_type=explicit_type,
            explicit_subtype=subtype,
            references_seen=references_seen,
            section_by_id=section_by_id,
        )
        if current is None or _must_open_region(
            current=current,
            target_type=target_type,
            target_subtype=target_subtype,
            element=element,
            explicit_type=explicit_type,
        ):
            if current is not None:
                regions[-1] = replace(current, end_order=element.order - 1)
            current = _new_region(
                regions,
                target_type,
                start_order=element.order,
                region_subtype=target_subtype,
            )
            regions.append(current)
            references_seen = False

        element_region_ids[element.id] = current.region_id
        if (
            current.region_type == REGION_REFERENCES
            and element.element_type == ElementType.REFERENCE
        ):
            references_seen = True

    assert current is not None
    regions[-1] = replace(current, end_order=ordered[-1].order)
    regions = _assign_citation_namespaces(regions)
    region_by_id = {region.region_id: region for region in regions}
    element_regions = {
        element.id: ElementRegionInfo(
            element_id=element.id,
            order=element.order,
            region_type=region_by_id[element_region_ids[element.id]].region_type,
            region_id=element_region_ids[element.id],
            citation_namespace_id=region_by_id[
                element_region_ids[element.id]
            ].citation_namespace_id,
            region_subtype=region_by_id[element_region_ids[element.id]].region_subtype,
        )
        for element in ordered
    }
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
    return info.region_type


def region_id_for_element(
    element_id: str,
    element_regions: dict[str, ElementRegionInfo],
    *,
    default: str | None = None,
) -> str | None:
    info = element_regions.get(element_id)
    return info.region_id if info else default


def citation_namespace_for_element(
    element_id: str,
    element_regions: dict[str, ElementRegionInfo],
) -> str | None:
    info = element_regions.get(element_id)
    return info.citation_namespace_id if info else None


def region_subtype_for_element(
    element_id: str,
    element_regions: dict[str, ElementRegionInfo],
) -> str | None:
    info = element_regions.get(element_id)
    return info.region_subtype if info else None


def bibliography_owner_region_id(region: DocumentRegion) -> str | None:
    return (
        region.owner_body_region_id
        if region.region_type == REGION_REFERENCES
        else region.region_id
    )


def _explicit_region(
    element: Element,
    section_by_id: dict[str, Section],
) -> tuple[str | None, str | None]:
    if element.element_type == ElementType.REFERENCE:
        return REGION_REFERENCES, None
    section_region, section_subtype = _section_region(element.section_id, section_by_id)
    if element.element_type == ElementType.TITLE:
        title = strip_heading_number(
            plain_text(element.text or element.markdown or "")
        )
        if _REFERENCES_MARKER.fullmatch(title):
            return REGION_REFERENCES, None
        if _APPENDIX_MARKER.match(title):
            return REGION_SUPPLEMENT, "appendix"
        if _SUPPLEMENT_MARKER.search(title):
            return REGION_SUPPLEMENT, None
        if _ABSTRACT_MARKER.fullmatch(title):
            return REGION_ABSTRACT, None
    return section_region, section_subtype


def _section_region(
    section_id: str | None,
    section_by_id: dict[str, Section],
) -> tuple[str | None, str | None]:
    current_id = section_id
    while current_id:
        section = section_by_id.get(current_id)
        if section is None:
            break
        title = strip_heading_number(section.title or "")
        section_type = (section.section_type or "").casefold()
        if (
            section_type in {"references", "bibliography"}
            or _REFERENCES_MARKER.fullmatch(title)
        ):
            return REGION_REFERENCES, None
        if _APPENDIX_MARKER.match(title):
            return REGION_SUPPLEMENT, "appendix"
        if _SUPPLEMENT_MARKER.search(title):
            return REGION_SUPPLEMENT, None
        if section_type == "abstract" or _ABSTRACT_MARKER.fullmatch(title):
            return REGION_ABSTRACT, None
        current_id = section.parent_section_id
    return None, None


def _next_state(
    *,
    current: DocumentRegion | None,
    element: Element,
    explicit_type: str | None,
    explicit_subtype: str | None,
    references_seen: bool,
    section_by_id: dict[str, Section],
) -> tuple[str, str | None]:
    if current is None:
        return explicit_type or REGION_MAIN, explicit_subtype
    if current.region_type == REGION_SUPPLEMENT and explicit_type == REGION_REFERENCES:
        # MinerU may leave all post-bibliography supplement sections parented to
        # the preceding References section.  Once a real supplement heading has
        # opened the body region, only a Reference element or an actual
        # References heading may enter a new bibliography region.
        title = strip_heading_number(
            plain_text(element.text or element.markdown or "")
        )
        is_bibliography_heading = (
            element.element_type == ElementType.TITLE
            and _REFERENCES_MARKER.fullmatch(title) is not None
        )
        if element.element_type != ElementType.REFERENCE and not is_bibliography_heading:
            return REGION_SUPPLEMENT, current.region_subtype
    if explicit_type is not None:
        return explicit_type, explicit_subtype
    if current.region_type == REGION_REFERENCES:
        if references_seen and element.element_type == ElementType.TITLE:
            section_type, section_subtype = _section_region(
                element.section_id, section_by_id
            )
            if section_type != REGION_REFERENCES:
                return section_type or REGION_MAIN, section_subtype
        return REGION_REFERENCES, current.region_subtype
    if current.region_type == REGION_ABSTRACT and element.element_type == ElementType.TITLE:
        return REGION_MAIN, None
    return current.region_type, current.region_subtype


def _must_open_region(
    *,
    current: DocumentRegion,
    target_type: str,
    target_subtype: str | None,
    element: Element,
    explicit_type: str | None,
) -> bool:
    if current.region_type != target_type or current.region_subtype != target_subtype:
        return True
    return (
        target_type == REGION_REFERENCES
        and explicit_type == REGION_REFERENCES
        and element.element_type == ElementType.TITLE
        and current.start_order != element.order
    )


def _new_region(
    regions: list[DocumentRegion],
    region_type: str,
    *,
    start_order: int,
    region_subtype: str | None,
) -> DocumentRegion:
    index = sum(1 for region in regions if region.region_type == region_type) + 1
    suffix = f"_{region_subtype}" if region_subtype else ""
    return DocumentRegion(
        region_id=f"region_{region_type}_{index}{suffix}",
        region_type=region_type,
        start_order=start_order,
        region_subtype=region_subtype,
    )


def _assign_citation_namespaces(
    regions: list[DocumentRegion],
) -> list[DocumentRegion]:
    bibliography_regions = [
        region for region in regions if region.region_type == REGION_REFERENCES
    ]
    namespace_by_region = {
        region.region_id: f"citation_namespace_{index}"
        for index, region in enumerate(bibliography_regions, start=1)
    }
    namespace_for_body: dict[str, str | None] = {}
    for body in (
        region for region in regions if region.region_type != REGION_REFERENCES
    ):
        following = [
            bibliography
            for bibliography in bibliography_regions
            if bibliography.start_order > (body.end_order or -1)
        ]
        if following:
            namespace_for_body[body.region_id] = namespace_by_region[
                min(following, key=lambda item: item.start_order).region_id
            ]
            continue
        previous = [
            bibliography
            for bibliography in bibliography_regions
            if bibliography.end_order is not None
            and bibliography.end_order < body.start_order
        ]
        namespace_for_body[body.region_id] = (
            namespace_by_region[
                max(previous, key=lambda item: item.end_order or -1).region_id
            ]
            if previous
            else None
        )

    assigned: list[DocumentRegion] = []
    previous_body: DocumentRegion | None = None
    for region in regions:
        if region.region_type == REGION_REFERENCES:
            namespace_id = namespace_by_region[region.region_id]
            assigned.append(
                replace(
                    region,
                    citation_namespace_id=namespace_id,
                    bibliography_scope_id=namespace_id,
                    owner_body_region_id=(
                        previous_body.region_id if previous_body else None
                    ),
                )
            )
        else:
            namespace_id = namespace_for_body[region.region_id]
            assigned.append(
                replace(
                    region,
                    citation_namespace_id=namespace_id,
                    bibliography_scope_id=namespace_id,
                )
            )
            previous_body = region
    return assigned
