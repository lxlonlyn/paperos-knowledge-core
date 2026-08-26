"""Citation namespaces derived exclusively from bibliography regions.

``BibliographyScope`` remains as a compatibility alias, but its semantics are
now exactly one ``REFERENCES`` region instance.  Citation code is not allowed
to choose or repair a namespace.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from paperos_core.ingestion.document_regions import (
        DocumentRegion,
        ElementRegionInfo,
    )

import re
from dataclasses import dataclass, field

from paperos_core.domain.canonical import Element, ReferenceEntry, Section

REGION_MAIN = "main"
REGION_SUPPLEMENT = "supplement"
REGION_ABSTRACT = "abstract"
REGION_REFERENCES = "references"
REGION_OTHER = "other"

FAILURE_NAMESPACE_NOT_ASSIGNED = "NAMESPACE_NOT_ASSIGNED"
# Kept for consumers that import the old constant; no new code emits it.
FAILURE_SCOPE_NOT_FOUND = FAILURE_NAMESPACE_NOT_ASSIGNED

_IndexT = TypeVar("_IndexT")

@dataclass(slots=True)
class CitationNamespace:
    namespace_id: str
    bibliography_region_id: str | None
    parent_region: str
    owner_body_region_id: str | None = None
    reference_ids: list[str] = field(default_factory=list)

    @property
    def scope_id(self) -> str:
        return self.namespace_id


BibliographyScope = CitationNamespace


@dataclass(slots=True)
class ScopedBibliography(Generic[_IndexT]):
    scopes: dict[str, CitationNamespace]
    reference_scope: dict[str, str]
    scope_indexes: dict[str, _IndexT]

    @property
    def namespaces(self) -> dict[str, CitationNamespace]:
        return self.scopes


def resolve_element_region(
    section_id: str | None,
    section_by_id: dict[str, Section],
) -> str:
    """Compatibility helper for non-citation callers."""
    current_id = section_id
    while current_id:
        section = section_by_id.get(current_id)
        if section is None:
            break
        section_type = (section.section_type or "").casefold()
        title = (section.title or "").casefold()
        if section_type == "abstract":
            return REGION_ABSTRACT
        if section_type in {"references", "bibliography"}:
            return REGION_REFERENCES
        if any(token in title for token in ("supplement", "appendix", "appendices")):
            return REGION_SUPPLEMENT
        current_id = section.parent_section_id
    return REGION_MAIN


def scopes_for_region(region: str, scoped: ScopedBibliography[Any]) -> list[str]:
    """Compatibility report helper; never used to route a citation."""
    accepted = (
        {REGION_SUPPLEMENT}
        if region == REGION_SUPPLEMENT
        else {REGION_MAIN, REGION_ABSTRACT}
    )
    return [
        namespace.namespace_id
        for namespace in scoped.scopes.values()
        if namespace.parent_region in accepted
    ]


def scope_for_element(
    element_id: str,
    element_regions: dict[str, ElementRegionInfo],
    scoped: ScopedBibliography[Any],
    document_regions: list[DocumentRegion],
) -> tuple[str | None, str | None]:
    """Read the namespace assigned during the region phase, without fallback."""
    _ = document_regions
    info = element_regions.get(element_id)
    namespace_id = info.citation_namespace_id if info is not None else None
    if namespace_id is None or namespace_id not in scoped.scope_indexes:
        return None, FAILURE_NAMESPACE_NOT_ASSIGNED
    return namespace_id, None


def assign_bibliography_scopes(
    *,
    references: list[ReferenceEntry],
    elements: list[Element],
    sections: list[Section],
) -> tuple[dict[str, str], dict[str, CitationNamespace]]:
    """Assign references to their already-built bibliography region namespace."""
    from paperos_core.ingestion.document_regions import build_document_regions

    if not elements:
        namespace = CitationNamespace(
            namespace_id="default",
            bibliography_region_id=None,
            parent_region=REGION_MAIN,
            reference_ids=[reference.id for reference in references],
        )
        return ({reference.id: "default" for reference in references}, {"default": namespace})

    regions, element_regions = build_document_regions(
        elements=elements, sections=sections
    )
    region_by_id = {region.region_id: region for region in regions}
    namespaces: dict[str, CitationNamespace] = {}
    for region in regions:
        if region.region_type != REGION_REFERENCES or not region.citation_namespace_id:
            continue
        owner = region_by_id.get(region.owner_body_region_id or "")
        namespaces[region.citation_namespace_id] = CitationNamespace(
            namespace_id=region.citation_namespace_id,
            bibliography_region_id=region.region_id,
            parent_region=owner.region_type if owner is not None else REGION_MAIN,
            owner_body_region_id=region.owner_body_region_id,
        )

    reference_scope: dict[str, str] = {}
    for reference in sorted(references, key=lambda item: (item.order, item.id)):
        info = element_regions.get(reference.source_element_id or "")
        namespace_id = info.citation_namespace_id if info is not None else None
        if namespace_id is None and len(namespaces) == 1:
            # Provider-neutral references created without source_element_id are
            # deterministic only when the document has one bibliography.
            namespace_id = next(iter(namespaces))
        if namespace_id is None or namespace_id not in namespaces:
            continue
        reference_scope[reference.id] = namespace_id
        namespaces[namespace_id].reference_ids.append(reference.id)
    return reference_scope, namespaces


def repair_numeric_label_sequence(
    references: list[ReferenceEntry],
    *,
    scope_id: str | None = None,
    reference_scope: dict[str, str] | None = None,
) -> list[ReferenceEntry]:
    """Repair a single missing numeric label between adjacent references."""
    scoped_refs = [
        reference
        for reference in references
        if reference_scope is None or reference_scope.get(reference.id) == scope_id
    ]
    scoped_refs.sort(key=lambda item: (item.order, item.id))
    repaired: dict[str, ReferenceEntry] = {}
    for index, reference in enumerate(scoped_refs):
        if _numeric_label(reference) is not None:
            continue
        previous = _nearest_numeric_label(scoped_refs, index, -1)
        following = _nearest_numeric_label(scoped_refs, index, 1)
        if previous is None or following is None or following - previous != 2:
            continue
        label = str(previous + 1)
        repaired[reference.id] = reference.model_copy(
            update={
                "citation_label": label,
                "parsed_fields": {
                    **reference.parsed_fields,
                    "citation_label": label,
                    "label_kind": "numeric",
                    "reference_number": previous + 1,
                    "label_source": "inferred_sequence",
                },
            }
        )
    return [repaired.get(reference.id, reference) for reference in references]


def _nearest_numeric_label(
    references: list[ReferenceEntry], index: int, step: int
) -> int | None:
    cursor = index + step
    while 0 <= cursor < len(references):
        label = _numeric_label(references[cursor])
        if label is not None:
            return label
        cursor += step
    return None


def _numeric_label(reference: ReferenceEntry) -> int | None:
    value = reference.citation_label
    if value is None:
        raw = reference.parsed_fields.get("reference_number")
        value = str(raw) if raw is not None else None
    if value is None:
        return None
    normalized = value.strip().strip("[]()")
    return int(normalized) if re.fullmatch(r"\d+", normalized) else None
