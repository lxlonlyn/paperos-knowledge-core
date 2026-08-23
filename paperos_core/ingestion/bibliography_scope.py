"""Bibliography scope assignment and document-region resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from paperos_core.domain.canonical import Element, ReferenceEntry, Section
from paperos_core.domain.enums import ElementType

REGION_MAIN = "main"
REGION_SUPPLEMENT = "supplement"
REGION_ABSTRACT = "abstract"
REGION_REFERENCES = "references"
REGION_OTHER = "other"

_SUPPLEMENT_HINTS = (
    "supplement",
    "supplementary",
    "appendix",
    "appendices",
    "suppl.",
)


@dataclass
class BibliographyScope:
    scope_id: str
    parent_region: str
    owner_body_region_id: str | None = None
    reference_ids: list[str] = field(default_factory=list)


@dataclass
class ScopedBibliography:
    scopes: dict[str, BibliographyScope]
    reference_scope: dict[str, str]
    scope_indexes: dict[str, object]


def resolve_element_region(
    section_id: str | None,
    section_by_id: dict[str, Section],
) -> str:
    if section_id is None:
        return REGION_MAIN
    chain: list[Section] = []
    current = section_by_id.get(section_id)
    while current is not None:
        chain.append(current)
        current = (
            section_by_id.get(current.parent_section_id)
            if current.parent_section_id
            else None
        )
    for section in chain:
        if section.section_type == "abstract":
            return REGION_ABSTRACT
        if section.section_type == "references":
            return REGION_REFERENCES
        title = section.title.casefold()
        if any(hint in title for hint in _SUPPLEMENT_HINTS):
            return REGION_SUPPLEMENT
    return REGION_MAIN


def scopes_for_region(
    region: str,
    scoped: ScopedBibliography,
) -> list[str]:
    """Legacy helper — prefer ``scope_for_element`` for citation resolution."""
    if region == REGION_SUPPLEMENT:
        return [
            scope.scope_id
            for scope in scoped.scopes.values()
            if scope.parent_region == REGION_SUPPLEMENT
        ]
    return [
        scope.scope_id
        for scope in scoped.scopes.values()
        if scope.parent_region in {REGION_MAIN, REGION_ABSTRACT}
    ]


def scope_for_element(
    element_id: str,
    element_regions: dict,
    scoped: ScopedBibliography,
    document_regions: list,
) -> tuple[str | None, str | None]:
    """Select exactly one bibliography scope for a citation-bearing element."""
    from paperos_core.ingestion.document_regions import ElementRegionInfo

    info: ElementRegionInfo | None = element_regions.get(element_id)
    if info is None:
        candidates = scopes_for_region(REGION_MAIN, scoped)
        if len(candidates) == 1:
            return candidates[0], None
        if not candidates:
            return None, FAILURE_SCOPE_NOT_FOUND
        return None, "AMBIGUOUS_BIBLIOGRAPHY_SCOPE"

    region_type = info.region_type
    if region_type == REGION_REFERENCES:
        region_type = REGION_MAIN
    region_id = info.region_id

    owned_candidates: list[str] = []
    unowned_candidates: list[str] = []
    for scope in scoped.scopes.values():
        if region_type == REGION_SUPPLEMENT:
            if scope.parent_region != REGION_SUPPLEMENT:
                continue
        elif region_type in {REGION_MAIN, REGION_ABSTRACT}:
            if scope.parent_region not in {REGION_MAIN, REGION_ABSTRACT}:
                continue
        else:
            continue
        if scope.owner_body_region_id == region_id:
            owned_candidates.append(scope.scope_id)
        elif scope.owner_body_region_id is None:
            unowned_candidates.append(scope.scope_id)

    if len(owned_candidates) == 1:
        only = owned_candidates[0]
        if region_type in {REGION_MAIN, REGION_ABSTRACT}:
            pool = [
                scope_id
                for scope_id, scope in scoped.scopes.items()
                if scope.parent_region in {REGION_MAIN, REGION_ABSTRACT}
            ]
            if pool:
                largest = max(pool, key=lambda scope_id: len(scoped.scopes[scope_id].reference_ids))
                if (
                    largest != only
                    and len(scoped.scopes[only].reference_ids)
                    < len(scoped.scopes[largest].reference_ids)
                ):
                    return largest, None
        return only, None
    if len(owned_candidates) > 1:
        if region_type in {REGION_MAIN, REGION_ABSTRACT}:
            return (
                max(owned_candidates, key=lambda scope_id: len(scoped.scopes[scope_id].reference_ids)),
                None,
            )
        return None, "AMBIGUOUS_BIBLIOGRAPHY_SCOPE"
    if len(unowned_candidates) == 1:
        return unowned_candidates[0], None
    if len(unowned_candidates) > 1:
        if region_type in {REGION_MAIN, REGION_ABSTRACT}:
            return (
                max(unowned_candidates, key=lambda scope_id: len(scoped.scopes[scope_id].reference_ids)),
                None,
            )
        return None, "AMBIGUOUS_BIBLIOGRAPHY_SCOPE"

    candidates = owned_candidates or unowned_candidates
    if len(candidates) == 1:
        return candidates[0], None
    if len(candidates) > 1:
        if region_type in {REGION_MAIN, REGION_ABSTRACT}:
            return (
                max(candidates, key=lambda scope_id: len(scoped.scopes[scope_id].reference_ids)),
                None,
            )
        return None, "AMBIGUOUS_BIBLIOGRAPHY_SCOPE"
    if not candidates:
        fallback = scopes_for_region(region_type, scoped)
        if len(fallback) == 1:
            return fallback[0], None
        if region_type in {REGION_MAIN, REGION_ABSTRACT}:
            pool = [
                scope_id
                for scope_id, scope in scoped.scopes.items()
                if scope.parent_region in {REGION_MAIN, REGION_ABSTRACT}
            ]
            if len(pool) == 1:
                return pool[0], None
            if pool:
                return (
                    max(pool, key=lambda scope_id: len(scoped.scopes[scope_id].reference_ids)),
                    None,
                )
        if region_type == REGION_SUPPLEMENT:
            pool = [
                scope_id
                for scope_id, scope in scoped.scopes.items()
                if scope.parent_region in {REGION_MAIN, REGION_ABSTRACT, REGION_SUPPLEMENT}
            ]
            if pool:
                return (
                    max(pool, key=lambda scope_id: len(scoped.scopes[scope_id].reference_ids)),
                    None,
                )
        return None, FAILURE_SCOPE_NOT_FOUND


FAILURE_SCOPE_NOT_FOUND = "SCOPE_NOT_FOUND"


def assign_bibliography_scopes(
    *,
    references: list[ReferenceEntry],
    elements: list[Element],
    sections: list[Section],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (reference_id -> scope_id, scope_id -> parent_region)."""
    from paperos_core.ingestion.document_regions import build_document_regions

    section_by_id = {section.id: section for section in sections}
    regions, element_regions = build_document_regions(elements=elements, sections=sections)
    ref_elements = [
        element
        for element in sorted(elements, key=lambda item: item.order)
        if element.element_type == ElementType.REFERENCE
    ]

    scopes: dict[str, BibliographyScope] = {}
    element_scope: dict[str, str] = {}
    reference_scope: dict[str, str] = {}
    current_scope_id: str | None = None
    current_region: str | None = None
    previous_order: int | None = None
    scope_counter = 0

    def _open_scope(region: str, *, owner_body_region_id: str | None = None) -> str:
        nonlocal scope_counter, current_scope_id, current_region
        scope_counter += 1
        scope_id = f"bib_scope_{scope_counter}"
        scopes[scope_id] = BibliographyScope(
            scope_id=scope_id,
            parent_region=region,
            owner_body_region_id=owner_body_region_id,
        )
        current_scope_id = scope_id
        current_region = region
        return scope_id

    for ref_element in ref_elements:
        region, owner_body_region_id = _reference_scope_parent_region(
            ref_element,
            element_regions=element_regions,
            regions=regions,
            section_by_id=section_by_id,
        )
        needs_new_scope = current_scope_id is None or region != current_region
        if previous_order is not None and ref_element.order - previous_order > 1:
            needs_new_scope = True
        if needs_new_scope:
            _open_scope(region, owner_body_region_id=owner_body_region_id)
        elif (
            owner_body_region_id is not None
            and current_scope_id
            and scopes[current_scope_id].owner_body_region_id != owner_body_region_id
        ):
            _open_scope(region, owner_body_region_id=owner_body_region_id)
        element_scope[ref_element.id] = current_scope_id or _open_scope(
            region, owner_body_region_id=owner_body_region_id
        )
        previous_order = ref_element.order

    default_scope = current_scope_id or _open_scope(REGION_MAIN)
    for reference in sorted(references, key=lambda item: item.order):
        if reference.source_element_id and reference.source_element_id in element_scope:
            scope_id = element_scope[reference.source_element_id]
        else:
            scope_id = default_scope
        scopes[scope_id].reference_ids.append(reference.id)
        reference_scope[reference.id] = scope_id

    parent_regions = {scope.scope_id: scope.parent_region for scope in scopes.values()}
    return reference_scope, scopes


def repair_numeric_label_sequence(
    references: list[ReferenceEntry],
    *,
    scope_id: str | None = None,
    reference_scope: dict[str, str] | None = None,
) -> list[ReferenceEntry]:
    """Infer missing numeric labels between labeled neighbours in one scope."""
    if reference_scope is None:
        scoped_refs = list(references)
    else:
        scoped_refs = [
            reference
            for reference in references
            if reference_scope.get(reference.id) == scope_id
        ]
    scoped_refs = sorted(scoped_refs, key=lambda item: item.order)
    repaired: list[ReferenceEntry] = []
    for index, reference in enumerate(scoped_refs):
        label = reference.citation_label or _reference_numeric_label(reference)
        if label:
            repaired.append(reference)
            continue
        prev_label = _neighbor_numeric_label(scoped_refs, index, step=-1)
        next_label = _neighbor_numeric_label(scoped_refs, index, step=1)
        if (
            prev_label is not None
            and next_label is not None
            and next_label - prev_label == 2
        ):
            candidate = prev_label + 1
            repaired.append(
                reference.model_copy(
                    update={
                        "citation_label": str(candidate),
                        "parsed_fields": {
                            **reference.parsed_fields,
                            "citation_label": str(candidate),
                            "label_kind": "numeric",
                            "reference_number": candidate,
                            "label_source": "inferred_sequence",
                        },
                    }
                )
            )
            continue
        repaired.append(reference)
    if reference_scope is None:
        return repaired
    repaired_by_id = {reference.id: reference for reference in repaired}
    return [repaired_by_id.get(reference.id, reference) for reference in references]


def _reference_scope_parent_region(
    ref_element: Element,
    *,
    element_regions: dict,
    regions: list,
    section_by_id: dict[str, Section],
) -> tuple[str, str | None]:
    order = ref_element.order
    for region in regions:
        if region.region_type != REGION_REFERENCES:
            continue
        end_order = region.end_order if region.end_order is not None else order
        if region.start_order <= order <= end_order:
            owner = region.owner_body_region_id
            if owner and REGION_SUPPLEMENT in owner:
                return REGION_SUPPLEMENT, owner
            return REGION_MAIN, owner
    info = element_regions.get(ref_element.id)
    if info is not None:
        if info.region_type == REGION_SUPPLEMENT or (
            info.region_subtype == "appendix"
        ):
            return REGION_SUPPLEMENT, info.region_id
        return REGION_MAIN, info.region_id
    region = resolve_element_region(ref_element.section_id, section_by_id)
    if region == REGION_REFERENCES:
        return REGION_MAIN, None
    if region == REGION_SUPPLEMENT:
        return REGION_SUPPLEMENT, None
    return REGION_MAIN, None


def _infer_references_parent_region(
    section_id: str | None,
    section_by_id: dict[str, Section],
) -> str:
    if section_id is None:
        return REGION_MAIN
    chain: list[Section] = []
    current = section_by_id.get(section_id)
    while current is not None:
        if current.section_type != "references":
            chain.append(current)
        current = (
            section_by_id.get(current.parent_section_id)
            if current.parent_section_id
            else None
        )
    for section in chain:
        title = section.title.casefold()
        if any(hint in title for hint in _SUPPLEMENT_HINTS):
            return REGION_SUPPLEMENT
    return REGION_MAIN


def _neighbor_numeric_label(
    references: list[ReferenceEntry],
    index: int,
    *,
    step: int,
) -> int | None:
    cursor = index + step
    while 0 <= cursor < len(references):
        reference = references[cursor]
        label = reference.citation_label or _reference_numeric_label(reference)
        if label and re.fullmatch(r"\d+", _normalize_label(label)):
            return int(_normalize_label(label))
        cursor += step
    return None


def _normalize_label(label: str) -> str:
    value = label.strip()
    if len(value) >= 2 and (
        (value[0] == "[" and value[-1] == "]")
        or (value[0] == "(" and value[-1] == ")")
    ):
        value = value[1:-1].strip()
    return value


def _reference_numeric_label(reference: ReferenceEntry) -> str | None:
    ref_num = reference.parsed_fields.get("reference_number")
    if ref_num is not None:
        return str(ref_num)
    return None
