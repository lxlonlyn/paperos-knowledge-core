"""Bibliography scope assignment and document-region resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from paperos_core.domain.canonical import Element, ReferenceEntry, Section
from paperos_core.domain.enums import ElementType
from paperos_core.ingestion.normalization import plain_text

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
    if region == REGION_SUPPLEMENT:
        return [
            scope.scope_id
            for scope in scoped.scopes.values()
            if scope.parent_region == REGION_SUPPLEMENT
        ]
    # Main body and abstract citations use main-bibliography scopes.
    return [
        scope.scope_id
        for scope in scoped.scopes.values()
        if scope.parent_region in {REGION_MAIN, REGION_ABSTRACT}
    ]


def assign_bibliography_scopes(
    *,
    references: list[ReferenceEntry],
    elements: list[Element],
    sections: list[Section],
) -> tuple[dict[str, str], dict[str, str]]:
    """Return (reference_id -> scope_id, scope_id -> parent_region)."""
    section_by_id = {section.id: section for section in sections}
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

    def _open_scope(region: str) -> str:
        nonlocal scope_counter, current_scope_id, current_region
        scope_counter += 1
        scope_id = f"bib_scope_{scope_counter}"
        scopes[scope_id] = BibliographyScope(scope_id=scope_id, parent_region=region)
        current_scope_id = scope_id
        current_region = region
        return scope_id

    for ref_element in ref_elements:
        region = resolve_element_region(ref_element.section_id, section_by_id)
        if region == REGION_REFERENCES:
            region = _infer_references_parent_region(
                ref_element.section_id, section_by_id
            )
        needs_new_scope = current_scope_id is None or region != current_region
        if previous_order is not None and ref_element.order - previous_order > 1:
            needs_new_scope = True
        if needs_new_scope:
            _open_scope(region)
        element_scope[ref_element.id] = current_scope_id or _open_scope(region)
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
    return reference_scope, parent_regions


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
