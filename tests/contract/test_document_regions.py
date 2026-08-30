"""DocumentRegion -> CitationNamespace architecture contracts."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.domain.canonical import Element, Section
from paperos_core.domain.enums import ElementType
from paperos_core.ingestion.document_regions import build_document_regions, region_for_element


def _section(
    order: int,
    title: str,
    *,
    section_type: str | None = None,
    parent_section_id: str | None = None,
) -> Section:
    return Section(
        id=f"section_{order}",
        document_id="doc",
        canonical_snapshot_id="snapshot",
        title=title,
        level=1,
        order=order,
        path=title,
        section_type=section_type,
        parent_section_id=parent_section_id,
    )


def _element(
    order: int,
    element_type: ElementType,
    text: str,
    section_id: str | None = None,
) -> Element:
    return Element(
        id=f"element_{order}",
        document_id="doc",
        canonical_snapshot_id="snapshot",
        element_type=element_type,
        order=order,
        text=text,
        section_id=section_id,
    )


def test_reference_order_gaps_do_not_split_namespace() -> None:
    sections = [_section(0, "Introduction"), _section(1, "References", section_type="references")]
    elements = [
        _element(0, ElementType.PARAGRAPH, "Body [1].", "section_0"),
        _element(1, ElementType.TITLE, "References", "section_1"),
        _element(2, ElementType.REFERENCE, "[1] One.", "section_1"),
        _element(3, ElementType.HEADER, "journal header", "section_1"),
        _element(4, ElementType.REFERENCE, "[2] Two.", "section_1"),
    ]
    regions, binding = build_document_regions(elements=elements, sections=sections)
    reference_regions = [region for region in regions if region.region_type == "references"]
    assert len(reference_regions) == 1
    assert binding["element_2"].citation_namespace_id == binding["element_4"].citation_namespace_id


def test_body_regions_use_nearest_following_bibliography() -> None:
    sections = [
        _section(0, "Main"),
        _section(1, "References", section_type="references"),
        _section(2, "Supplementary Material"),
        _section(3, "References", section_type="references"),
    ]
    elements = [
        _element(0, ElementType.PARAGRAPH, "Main [1].", "section_0"),
        _element(1, ElementType.TITLE, "References", "section_1"),
        _element(2, ElementType.REFERENCE, "[1] Main ref.", "section_1"),
        _element(3, ElementType.TITLE, "Supplementary Material", "section_2"),
        _element(4, ElementType.PARAGRAPH, "Supplement [1].", "section_2"),
        _element(5, ElementType.TITLE, "References", "section_3"),
        _element(6, ElementType.REFERENCE, "[1] Supplement ref.", "section_3"),
    ]
    _, binding = build_document_regions(elements=elements, sections=sections)
    assert binding["element_0"].citation_namespace_id == "citation_namespace_1"
    assert binding["element_4"].citation_namespace_id == "citation_namespace_2"


def test_appendix_after_references_inherits_previous_namespace() -> None:
    sections = [
        _section(0, "Main"),
        _section(1, "References", section_type="references"),
        _section(2, "Appendix A"),
    ]
    elements = [
        _element(0, ElementType.PARAGRAPH, "Main [1].", "section_0"),
        _element(1, ElementType.TITLE, "References", "section_1"),
        _element(2, ElementType.REFERENCE, "[1] Ref.", "section_1"),
        _element(3, ElementType.TITLE, "Appendix A", "section_2"),
        _element(4, ElementType.PARAGRAPH, "Appendix [1].", "section_2"),
    ]
    _, binding = build_document_regions(elements=elements, sections=sections)
    assert binding["element_0"].citation_namespace_id == "citation_namespace_1"
    assert binding["element_4"].region_type == "supplement"
    assert binding["element_4"].citation_namespace_id == "citation_namespace_1"


def test_supplement_heading_repairs_stale_references_section_ancestry() -> None:
    sections = [
        _section(0, "Main"),
        _section(1, "References", section_type="references"),
        _section(2, "Minimal surfaces", parent_section_id="section_1"),
        _section(3, "References", section_type="references"),
    ]
    elements = [
        _element(0, ElementType.PARAGRAPH, "Main [1].", "section_0"),
        _element(1, ElementType.TITLE, "References", "section_1"),
        _element(2, ElementType.REFERENCE, "[1] Main ref.", "section_1"),
        _element(
            3,
            ElementType.TITLE,
            "Paper Title – Supplementary Material –",
            "section_1",
        ),
        _element(4, ElementType.TITLE, "1. Minimal surfaces", "section_2"),
        _element(5, ElementType.PARAGRAPH, "Supplement [1].", "section_2"),
        _element(6, ElementType.TITLE, "References", "section_3"),
        _element(7, ElementType.REFERENCE, "[1] Supplement ref.", "section_3"),
    ]
    regions, binding = build_document_regions(elements=elements, sections=sections)
    assert len([region for region in regions if region.region_type == "references"]) == 2
    assert binding["element_5"].region_type == "supplement"
    assert binding["element_5"].citation_namespace_id == "citation_namespace_2"


def test_abstract_region_is_preserved_while_sharing_main_namespace() -> None:
    sections = [
        _section(0, "Abstract", section_type="abstract"),
        _section(1, "Introduction"),
        _section(2, "References", section_type="references"),
    ]
    elements = [
        _element(0, ElementType.PARAGRAPH, "Abstract citation [1].", "section_0"),
        _element(1, ElementType.PARAGRAPH, "Main citation [1].", "section_1"),
        _element(2, ElementType.TITLE, "References", "section_2"),
        _element(3, ElementType.REFERENCE, "[1] Shared ref.", "section_2"),
    ]
    _, binding = build_document_regions(elements=elements, sections=sections)
    assert region_for_element("element_0", binding) == "abstract"
    assert binding["element_0"].citation_namespace_id == binding["element_1"].citation_namespace_id
