"""Validation against the genuine-corpus minimum structural expectations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from paperos_core.domain.canonical import CanonicalBundle
from paperos_core.domain.documents import SourceFile
from paperos_core.domain.enums import ElementType
from paperos_core.errors import CanonicalValidationError
from paperos_core.ingestion.normalization import normalized_match_text


def validate_expected_case(
    *,
    bundle: CanonicalBundle,
    source: SourceFile,
    expected_path: Path,
) -> dict[str, Any]:
    """Apply one checked-in expectation file without repairing or seeding data."""
    try:
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalValidationError(
            f"Unable to read expected-corpus definition: {exc}",
            affected=expected_path,
        ) from exc
    failures: list[str] = []
    integrity = expected["source_integrity"]
    actual_sha = hashlib.sha256(source.storage_path.read_bytes()).hexdigest()
    _check(
        actual_sha == integrity["sha256"] == source.sha256,
        "source SHA-256 does not match the expected corpus",
        failures,
    )
    _check(
        source.storage_path.read_bytes()[:5] == b"%PDF-",
        "stored source does not have a PDF header",
        failures,
    )
    pages = [element.page for element in bundle.elements if element.page is not None]
    actual_pages = max(pages, default=0)
    expected_pages = int(integrity["expected_page_count"])
    tolerance = int(integrity.get("page_count_tolerance", 0))
    _check(
        abs(actual_pages - expected_pages) <= tolerance,
        f"page count {actual_pages} is outside expected {expected_pages}±{tolerance}",
        failures,
    )

    document_expected = expected["document"]
    document = bundle.document
    _check(
        normalized_match_text(document.title)
        == normalized_match_text(document_expected["expected_title"]),
        f"title mismatch: {document.title!r}",
        failures,
    )
    _check(
        document.year == document_expected["expected_year"],
        f"year mismatch: {document.year!r}",
        failures,
    )
    _check(
        (document.doi or "").casefold() == (document_expected["expected_doi"] or "").casefold(),
        f"DOI mismatch: {document.doi!r}",
        failures,
    )
    _check(
        document.language == document_expected["language"],
        f"language mismatch: {document.language!r}",
        failures,
    )
    _check(
        document.document_type == document_expected["document_type"],
        f"document type mismatch: {document.document_type!r}",
        failures,
    )
    authors_expected = document_expected["authors"]
    if authors_expected["mode"] == "contains_all":
        actual_authors = {normalized_match_text(author.display_name) for author in document.authors}
        for author in authors_expected["values"]:
            _check(
                normalized_match_text(author) in actual_authors,
                f"expected author is missing: {author}",
                failures,
            )

    structure = expected["structure"]
    section_titles = [normalized_match_text(section.title) for section in bundle.sections]
    for required in structure["required_sections"]:
        needle = normalized_match_text(required["title"])
        if required["match"] == "normalized_contains":
            matched = any(needle in value for value in section_titles)
        else:
            matched = needle in section_titles
        _check(matched, f"required section is missing: {required['title']}", failures)
    _check(
        len(bundle.sections) >= structure["minimum_section_count"],
        f"section count {len(bundle.sections)} is below minimum",
        failures,
    )
    _check(
        len(bundle.chunks) >= structure["minimum_chunk_count"],
        f"chunk count {len(bundle.chunks)} is below minimum",
        failures,
    )
    _check(
        len(bundle.references) >= structure["minimum_reference_count"],
        f"reference count {len(bundle.references)} is below minimum",
        failures,
    )

    elements_expected = expected["elements"]
    type_counts = {
        kind.value: sum(element.element_type == kind for element in bundle.elements)
        for kind in ElementType
    }
    for required_type in elements_expected["must_contain"]:
        _check(
            type_counts.get(required_type, 0) > 0,
            f"required element type is missing: {required_type}",
            failures,
        )
    _check(
        type_counts["figure"] >= elements_expected["minimum_figure_count"],
        f"figure count {type_counts['figure']} is below minimum",
        failures,
    )
    _check(
        type_counts["formula"] >= elements_expected["minimum_formula_count"],
        f"formula count {type_counts['formula']} is below minimum",
        failures,
    )
    if elements_expected.get("require_figure_captions"):
        _check(
            any(
                element.element_type == ElementType.FIGURE and element.caption_element_ids
                for element in bundle.elements
            ),
            "no figure retains a caption relation",
            failures,
        )
    if elements_expected.get("require_reference_entries"):
        _check(bool(bundle.references), "reference entries are missing", failures)

    searchable = normalized_match_text(
        "\n".join(
            [
                document.title,
                document.abstract or "",
                *(element.text or element.latex or "" for element in bundle.elements),
            ]
        )
    )
    for check in expected.get("content_checks", []):
        if not check.get("required"):
            continue
        matched = any(normalized_match_text(value) in searchable for value in check["any_of"])
        _check(
            matched,
            "required content is missing: " + " | ".join(check["any_of"]),
            failures,
        )

    canonical = expected["canonical_requirements"]
    if canonical.get("require_section_paths"):
        _check(
            all(section.path for section in bundle.sections),
            "one or more sections have no path",
            failures,
        )
    if canonical.get("require_page_ranges"):
        _check(
            all(
                section.page_start is not None and section.page_end is not None
                for section in bundle.sections
            )
            and all(
                chunk.page_start is not None and chunk.page_end is not None
                for chunk in bundle.chunks
            ),
            "one or more sections/chunks have no page range",
            failures,
        )
    if canonical.get("require_chunk_element_links"):
        _check(
            all(chunk.element_ids for chunk in bundle.chunks),
            "one or more chunks have no element links",
            failures,
        )
    if canonical.get("require_source_spans_when_available"):
        _check(
            all(element.source_span is not None for element in bundle.elements),
            "one or more elements have no source span",
            failures,
        )
    if failures:
        raise CanonicalValidationError(
            f"Canonical snapshot failed {len(failures)} expected-corpus checks.",
            affected=expected_path,
            details={"failures": failures},
        )
    return {
        "case_id": expected["case_id"],
        "expected_path": str(expected_path),
        "passed": True,
        "counts": {
            "pages": actual_pages,
            "sections": len(bundle.sections),
            "chunks": len(bundle.chunks),
            "references": len(bundle.references),
            "element_types": type_counts,
        },
    }


def expected_path_for_source(expected_dir: Path, source: SourceFile) -> Path:
    for path in sorted(expected_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if payload.get("pdf_file") == source.original_filename:
            return path
    raise CanonicalValidationError(
        "No expected-corpus definition matches the source PDF.",
        affected=source.original_filename,
    )


def _check(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)
