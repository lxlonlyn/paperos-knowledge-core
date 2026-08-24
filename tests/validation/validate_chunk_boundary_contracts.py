"""Validate abstract, formula-cohesion, and table-part chunk contracts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.domain.canonical import Document, Element, Section
from paperos_core.domain.enums import ElementType
from paperos_core.ingestion.chunk_eligibility import classify_chunk_eligibility
from paperos_core.ingestion.chunking import _is_subsection_boundary, build_chunks
from paperos_core.ingestion.document_regions import (
    build_document_regions,
    region_id_for_element,
)
from paperos_core.ingestion.sentence_units import (
    SentenceUnit,
    formula_cohesion_boundary,
    resolve_major_section_id,
    units_for_element,
)
from tests.validation.chunk_corpus_review import (
    _load_bundle_from_snapshot_dir,
    _resolve_tokenizer,
)
from tests.validation.validate_chunk_regions_scopes import _chunks_for_snapshot


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _unit_groups(*, bundle: Any, count: Any, hard_max_tokens: int) -> list[list[SentenceUnit]]:
    section_by_id = {section.id: section for section in bundle.sections}
    _, element_regions = build_document_regions(elements=bundle.elements, sections=bundle.sections)
    eligible = []
    for element in sorted(bundle.elements, key=lambda item: (item.order, item.id)):
        info = element_regions.get(element.id)
        eligibility = classify_chunk_eligibility(
            element,
            section_by_id=section_by_id,
            region_type=info.region_type if info else None,
        )
        if eligibility.eligible:
            eligible.append(element)

    grouped: dict[tuple[str, str], list[Element]] = {}
    for element in eligible:
        major_id = resolve_major_section_id(element.section_id, section_by_id)
        if major_id is None:
            continue
        region_id = region_id_for_element(element.id, element_regions) or "region_main_1"
        grouped.setdefault((major_id, region_id), []).append(element)

    groups: list[list[SentenceUnit]] = []
    for key in sorted(grouped):
        elements = grouped[key]
        units: list[SentenceUnit] = []
        for index, element in enumerate(elements):
            section = section_by_id.get(element.section_id or "")
            units.extend(
                units_for_element(
                    element,
                    count=count,
                    hard_max_tokens=hard_max_tokens,
                    section_id=element.section_id,
                    section_path=section.path if section else None,
                    subsection_end=_is_subsection_boundary(elements, index, section_by_id),
                )
            )
        groups.append(units)
    return groups


def _paper_contracts(
    *, bundle: Any, chunks_json: dict[str, Any], count: Any, hard_max_tokens: int
) -> dict[str, Any]:
    _, element_regions = build_document_regions(elements=bundle.elements, sections=bundle.sections)
    abstract_errors: list[dict[str, Any]] = []
    for chunk in chunks_json.get("chunks", []):
        infos = [element_regions.get(span.get("element_id")) for span in chunk.get("spans", [])]
        if (
            infos
            and all(info is not None and info.region_type == "abstract" for info in infos)
            and chunk.get("document_region") != "abstract"
        ):
            abstract_errors.append(
                {
                    "chunk_id": chunk.get("id"),
                    "actual": chunk.get("document_region"),
                }
            )

    span_chunks: dict[str, set[str]] = {}
    for chunk in chunks_json.get("chunks", []):
        for span in chunk.get("spans", []):
            span_chunks.setdefault(span["id"], set()).add(chunk["id"])

    cohesion_cases = 0
    cohesion_breaks: list[dict[str, Any]] = []
    table_misclassifications: list[dict[str, Any]] = []
    unit_by_span: dict[str, SentenceUnit] = {}
    for units in _unit_groups(bundle=bundle, count=count, hard_max_tokens=hard_max_tokens):
        for unit in units:
            unit_by_span[unit.span_id] = unit
            if unit.split_type == "TABLE_PART" and unit.emergency_split:
                table_misclassifications.append(
                    {"span_id": unit.span_id, "split_type": unit.split_type}
                )
        for index in range(len(units) - 1):
            if not formula_cohesion_boundary(units[index], units[index + 1]):
                continue
            group_start = index
            while group_start > 0 and formula_cohesion_boundary(
                units[group_start - 1], units[group_start]
            ):
                group_start -= 1
            group_end = index + 2
            while group_end < len(units) and formula_cohesion_boundary(
                units[group_end - 1], units[group_end]
            ):
                group_end += 1
            group = units[group_start:group_end]
            if sum(unit.tokens for unit in group) > hard_max_tokens:
                continue
            cohesion_cases += 1
            shared_chunks = span_chunks.get(units[index].span_id, set()) & span_chunks.get(
                units[index + 1].span_id, set()
            )
            if not shared_chunks:
                cohesion_breaks.append(
                    {
                        "left_span_id": units[index].span_id,
                        "right_span_id": units[index + 1].span_id,
                        "group_tokens": sum(unit.tokens for unit in group),
                    }
                )

    metadata_errors: list[dict[str, Any]] = []
    table_parts = 0
    real_emergency_splits = 0
    for chunk in chunks_json.get("chunks", []):
        chunk_units = [
            unit_by_span[span["id"]]
            for span in chunk.get("spans", [])
            if span["id"] in unit_by_span
        ]
        expected_table = sum(unit.split_type == "TABLE_PART" for unit in chunk_units)
        expected_emergency = sum(unit.emergency_split for unit in chunk_units)
        table_parts += expected_table
        real_emergency_splits += expected_emergency
        metadata = chunk.get("metadata") or {}
        if int(metadata.get("table_parts") or 0) != expected_table:
            metadata_errors.append(
                {"chunk_id": chunk.get("id"), "failure_type": "TABLE_PART_COUNT"}
            )
        if (
            int(metadata.get("real_emergency_splits") or 0) != expected_emergency
            or int(metadata.get("emergency_oversized_sentence_splits") or 0) != expected_emergency
        ):
            metadata_errors.append(
                {"chunk_id": chunk.get("id"), "failure_type": "REAL_EMERGENCY_COUNT"}
            )

    return {
        "abstract_region_errors": len(abstract_errors),
        "abstract_failures": abstract_errors,
        "formula_cohesion_cases": cohesion_cases,
        "avoidable_formula_cohesion_breaks": len(cohesion_breaks),
        "formula_cohesion_failures": cohesion_breaks,
        "table_parts": table_parts,
        "real_emergency_splits": real_emergency_splits,
        "table_part_emergency_misclassification": (
            len(table_misclassifications) + len(metadata_errors)
        ),
        "table_failures": [*table_misclassifications, *metadata_errors],
    }


class _CharacterTokenizer:
    def count_tokens(self, text: str) -> int:
        return len(text)


def synthetic_multi_part_table_contract() -> dict[str, Any]:
    header = "| Col A | Col B |\n| --- | --- |\n"
    rows = "".join(f"| row{index:02d} value | payload{index:02d} |\n" for index in range(1, 7))
    source = header + rows
    section = Section(
        id="section_table",
        document_id="doc_table",
        canonical_snapshot_id="snapshot_table",
        title="Results",
        level=1,
        order=0,
        path="Results",
    )
    table = Element(
        id="element_table",
        document_id="doc_table",
        canonical_snapshot_id="snapshot_table",
        element_type=ElementType.TABLE,
        order=0,
        section_id=section.id,
        markdown=source,
    )
    tokenizer = _CharacterTokenizer()
    units = units_for_element(
        table,
        count=tokenizer.count_tokens,
        hard_max_tokens=75,
        section_id=section.id,
        section_path=section.path,
        subsection_end=True,
    )
    failures: list[str] = []
    if len(units) < 3:
        failures.append(f"table_part_count:{len(units)}")
    cursor = 0
    reconstructed: list[str] = []
    for unit in units:
        if unit.character_start_in_element != cursor:
            failures.append(f"source_gap_or_overlap:{cursor}:{unit.character_start_in_element}")
        reconstructed.append(unit.text)
        cursor = unit.character_end_in_element
        if unit.split_type != "TABLE_PART" or unit.emergency_split:
            failures.append(f"emergency_misclassification:{unit.span_id}")
    if cursor != len(source) or "".join(reconstructed) != source:
        failures.append("authoritative_source_coverage")
    for unit in units[1:]:
        if not (unit.display_text or "").startswith(header.rstrip("\n")):
            failures.append(f"display_header_missing:{unit.span_id}")
        if unit.text.startswith(header.rstrip("\n")):
            failures.append(f"authoritative_header_repeated:{unit.span_id}")

    document = Document(
        id="doc_table",
        source_file_id="source_table",
        parse_run_id="parse_table",
        canonical_snapshot_id="snapshot_table",
        language="en",
        title="Synthetic table contract",
    )
    chunks, _ = build_chunks(
        document=document,
        snapshot_id="snapshot_table",
        sections=[section],
        elements=[table],
        references=[],
        target_tokens=60,
        hard_max_tokens=75,
        overlap_tokens=0,
        tokenizer=tokenizer,
    )
    for chunk in chunks:
        starts_after_header = any(span.character_start_in_element > 0 for span in chunk.spans)
        if starts_after_header:
            if header.rstrip("\n") not in (chunk.retrieval_text or ""):
                failures.append(f"retrieval_header_missing:{chunk.id}")
            if chunk.text.startswith(header.rstrip("\n")):
                failures.append(f"chunk_authoritative_header_repeated:{chunk.id}")
        if int(chunk.metadata.get("real_emergency_splits") or 0) != 0:
            failures.append(f"chunk_emergency_misclassification:{chunk.id}")

    return {
        "multi_part_table_provenance_errors": len(failures),
        "table_part_count": len(units),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/validation/runs/chunk"))
    args = parser.parse_args()
    tokenizer = _resolve_tokenizer()
    paper_results: list[dict[str, Any]] = []
    for src_dir in sorted((args.run_dir / "canonical").glob("src_*")):
        snapshot_dirs = sorted(src_dir.glob("snapshot_*"))
        if len(snapshot_dirs) != 1:
            raise RuntimeError(
                f"Expected one canonical snapshot in {src_dir}, got {len(snapshot_dirs)}"
            )
        bundle = _load_bundle_from_snapshot_dir(snapshot_dirs[0])
        chunks_json = _chunks_for_snapshot(args.run_dir, bundle.snapshot.id)
        hard_max = 1200
        paper_results.append(
            {
                "source_id": src_dir.name,
                "title": bundle.document.title,
                **_paper_contracts(
                    bundle=bundle,
                    chunks_json=chunks_json,
                    count=tokenizer.count_tokens,
                    hard_max_tokens=hard_max,
                ),
            }
        )

    synthetic = synthetic_multi_part_table_contract()
    report = {
        "git_commit": _git_commit(),
        "paper_count": len(paper_results),
        "abstract_region_errors": sum(item["abstract_region_errors"] for item in paper_results),
        "formula_cohesion_cases": sum(item["formula_cohesion_cases"] for item in paper_results),
        "avoidable_formula_cohesion_breaks": sum(
            item["avoidable_formula_cohesion_breaks"] for item in paper_results
        ),
        "table_parts": sum(item["table_parts"] for item in paper_results),
        "real_emergency_splits": sum(item["real_emergency_splits"] for item in paper_results),
        "table_part_emergency_misclassification": sum(
            item["table_part_emergency_misclassification"] for item in paper_results
        ),
        **synthetic,
        "papers": paper_results,
    }
    report["pass"] = all(
        report[key] == 0
        for key in (
            "abstract_region_errors",
            "avoidable_formula_cohesion_breaks",
            "table_part_emergency_misclassification",
            "multi_part_table_provenance_errors",
        )
    )
    output = args.run_dir / "chunk-boundary-contracts.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                **{key: report[key] for key in report if key not in {"papers", "failures"}},
                "report": str(output),
            },
            indent=2,
        )
    )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
