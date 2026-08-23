#!/usr/bin/env python3
"""Validate MinerU source fields → Canonical provenance using Gold v3 anchors."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.domain.enums import ElementType  # noqa: E402
from paperos_core.ingestion.normalization import source_evidence_text  # noqa: E402
from paperos_core.ingestion.sentence_units import element_text  # noqa: E402
from tests.validation.build_citation_gold_v3 import (  # noqa: E402
    _flex_occurrences,
    _mineru_content,
    _source_fields,
)
from tests.validation.chunk_corpus_review import (  # noqa: E402
    _load_bundle_from_snapshot_dir,
)


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


def _field_key(record: dict[str, Any]) -> tuple[int, str, int | None]:
    return (
        record["source_item"],
        record["source_domain"],
        record.get("source_subindex"),
    )


def _span_key(span: Any) -> tuple[int, str, int | None]:
    return (span.item_index, span.source_domain, span.source_subindex)


def _expected_element_text(element_type: ElementType, source: str) -> str:
    if element_type == ElementType.FORMULA:
        value = source_evidence_text(source)
        value = re.sub(r"^\$\$\s*", "", value)
        return re.sub(r"\s*\$\$$", "", value).strip()
    return source_evidence_text(source)


def _check_element_provenance(
    *,
    paper_key: str,
    bundle: Any,
    fields_by_key: dict[tuple[int, str, int | None], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for element in bundle.elements:
        evidence = element_text(element)
        if not evidence:
            continue
        span = element.source_span
        if span is None:
            failures.append(
                {
                    "failure_type": "CANONICAL_SOURCE_PROVENANCE_MISSING",
                    "paper": paper_key,
                    "element_id": element.id,
                }
            )
            continue
        fields = fields_by_key.get(_span_key(span), [])
        if len(fields) != 1:
            failures.append(
                {
                    "failure_type": "CANONICAL_SOURCE_FIELD_MISSING",
                    "paper": paper_key,
                    "element_id": element.id,
                    "source_item": span.item_index,
                    "source_domain": span.source_domain,
                    "source_subindex": span.source_subindex,
                    "field_matches": len(fields),
                }
            )
            continue
        source = fields[0]["value"]
        start = span.character_start
        end = span.character_end
        if start is None or end is None or not (0 <= start <= end <= len(source)):
            failures.append(
                {
                    "failure_type": "CANONICAL_SOURCE_RANGE_INVALID",
                    "paper": paper_key,
                    "element_id": element.id,
                    "source_item": span.item_index,
                    "start": start,
                    "end": end,
                    "source_length": len(source),
                }
            )
            continue
        source_slice = source[start:end]
        expected = (
            source_slice
            if element.element_type == ElementType.TABLE
            else _expected_element_text(element.element_type, source_slice)
        )
        if evidence != expected:
            failures.append(
                {
                    "failure_type": "CANONICAL_SOURCE_CONTENT_MISMATCH",
                    "paper": paper_key,
                    "element_id": element.id,
                    "source_item": span.item_index,
                    "source_domain": span.source_domain,
                    "canonical_length": len(evidence),
                    "expected_length": len(expected),
                }
            )
    return failures


def _check_gold_anchors(
    *,
    paper_key: str,
    paper: dict[str, Any],
    bundle: Any,
    fields_by_key: dict[tuple[int, str, int | None], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    elements_by_key: dict[tuple[int, str, int | None], list[Any]] = defaultdict(list)
    for element in bundle.elements:
        if element.source_span is not None:
            elements_by_key[_span_key(element.source_span)].append(element)

    failures: list[dict[str, Any]] = []
    for occurrence in paper["occurrences"]:
        key = _field_key(occurrence)
        fields = fields_by_key.get(key, [])
        start = occurrence["start"]
        end = occurrence["end"]
        source_valid = (
            len(fields) == 1
            and 0 <= start <= end <= len(fields[0]["value"])
            and fields[0]["value"][start:end] == occurrence["surface"]
        )
        if not source_valid:
            failures.append(
                {
                    "failure_type": "GOLD_MINERU_ANCHOR_INVALID",
                    "paper": paper_key,
                    "source_item": occurrence["source_item"],
                    "source_domain": occurrence["source_domain"],
                    "start": start,
                    "end": end,
                    "surface": occurrence["surface"],
                }
            )
            continue

        mapped = False
        for element in elements_by_key.get(key, []):
            span = element.source_span
            if span.character_start is None or span.character_end is None:
                continue
            if not (span.character_start <= start and end <= span.character_end):
                continue
            if _flex_occurrences(element_text(element), occurrence["surface"]):
                mapped = True
                break
        if not mapped:
            failures.append(
                {
                    "failure_type": "GOLD_CANONICAL_SOURCE_LOSS",
                    "paper": paper_key,
                    "source_item": occurrence["source_item"],
                    "source_domain": occurrence["source_domain"],
                    "source_subindex": occurrence.get("source_subindex"),
                    "start": start,
                    "end": end,
                    "surface": occurrence["surface"],
                }
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/validation/runs/chunk"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/validation/corpus/chunk"))
    parser.add_argument("--gold", type=Path, default=Path("tests/fixtures/chunk/citation_gold_v3.json"))
    args = parser.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    papers_report: dict[str, dict[str, Any]] = {}
    for paper_key, paper in gold["papers"].items():
        snapshot_dirs = sorted((args.run_dir / "canonical" / paper["source_id"]).glob("snapshot_*"))
        if len(snapshot_dirs) != 1:
            raise RuntimeError(f"Expected one canonical snapshot for {paper_key}, got {len(snapshot_dirs)}")
        canonical_dir = snapshot_dirs[0]
        bundle = _load_bundle_from_snapshot_dir(canonical_dir)
        fields = _source_fields(_mineru_content(args.run_dir, paper["source_id"], canonical_dir))
        fields_by_key: dict[tuple[int, str, int | None], list[dict[str, Any]]] = defaultdict(list)
        for field in fields:
            fields_by_key[_field_key(field)].append(field)
        element_failures = _check_element_provenance(
            paper_key=paper_key,
            bundle=bundle,
            fields_by_key=fields_by_key,
        )
        gold_failures = _check_gold_anchors(
            paper_key=paper_key,
            paper=paper,
            bundle=bundle,
            fields_by_key=fields_by_key,
        )
        paper_failures = [*element_failures, *gold_failures]
        failures.extend(paper_failures)
        papers_report[paper_key] = {
            "source_id": paper["source_id"],
            "snapshot_id": bundle.snapshot.id,
            "canonical_elements_checked": sum(bool(element_text(element)) for element in bundle.elements),
            "gold_occurrences_checked": len(paper["occurrences"]),
            "canonical_source_loss": len(element_failures),
            "gold_canonical_source_loss": len(gold_failures),
            "failures": paper_failures[:20],
            "status": "PASS" if not paper_failures else "FAIL",
        }

    report = {
        "git_commit": _git_commit(),
        "gold_version": gold.get("gold_version"),
        "canonical_source_loss": sum(item["canonical_source_loss"] for item in papers_report.values()),
        "gold_canonical_source_loss": sum(item["gold_canonical_source_loss"] for item in papers_report.values()),
        "failure_count": len(failures),
        "papers": papers_report,
        "failures": failures[:100],
        "pass": not failures,
    }
    output = args.run_dir / "canonical-source-survival.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "failures": len(failures), "report": str(output)}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
