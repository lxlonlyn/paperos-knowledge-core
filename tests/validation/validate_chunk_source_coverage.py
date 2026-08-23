#!/usr/bin/env python3
"""Validate canonical element text coverage by chunk authoritative spans."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.domain.enums import ElementType
from tests.validation.chunk_corpus_review import _load_bundle_from_snapshot_dir, _guess_pdf_for_bundle


EXCLUDED_TYPES = {
    ElementType.REFERENCE,
    ElementType.HEADER,
    ElementType.FOOTER,
    ElementType.PAGE_NUMBER,
}


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


def _is_publication_metadata(text: str) -> bool:
    lowered = text.casefold()
    markers = (
        "received ",
        "revised ",
        "accepted ",
        "acm reference format",
        "copyright",
        "to cite this version",
    )
    return any(marker in lowered for marker in markers) and len(text) < 400


def validate_paper(*, bundle, chunks_json: dict) -> dict:
    chunks = chunks_json["chunks"]
    eligible = []
    excluded = []
    for element in bundle.elements:
        if element.element_type in EXCLUDED_TYPES:
            excluded.append({"element_id": element.id, "reason": element.element_type.value})
            continue
        text = (element.text or element.markdown or "").strip()
        if not text:
            excluded.append({"element_id": element.id, "reason": "empty_text"})
            continue
        if element.element_type == ElementType.TITLE and len(text) < 120:
            excluded.append({"element_id": element.id, "reason": "container_only_heading"})
            continue
        if element.element_type not in {
            ElementType.PARAGRAPH,
            ElementType.TITLE,
            ElementType.LIST,
            ElementType.LIST_ITEM,
            ElementType.CAPTION,
            ElementType.FOOTNOTE,
            ElementType.TABLE,
            ElementType.FORMULA,
        }:
            excluded.append({"element_id": element.id, "reason": f"type:{element.element_type.value}"})
            continue
        eligible.append(element)

    failures = []
    holes = 0
    overlaps = 0
    for element in eligible:
        source = element.text or element.markdown or ""
        spans = []
        for chunk in chunks:
            for span in chunk.get("spans", []):
                if span["element_id"] == element.id:
                    spans.append(span)
        spans.sort(key=lambda item: item["character_start_in_element"])
        cursor = 0
        reconstructed_parts = []
        for span in spans:
            start = span["character_start_in_element"]
            end = span["character_end_in_element"]
            if start > cursor:
                holes += 1
                failures.append(
                    {
                        "element_id": element.id,
                        "failure_type": "chunk_source_hole",
                        "hole_start": cursor,
                        "hole_end": start,
                    }
                )
            if start < cursor:
                overlaps += 1
                failures.append(
                    {
                        "element_id": element.id,
                        "failure_type": "chunk_source_overlap",
                        "overlap_start": start,
                        "overlap_end": cursor,
                    }
                )
            reconstructed_parts.append(source[start:end])
            cursor = max(cursor, end)
        if cursor < len(source):
            holes += 1
            failures.append(
                {
                    "element_id": element.id,
                    "failure_type": "chunk_source_hole",
                    "hole_start": cursor,
                    "hole_end": len(source),
                }
            )
        reconstructed = "".join(reconstructed_parts)
        if reconstructed != source:
            failures.append(
                {
                    "element_id": element.id,
                    "failure_type": "chunk_source_mismatch",
                    "source_len": len(source),
                    "reconstructed_len": len(reconstructed),
                }
            )

    return {
        "eligible_elements": len(eligible),
        "excluded_elements": excluded,
        "chunk_source_holes": holes,
        "chunk_source_overlaps": overlaps,
        "failures": failures,
        "pass": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/validation/runs/chunk"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/validation/corpus/chunk"))
    args = parser.parse_args()

    papers = []
    total_failures = 0
    for src_dir in sorted((args.run_dir / "canonical").glob("src_*")):
        snapshot_dirs = sorted(src_dir.glob("snapshot_*"))
        if not snapshot_dirs:
            continue
        bundle = _load_bundle_from_snapshot_dir(snapshot_dirs[-1])
        pdf_path = _guess_pdf_for_bundle(bundle, args.corpus_dir)
        chunk_candidates = list(args.run_dir.glob(f"*{pdf_path.stem}*.chunks.json"))
        if not chunk_candidates:
            chunk_candidates = [
                path
                for path in args.run_dir.glob("*.chunks.json")
                if bundle.snapshot.id in path.read_text(encoding="utf-8")
            ]
        if len(chunk_candidates) != 1:
            raise RuntimeError(f"Unable to locate chunks json for {pdf_path.name}")
        chunks_json = json.loads(chunk_candidates[0].read_text(encoding="utf-8"))
        result = validate_paper(bundle=bundle, chunks_json=chunks_json)
        total_failures += len(result["failures"])
        papers.append({"pdf": str(pdf_path), "snapshot_id": bundle.snapshot.id, **result})

    report = {
        "git_commit": _git_commit(),
        "chunk_source_holes": sum(item["chunk_source_holes"] for item in papers),
        "chunk_source_overlaps": sum(item["chunk_source_overlaps"] for item in papers),
        "failure_count": total_failures,
        "papers": papers,
        "pass": total_failures == 0,
    }
    output = args.run_dir / "chunk-source-coverage.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "failures": total_failures, "report": str(output)}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
