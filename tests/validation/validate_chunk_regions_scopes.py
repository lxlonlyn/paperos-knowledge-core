#!/usr/bin/env python3
"""Validate DocumentRegion boundaries and preassigned CitationNamespace flow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.ingestion.bibliography_scope import (  # noqa: E402
    FAILURE_NAMESPACE_NOT_ASSIGNED,
    REGION_REFERENCES,
)
from paperos_core.ingestion.document_regions import (  # noqa: E402
    build_document_regions,
    region_for_element,
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


def _chunks_for_snapshot(run_dir: Path, snapshot_id: str) -> dict[str, Any]:
    matches = []
    for path in run_dir.glob("*.chunks.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("snapshot_id") == snapshot_id:
            matches.append(payload)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one chunks file for {snapshot_id}, got {len(matches)}")
    return matches[0]


def validate_paper(*, bundle: Any, chunks_json: dict[str, Any]) -> dict[str, Any]:
    regions, element_regions = build_document_regions(
        elements=bundle.elements,
        sections=bundle.sections,
    )
    failures: list[dict[str, Any]] = []
    wrong_regions = 0
    wrong_namespaces = 0

    for chunk in chunks_json.get("chunks", []):
        span_infos = [
            element_regions.get(span["element_id"])
            for span in chunk.get("spans", [])
        ]
        if not span_infos or any(info is None for info in span_infos):
            wrong_regions += 1
            failures.append({"failure_type": "CHUNK_REGION_ELEMENT_MISSING", "chunk_id": chunk.get("id")})
            continue
        region_ids = {info.region_id for info in span_infos if info is not None}
        namespaces = {info.citation_namespace_id for info in span_infos if info is not None}
        region_types = {
            region_for_element(info.element_id, element_regions)
            for info in span_infos
            if info is not None
        }
        if len(region_ids) != 1 or len(region_types) != 1 or REGION_REFERENCES in region_types:
            wrong_regions += 1
            failures.append(
                {
                    "failure_type": "MIXED_OR_REFERENCE_REGION_CHUNK",
                    "chunk_id": chunk.get("id"),
                    "region_ids": sorted(region_ids),
                    "region_types": sorted(region_types),
                }
            )
        expected_region_id = next(iter(region_ids))
        if chunk.get("metadata", {}).get("region_instance_id") != expected_region_id:
            wrong_regions += 1
            failures.append(
                {
                    "failure_type": "WRONG_CHUNK_REGION_INSTANCE",
                    "chunk_id": chunk.get("id"),
                    "expected": expected_region_id,
                    "actual": chunk.get("metadata", {}).get("region_instance_id"),
                }
            )
        if len(namespaces) != 1 or None in namespaces:
            wrong_namespaces += 1
            failures.append(
                {
                    "failure_type": FAILURE_NAMESPACE_NOT_ASSIGNED,
                    "chunk_id": chunk.get("id"),
                    "namespaces": sorted(value or "<none>" for value in namespaces),
                }
            )
        else:
            expected_namespace = next(iter(namespaces))
            if chunk.get("citation_namespace_id") != expected_namespace:
                wrong_namespaces += 1
                failures.append(
                    {
                        "failure_type": "WRONG_CHUNK_NAMESPACE",
                        "chunk_id": chunk.get("id"),
                        "expected": expected_namespace,
                        "actual": chunk.get("citation_namespace_id"),
                    }
                )

    for mention in chunks_json.get("citation_mentions", []):
        info = element_regions.get(mention.get("element_id"))
        diagnostic = (mention.get("metadata") or {}).get("bibliography_scope_diagnostic")
        if info is None:
            wrong_regions += 1
            failures.append({"failure_type": "MENTION_REGION_ELEMENT_MISSING", "mention_id": mention.get("id")})
            continue
        expected_region = region_for_element(info.element_id, element_regions)
        if mention.get("document_region") != expected_region or info.region_type == REGION_REFERENCES:
            wrong_regions += 1
            failures.append(
                {
                    "failure_type": "WRONG_MENTION_REGION",
                    "mention_id": mention.get("id"),
                    "expected": expected_region,
                    "actual": mention.get("document_region"),
                }
            )
        if (
            info.citation_namespace_id is None
            or mention.get("citation_namespace_id") != info.citation_namespace_id
            or mention.get("bibliography_scope_id") != info.citation_namespace_id
            or diagnostic == FAILURE_NAMESPACE_NOT_ASSIGNED
        ):
            wrong_namespaces += 1
            failures.append(
                {
                    "failure_type": FAILURE_NAMESPACE_NOT_ASSIGNED if info.citation_namespace_id is None else "WRONG_MENTION_NAMESPACE",
                    "mention_id": mention.get("id"),
                    "surface": mention.get("surface_text"),
                    "expected": info.citation_namespace_id,
                    "actual": mention.get("citation_namespace_id"),
                    "diagnostic": diagnostic,
                }
            )

    return {
        "regions": [asdict(region) for region in regions],
        "wrong_regions": wrong_regions,
        "wrong_namespaces": wrong_namespaces,
        "wrong_bibliography_scopes": wrong_namespaces,
        "failures": failures,
        "pass": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/validation/runs/chunk"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/validation/corpus/chunk"))
    args = parser.parse_args()

    papers: list[dict[str, Any]] = []
    for src_dir in sorted((args.run_dir / "canonical").glob("src_*")):
        snapshot_dirs = sorted(src_dir.glob("snapshot_*"))
        if len(snapshot_dirs) != 1:
            raise RuntimeError(f"Expected one canonical snapshot in {src_dir}, got {len(snapshot_dirs)}")
        bundle = _load_bundle_from_snapshot_dir(snapshot_dirs[0])
        result = validate_paper(
            bundle=bundle,
            chunks_json=_chunks_for_snapshot(args.run_dir, bundle.snapshot.id),
        )
        papers.append(
            {
                "source_id": src_dir.name,
                "title": bundle.document.title,
                "snapshot_id": bundle.snapshot.id,
                **result,
            }
        )

    report = {
        "git_commit": _git_commit(),
        "wrong_regions": sum(item["wrong_regions"] for item in papers),
        "wrong_namespaces": sum(item["wrong_namespaces"] for item in papers),
        "wrong_bibliography_scopes": sum(item["wrong_namespaces"] for item in papers),
        "papers": papers,
    }
    report["pass"] = report["wrong_regions"] == 0 and report["wrong_namespaces"] == 0
    output = args.run_dir / "chunk-regions-scopes.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "wrong_regions": report["wrong_regions"], "wrong_namespaces": report["wrong_namespaces"], "report": str(output)}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
