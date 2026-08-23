#!/usr/bin/env python3
"""Validate chunk region boundaries and bibliography scope assignment on citations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.validation.chunk_corpus_review import _guess_pdf_for_bundle, _load_bundle_from_snapshot_dir


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


def validate_paper(*, chunks_json: dict) -> dict:
    failures: list[dict] = []
    wrong_regions = 0
    wrong_scopes = 0

    for chunk in chunks_json.get("chunks", []):
        if chunk.get("metadata", {}).get("mixed_region_chunk"):
            wrong_regions += 1
            failures.append(
                {
                    "failure_type": "MIXED_REGION_CHUNK",
                    "chunk_id": chunk.get("id"),
                    "region_instance_id": chunk.get("metadata", {}).get("region_instance_id"),
                }
            )

    for mention in chunks_json.get("citation_mentions", []):
        diagnostic = (mention.get("metadata") or {}).get("bibliography_scope_diagnostic")
        if diagnostic in {"AMBIGUOUS_BIBLIOGRAPHY_SCOPE", "SCOPE_NOT_FOUND"}:
            wrong_scopes += 1
            failures.append(
                {
                    "failure_type": diagnostic,
                    "surface": mention.get("surface_text"),
                    "element_id": mention.get("element_id"),
                    "page": mention.get("page"),
                    "document_region": mention.get("document_region"),
                }
            )

    return {
        "wrong_regions": wrong_regions,
        "wrong_bibliography_scopes": wrong_scopes,
        "failures": failures,
        "pass": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/validation/runs/chunk"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/validation/corpus/chunk"))
    args = parser.parse_args()

    papers = []
    total_regions = 0
    total_scopes = 0
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
        result = validate_paper(chunks_json=chunks_json)
        total_regions += result["wrong_regions"]
        total_scopes += result["wrong_bibliography_scopes"]
        papers.append({"pdf": str(pdf_path), "snapshot_id": bundle.snapshot.id, **result})

    report = {
        "git_commit": _git_commit(),
        "wrong_regions": total_regions,
        "wrong_bibliography_scopes": total_scopes,
        "papers": papers,
        "pass": total_regions == 0 and total_scopes == 0,
    }
    output = args.run_dir / "chunk-regions-scopes.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "report": str(output)}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
