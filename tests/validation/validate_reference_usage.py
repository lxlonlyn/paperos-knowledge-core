#!/usr/bin/env python3
"""Validate citation reference usage integrity."""

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


def validate_paper(*, bundle, chunks_json: dict) -> dict:
    ref_ids = {reference.id for reference in bundle.references}
    cited_ref_ids: set[str] = set()
    unexpected: list[dict] = []
    for mention in chunks_json.get("citation_mentions", []):
        ref_id = mention.get("reference_entry_id")
        if ref_id:
            cited_ref_ids.add(ref_id)
            if ref_id not in ref_ids:
                unexpected.append(
                    {
                        "failure_type": "unexpected_used_reference",
                        "surface": mention.get("surface_text"),
                        "reference_entry_id": ref_id,
                    }
                )
    missed: list[dict] = []
    for reference in bundle.references:
        if reference.id not in cited_ref_ids:
            missed.append(
                {
                    "failure_type": "missed_used_reference",
                    "reference_id": reference.id,
                    "citation_label": reference.citation_label,
                }
            )
    return {
        "unexpected_used_references": len(unexpected),
        "missed_used_references": len(missed),
        "failures": unexpected + missed,
        "pass": not unexpected and not missed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/validation/runs/chunk"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/validation/corpus/chunk"))
    args = parser.parse_args()

    papers = []
    missed_total = 0
    unexpected_total = 0
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
        missed_total += result["missed_used_references"]
        unexpected_total += result["unexpected_used_references"]
        papers.append({"pdf": str(pdf_path), "snapshot_id": bundle.snapshot.id, **result})

    report = {
        "git_commit": _git_commit(),
        "missed_used_references": missed_total,
        "unexpected_used_references": unexpected_total,
        "papers": papers,
        "pass": missed_total == 0 and unexpected_total == 0,
    }
    output = args.run_dir / "reference-usage.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "report": str(output)}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
