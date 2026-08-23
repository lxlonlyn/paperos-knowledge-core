#!/usr/bin/env python3
"""Validate used ReferenceEntry identities against Gold v3 citation targets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.validation.chunk_corpus_review import _load_bundle_from_snapshot_dir  # noqa: E402


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


def validate_paper(*, bundle: Any, chunks_json: dict[str, Any], paper: dict[str, Any]) -> dict[str, Any]:
    fingerprint_by_order = {reference["order"]: reference["fingerprint"] for reference in paper["references"]}
    order_by_reference_id = {reference.id: reference.order for reference in bundle.references}
    actual: set[str] = set()
    invalid_reference_ids: list[str] = []
    for mention in chunks_json.get("citation_mentions", []):
        reference_id = mention.get("reference_entry_id")
        if not reference_id:
            continue
        order = order_by_reference_id.get(reference_id)
        fingerprint = fingerprint_by_order.get(order)
        if fingerprint is None:
            invalid_reference_ids.append(reference_id)
        else:
            actual.add(fingerprint)

    expected_groups: set[tuple[str, ...]] = set()
    for occurrence in paper["occurrences"]:
        for target in occurrence["targets"]:
            accepted = target.get("acceptable_fingerprints") or [target.get("fingerprint")]
            expected_groups.add(tuple(sorted(value for value in accepted if value)))
    expected_union = {fingerprint for group in expected_groups for fingerprint in group}
    unexpected = sorted(actual - expected_union)
    missed = sorted(group for group in expected_groups if not actual.intersection(group))
    failures: list[dict[str, Any]] = [
        {"failure_type": "UNEXPECTED_USED_REFERENCE", "fingerprint": fingerprint}
        for fingerprint in unexpected
    ]
    failures.extend(
        {"failure_type": "MISSED_EXPECTED_USED_REFERENCE", "acceptable_fingerprints": list(group)}
        for group in missed
    )
    failures.extend(
        {"failure_type": "REFERENCE_ID_NOT_IN_CANONICAL_BIBLIOGRAPHY", "reference_entry_id": reference_id}
        for reference_id in sorted(set(invalid_reference_ids))
    )
    return {
        "expected_used_reference_identities": len(expected_groups),
        "actual_used_reference_identities": len(actual),
        "unexpected_used_references": len(unexpected) + len(set(invalid_reference_ids)),
        "missed_used_references": len(missed),
        "failures": failures,
        "pass": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/validation/runs/chunk"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/validation/corpus/chunk"))
    parser.add_argument("--gold", type=Path, default=Path("tests/fixtures/chunk/citation_gold_v3.json"))
    args = parser.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    papers: dict[str, dict[str, Any]] = {}
    for paper_key, paper in gold["papers"].items():
        snapshots = sorted((args.run_dir / "canonical" / paper["source_id"]).glob("snapshot_*"))
        if len(snapshots) != 1:
            raise RuntimeError(f"Expected one canonical snapshot for {paper_key}, got {len(snapshots)}")
        bundle = _load_bundle_from_snapshot_dir(snapshots[0])
        papers[paper_key] = validate_paper(
            bundle=bundle,
            chunks_json=_chunks_for_snapshot(args.run_dir, bundle.snapshot.id),
            paper=paper,
        )

    report = {
        "git_commit": _git_commit(),
        "gold_version": gold.get("gold_version"),
        "missed_used_references": sum(item["missed_used_references"] for item in papers.values()),
        "unexpected_used_references": sum(item["unexpected_used_references"] for item in papers.values()),
        "papers": papers,
    }
    report["pass"] = report["missed_used_references"] == 0 and report["unexpected_used_references"] == 0
    output = args.run_dir / "reference-usage.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "missed": report["missed_used_references"], "unexpected": report["unexpected_used_references"], "report": str(output)}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
