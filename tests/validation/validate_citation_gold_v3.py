#!/usr/bin/env python3
"""Validate production citations against MinerU-anchored Gold v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.validation.build_citation_gold_v3 import (  # noqa: E402
    _canonical_dir,
    _flex_occurrences,
    _json_lines,
    _mineru_content,
    _norm,
    _source_fields,
)


def _key(record: dict[str, Any]) -> tuple[int, str, int | None, int, int]:
    return (
        record["source_item"],
        record["source_domain"],
        record.get("source_subindex"),
        record["start"],
        record["end"],
    )


def _chunks_for_snapshot(run_dir: Path, snapshot_id: str) -> dict[str, Any]:
    matches = []
    for path in run_dir.glob("*.chunks.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("snapshot_id") == snapshot_id:
            matches.append(payload)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one chunks file for {snapshot_id}, got {len(matches)}")
    return matches[0]


def _element_domain(element: dict[str, Any]) -> tuple[str, int | None]:
    metadata = element.get("metadata") or {}
    if "caption_index" in metadata:
        return "caption", metadata["caption_index"]
    if "footnote_index" in metadata:
        return "footnote", metadata["footnote_index"]
    if element.get("element_type") == "table" and element.get("html"):
        return "table_body", None
    return "text", None


def _actual_source_locator(
    mention: dict[str, Any],
    element: dict[str, Any],
    fields_by_item: dict[int, list[dict[str, Any]]],
    used: set[tuple[int, str, int | None, int, int]],
) -> dict[str, Any] | None:
    source_span = element.get("source_span") or {}
    item_index = source_span.get("item_index")
    if item_index is None:
        return None
    domain, subindex = _element_domain(element)
    element_text = element.get("html") if domain == "table_body" else element.get("text")
    element_text = element_text or element.get("markdown") or ""
    left = element_text[max(0, mention["character_start"] - 80) : mention["character_start"]]
    right = element_text[mention["character_end"] : mention["character_end"] + 80]
    candidates: list[tuple[float, dict[str, Any], int, int]] = []
    for field in fields_by_item.get(item_index, []):
        if field["source_domain"] != domain or field["source_subindex"] != subindex:
            continue
        for start, end in _flex_occurrences(field["value"], mention["surface_text"]):
            candidate_key = (item_index, domain, subindex, start, end)
            if candidate_key in used:
                continue
            score = SequenceMatcher(None, _norm(left)[-60:], _norm(field["value"][max(0, start - 100) : start])[-60:]).ratio()
            score += SequenceMatcher(None, _norm(right)[:60], _norm(field["value"][end : end + 100])[:60]).ratio()
            candidates.append((score, field, start, end))
    if not candidates:
        return None
    _, field, start, end = max(candidates, key=lambda item: (item[0], -item[2]))
    used.add((item_index, domain, subindex, start, end))
    return {
        "source_item": item_index,
        "source_domain": domain,
        **({"source_subindex": subindex} if subindex is not None else {}),
        "start": start,
        "end": end,
        "surface": field["value"][start:end],
    }


def validate_paper(paper: dict[str, Any], *, run_dir: Path) -> dict[str, Any]:
    canonical_dir = _canonical_dir(run_dir, paper["source_id"])
    snapshot = json.loads((canonical_dir / "snapshot.json").read_text(encoding="utf-8"))
    elements = _json_lines(canonical_dir / "elements.jsonl")
    references = _json_lines(canonical_dir / "references.jsonl")
    element_by_id = {element["id"]: element for element in elements}
    reference_order = {reference["id"]: reference["order"] for reference in references}
    fingerprint_by_order = {reference["order"]: reference["fingerprint"] for reference in paper["references"]}
    fields = _source_fields(_mineru_content(run_dir, paper["source_id"], canonical_dir))
    fields_by_item: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for field in fields:
        fields_by_item[field["source_item"]].append(field)

    chunks = _chunks_for_snapshot(run_dir, snapshot["id"])
    mentions_by_span: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in chunks.get("citation_mentions", []):
        mentions_by_span[mention["citation_span_id"]].append(mention)
    used: set[tuple[int, str, int | None, int, int]] = set()
    actual: dict[tuple[int, str, int | None, int, int], dict[str, Any]] = {}
    source_mapping_failures: list[dict[str, Any]] = []
    for span_mentions in sorted(
        mentions_by_span.values(),
        key=lambda rows: (element_by_id[rows[0]["element_id"]]["order"], rows[0]["character_start"]),
    ):
        first = min(span_mentions, key=lambda row: row["group_index"])
        locator = _actual_source_locator(first, element_by_id[first["element_id"]], fields_by_item, used)
        if locator is None:
            source_mapping_failures.append({"surface": first["surface_text"], "element_id": first["element_id"]})
            continue
        targets = []
        for mention in sorted(span_mentions, key=lambda row: row["group_index"]):
            order = reference_order.get(mention.get("reference_entry_id"))
            targets.append(
                {
                    "atomic_key": mention["atomic_key"],
                    "fingerprint": fingerprint_by_order.get(order),
                    "resolution_status": mention["resolution_status"],
                    "chunk_id": mention.get("chunk_id"),
                }
            )
        actual[_key(locator)] = {
            **locator,
            "targets": targets,
            "citation_namespace_id": first.get("citation_namespace_id") or first.get("bibliography_scope_id"),
        }

    expected = {_key(item): item for item in paper["occurrences"]}
    missing_keys = sorted(set(expected) - set(actual))
    extra_keys = sorted(set(actual) - set(expected))
    failures: list[dict[str, Any]] = []
    for item_key in missing_keys:
        failures.append({"failure_type": "MISSING_OCCURRENCE", "expected": expected[item_key]})
    for item_key in extra_keys:
        failures.append({"failure_type": "EXTRA_OCCURRENCE", "actual": actual[item_key]})
    wrong_targets = 0
    unresolved_expected = 0
    unattached = 0
    wrong_namespaces = 0
    for item_key in sorted(set(expected) & set(actual)):
        expected_fingerprints = [
            target.get("acceptable_fingerprints")
            or [target.get("fingerprint")]
            for target in expected[item_key]["targets"]
        ]
        actual_fingerprints = [target["fingerprint"] for target in actual[item_key]["targets"]]
        targets_match = len(expected_fingerprints) == len(actual_fingerprints) and all(
            actual_fingerprint in accepted
            for actual_fingerprint, accepted in zip(actual_fingerprints, expected_fingerprints, strict=True)
        )
        if not targets_match:
            wrong_targets += 1
            failures.append({"failure_type": "WRONG_TARGETS", "expected": expected[item_key], "actual": actual[item_key]})
        unresolved_expected += sum(target["resolution_status"] != "resolved" for target in actual[item_key]["targets"])
        unattached += sum(not target["chunk_id"] for target in actual[item_key]["targets"])
        if expected[item_key]["citation_namespace_id"] != actual[item_key]["citation_namespace_id"]:
            wrong_namespaces += 1
            failures.append({"failure_type": "WRONG_NAMESPACE", "expected": expected[item_key], "actual": actual[item_key]})
    return {
        "expected_spans": len(expected),
        "actual_spans": len(actual),
        "missing_occurrences": len(missing_keys),
        "extra_occurrences": len(extra_keys),
        "wrong_targets": wrong_targets,
        "unresolved_expected_targets": unresolved_expected,
        "unattached_targets": unattached,
        "wrong_namespaces": wrong_namespaces,
        "source_mapping_failures": source_mapping_failures,
        "source_anchor_digest": hashlib.sha256(
            json.dumps(
                [actual[key] for key in sorted(actual)],
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "failures": failures,
        "status": "PASS" if not failures and not source_mapping_failures and not unresolved_expected and not unattached else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=ROOT / "tests/fixtures/chunk/citation_gold_v3.json")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/validation/runs/chunk")
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    papers = {key: validate_paper(paper, run_dir=args.run_dir) for key, paper in gold["papers"].items()}
    report = {
        "gold_version": "citation-gold-v3",
        "papers": papers,
        "missing_occurrences": sum(item["missing_occurrences"] for item in papers.values()),
        "extra_occurrences": sum(item["extra_occurrences"] for item in papers.values()),
        "wrong_targets": sum(item["wrong_targets"] for item in papers.values()),
        "unresolved_expected_targets": sum(item["unresolved_expected_targets"] for item in papers.values()),
        "unattached_targets": sum(item["unattached_targets"] for item in papers.values()),
        "wrong_namespaces": sum(item["wrong_namespaces"] for item in papers.values()),
        "source_mapping_failures": sum(len(item["source_mapping_failures"]) for item in papers.values()),
    }
    report["pass"] = all(item["status"] == "PASS" for item in papers.values())
    output = args.run_dir / "citation-gold-v3-validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "report": str(output), **{key: report[key] for key in ("missing_occurrences", "extra_occurrences", "wrong_targets", "wrong_namespaces", "source_mapping_failures")}}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
