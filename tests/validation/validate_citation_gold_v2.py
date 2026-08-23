#!/usr/bin/env python3
"""Validate production citations against frozen citation_gold_v2 (occurrence-level)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def norm_text(value: str) -> str:
    value = re.sub(r"</?(?:sub|sup)>", "", value or "", flags=re.I)
    value = unicodedata.normalize("NFKC", value)
    value = (
        value.replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("∗", "*")
        .replace("⋆", "*")
        .replace("\\*", "*")
    )
    return re.sub(r"\s+", " ", value).strip()


def ref_fp(raw: str) -> str:
    return hashlib.sha256(norm_text(raw).casefold().encode("utf-8")).hexdigest()[:16]


from tests.validation.gold_audit_candidate_builder import context_hash, norm_context
def locate_references(run_dir: Path, snapshot_id: str) -> Path:
    matches = list(run_dir.rglob(f"{snapshot_id}/references.jsonl"))
    matches = [path for path in matches if "/src_" in str(path)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one references.jsonl for {snapshot_id}, found {len(matches)}"
        )
    return matches[0]


def choose_result_file(run_dir: Path, pdf_basename: str) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for path in run_dir.glob("*.chunks.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if Path(data.get("pdf", "")).name == pdf_basename:
            candidates.append((path, data))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one chunks.json for {pdf_basename}, found {len(candidates)}")
    return candidates[0]


def canonicalizer(paper_gold: dict[str, Any]):
    aliases: dict[str, str] = {}
    for eq_class in paper_gold.get("reference_equivalence_classes", []):
        canonical = sorted(eq_class)[0]
        for fp in eq_class:
            aliases[fp] = canonical
    return lambda fp: aliases.get(fp, fp)


def expected_targets(group: dict[str, Any], canon) -> tuple[str, ...]:
    keys: list[str] = []
    for target in group["targets"]:
        fps = target.get("acceptable_fingerprints") or []
        if not fps and target.get("fingerprint"):
            fps = [target["fingerprint"]]
        if fps:
            choices = {canon(fp) for fp in fps}
            if len(choices) != 1:
                raise RuntimeError(
                    f"Gold target alternatives cross semantic equivalence classes: {target}"
                )
            keys.append(f"resolved:{next(iter(choices))}")
        else:
            keys.append(f"unresolved:{target.get('atomic_key', '')}")
    return tuple(sorted(keys))


def _target_details(
    target_keys: tuple[str, ...],
    *,
    ref_by_id: dict[str, dict[str, Any]],
    gold_targets: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    gold_by_key: dict[str, dict[str, Any]] = {}
    if gold_targets:
        for target in gold_targets:
            if target.get("unresolved"):
                gold_by_key[f"unresolved:{target.get('atomic_key', '')}"] = target
            else:
                fps = target.get("acceptable_fingerprints") or []
                if fps:
                    gold_by_key[f"resolved:{fps[0]}"] = target
    for key in target_keys:
        if key.startswith("unresolved:"):
            details.append(
                {
                    "atomic_key": key.removeprefix("unresolved:"),
                    "resolution_status": "unresolved",
                }
            )
            continue
        fp = key.removeprefix("resolved:")
        gold_target = gold_by_key.get(key)
        label = gold_target.get("atomic_key") if gold_target else None
        title = None
        for ref in ref_by_id.values():
            if ref_fp(ref["raw_text"]) == fp:
                title = ref.get("title") or ref.get("raw_text", "")[:120]
                label = label or ref.get("citation_label")
                break
        details.append(
            {
                "label": label,
                "title": title,
                "fingerprint": fp,
                "resolution_status": "resolved",
            }
        )
    return details


def actual_span_record(
    mentions: list[dict[str, Any]],
    ref_by_id: dict[str, dict[str, Any]],
    canon,
    element_texts: dict[str, str],
) -> dict[str, Any]:
    mentions = sorted(mentions, key=lambda item: (item.get("element_id", ""), item.get("character_start", 0)))
    first = mentions[0]
    element_id = first.get("element_id", "")
    text = element_texts.get(element_id, "")
    start = int(first.get("character_start", 0))
    end = int(first.get("character_end", start))
    surface = first.get("surface_text", "")
    left = text[max(0, start - 80) : start]
    right = text[end : end + 80]
    targets: list[str] = []
    for mention in mentions:
        ref_id = mention.get("reference_entry_id")
        if ref_id and ref_id in ref_by_id:
            targets.append(f"resolved:{canon(ref_fp(ref_by_id[ref_id]['raw_text']))}")
        else:
            targets.append(f"unresolved:{mention.get('atomic_key', '')}")
    return {
        "region": first.get("document_region") or "main",
        "surface": norm_text(surface),
        "targets": tuple(sorted(targets)),
        "left_context": norm_context(left)[-60:],
        "right_context": norm_context(right)[:60],
        "context_hash": context_hash(left, surface, right),
        "page": first.get("page"),
        "source_domain": first.get("metadata", {}).get("source_domain"),
        "bibliography_scope_id": first.get("bibliography_scope_id"),
        "element_id": element_id,
        "mentions": mentions,
    }


def load_element_texts(run_dir: Path, snapshot_id: str) -> dict[str, str]:
    matches = list(run_dir.rglob(f"{snapshot_id}/elements.jsonl"))
    matches = [path for path in matches if "/src_" in str(path)]
    if len(matches) != 1:
        return {}
    texts: dict[str, str] = {}
    for line in matches[0].read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        texts[row["id"]] = row.get("text") or row.get("markdown") or ""
    return texts


def occurrence_key(record: dict[str, Any], *, use_context: bool) -> tuple:
    element_id = record.get("element_id")
    if use_context and record.get("context_hash"):
        if element_id:
            return ("element_hash", element_id, record["context_hash"])
        return ("hash", record["context_hash"], record["surface"])
    domain = record.get("source_domain") or record.get("region") or "main"
    return (
        "fallback",
        element_id,
        domain,
        record["surface"],
        record.get("left_context", ""),
        record.get("right_context", ""),
    )


def legacy_group_key(record: dict[str, Any]) -> tuple:
    return (
        record.get("region") or "main",
        record["surface"],
    )


def record_match_key(record: dict[str, Any]) -> tuple:
    if record.get("context_hash"):
        return occurrence_key(record, use_context=True)
    return legacy_group_key(record)


def _match_citation_records(
    expected_records: list[dict[str, Any]],
    actual_records: list[dict[str, Any]],
) -> tuple[int, int, list[dict], list[dict], list[tuple[dict, dict]]]:
    """Greedy occurrence matching with separate locator and legacy keys."""
    unmatched_actual = list(actual_records)
    missing_records: list[dict] = []
    extra_records: list[dict] = []
    matched_pairs: list[tuple[dict, dict]] = []

    for expected in expected_records:
        if expected.get("context_hash"):
            key_fn = record_match_key
        else:
            key_fn = legacy_group_key
        expected_key = key_fn(expected)
        match_index = next(
            (
                index
                for index, actual in enumerate(unmatched_actual)
                if key_fn(actual) == expected_key
            ),
            None,
        )
        if match_index is None:
            missing_records.append(expected)
            continue
        actual = unmatched_actual.pop(match_index)
        matched_pairs.append((expected, actual))

    extra_records.extend(unmatched_actual)
    return (
        len(missing_records),
        len(extra_records),
        missing_records,
        extra_records,
        matched_pairs,
    )

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, default=Path("tests/fixtures/chunk/citation_gold_v2.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/validation/runs/chunk"))
    args = parser.parse_args()

    if not args.gold.exists():
        print(f"Gold v2 not found at {args.gold}; run build_gold_v2_from_v1.py first")
        return 2

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "gold_version": gold.get("gold_version"),
        "papers": {},
        "failure_count": 0,
        "missing_occurrences": 0,
        "extra_occurrences": 0,
        "wrong_targets": 0,
        "unresolved_expected_targets": 0,
        "unattached_targets": 0,
    }

    for paper_key, paper in gold["papers"].items():
        result_path, result = choose_result_file(args.run_dir, paper["pdf_basename"])
        refs_path = locate_references(args.run_dir, result["snapshot_id"])
        refs = [json.loads(line) for line in refs_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        ref_by_id = {row["id"]: row for row in refs}
        element_texts = load_element_texts(args.run_dir, result["snapshot_id"])
        canon = canonicalizer(paper)

        by_span: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for mention in result["citation_mentions"]:
            by_span[mention["citation_span_id"]].append(mention)

        actual_records = [
            actual_span_record(group, ref_by_id, canon, element_texts)
            for group in by_span.values()
        ]

        expected_records = []
        for span in paper.get("citation_spans", paper.get("citation_groups", [])):
            targets = expected_targets(span, canon)
            locator = span.get("locator") or {}
            expected_records.append(
                {
                    "region": span["region"],
                    "surface": norm_text(span.get("surface_text") or span.get("surface", "")),
                    "targets": targets,
                    "left_context": locator.get("left_context", ""),
                    "right_context": locator.get("right_context", ""),
                    "context_hash": locator.get("context_hash"),
                    "page": locator.get("page", span.get("page_idx")),
                    "source_domain": locator.get("source_domain") or span.get("region"),
                    "element_id": locator.get("element_id"),
                    "span_id": span.get("span_id"),
                    "gold_targets": span.get("targets", []),
                    "expected_bibliography_scope_id": span.get("bibliography_scope_id"),
                }
            )

        missing_count, extra_count, missing_records, extra_records, matched_pairs = (
            _match_citation_records(expected_records, actual_records)
        )
        paper_failures: list[dict[str, Any]] = []

        for record in missing_records:
            paper_failures.append(
                {
                    "failure_type": "MISSING_OCCURRENCE",
                    "paper": paper_key,
                    "page": record.get("page"),
                    "source_domain": record.get("source_domain") or record.get("region"),
                    "surface": record.get("surface"),
                    "left_context": record.get("left_context"),
                    "right_context": record.get("right_context"),
                    "expected_region": record.get("region"),
                    "expected_bibliography_scope": record.get("expected_bibliography_scope_id"),
                    "expected_targets": _target_details(
                        record.get("targets", ()),
                        ref_by_id=ref_by_id,
                        gold_targets=record.get("gold_targets"),
                    ),
                }
            )

        for record in extra_records:
            paper_failures.append(
                {
                    "failure_type": "EXTRA_OCCURRENCE",
                    "paper": paper_key,
                    "page": record.get("page"),
                    "source_domain": record.get("source_domain") or record.get("region"),
                    "surface": record.get("surface"),
                    "left_context": record.get("left_context"),
                    "right_context": record.get("right_context"),
                    "actual_region": record.get("region"),
                    "actual_bibliography_scope": record.get("bibliography_scope_id"),
                    "actual_targets": _target_details(
                        record.get("targets", ()),
                        ref_by_id=ref_by_id,
                    ),
                }
            )

        wrong_target_count = 0
        for expected, actual in matched_pairs:
            if expected["targets"] != actual["targets"]:
                wrong_target_count += 1
                paper_failures.append(
                    {
                        "failure_type": "WRONG_TARGETS",
                        "paper": paper_key,
                        "page": expected.get("page"),
                        "source_domain": expected.get("source_domain"),
                        "surface": expected.get("surface"),
                        "left_context": expected.get("left_context"),
                        "right_context": expected.get("right_context"),
                        "expected_region": expected.get("region"),
                        "actual_region": actual.get("region"),
                        "expected_bibliography_scope": expected.get("expected_bibliography_scope_id"),
                        "actual_bibliography_scope": actual.get("bibliography_scope_id"),
                        "expected_targets": _target_details(
                            expected["targets"],
                            ref_by_id=ref_by_id,
                            gold_targets=expected.get("gold_targets"),
                        ),
                        "actual_targets": _target_details(
                            actual["targets"],
                            ref_by_id=ref_by_id,
                        ),
                    }
                )
            for target in expected["targets"]:
                if target.startswith("unresolved:") and target not in actual["targets"]:
                    paper_failures.append(
                        {
                            "failure_type": "UNRESOLVED_EXPECTED_TARGET",
                            "paper": paper_key,
                            "page": expected.get("page"),
                            "surface": expected.get("surface"),
                            "expected_target": target,
                            "actual_targets": list(actual["targets"]),
                        }
                    )

        unattached = [
            mention
            for mention in result["citation_mentions"]
            if not mention.get("chunk_id")
        ]
        for mention in unattached:
            paper_failures.append(
                {
                    "failure_type": "UNATTACHED_TARGET",
                    "paper": paper_key,
                    "page": mention.get("page"),
                    "surface": mention.get("surface_text"),
                    "atomic_key": mention.get("atomic_key"),
                }
            )

        wrong_target_count = sum(1 for item in paper_failures if item["failure_type"] == "WRONG_TARGETS")
        unresolved_expected = sum(
            1 for item in paper_failures if item["failure_type"] == "UNRESOLVED_EXPECTED_TARGET"
        )
        failures.extend(paper_failures)
        report["papers"][paper_key] = {
            "result_file": str(result_path),
            "expected_spans": len(expected_records),
            "actual_spans": len(actual_records),
            "missing_occurrences": missing_count,
            "extra_occurrences": extra_count,
            "wrong_targets": wrong_target_count,
            "unresolved_expected_targets": unresolved_expected,
            "unattached_targets": len(unattached),
            "status": "PASS" if not paper_failures else "FAIL",
            "failures": paper_failures[:50],
        }
        report["missing_occurrences"] += missing_count
        report["extra_occurrences"] += extra_count
        report["wrong_targets"] += wrong_target_count
        report["unresolved_expected_targets"] += unresolved_expected
        report["unattached_targets"] += len(unattached)

        print(f"\n=== {paper_key}: {report['papers'][paper_key]['status']} ===")
        print(
            f"spans expected/actual={len(expected_records)}/{len(actual_records)} "
            f"missing={missing_count} extra={extra_count} wrong_targets={wrong_target_count} "
            f"unattached={len(unattached)}"
        )
        for failure in paper_failures[:5]:
            print(json.dumps(failure, ensure_ascii=False))

    report["failure_count"] = len(failures)
    output = args.run_dir / "citation-gold-v2-validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {output}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
