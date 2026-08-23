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


def context_hash(left: str, surface: str, right: str) -> str:
    payload = f"{norm_text(left)}|{norm_text(surface)}|{norm_text(right)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


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
        "left_context": norm_text(left)[-60:],
        "right_context": norm_text(right)[:60],
        "context_hash": context_hash(left, surface, right),
        "page": first.get("page"),
        "bibliography_scope_id": first.get("bibliography_scope_id"),
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


def match_key(record: dict[str, Any], *, use_context: bool) -> tuple:
    base = (record["region"], record["surface"], record["targets"])
    if use_context and record.get("context_hash"):
        return base + (record["context_hash"],)
    return base


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
        actual_counter = Counter(match_key(record, use_context=False) for record in actual_records)

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
                    "source_domain": locator.get("source_domain"),
                    "span_id": span.get("span_id"),
                }
            )
        expected_counter = Counter(
            match_key(record, use_context=False) for record in expected_records
        )

        missing = expected_counter - actual_counter
        extra = actual_counter - expected_counter
        paper_failures: list[dict[str, Any]] = []

        for key, count in missing.items():
            for _ in range(count):
                paper_failures.append(
                    {
                        "failure_type": "missing_span",
                        "paper": paper_key,
                        "region": key[0],
                        "surface": key[1],
                        "expected_targets": list(key[2]),
                        "context_hash": key[3] if len(key) > 3 else None,
                    }
                )
        for key, count in extra.items():
            for _ in range(count):
                paper_failures.append(
                    {
                        "failure_type": "extra_span",
                        "paper": paper_key,
                        "region": key[0],
                        "surface": key[1],
                        "actual_targets": list(key[2]),
                        "context_hash": key[3] if len(key) > 3 else None,
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
                    "failure_type": "unattached_target",
                    "paper": paper_key,
                    "surface": mention.get("surface_text"),
                    "atomic_key": mention.get("atomic_key"),
                }
            )

        failures.extend(paper_failures)
        report["papers"][paper_key] = {
            "result_file": str(result_path),
            "expected_spans": len(expected_records),
            "actual_spans": len(actual_records),
            "missing_spans": sum(missing.values()),
            "extra_spans": sum(extra.values()),
            "unattached_targets": len(unattached),
            "status": "PASS" if not paper_failures else "FAIL",
            "failures": paper_failures[:50],
        }
        print(f"\n=== {paper_key}: {report['papers'][paper_key]['status']} ===")
        print(
            f"spans expected/actual={len(expected_records)}/{len(actual_records)} "
            f"missing={sum(missing.values())} extra={sum(extra.values())} "
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
