#!/usr/bin/env python3
"""Validate PaperOS production citation output against frozen citation_gold_v1.json.

Usage:
  PYTHONPATH=. python tests/validation/validate_citation_gold.py \
      --gold tests/validation/gold/citation_gold_v1.json \
      --run-dir data/validation/runs/chunk

This script NEVER generates gold.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def norm_text(s: str) -> str:
    s = re.sub(r"</?(?:sub|sup)>", "", s or "", flags=re.I)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("−", "-").replace("–", "-").replace("—", "-").replace("∗", "*").replace("⋆", "*")
    s = s.replace("\\*", "*")
    return re.sub(r"\s+", " ", s).strip()


def ref_fp(raw: str) -> str:
    return hashlib.sha256(norm_text(raw).casefold().encode("utf-8")).hexdigest()[:16]


def locate_references(run_dir: Path, snapshot_id: str) -> Path:
    roots = [run_dir, run_dir.parent, run_dir.parent.parent, Path("data")]
    matches = []
    for root in roots:
        if root.exists():
            matches.extend(root.rglob(f"{snapshot_id}/references.jsonl"))
    matches = list(dict.fromkeys(matches))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one references.jsonl for {snapshot_id}, found {len(matches)}: {matches}"
        )
    return matches[0]


def load_refs(path: Path):
    refs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return refs, {r["id"]: r for r in refs}


def canonicalizer(paper_gold):
    aliases = {}
    for eq_class in paper_gold.get("reference_equivalence_classes", []):
        canonical = sorted(eq_class)[0]
        for fp in eq_class:
            aliases[fp] = canonical
    return lambda fp: aliases.get(fp, fp)


def expected_group_counter(paper_gold):
    canon = canonicalizer(paper_gold)
    counter = collections.Counter()
    for group in paper_gold["citation_groups"]:
        targets = []
        for target in group["targets"]:
            fps = target.get("acceptable_fingerprints") or [target["fingerprint"]]
            choices = {canon(fp) for fp in fps}
            if len(choices) != 1:
                raise RuntimeError(f"Gold target alternatives cross semantic equivalence classes: {target}")
            targets.append(next(iter(choices)))
        counter[(group["region"], tuple(sorted(targets)))] += 1
    return counter


def actual_group_counter(result, ref_by_id, paper_gold):
    canon = canonicalizer(paper_gold)
    by_span = collections.defaultdict(list)
    unattached = []
    for mention in result["citation_mentions"]:
        by_span[mention["citation_span_id"]].append(mention)
        if not mention.get("chunk_id"):
            unattached.append(mention)

    counter = collections.Counter()
    details = []
    for span_id, mentions in by_span.items():
        targets = []
        unresolved = []
        for mention in mentions:
            ref_id = mention.get("reference_entry_id")
            if not ref_id:
                unresolved.append(mention.get("atomic_key") or mention.get("surface_text"))
                continue
            if ref_id not in ref_by_id:
                raise RuntimeError(f"Reference id {ref_id} missing from canonical references")
            targets.append(canon(ref_fp(ref_by_id[ref_id]["raw_text"])))
        targets.extend(f"__UNRESOLVED__:{x}" for x in unresolved)
        region = mentions[0].get("document_region") or "main"
        sig = (region, tuple(sorted(targets)))
        counter[sig] += 1
        details.append((span_id, norm_text(mentions[0].get("surface_text", "")), sig, mentions))
    return counter, details, unattached


def choose_result_file(run_dir: Path, pdf_basename: str):
    candidates = []
    for path in run_dir.glob("*.chunks.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if Path(data.get("pdf", "")).name == pdf_basename:
            candidates.append((path, data))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one final chunks.json for {pdf_basename}, found {len(candidates)}: "
            f"{[str(p) for p, _ in candidates]}"
        )
    return candidates[0]


def fmt_sig(sig):
    region, targets = sig
    return f"{region}: {list(targets)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    total_failures = 0
    report = {"gold_version": gold["gold_version"], "papers": {}}

    for paper_key, pg in gold["papers"].items():
        result_path, result = choose_result_file(args.run_dir, pg["pdf_basename"])
        refs_path = locate_references(args.run_dir, result["snapshot_id"])
        refs, ref_by_id = load_refs(refs_path)

        expected = expected_group_counter(pg)
        actual, details, unattached = actual_group_counter(result, ref_by_id, pg)
        missing = expected - actual
        extra = actual - expected

        canon = canonicalizer(pg)
        expected_used = set()
        for group in pg["citation_groups"]:
            for target in group["targets"]:
                fps = target.get("acceptable_fingerprints") or [target["fingerprint"]]
                expected_used.add(next(iter({canon(fp) for fp in fps})))

        actual_used = set()
        for mention in result["citation_mentions"]:
            ref_id = mention.get("reference_entry_id")
            if ref_id and ref_id in ref_by_id:
                actual_used.add(canon(ref_fp(ref_by_id[ref_id]["raw_text"])))

        missed_used = expected_used - actual_used
        unexpected_used = actual_used - expected_used

        actual_surfaces = collections.Counter(
            norm_text(m.get("surface_text", "")) for m in result["citation_mentions"]
        )
        negative_hits = []
        for neg in pg.get("negative_brackets", []):
            surface = norm_text(neg["surface"])
            if actual_surfaces.get(surface):
                negative_hits.append((surface, actual_surfaces[surface]))
        for forbidden in pg.get("negative_nonbracket_surfaces", []):
            for surface, count in actual_surfaces.items():
                if norm_text(forbidden) in surface or surface in norm_text(forbidden):
                    negative_hits.append((surface, count))

        paper_failures = (
            sum(missing.values()) + sum(extra.values()) + len(unattached)
            + len(missed_used) + len(unexpected_used)
            + sum(count for _, count in negative_hits)
        )
        total_failures += paper_failures

        info = {
            "result_file": str(result_path),
            "gold_spans": pg["expected_citation_span_count"],
            "actual_spans": len({m["citation_span_id"] for m in result["citation_mentions"]}),
            "gold_atomic_targets": pg["expected_atomic_target_count"],
            "actual_atomic_targets": len(result["citation_mentions"]),
            "missing_group_count": sum(missing.values()),
            "extra_group_count": sum(extra.values()),
            "unattached_atomic_targets": len(unattached),
            "missed_used_references": sorted(missed_used),
            "unexpected_used_references": sorted(unexpected_used),
            "negative_hits": negative_hits,
            "status": "PASS" if paper_failures == 0 else "FAIL",
        }
        report["papers"][paper_key] = info

        print(f"\n=== {paper_key}: {info['status']} ===")
        print(
            f"spans gold/actual={info['gold_spans']}/{info['actual_spans']}; "
            f"atoms={info['gold_atomic_targets']}/{info['actual_atomic_targets']}; "
            f"unattached={info['unattached_atomic_targets']}"
        )
        if missing:
            print("MISSING GROUPS:")
            for sig, count in missing.items():
                print(f"  {count}x {fmt_sig(sig)}")
        if extra:
            print("EXTRA / WRONG GROUPS:")
            for sig, count in extra.items():
                print(f"  {count}x {fmt_sig(sig)}")
        if missed_used:
            print("MISSED USED REFERENCES:", sorted(missed_used))
        if unexpected_used:
            print("UNEXPECTED USED REFERENCES:", sorted(unexpected_used))
        if negative_hits:
            print("NEGATIVE CITATION HITS:", negative_hits)
        if unattached:
            print("UNATTACHED TARGETS (first 10):")
            for m in unattached[:10]:
                print(
                    " ", m.get("surface_text"), "atomic=", m.get("atomic_key"),
                    "reason=", (m.get("metadata") or {}).get("chunk_attachment_diagnostic")
                )

    output = args.run_dir / "citation-gold-validation.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {output}")
    if total_failures:
        print(f"FAIL: {total_failures} gold violations")
        return 1
    print("PASS: all frozen citation gold checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
