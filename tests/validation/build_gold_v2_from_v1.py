#!/usr/bin/env python3
"""Build citation_gold_v2.json from v1 + MinerU source locators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tests.validation.gold_audit_candidate_builder import load_content_list, scan_content_list


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v1", type=Path, default=Path("tests/fixtures/chunk/citation_gold_v1.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("data/validation/runs/chunk"))
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/chunk/citation_gold_v2.json"))
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("tests/fixtures/chunk/citation_gold_v1_to_v2_audit.md"),
    )
    args = parser.parse_args()

    v1 = json.loads(args.v1.read_text(encoding="utf-8"))
    audit_lines = [
        "# citation_gold_v1 → citation_gold_v2 audit",
        "",
        "Auto-generated locator enrichment from MinerU source scans.",
        "",
    ]
    v2 = {
        "gold_version": "citation-gold-v2",
        "source_basis": v1.get("source_basis", "") + " + MinerU locator enrichment.",
        "papers": {},
    }

    for paper_key, paper in v1["papers"].items():
        source_id = paper["source_id"]
        parsed_dirs = sorted((args.run_dir / "parsed" / source_id).glob("parse_*"))
        if not parsed_dirs:
            raise RuntimeError(f"No parsed dir for {paper_key} ({source_id})")
        candidates = scan_content_list(load_content_list(parsed_dirs[-1]))
        spans: list[dict] = []
        for group_index, group in enumerate(paper.get("citation_groups", [])):
            surface = group.get("surface") or group.get("surface_text") or ""
            locator = _best_locator(candidates, surface)
            span = {
                "span_id": f"{paper_key}:{group_index}",
                "region": group["region"],
                "surface_text": surface or locator.get("surface_text", ""),
                "targets": group["targets"],
                "locator": locator,
            }
            spans.append(span)
            audit_lines.append(f"## {paper_key} span {group_index}")
            audit_lines.append(f"- region: {group['region']}")
            audit_lines.append(f"- surface: `{span['surface_text']}`")
            audit_lines.append(f"- locator: `{json.dumps(locator, ensure_ascii=False)}`")
            audit_lines.append("")

        v2_paper = dict(paper)
        v2_paper["citation_spans"] = spans
        v2_paper["expected_citation_span_count"] = len(spans)
        v2["papers"][paper_key] = v2_paper

    args.output.write_text(json.dumps(v2, ensure_ascii=False, indent=2), encoding="utf-8")
    args.audit.write_text("\n".join(audit_lines), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.audit}")
    return 0


def _best_locator(candidates: list[dict], surface: str) -> dict:
    if not surface:
        return {}
    normalized = surface.strip()
    for candidate in candidates:
        if candidate["surface_text"] == normalized:
            return {
                "page": candidate["page"],
                "source_domain": candidate["source_domain"],
                "left_context": candidate["left_context"],
                "right_context": candidate["right_context"],
                "context_hash": candidate["context_hash"],
                "item_index": candidate["item_index"],
            }
    return {}


if __name__ == "__main__":
    raise SystemExit(main())
