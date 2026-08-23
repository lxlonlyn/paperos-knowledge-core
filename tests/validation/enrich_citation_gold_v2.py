#!/usr/bin/env python3
"""Source-review enrichment for citation_gold_v2 (caption/table/appendix regions)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from tests.validation.gold_audit_candidate_builder import load_content_list, scan_content_list
from tests.validation.validate_citation_gold import ref_fp, norm_text


def _label_targets(inner: str, references: list[dict]) -> list[dict]:
    inner = inner.strip()
    atoms: list[str] = []
    if re.fullmatch(r"\d+\s*[-–−—]\s*\d+", inner):
        start, end = re.split(r"\s*[-–−—]\s*", inner)
        atoms = [str(i) for i in range(int(start), int(end) + 1)]
    elif "," in inner:
        atoms = [part.strip() for part in inner.split(",") if part.strip()]
    else:
        atoms = [inner]
    targets = []
    label_index = {
        (ref.get("citation_label") or "").strip(): ref for ref in references
    }
    for atom in atoms:
        label = re.sub(r"^\[|\]$", "", atom).strip()
        ref = label_index.get(label)
        if ref:
            targets.append(
                {
                    "atomic_key": label,
                    "reference_order": ref.get("order"),
                    "fingerprint": ref_fp(ref["raw_text"]),
                }
            )
        else:
            targets.append({"atomic_key": label, "unresolved": True})
    return targets


def _author_year_targets(inner: str, references: list[dict]) -> list[dict]:
    targets = []
    for part in re.split(r"\s*;\s*", inner):
        part = part.strip().strip("[]")
        year = re.search(r"([12]\d{3}[a-d]?)", part)
        if not year:
            continue
        author = part[: year.start()].strip()
        key = f"{author.casefold()}:{year.group(1).casefold()}"
        matched = None
        for ref in references:
            raw = norm_text(ref["raw_text"]).casefold()
            if year.group(1) in raw and author.split()[0].casefold() in raw:
                matched = ref
                break
        if matched:
            targets.append(
                {
                    "atomic_key": part,
                    "reference_order": matched.get("order"),
                    "fingerprint": ref_fp(matched["raw_text"]),
                }
            )
        else:
            targets.append({"atomic_key": part, "unresolved": True})
    return targets


def candidate_to_group(candidate: dict, references: list[dict], *, region: str) -> dict | None:
    inner = candidate["surface_text"].strip("[]")
    if re.search(r"[A-Za-z].*[12]\d{3}", inner):
        targets = _author_year_targets(inner, references)
    else:
        targets = _label_targets(inner, references)
    if not targets:
        return None
    return {
        "region": region,
        "page_idx": (candidate.get("page") or 1) - 1,
        "surface": candidate["surface_text"],
        "source_domain": candidate["source_domain"],
        "locator": {
            "page": candidate.get("page"),
            "source_domain": candidate["source_domain"],
            "left_context": candidate.get("left_context"),
            "right_context": candidate.get("right_context"),
            "context_hash": candidate.get("context_hash"),
            "item_index": candidate.get("item_index"),
        },
        "targets": targets,
    }


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
    audit: list[str] = ["# citation_gold_v1 → citation_gold_v2 audit", ""]
    v2 = {
        "gold_version": "citation-gold-v2",
        "gold_hash": "",
        "source_basis": v1.get("source_basis", "")
        + " Enriched with MinerU caption/table/appendix source review.",
        "papers": {},
    }

    for paper_key, paper in v1["papers"].items():
        source_id = paper["source_id"]
        parsed_dirs = sorted((args.run_dir / "parsed" / source_id).glob("parse_*"))
        candidates = (
            scan_content_list(load_content_list(parsed_dirs[-1])) if parsed_dirs else []
        )
        groups = [dict(group) for group in paper.get("citation_groups", [])]
        existing_surfaces = {
            (group["region"], norm_text(group.get("surface", ""))) for group in groups
        }
        appendix_start_index: int | None = None

        if paper_key == "buonomo":
            for candidate in candidates:
                if candidate["source_domain"] == "heading" and re.search(
                    r"appendix", candidate.get("surface_text", ""), re.I
                ):
                    appendix_start_index = candidate["item_index"]
            if appendix_start_index is not None:
                for group in groups:
                    for candidate in candidates:
                        if norm_text(candidate["surface_text"]) != norm_text(
                            group.get("surface", "")
                        ):
                            continue
                        if candidate["item_index"] >= appendix_start_index:
                            if group["region"] == "main":
                                audit.append(
                                    f"- buonomo region fix: `{group.get('surface')}` main → supplement (appendix item_index)"
                                )
                                group["region"] = "supplement"

        for candidate in candidates:
            domain = candidate["source_domain"]
            if domain not in {
                "image_caption",
                "table_caption",
                "table_body",
                "image_footnote",
                "table_footnote",
            }:
                continue
            inner = candidate["surface_text"].strip("[]")
            region = "main"
            if appendix_start_index is not None and candidate["item_index"] >= appendix_start_index:
                region = "supplement"
            surface = norm_text(candidate["surface_text"])
            key = (region, surface)
            if key in existing_surfaces:
                continue
            group = candidate_to_group(candidate, paper.get("references", []), region=region)
            if group is None:
                continue
            groups.append(group)
            existing_surfaces.add(key)
            audit.append(
                f"- {paper_key} ADD `{surface}` domain={candidate['source_domain']} page={candidate.get('page')}"
            )

        spans = []
        for index, group in enumerate(groups):
            locator = group.get("locator") or {}
            if not locator:
                for candidate in candidates:
                    if norm_text(candidate["surface_text"]) == norm_text(
                        group.get("surface", "")
                    ):
                        locator = {
                            "page": candidate.get("page"),
                            "source_domain": candidate.get("source_domain", "text"),
                            "left_context": candidate.get("left_context"),
                            "right_context": candidate.get("right_context"),
                            "context_hash": candidate.get("context_hash"),
                            "item_index": candidate.get("item_index"),
                        }
                        break
            spans.append(
                {
                    "span_id": f"{paper_key}:{index}",
                    "region": group["region"],
                    "surface_text": group.get("surface") or group.get("surface_text", ""),
                    "source_domain": group.get("source_domain", locator.get("source_domain")),
                    "targets": group["targets"],
                    "locator": locator,
                }
            )

        paper_v2 = dict(paper)
        paper_v2["citation_groups"] = groups
        paper_v2["citation_spans"] = spans
        paper_v2["expected_citation_span_count"] = len(spans)
        paper_v2["expected_atomic_target_count"] = sum(
            len(span["targets"]) for span in spans
        )
        v2["papers"][paper_key] = paper_v2

    payload = json.dumps(v2, ensure_ascii=False, indent=2)
    v2["gold_hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    args.output.write_text(
        json.dumps(v2, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.audit.write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} spans={sum(len(p['citation_spans']) for p in v2['papers'].values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
