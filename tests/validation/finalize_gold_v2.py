#!/usr/bin/env python3
"""Build citation_gold_v2 from v1 + canonical caption/table occurrence review."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from paperos_core.domain.enums import ElementType
from paperos_core.ingestion.document_regions import build_document_regions, region_for_element
from tests.validation.chunk_corpus_review import _load_bundle_from_snapshot_dir
from tests.validation.gold_audit_candidate_builder import context_hash, norm_context
from tests.validation.validate_citation_gold import canonicalizer, ref_fp


def _label_targets(inner: str, references: list[dict]) -> list[dict]:
    inner = inner.strip().strip("[]")
    atoms: list[str] = []
    if re.fullmatch(r"\d+\s*[-–−—]\s*\d+", inner):
        start, end = re.split(r"\s*[-–−—]\s*", inner)
        atoms = [str(i) for i in range(int(start), int(end) + 1)]
    elif "," in inner:
        atoms = [part.strip() for part in inner.split(",") if part.strip()]
    else:
        atoms = [inner]
    label_index = {(ref.get("citation_label") or "").strip(): ref for ref in references}
    targets = []
    for atom in atoms:
        label = re.sub(r"^\[|\]$", "", atom).strip()
        ref = label_index.get(label)
        if ref:
            targets.append(
                {
                    "atomic_key": label,
                    "acceptable_reference_orders": [ref.get("order")],
                    "acceptable_fingerprints": [
                        ref.get("fingerprint") or ref_fp(ref["raw_text"])
                    ],
                }
            )
        else:
            targets.append({"atomic_key": label, "unresolved": True})
    return targets


def _bracket_spans(text: str) -> list[tuple[str, str, str, str]]:
    spans: list[tuple[str, str, str, str]] = []
    for match in re.finditer(r"\[[^\[\]\n]{1,240}\]", text):
        surface = match.group(0)
        inner = surface[1:-1]
        if re.search(r"\b(?:sec|fig|eq)\b", inner, re.I):
            continue
        if re.search(r"[12]\d{3}", inner) and re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", inner):
            continue
        left = text[max(0, match.start() - 80) : match.start()]
        right = text[match.end() : match.end() + 80]
        spans.append(
            (
                surface,
                norm_context(left)[-60:],
                norm_context(right)[:60],
                context_hash(left, surface, right),
            )
        )
    return spans


def _group_signature(region: str, surface: str, targets: list[dict], canon) -> tuple:
    keys: list[str] = []
    for target in targets:
        fps = target.get("acceptable_fingerprints") or []
        if not fps and target.get("fingerprint"):
            fps = [target["fingerprint"]]
        if fps:
            keys.append(canon(next(iter({canon(fp) for fp in fps}))))
        else:
            keys.append(f"__UNRESOLVED__:{target.get('atomic_key', '')}")
    return (region, surface, tuple(sorted(keys)))


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
    audit = ["# citation_gold_v1 → citation_gold_v2 audit", ""]
    v2 = {
        "gold_version": "citation-gold-v2",
        "source_basis": v1.get("source_basis", "") + " Canonical caption/table occurrence review.",
        "papers": {},
    }

    for paper_key, paper in v1["papers"].items():
        source_id = paper["source_id"]
        snap_dir = sorted((args.run_dir / "canonical" / source_id).glob("snapshot_*"))[-1]
        bundle = _load_bundle_from_snapshot_dir(snap_dir)
        _, element_regions = build_document_regions(elements=bundle.elements, sections=bundle.sections)
        canon = canonicalizer(paper)

        groups = [dict(g) for g in paper.get("citation_groups", [])]
        used_locators: set[tuple[str | None, str | None]] = set()

        for group in groups:
            surface = group.get("surface") or ""
            if not surface:
                continue
            for element in bundle.elements:
                text = element.text or element.markdown or element.html or ""
                if surface not in text:
                    continue
                for span_surface, left, right, chash in _bracket_spans(text):
                    if span_surface != surface:
                        continue
                    locator_key = (element.id, chash)
                    if locator_key in used_locators:
                        continue
                    region = region_for_element(element.id, element_regions)
                    used_locators.add(locator_key)
                    group["locator"] = {
                        "page": element.page,
                        "source_domain": group.get("source_domain", "text"),
                        "left_context": left,
                        "right_context": right,
                        "context_hash": chash,
                        "element_id": element.id,
                    }
                    break
                if group.get("locator", {}).get("context_hash"):
                    break

        v1_counter = Counter(
            _group_signature(g["region"], g.get("surface", ""), g["targets"], canon)
            for g in groups
        )
        caption_table_occurrences: list[dict] = []
        for element in bundle.elements:
            if element.element_type not in {ElementType.CAPTION, ElementType.TABLE, ElementType.FOOTNOTE}:
                continue
            text = element.text or element.markdown or element.html or ""
            if not text.strip():
                continue
            region = region_for_element(element.id, element_regions)
            domain = {
                ElementType.CAPTION: "image_caption",
                ElementType.TABLE: "table_body",
                ElementType.FOOTNOTE: "table_footnote",
            }[element.element_type]
            for surface, left, right, chash in _bracket_spans(text):
                inner = surface.strip("[]")
                targets = _label_targets(inner, paper.get("references", []))
                if not targets or all(t.get("unresolved") for t in targets):
                    continue
                caption_table_occurrences.append(
                    {
                        "region": region,
                        "page_idx": (element.page or 1) - 1,
                        "surface": surface,
                        "source_domain": domain,
                        "locator": {
                            "page": element.page,
                            "source_domain": domain,
                            "left_context": left,
                            "right_context": right,
                            "context_hash": chash,
                            "element_id": element.id,
                        },
                        "targets": targets,
                    }
                )

        for occurrence in caption_table_occurrences:
            locator = occurrence["locator"]
            locator_key = (locator.get("element_id"), locator.get("context_hash"))
            if locator_key in used_locators:
                continue
            groups.append(occurrence)
            used_locators.add(locator_key)
            audit.append(
                f"- {paper_key} ADD `{occurrence['surface']}` from canonical "
                f"{occurrence['source_domain']} element={locator.get('element_id')}"
            )

        spans = []
        for index, group in enumerate(groups):
            locator = group.get("locator") or {}
            spans.append(
                {
                    "span_id": f"{paper_key}:{index}",
                    "region": group["region"],
                    "surface_text": group.get("surface") or group.get("surface_text", ""),
                    "source_domain": group.get("source_domain", locator.get("source_domain", "text")),
                    "targets": group["targets"],
                    "locator": locator,
                }
            )

        paper_v2 = dict(paper)
        paper_v2["citation_groups"] = groups
        paper_v2["citation_spans"] = spans
        paper_v2["expected_citation_span_count"] = len(spans)
        paper_v2["expected_atomic_target_count"] = sum(len(s["targets"]) for s in spans)
        v2["papers"][paper_key] = paper_v2

    payload = json.dumps(v2, ensure_ascii=False, sort_keys=True, indent=2)
    v2["gold_hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    args.output.write_text(json.dumps(v2, ensure_ascii=False, indent=2), encoding="utf-8")
    args.audit.write_text("\n".join(audit) + "\n", encoding="utf-8")
    print(
        f"Wrote {args.output} ({sum(len(p['citation_spans']) for p in v2['papers'].values())} spans)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
