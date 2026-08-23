#!/usr/bin/env python3
"""Build MinerU-anchored citation Gold v3 by an independent source audit.

This script intentionally imports no ``paperos_core`` citation code. Historical
Gold v2 supplies reviewed target identities only; every occurrence is located
again in raw MinerU fields and bibliography markers in ``ref_text`` are dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
_YEAR_ONLY = re.compile(r"^\[(?:19|20)\d{2}[a-d]?\]$", re.IGNORECASE)
_LEFT_AUTHOR = re.compile(
    r"(?P<author>[A-ZÀ-ÖØ-Þ][\w'’\-]+(?:\s+et\s+al\.?)?"
    r"(?:\s+(?:and|&)\s+[A-ZÀ-ÖØ-Þ][\w'’\-]+)?)\s*$"
)


def _json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _canonical_dir(run_dir: Path, source_id: str) -> Path:
    snapshots = sorted((run_dir / "canonical" / source_id).glob("snapshot_*"))
    if not snapshots:
        raise FileNotFoundError(f"No canonical snapshot for {source_id}")
    return snapshots[-1]


def _mineru_content(run_dir: Path, source_id: str, canonical_dir: Path) -> list[dict[str, Any]]:
    snapshot = json.loads((canonical_dir / "snapshot.json").read_text(encoding="utf-8"))
    parse_id = snapshot["parse_run_id"]
    parse_dir = run_dir / "parsed" / source_id / parse_id
    manifest = json.loads((parse_dir / "manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact.get("type") != "content_list":
            continue
        path = parse_dir / artifact["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and any(isinstance(row, dict) and "page_idx" in row for row in payload):
            return payload
    raise RuntimeError(f"No MinerU content_list for {source_id}")


def _source_fields(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    for item_index, item in enumerate(content):
        page_idx = item.get("page_idx")
        item_type = str(item.get("type") or "other")
        values: list[tuple[str, int | None, str]] = []
        if isinstance(item.get("text"), str):
            values.append(("text", None, item["text"]))
        if isinstance(item.get("table_body"), str):
            values.append(("table_body", None, item["table_body"]))
        for key, domain in (
            ("image_caption", "caption"),
            ("chart_caption", "caption"),
            ("table_caption", "caption"),
            ("image_footnote", "footnote"),
            ("chart_footnote", "footnote"),
            ("table_footnote", "footnote"),
        ):
            value = item.get(key)
            if isinstance(value, list):
                values.extend((domain, index, text) for index, text in enumerate(value) if isinstance(text, str))
        for domain, subindex, value in values:
            fields.append(
                {
                    "source_item": item_index,
                    "source_domain": domain,
                    "source_subindex": subindex,
                    "page_idx": page_idx,
                    "item_type": item_type,
                    "is_heading": item.get("text_level") is not None,
                    "value": value,
                }
            )
    return fields


def _norm(value: str) -> str:
    value = html.unescape(value).replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", value).strip()


def _occurrences(value: str, needle: str) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    cursor = 0
    while needle and (start := value.find(needle, cursor)) >= 0:
        found.append((start, start + len(needle)))
        cursor = start + max(1, len(needle))
    return found


def _fold_with_map(value: str) -> tuple[str, list[int]]:
    folded: list[str] = []
    source_indexes: list[int] = []
    index = 0
    while index < len(value):
        if value[index] == "<":
            tag = re.match(r"</?(?:sub|sup|span|i|b|em|strong)\b[^>]*>", value[index:], re.IGNORECASE)
            if tag is not None:
                index += tag.end()
                continue
        lowered = value[index:].casefold()
        skipped = next(
            (
                len(token)
                for token in (r"\mathrm", r"\text", r"\ast")
                if lowered.startswith(token)
            ),
            0,
        )
        if skipped:
            if lowered.startswith(r"\ast"):
                folded.append("*")
                source_indexes.append(index)
            index += skipped
            continue
        char = value[index]
        if char in "−–—‑":
            char = "-"
        elif char in "∗⋆":
            char = "*"
        normalized = unicodedata.normalize("NFKD", char.casefold())
        for item in normalized:
            if unicodedata.combining(item) or item.isspace() or item in "{}^\\":
                continue
            folded.append(item)
            source_indexes.append(index)
        index += 1
    return "".join(folded), source_indexes


def _flex_occurrences(value: str, needle: str) -> list[tuple[int, int]]:
    exact = _occurrences(value, needle)
    if exact:
        return exact
    folded_value, mapping = _fold_with_map(value)
    folded_needle, _ = _fold_with_map(needle)
    matches: list[tuple[int, int]] = []
    for start, end in _occurrences(folded_value, folded_needle):
        if start < len(mapping) and end > 0:
            matches.append((mapping[start], mapping[end - 1] + 1))
    return matches


def _expand_year_surface(value: str, start: int, end: int) -> tuple[int, int]:
    window_start = max(0, start - 120)
    raw_prefix = value[window_start:start]
    visible: list[str] = []
    mapping: list[int] = []
    cursor = 0
    while cursor < len(raw_prefix):
        if raw_prefix[cursor] == "<":
            tag = re.match(r"</?(?:sub|sup|span|i|b|em|strong)\b[^>]*>", raw_prefix[cursor:], re.IGNORECASE)
            if tag is not None:
                cursor += tag.end()
                continue
        visible.append(raw_prefix[cursor])
        mapping.append(cursor)
        cursor += 1
    visible_prefix = "".join(visible).rstrip()
    match = _LEFT_AUTHOR.search(visible_prefix)
    if match is None:
        return start, end
    return window_start + mapping[match.start("author")], end


def _context_score(field: dict[str, Any], start: int, end: int, group: dict[str, Any]) -> float:
    locator = group.get("locator") or {}
    left = locator.get("left_context", "")
    right = locator.get("right_context", "")
    value = field["value"]
    score = 0.0
    if left:
        score += SequenceMatcher(None, _norm(left)[-60:], _norm(value[max(0, start - 80) : start])[-60:]).ratio()
    if right:
        score += SequenceMatcher(None, _norm(right)[:60], _norm(value[end : end + 80])[:60]).ratio()
    return score


def _locate_group(
    group: dict[str, Any],
    fields: list[dict[str, Any]],
    element_item: dict[str, int],
    used: set[tuple[int, str, int | None, int, int]],
) -> tuple[dict[str, Any] | None, str]:
    surface = group["surface"]
    locator = group.get("locator") or {}
    preferred_item = element_item.get(locator.get("element_id", ""))
    if preferred_item is not None and any(
        field["source_item"] == preferred_item and field["item_type"] == "ref_text"
        for field in fields
    ):
        return None, "DROPPED_REFERENCE_MARKER"
    candidates: list[tuple[float, dict[str, Any], int, int]] = []
    reference_candidates = 0
    heading_candidates = 0
    for field in fields:
        if preferred_item is not None and field["source_item"] != preferred_item:
            continue
        if group.get("page_idx") is not None and field["page_idx"] != group["page_idx"]:
            continue
        for start, end in _flex_occurrences(field["value"], surface):
            if _YEAR_ONLY.fullmatch(surface):
                start, end = _expand_year_surface(field["value"], start, end)
            key = (field["source_item"], field["source_domain"], field["source_subindex"], start, end)
            if key in used:
                continue
            if field["item_type"] == "ref_text":
                reference_candidates += 1
                continue
            if field["is_heading"]:
                heading_candidates += 1
                continue
            candidates.append((_context_score(field, start, end, group), field, start, end))
    if not candidates and preferred_item is not None:
        # A stale Canonical locator must not prevent source recovery.
        return _locate_group(group, fields, {}, used)
    if not candidates:
        if reference_candidates:
            return None, "DROPPED_REFERENCE_MARKER"
        if heading_candidates:
            return None, "DROPPED_CONTAINER_HEADING"
        return None, "SOURCE_OCCURRENCE_NOT_FOUND"
    _, field, start, end = max(
        candidates,
        key=lambda item: (item[0], -item[1]["source_item"], -item[2]),
    )
    used.add((field["source_item"], field["source_domain"], field["source_subindex"], start, end))
    return {
        "source_item": field["source_item"],
        "source_domain": field["source_domain"],
        **({"source_subindex": field["source_subindex"]} if field["source_subindex"] is not None else {}),
        "page_idx": field["page_idx"],
        "start": start,
        "end": end,
        "surface": field["value"][start:end],
    }, "AUDITED"


def _reference_targets(
    paper: dict[str, Any], references_jsonl: list[dict[str, Any]], element_item: dict[str, int]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    by_order = {reference["order"]: reference for reference in references_jsonl}
    enriched: list[dict[str, Any]] = []
    by_fingerprint: dict[str, dict[str, Any]] = {}
    for reference in paper["references"]:
        canonical = by_order.get(reference["order"], {})
        record = {
            **reference,
            "source_item": element_item.get(canonical.get("source_element_id", "")),
            "normalized_title_year": f"{_norm(reference['raw_text']).casefold()}|{reference.get('year') or ''}",
        }
        enriched.append(record)
        by_fingerprint[record["fingerprint"]] = record
    return enriched, by_fingerprint


def _alias_targets(paper: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    aliases: dict[str, list[dict[str, Any]]] = {}
    reference_by_order = {reference["order"]: reference for reference in paper["references"]}
    for reference in paper["references"]:
        label = reference.get("citation_label")
        if label:
            aliases[_fold_with_map(str(label))[0]] = [
                {
                    "atomic_key": str(label),
                    "reference_order": reference["order"],
                    "fingerprint": reference["fingerprint"],
                }
            ]
    for group in paper["citation_groups"]:
        for target in group["targets"]:
            key = _fold_with_map(str(target.get("atomic_key") or ""))[0]
            if not key:
                continue
            record = dict(target)
            if "reference_order" not in record and record.get("acceptable_reference_orders"):
                record["reference_order"] = record["acceptable_reference_orders"][0]
            if "fingerprint" not in record and record.get("acceptable_fingerprints"):
                record["fingerprint"] = record["acceptable_fingerprints"][0]
            if record.get("reference_order") in reference_by_order:
                aliases.setdefault(key, [record])
    author_year_aliases: dict[str, list[dict[str, Any]]] = {}
    for reference in paper["references"]:
        year = reference.get("year")
        if not year:
            continue
        first_author = reference.get("raw_text", "").split(",", 1)[0].strip()
        surname_match = re.search(r"([A-ZÀ-ÖØ-Þ][\w'’\-]+)\s*$", first_author)
        if surname_match is None:
            continue
        surname = surname_match.group(1)
        target = {
            "atomic_key": f"{surname} et al. {year}",
            "reference_order": reference["order"],
            "fingerprint": reference["fingerprint"],
        }
        for semantic in (f"{surname} et al. {year}", f"{surname} {year}"):
            author_year_aliases.setdefault(_fold_with_map(semantic)[0], []).append(target)
    for key, targets in author_year_aliases.items():
        if len(targets) == 1:
            aliases.setdefault(key, targets)
    return aliases


def _source_math_spans(value: str) -> list[tuple[int, int]]:
    """Independently identify raw MinerU math delimiters for Gold discovery."""
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(value):
        delimiters: tuple[str, str] | None = None
        if value.startswith("$$", index):
            delimiters = ("$$", "$$")
        elif value[index] == "$":
            delimiters = ("$", "$")
        elif value.startswith(r"\(", index):
            delimiters = (r"\(", r"\)")
        elif value.startswith(r"\[", index):
            delimiters = (r"\[", r"\]")
        if delimiters is None:
            index += 1
            continue
        opener, closer = delimiters
        end = value.find(closer, index + len(opener))
        if end < 0:
            index += len(opener)
            continue
        spans.append((index, end + len(closer)))
        index = end + len(closer)
    return spans


def _resolved_source_candidate(
    value: str,
    start: int,
    end: int,
    aliases: dict[str, list[dict[str, Any]]],
) -> tuple[int, int, list[dict[str, Any]]] | None:
    inner = value[start + 1 : end - 1]
    folded = _fold_with_map(inner)[0]
    if folded in aliases:
        return start, end, aliases[folded]
    visible = re.sub(r"</?(?:sub|sup|span|i|b|em|strong)\b[^>]*>", "", inner, flags=re.IGNORECASE)
    parts = [part.strip() for part in re.split(r"[;,]", visible) if part.strip()]
    targets: list[dict[str, Any]] = []
    for part in parts:
        range_match = re.fullmatch(r"(\d+)\s*[-–−—]\s*(\d+)", part)
        expanded = [str(item) for item in range(int(range_match.group(1)), int(range_match.group(2)) + 1)] if range_match else [part]
        for atom in expanded:
            resolved = aliases.get(_fold_with_map(atom)[0])
            if not resolved:
                targets = []
                break
            targets.extend(resolved)
        if not targets:
            break
    if targets:
        return start, end, targets
    if _YEAR_ONLY.fullmatch(f"[{visible.strip()}]"):
        expanded_start, _ = _expand_year_surface(value, start, end)
        surface = re.sub(
            r"</?(?:sub|sup|span|i|b|em|strong)\b[^>]*>",
            "",
            value[expanded_start:end],
            flags=re.IGNORECASE,
        )
        semantic = surface.replace("[", " ").replace("]", " ")
        resolved = aliases.get(_fold_with_map(semantic)[0])
        if resolved:
            return expanded_start, end, resolved
    return None


def _discover_source_occurrences(
    *,
    paper: dict[str, Any],
    fields: list[dict[str, Any]],
    used: set[tuple[int, str, int | None, int, int]],
    reference_by_fp: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    aliases = _alias_targets(paper)
    discovered: list[dict[str, Any]] = []
    for field in fields:
        if field["item_type"] in {"ref_text", "equation"} or field["is_heading"]:
            continue
        value = field["value"]
        math_spans = _source_math_spans(value)
        for match in re.finditer(r"\[[^\[\]\n]{1,400}\]", value):
            if any(start <= match.start() < end for start, end in math_spans):
                continue
            resolved = _resolved_source_candidate(value, match.start(), match.end(), aliases)
            if resolved is None:
                continue
            start, end, targets = resolved
            item_key = (
                field["source_item"],
                field["source_domain"],
                field["source_subindex"],
                start,
                end,
            )
            if item_key in used:
                continue
            used.add(item_key)
            scopes = {
                reference_by_fp.get(target.get("fingerprint", ""), {}).get("scope", "main")
                for target in targets
            }
            discovered.append(
                {
                    "source_item": field["source_item"],
                    "source_domain": field["source_domain"],
                    **({"source_subindex": field["source_subindex"]} if field["source_subindex"] is not None else {}),
                    "page_idx": field["page_idx"],
                    "start": start,
                    "end": end,
                    "surface": value[start:end],
                    "region": "supplement" if scopes == {"supplement"} else "main",
                    "citation_namespace_id": "citation_namespace_2" if scopes == {"supplement"} else "citation_namespace_1",
                    "targets": [
                        {
                            **target,
                            "reference_source_item": reference_by_fp.get(target.get("fingerprint", ""), {}).get("source_item"),
                        }
                        for target in targets
                    ],
                    "audit_origin": "independent_source_discovery",
                }
            )
    return discovered


def build(args: argparse.Namespace) -> dict[str, Any]:
    historical = json.loads(args.historical.read_text(encoding="utf-8"))
    output: dict[str, Any] = {
        "gold_version": "citation-gold-v3",
        "source_basis": "Raw MinerU content_list source items and offsets; target identity independently carried from reviewed v2 bibliography audit.",
        "builder_contract": "No production citation detector or resolver imports.",
        "papers": {},
    }
    audit: dict[str, Any] = {"gold_version": "citation-gold-v3", "papers": {}}
    hard_failures = 0
    for paper_key, paper in historical["papers"].items():
        canonical_dir = _canonical_dir(args.run_dir, paper["source_id"])
        elements = _json_lines(canonical_dir / "elements.jsonl")
        references_jsonl = _json_lines(canonical_dir / "references.jsonl")
        element_item = {
            element["id"]: element.get("source_span", {}).get("item_index")
            for element in elements
            if element.get("source_span")
        }
        content = _mineru_content(args.run_dir, paper["source_id"], canonical_dir)
        fields = _source_fields(content)
        references, reference_by_fp = _reference_targets(paper, references_jsonl, element_item)
        used: set[tuple[int, str, int | None, int, int]] = set()
        occurrences: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        for group_index, group in enumerate(paper["citation_groups"]):
            locator, decision = _locate_group(group, fields, element_item, used)
            decisions.append({"historical_group": group_index, "surface": group["surface"], "decision": decision, "source_locator": locator})
            if locator is None:
                if decision not in {"DROPPED_REFERENCE_MARKER", "DROPPED_CONTAINER_HEADING"}:
                    hard_failures += 1
                continue
            target_scopes = {
                reference_by_fp[target["fingerprint"]].get("scope", "main")
                for target in group["targets"]
                if target.get("fingerprint") in reference_by_fp
            }
            namespace_id = "citation_namespace_2" if target_scopes == {"supplement"} else "citation_namespace_1"
            occurrences.append(
                {
                    **locator,
                    "region": group["region"],
                    "citation_namespace_id": namespace_id,
                    "targets": [
                        {
                            **target,
                            "reference_source_item": reference_by_fp.get(target.get("fingerprint", ""), {}).get("source_item"),
                        }
                        for target in group["targets"]
                    ],
                }
            )
        discovered = _discover_source_occurrences(
            paper=paper,
            fields=fields,
            used=used,
            reference_by_fp=reference_by_fp,
        )
        occurrences.extend(discovered)
        occurrences.sort(
            key=lambda item: (
                item["source_item"],
                item["source_domain"],
                item.get("source_subindex", -1),
                item["start"],
            )
        )
        output["papers"][paper_key] = {
            "title": paper["title"],
            "pdf_basename": paper["pdf_basename"],
            "source_id": paper["source_id"],
            "style": paper["style"],
            "expected_citation_span_count": len(occurrences),
            "expected_atomic_target_count": sum(len(item["targets"]) for item in occurrences),
            "references": references,
            "occurrences": occurrences,
        }
        audit["papers"][paper_key] = {
            "audited": len(occurrences),
            "dropped_reference_markers": sum(item["decision"] == "DROPPED_REFERENCE_MARKER" for item in decisions),
            "dropped_container_headings": sum(item["decision"] == "DROPPED_CONTAINER_HEADING" for item in decisions),
            "discovered_source_citations": len(discovered),
            "source_failures": sum(item["decision"] == "SOURCE_OCCURRENCE_NOT_FOUND" for item in decisions),
            "decisions": decisions,
        }
    if hard_failures:
        args.audit.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(f"Gold v3 source audit has {hard_failures} unmatched non-reference occurrences")
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    args.audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    audit_lines = [
        "# Citation Gold v3 source audit",
        "",
        f"Gold SHA-256: `{digest}`",
        "",
        "All retained occurrences were re-located in raw MinerU source fields. "
        "Reference-region leading markers and container-only headings were removed. "
        "Independent discovery excludes formula/inline-math domains and does not import production citation code.",
        "",
        "| Paper | Retained | Dropped reference markers | Dropped container headings | Independently discovered | Source failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for paper_key, paper_audit in audit["papers"].items():
        audit_lines.append(
            f"| {paper_key} | {paper_audit['audited']} | "
            f"{paper_audit['dropped_reference_markers']} | "
            f"{paper_audit['dropped_container_headings']} | "
            f"{paper_audit['discovered_source_citations']} | "
            f"{paper_audit['source_failures']} |"
        )
    audit_lines.extend(
        [
            "",
            "The full JSON audit records every historical decision and recovered MinerU locator. "
            "Notable corrections include NISE bibliography-marker removal, NISE supplement-body citations, "
            "Buonomo caption citations, and 4Deform table-body citations.",
            "",
        ]
    )
    args.audit_md.write_text("\n".join(audit_lines), encoding="utf-8")
    return {"gold": str(args.output), "audit": str(args.audit), "sha256": digest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", type=Path, default=ROOT / "tests/fixtures/chunk/citation_gold_v2.json")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "data/validation/runs/chunk")
    parser.add_argument("--output", type=Path, default=ROOT / "tests/fixtures/chunk/citation_gold_v3.json")
    parser.add_argument("--audit", type=Path, default=ROOT / "tests/fixtures/chunk/gold-v3-audit.json")
    parser.add_argument("--audit-md", type=Path, default=ROOT / "tests/fixtures/chunk/gold-v3-audit.md")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
