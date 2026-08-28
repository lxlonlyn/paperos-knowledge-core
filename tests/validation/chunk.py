from __future__ import annotations

"Build MinerU-anchored citation Gold v3 by an independent source audit.\n\nThis script intentionally imports no ``paperos_core`` citation code. Historical\nGold v2 supplies reviewed target identities only; every occurrence is located\nagain in raw MinerU fields and bibliography markers in ``ref_text`` are dropped.\n"
import argparse
import hashlib
import html
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

gold_builder__ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(gold_builder__ROOT))
gold_builder___YEAR_ONLY = re.compile("^\\[(?:19|20)\\d{2}[a-d]?\\]$", re.IGNORECASE)
gold_builder___LEFT_AUTHOR = re.compile(
    "(?P<author>[A-ZÀ-ÖØ-Þ][\\w'’\\-]+(?:\\s+et\\s+al\\.?)?(?:\\s+(?:and|&)\\s+[A-ZÀ-ÖØ-Þ][\\w'’\\-]+)?)\\s*$"
)


def gold_builder___json_lines(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def gold_builder___canonical_dir(run_dir: Path, source_id: str) -> Path:
    snapshots = sorted((run_dir / "canonical" / source_id).glob("snapshot_*"))
    if not snapshots:
        raise FileNotFoundError(f"""No canonical snapshot for {source_id }""")
    return snapshots[-1]


def gold_builder___mineru_content(
    run_dir: Path, source_id: str, canonical_dir: Path
) -> list[dict[str, Any]]:
    snapshot = json.loads((canonical_dir / "snapshot.json").read_text(encoding="utf-8"))
    parse_id = snapshot["parse_run_id"]
    parse_dir = run_dir / "parsed" / source_id / parse_id
    manifest = json.loads((parse_dir / "manifest.json").read_text(encoding="utf-8"))
    for artifact in manifest["artifacts"]:
        if artifact.get("type") != "content_list":
            continue
        path = parse_dir / artifact["path"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list) and any(
            isinstance(row, dict) and "page_idx" in row for row in payload
        ):
            return payload
    raise RuntimeError(f"""No MinerU content_list for {source_id }""")


def gold_builder___source_fields(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
                values.extend(
                    (
                        (domain, index, text)
                        for index, text in enumerate(value)
                        if isinstance(text, str)
                    )
                )
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


def gold_builder___norm(value: str) -> str:
    value = html.unescape(value).replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub("\\s+", " ", value).strip()


def gold_builder___occurrences(value: str, needle: str) -> list[tuple[int, int]]:
    found: list[tuple[int, int]] = []
    cursor = 0
    while needle and (start := value.find(needle, cursor)) >= 0:
        found.append((start, start + len(needle)))
        cursor = start + max(1, len(needle))
    return found


def gold_builder___fold_with_map(value: str) -> tuple[str, list[int]]:
    folded: list[str] = []
    source_indexes: list[int] = []
    index = 0
    while index < len(value):
        if value[index] == "<":
            tag = re.match(
                "</?(?:sub|sup|span|i|b|em|strong)\\b[^>]*>",
                value[index:],
                re.IGNORECASE,
            )
            if tag is not None:
                index += tag.end()
                continue
        lowered = value[index:].casefold()
        skipped = next(
            (
                len(token)
                for token in ("\\mathrm", "\\text", "\\ast")
                if lowered.startswith(token)
            ),
            0,
        )
        if skipped:
            if lowered.startswith("\\ast"):
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
    return ("".join(folded), source_indexes)


def gold_builder___flex_occurrences(value: str, needle: str) -> list[tuple[int, int]]:
    exact = gold_builder___occurrences(value, needle)
    if exact:
        return exact
    folded_value, mapping = gold_builder___fold_with_map(value)
    folded_needle, _ = gold_builder___fold_with_map(needle)
    matches: list[tuple[int, int]] = []
    for start, end in gold_builder___occurrences(folded_value, folded_needle):
        if start < len(mapping) and end > 0:
            matches.append((mapping[start], mapping[end - 1] + 1))
    return matches


def gold_builder___expand_year_surface(
    value: str, start: int, end: int
) -> tuple[int, int]:
    window_start = max(0, start - 120)
    raw_prefix = value[window_start:start]
    visible: list[str] = []
    mapping: list[int] = []
    cursor = 0
    while cursor < len(raw_prefix):
        if raw_prefix[cursor] == "<":
            tag = re.match(
                "</?(?:sub|sup|span|i|b|em|strong)\\b[^>]*>",
                raw_prefix[cursor:],
                re.IGNORECASE,
            )
            if tag is not None:
                cursor += tag.end()
                continue
        visible.append(raw_prefix[cursor])
        mapping.append(cursor)
        cursor += 1
    visible_prefix = "".join(visible).rstrip()
    match = gold_builder___LEFT_AUTHOR.search(visible_prefix)
    if match is None:
        return (start, end)
    return (window_start + mapping[match.start("author")], end)


def gold_builder___context_score(
    field: dict[str, Any], start: int, end: int, group: dict[str, Any]
) -> float:
    locator = group.get("locator") or {}
    left = locator.get("left_context", "")
    right = locator.get("right_context", "")
    value = field["value"]
    score = 0.0
    if left:
        score += SequenceMatcher(
            None,
            gold_builder___norm(left)[-60:],
            gold_builder___norm(value[max(0, start - 80) : start])[-60:],
        ).ratio()
    if right:
        score += SequenceMatcher(
            None,
            gold_builder___norm(right)[:60],
            gold_builder___norm(value[end : end + 80])[:60],
        ).ratio()
    return score


def gold_builder___locate_group(
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
        return (None, "DROPPED_REFERENCE_MARKER")
    candidates: list[tuple[float, dict[str, Any], int, int]] = []
    reference_candidates = 0
    heading_candidates = 0
    for field in fields:
        if preferred_item is not None and field["source_item"] != preferred_item:
            continue
        if group.get("page_idx") is not None and field["page_idx"] != group["page_idx"]:
            continue
        for start, end in gold_builder___flex_occurrences(field["value"], surface):
            if gold_builder___YEAR_ONLY.fullmatch(surface):
                start, end = gold_builder___expand_year_surface(
                    field["value"], start, end
                )
            key = (
                field["source_item"],
                field["source_domain"],
                field["source_subindex"],
                start,
                end,
            )
            if key in used:
                continue
            if field["item_type"] == "ref_text":
                reference_candidates += 1
                continue
            if field["is_heading"]:
                heading_candidates += 1
                continue
            candidates.append(
                (
                    gold_builder___context_score(field, start, end, group),
                    field,
                    start,
                    end,
                )
            )
    if not candidates and preferred_item is not None:
        return gold_builder___locate_group(group, fields, {}, used)
    if not candidates:
        if reference_candidates:
            return (None, "DROPPED_REFERENCE_MARKER")
        if heading_candidates:
            return (None, "DROPPED_CONTAINER_HEADING")
        return (None, "SOURCE_OCCURRENCE_NOT_FOUND")
    _, field, start, end = max(
        candidates, key=lambda item: (item[0], -item[1]["source_item"], -item[2])
    )
    used.add(
        (
            field["source_item"],
            field["source_domain"],
            field["source_subindex"],
            start,
            end,
        )
    )
    return (
        {
            "source_item": field["source_item"],
            "source_domain": field["source_domain"],
            **(
                {"source_subindex": field["source_subindex"]}
                if field["source_subindex"] is not None
                else {}
            ),
            "page_idx": field["page_idx"],
            "start": start,
            "end": end,
            "surface": field["value"][start:end],
        },
        "AUDITED",
    )


def gold_builder___reference_targets(
    paper: dict[str, Any],
    references_jsonl: list[dict[str, Any]],
    element_item: dict[str, int],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    by_order = {reference["order"]: reference for reference in references_jsonl}
    enriched: list[dict[str, Any]] = []
    by_fingerprint: dict[str, dict[str, Any]] = {}
    for reference in paper["references"]:
        canonical = by_order.get(reference["order"], {})
        record = {
            **reference,
            "source_item": element_item.get(canonical.get("source_element_id", "")),
            "normalized_title_year": f"""{gold_builder___norm (reference ['raw_text']).casefold ()}|{reference .get ('year')or ''}""",
        }
        enriched.append(record)
        by_fingerprint[record["fingerprint"]] = record
    return (enriched, by_fingerprint)


def gold_builder___alias_targets(
    paper: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    aliases: dict[str, list[dict[str, Any]]] = {}
    reference_by_order = {
        reference["order"]: reference for reference in paper["references"]
    }
    for reference in paper["references"]:
        label = reference.get("citation_label")
        if label:
            aliases[gold_builder___fold_with_map(str(label))[0]] = [
                {
                    "atomic_key": str(label),
                    "reference_order": reference["order"],
                    "fingerprint": reference["fingerprint"],
                }
            ]
    for group in paper["citation_groups"]:
        for target in group["targets"]:
            key = gold_builder___fold_with_map(str(target.get("atomic_key") or ""))[0]
            if not key:
                continue
            record = dict(target)
            if "reference_order" not in record and record.get(
                "acceptable_reference_orders"
            ):
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
        surname_match = re.search("([A-ZÀ-ÖØ-Þ][\\w'’\\-]+)\\s*$", first_author)
        if surname_match is None:
            continue
        surname = surname_match.group(1)
        target = {
            "atomic_key": f"""{surname } et al. {year }""",
            "reference_order": reference["order"],
            "fingerprint": reference["fingerprint"],
        }
        for semantic in (f"""{surname } et al. {year }""", f"""{surname } {year }"""):
            author_year_aliases.setdefault(
                gold_builder___fold_with_map(semantic)[0], []
            ).append(target)
    for key, targets in author_year_aliases.items():
        if len(targets) == 1:
            aliases.setdefault(key, targets)
    return aliases


def gold_builder___source_math_spans(value: str) -> list[tuple[int, int]]:
    """Independently identify raw MinerU math delimiters for Gold discovery."""
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(value):
        delimiters: tuple[str, str] | None = None
        if value.startswith("$$", index):
            delimiters = ("$$", "$$")
        elif value[index] == "$":
            delimiters = ("$", "$")
        elif value.startswith("\\(", index):
            delimiters = ("\\(", "\\)")
        elif value.startswith("\\[", index):
            delimiters = ("\\[", "\\]")
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


def gold_builder___resolved_source_candidate(
    value: str, start: int, end: int, aliases: dict[str, list[dict[str, Any]]]
) -> tuple[int, int, list[dict[str, Any]]] | None:
    inner = value[start + 1 : end - 1]
    folded = gold_builder___fold_with_map(inner)[0]
    if folded in aliases:
        return (start, end, aliases[folded])
    visible = re.sub(
        "</?(?:sub|sup|span|i|b|em|strong)\\b[^>]*>", "", inner, flags=re.IGNORECASE
    )
    parts = [part.strip() for part in re.split("[;,]", visible) if part.strip()]
    targets: list[dict[str, Any]] = []
    for part in parts:
        range_match = re.fullmatch("(\\d+)\\s*[-–−—]\\s*(\\d+)", part)
        expanded = (
            [
                str(item)
                for item in range(
                    int(range_match.group(1)), int(range_match.group(2)) + 1
                )
            ]
            if range_match
            else [part]
        )
        for atom in expanded:
            resolved = aliases.get(gold_builder___fold_with_map(atom)[0])
            if not resolved:
                targets = []
                break
            targets.extend(resolved)
        if not targets:
            break
    if targets:
        return (start, end, targets)
    if gold_builder___YEAR_ONLY.fullmatch(f"""[{visible .strip ()}]"""):
        expanded_start, _ = gold_builder___expand_year_surface(value, start, end)
        surface = re.sub(
            "</?(?:sub|sup|span|i|b|em|strong)\\b[^>]*>",
            "",
            value[expanded_start:end],
            flags=re.IGNORECASE,
        )
        semantic = surface.replace("[", " ").replace("]", " ")
        resolved = aliases.get(gold_builder___fold_with_map(semantic)[0])
        if resolved:
            return (expanded_start, end, resolved)
    return None


def gold_builder___discover_source_occurrences(
    *,
    paper: dict[str, Any],
    fields: list[dict[str, Any]],
    used: set[tuple[int, str, int | None, int, int]],
    reference_by_fp: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    aliases = gold_builder___alias_targets(paper)
    discovered: list[dict[str, Any]] = []
    for field in fields:
        if field["item_type"] in {"ref_text", "equation"} or field["is_heading"]:
            continue
        value = field["value"]
        math_spans = gold_builder___source_math_spans(value)
        for match in re.finditer("\\[[^\\[\\]\\n]{1,400}\\]", value):
            if any((start <= match.start() < end for start, end in math_spans)):
                continue
            resolved = gold_builder___resolved_source_candidate(
                value, match.start(), match.end(), aliases
            )
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
                reference_by_fp.get(target.get("fingerprint", ""), {}).get(
                    "scope", "main"
                )
                for target in targets
            }
            discovered.append(
                {
                    "source_item": field["source_item"],
                    "source_domain": field["source_domain"],
                    **(
                        {"source_subindex": field["source_subindex"]}
                        if field["source_subindex"] is not None
                        else {}
                    ),
                    "page_idx": field["page_idx"],
                    "start": start,
                    "end": end,
                    "surface": value[start:end],
                    "region": "supplement" if scopes == {"supplement"} else "main",
                    "citation_namespace_id": (
                        "citation_namespace_2"
                        if scopes == {"supplement"}
                        else "citation_namespace_1"
                    ),
                    "targets": [
                        {
                            **target,
                            "reference_source_item": reference_by_fp.get(
                                target.get("fingerprint", ""), {}
                            ).get("source_item"),
                        }
                        for target in targets
                    ],
                    "audit_origin": "independent_source_discovery",
                }
            )
    return discovered


def gold_builder__build(args: argparse.Namespace) -> dict[str, Any]:
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
        canonical_dir = gold_builder___canonical_dir(args.run_dir, paper["source_id"])
        elements = gold_builder___json_lines(canonical_dir / "elements.jsonl")
        references_jsonl = gold_builder___json_lines(canonical_dir / "references.jsonl")
        element_item = {
            element["id"]: element.get("source_span", {}).get("item_index")
            for element in elements
            if element.get("source_span")
        }
        content = gold_builder___mineru_content(
            args.run_dir, paper["source_id"], canonical_dir
        )
        fields = gold_builder___source_fields(content)
        references, reference_by_fp = gold_builder___reference_targets(
            paper, references_jsonl, element_item
        )
        used: set[tuple[int, str, int | None, int, int]] = set()
        occurrences: list[dict[str, Any]] = []
        decisions: list[dict[str, Any]] = []
        for group_index, group in enumerate(paper["citation_groups"]):
            locator, decision = gold_builder___locate_group(
                group, fields, element_item, used
            )
            decisions.append(
                {
                    "historical_group": group_index,
                    "surface": group["surface"],
                    "decision": decision,
                    "source_locator": locator,
                }
            )
            if locator is None:
                if decision not in {
                    "DROPPED_REFERENCE_MARKER",
                    "DROPPED_CONTAINER_HEADING",
                }:
                    hard_failures += 1
                continue
            target_scopes = {
                reference_by_fp[target["fingerprint"]].get("scope", "main")
                for target in group["targets"]
                if target.get("fingerprint") in reference_by_fp
            }
            namespace_id = (
                "citation_namespace_2"
                if target_scopes == {"supplement"}
                else "citation_namespace_1"
            )
            occurrences.append(
                {
                    **locator,
                    "region": group["region"],
                    "citation_namespace_id": namespace_id,
                    "targets": [
                        {
                            **target,
                            "reference_source_item": reference_by_fp.get(
                                target.get("fingerprint", ""), {}
                            ).get("source_item"),
                        }
                        for target in group["targets"]
                    ],
                }
            )
        discovered = gold_builder___discover_source_occurrences(
            paper=paper, fields=fields, used=used, reference_by_fp=reference_by_fp
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
            "expected_atomic_target_count": sum(
                len(item["targets"]) for item in occurrences
            ),
            "references": references,
            "occurrences": occurrences,
        }
        audit["papers"][paper_key] = {
            "audited": len(occurrences),
            "dropped_reference_markers": sum(
                item["decision"] == "DROPPED_REFERENCE_MARKER" for item in decisions
            ),
            "dropped_container_headings": sum(
                item["decision"] == "DROPPED_CONTAINER_HEADING" for item in decisions
            ),
            "discovered_source_citations": len(discovered),
            "source_failures": sum(
                item["decision"] == "SOURCE_OCCURRENCE_NOT_FOUND"
                for item in decisions
            ),
            "decisions": decisions,
        }
    if hard_failures:
        args.audit.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        raise RuntimeError(
            f"""Gold v3 source audit has {hard_failures } unmatched non-reference occurrences"""
        )
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    audit_lines = [
        "# Citation Gold v3 source audit",
        "",
        f"""Gold SHA-256: `{digest }`""",
        "",
        "All retained occurrences were re-located in raw MinerU source fields. Reference-region leading markers and container-only headings were removed. Independent discovery excludes formula/inline-math domains and does not import production citation code.",
        "",
        "| Paper | Retained | Dropped reference markers | Dropped container headings | Independently discovered | Source failures |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for paper_key, paper_audit in audit["papers"].items():
        audit_lines.append(
            f"""| {paper_key } | {paper_audit ['audited']} | {paper_audit ['dropped_reference_markers']} | {paper_audit ['dropped_container_headings']} | {paper_audit ['discovered_source_citations']} | {paper_audit ['source_failures']} |"""
        )
    audit_lines.extend(
        [
            "",
            "The full JSON audit records every historical decision and recovered MinerU locator. Notable corrections include NISE bibliography-marker removal, NISE supplement-body citations, Buonomo caption citations, and 4Deform table-body citations.",
            "",
        ]
    )
    args.audit_md.write_text("\n".join(audit_lines), encoding="utf-8")
    return {"gold": str(args.output), "audit": str(args.audit), "sha256": digest}


def gold_builder__main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--historical",
        type=Path,
        default=gold_builder__ROOT / "data/validation/chunk/config/citation_gold.json",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=gold_builder__ROOT / "data/validation/chunk/output",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=gold_builder__ROOT / "data/validation/chunk/config/citation_gold.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=gold_builder__ROOT / "data/validation/chunk/config/gold-audit.json",
    )
    parser.add_argument(
        "--audit-md",
        type=Path,
        default=gold_builder__ROOT / "data/validation/chunk/config/gold-audit.md",
    )
    args = parser.parse_args()
    print(json.dumps(gold_builder__build(args), indent=2))
    return 0


"Rebuild canonical snapshots from cached MinerU parsed artifacts (no re-OCR)."
from datetime import datetime
from pathlib import Path

from paperos_core.adapters.mineru.mapper import MinerUCanonicalMapper
from paperos_core.domain.documents import SourceFile
from paperos_core.domain.enums import ParserArtifactType, ParseRunStatus
from paperos_core.domain.ids import canonical_snapshot_id
from paperos_core.domain.parsing import ParserArtifact, ParseRun
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.paths import build_data_paths


def rebuild__rebuild_all_canonical_snapshots(
    *, run_dir: Path, dataset_id: str = "paperos-chunk-corpus-review"
) -> list[dict[str, Any]]:
    """Re-map every cached parse run under ``run_dir/parsed`` into canonical."""
    paths = build_data_paths(run_dir)
    mapper = MinerUCanonicalMapper()
    repository = CanonicalRepository(paths)
    results: list[dict[str, Any]] = []
    parsed_root = run_dir / "parsed"
    for src_dir in sorted(parsed_root.glob("src_*")):
        parse_dirs = sorted(src_dir.glob("parse_*"))
        if not parse_dirs:
            continue
        parse_dir = parse_dirs[-1]
        source_file_id = src_dir.name.removeprefix("src_")
        result = rebuild__rebuild_canonical_from_parse_dir(
            parse_dir=parse_dir,
            source_file_id=source_file_id,
            canonical_source_dir=src_dir.name,
            mapper=mapper,
            repository=repository,
            dataset_id=dataset_id,
            raw_root=run_dir / "raw",
        )
        results.append(result)
    return results


def rebuild__rebuild_canonical_from_parse_dir(
    *,
    parse_dir: Path,
    source_file_id: str,
    canonical_source_dir: str,
    mapper: MinerUCanonicalMapper,
    repository: CanonicalRepository,
    dataset_id: str,
    raw_root: Path,
) -> dict[str, Any]:
    manifest = json.loads((parse_dir / "manifest.json").read_text(encoding="utf-8"))
    parse_run_id = manifest["parse_run_id"]
    artifacts = rebuild___artifacts_from_manifest(parse_dir, manifest)
    parse_run = ParseRun(
        id=parse_run_id,
        source_file_id=source_file_id,
        provider=manifest.get("provider", "mineru"),
        backend=manifest.get("backend", "mineru"),
        status=ParseRunStatus.COMPLETED,
        request_options={},
        created_at=datetime.fromisoformat("2020-01-01T00:00:00+00:00"),
        completed_at=datetime.fromisoformat("2020-01-01T00:00:00+00:00"),
        artifact_manifest_path=parse_dir / "manifest.json",
    )
    raw_path = raw_root / f"""src_{source_file_id }"""
    pdf_candidates = sorted(raw_path.glob("*.pdf"))
    source = SourceFile(
        id=source_file_id,
        original_filename=(
            pdf_candidates[0].name if pdf_candidates else f"""{source_file_id }.pdf"""
        ),
        storage_path=pdf_candidates[0] if pdf_candidates else raw_path,
        sha256="0" * 64,
        size_bytes=max(pdf_candidates[0].stat().st_size, 1) if pdf_candidates else 1,
        media_type="application/pdf",
        created_at=datetime.fromisoformat("2020-01-01T00:00:00+00:00"),
    )
    manifest_path = (
        repository.paths.canonical
        / canonical_source_dir
        / canonical_snapshot_id(parse_run_id)
        / "manifest.json"
    )
    bundle = mapper.build_canonical_snapshot(
        source=source,
        parse_run=parse_run,
        artifacts=artifacts,
        manifest_path=manifest_path,
        dataset_id=dataset_id,
    )
    rebuild___overwrite_snapshot(repository, bundle)
    return {
        "source_file_id": source_file_id,
        "canonical_source_dir": canonical_source_dir,
        "parse_run_id": parse_run_id,
        "snapshot_id": bundle.snapshot.id,
        "element_count": len(bundle.elements),
        "reference_count": len(bundle.references),
        "status": "rebuilt",
    }


def rebuild___artifacts_from_manifest(
    parse_dir: Path, manifest: dict[str, Any]
) -> list[ParserArtifact]:
    artifacts: list[ParserArtifact] = []
    for item in manifest.get("artifacts", []):
        artifact_type = ParserArtifactType(item["type"])
        storage_path = (parse_dir / item["path"]).resolve()
        artifacts.append(
            ParserArtifact(
                id=item["id"],
                parse_run_id=manifest["parse_run_id"],
                artifact_type=artifact_type,
                storage_path=storage_path,
                sha256=item["sha256"],
                size_bytes=item["size_bytes"],
                created_at=datetime.fromisoformat("2020-01-01T00:00:00+00:00"),
            )
        )
    return artifacts


def rebuild___overwrite_snapshot(repository: CanonicalRepository, bundle) -> None:
    """Write canonical files, replacing prior snapshot content when normalization changes."""
    snapshot = bundle.snapshot
    root = snapshot.manifest_path.parent
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "snapshot.json": repository._json_bytes(snapshot),
        "document.json": repository._json_bytes(bundle.document),
        "sections.jsonl": repository._jsonl_bytes(bundle.sections),
        "elements.jsonl": repository._jsonl_bytes(bundle.elements),
        "references.jsonl": repository._jsonl_bytes(bundle.references),
        "warnings.json": json.dumps(
            bundle.warnings, ensure_ascii=False, sort_keys=True, indent=2
        ).encode(),
    }
    for name, content in payloads.items():
        (root / name).write_bytes(content)
    manifest = {
        "schema_version": snapshot.schema_version,
        "id_version": snapshot.id_version,
        "pipeline_version": snapshot.pipeline_version,
        "canonical_snapshot_id": snapshot.id,
        "source_file_id": snapshot.source_file_id,
        "parse_run_id": snapshot.parse_run_id,
        "document_id": snapshot.document_id,
        "dataset_id": snapshot.dataset_id,
        "counts": {
            "sections": len(bundle.sections),
            "elements": len(bundle.elements),
            "references": len(bundle.references),
        },
        "files": [
            {
                "path": name,
                "sha256": __import__("hashlib").sha256(content).hexdigest(),
                "size_bytes": len(content),
            }
            for name, content in payloads.items()
        ],
    }
    snapshot.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def rebuild__isogeometric_regression_text(bundle_elements: list) -> dict[str, bool]:
    """Check Isogeometric paragraph markers survived canonical mapping."""
    combined = "\n".join(
        element.text or element.markdown or ""
        for element in bundle_elements
        if element.text or element.markdown
    )
    markers = ["[29]", "[32]", "[30]", "[31]", "[34]", "T <", "J_W", "16\\pi", "16π"]
    return {marker: marker in combined for marker in markers}


"Real-PDF chunk corpus review: MinerU → Canonical → production chunk builder.\n\nAll runtime data MUST stay under a validation run root (never production ``data/``).\n\n    PYTHONPATH=. python tests/validation/chunk.py review \\\n      --run-dir data/validation/chunk/output\n"
import asyncio
import statistics
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

review__REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(review__REPOSITORY_ROOT))
from paperos_core.config import load_settings
from paperos_core.domain.canonical import (
    CanonicalBundle,
    CanonicalSnapshot,
    Document,
    Element,
    ReferenceEntry,
    Section,
)
from paperos_core.domain.enums import ElementType
from paperos_core.domain.ids import CHUNKING_VERSION
from paperos_core.ingestion.chunk_dp import TINY_TOKEN_THRESHOLD
from paperos_core.ingestion.chunk_eligibility import classify_chunk_eligibility
from paperos_core.ingestion.chunk_markdown import render_chunk_review_markdown
from paperos_core.ingestion.chunking import build_chunks
from paperos_core.ingestion.document_regions import build_document_regions
from paperos_core.ingestion.sentence_units import (
    element_text,
    figure_caption_element_ids,
    resolve_major_section_id,
)
from paperos_core.ingestion.tokenization import AUTHORITATIVE_CHUNK_TOKENIZER


def review___resolve_tokenizer() -> Any:
    return AUTHORITATIVE_CHUNK_TOKENIZER


review__DEFAULT_RUN_DIR = Path("data/validation/chunk/output")
review__DEFAULT_DATASET = "paperos-chunk-corpus-review"


def review___assert_validation_run_dir(run_dir: Path) -> None:
    """Refuse to write artifacts outside a task-owned validation output."""
    resolved = run_dir.expanduser().resolve()
    parts = resolved.parts
    if "validation" not in parts:
        raise RuntimeError(
            f"""--run-dir must be a validation task output (got {resolved })"""
        )
    validation_index = parts.index("validation")
    if validation_index + 2 >= len(parts) or parts[validation_index + 2] != "output":
        raise RuntimeError(
            f"""--run-dir must be data/validation/<task>/output (got {resolved })"""
        )


def review___selected_pdfs(corpus_dir: Path, papers_config: Path) -> list[Path]:
    corpus_root = corpus_dir.parent if corpus_dir.name == "papers" else corpus_dir
    manifest = json.loads((corpus_root / "manifest.json").read_text(encoding="utf-8"))
    selected = json.loads(papers_config.read_text(encoding="utf-8"))["papers"]
    paths = []
    for paper_id in selected:
        entry = manifest[paper_id]
        path = (corpus_root / entry["file"]).resolve()
        if not path.is_file():
            raise RuntimeError(f"""Missing validation PDF for {paper_id }: {path }""")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != entry["sha256"]:
            raise RuntimeError(
                f"""Validation PDF checksum mismatch for {paper_id }: {path }"""
            )
        paths.append(path)
    return paths


def review___slugify(name: str) -> str:
    stem = Path(name).stem
    return "".join(char if char.isalnum() else "_" for char in stem).strip("_")[:120]


def review___git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=review__REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def review___cleanup_review_artifacts(
    output_dir: Path, *, active_slugs: set[str]
) -> None:
    for pattern in ("*.chunks.md", "*.chunks.json"):
        for path in output_dir.glob(pattern):
            slug = path.name.split(".chunks.", 1)[0]
            if slug not in active_slugs:
                path.unlink(missing_ok=True)


def review___citation_metrics(mentions: list[Any]) -> dict[str, Any]:
    spans: dict[str, list[Any]] = defaultdict(list)
    for mention in mentions:
        spans[mention.citation_span_id].append(mention)
    span_status = Counter()
    for group in spans.values():
        span_status[group[0].span_resolution_status] += 1
    atomic_resolved_reference = sum(
        1 for mention in mentions if mention.reference_entry_id
    )
    atomic_resolved_work = sum(1 for mention in mentions if mention.resolved_work_id)
    failure_reasons = Counter(
        mention.failure_reason for mention in mentions if mention.failure_reason
    )
    unresolved_targets = [
        {
            "surface_text": mention.surface_text,
            "atomic_key": mention.atomic_key,
            "bibliography_scope_id": mention.bibliography_scope_id,
            "document_region": mention.document_region,
            "failure_reason": mention.failure_reason,
            "resolution_status": mention.resolution_status,
        }
        for mention in mentions
        if mention.resolution_status != "resolved"
    ]
    return {
        "citation_span_count": len(spans),
        "fully_resolved_span_count": span_status.get("resolved", 0),
        "partially_resolved_span_count": span_status.get("partially_resolved", 0),
        "unresolved_span_count": span_status.get("unresolved", 0),
        "atomic_target_count": len(mentions),
        "resolved_reference_target_count": atomic_resolved_reference,
        "resolved_work_target_count": atomic_resolved_work,
        "unresolved_target_count": len(mentions) - atomic_resolved_reference,
        "failure_reasons": dict(failure_reasons),
        "unresolved_targets": unresolved_targets,
    }


def review___validate_chunks(
    *, chunks: list[Any], hard_max_tokens: int, section_by_id: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    if not chunks:
        errors.append("chunk_count_zero")
    for chunk in chunks:
        if not (chunk.text or "").strip():
            errors.append(f"""empty_chunk:{chunk .id }""")
        if (chunk.token_count or 0) > hard_max_tokens:
            errors.append(
                f"""hard_max_violation:{chunk .id }:{chunk .token_count }>{hard_max_tokens }"""
            )
        if chunk.retrieval_text and any(
            span.text not in chunk.retrieval_text for span in chunk.spans
        ):
            errors.append(f"""retrieval_missing_authoritative:{chunk .id }""")
        if chunk.text.startswith("Paper:") or "\nSection:\n" in chunk.text[:80]:
            errors.append(f"""authoritative_has_header:{chunk .id }""")
    return {"pass": not errors, "errors": errors, "chunk_count": len(chunks)}


def review___element_source_field(element: Element) -> str:
    if element.element_type == ElementType.TABLE:
        if element.markdown is not None:
            return "markdown"
        if element.text is not None:
            return "text"
        return "html"
    if element.element_type == ElementType.FORMULA:
        if element.latex is not None:
            return "latex"
        if element.text is not None:
            return "text"
        return "markdown"
    return "text" if element.text is not None else "markdown"


def review___source_text(element: Element, source_field: str | None) -> str:
    if source_field is None:
        raise ValueError("source span has no source_field")
    if source_field.startswith("metadata."):
        value = element.metadata.get(source_field.removeprefix("metadata."))
    else:
        value = getattr(element, source_field, None)
    if not isinstance(value, str):
        raise TypeError(
            f"source field {source_field!r} is not text on {element.id}"
        )
    return value


def review___figure_text_sources(
    figure: Element, *, elements_by_id: dict[str, Element]
) -> list[tuple[str, str, int, int, str]]:
    captions: list[tuple[str, str, int, int, str]] = []
    for caption_id in figure.caption_element_ids:
        caption = elements_by_id.get(caption_id)
        if caption is None or caption.element_type != ElementType.CAPTION:
            continue
        value = element_text(caption)
        if value.strip():
            captions.append(
                (
                    caption.id,
                    review___element_source_field(caption),
                    0,
                    len(value),
                    value,
                )
            )
    if captions:
        return captions
    for key in ("alt", "alt_text", "description"):
        value = figure.metadata.get(key)
        if isinstance(value, str) and value.strip():
            stripped = value.strip()
            start = value.index(stripped)
            return [
                (
                    figure.id,
                    f"metadata.{key}",
                    start,
                    start + len(stripped),
                    stripped,
                )
            ]
    return []


def review___projection_metrics(
    *, bundle: CanonicalBundle, chunks: list[Any], mentions: list[Any]
) -> dict[str, Any]:
    elements_by_id = {element.id: element for element in bundle.elements}
    section_by_id = {section.id: section for section in bundle.sections}
    bound_captions = figure_caption_element_ids(bundle.elements)
    _, element_regions = build_document_regions(
        elements=bundle.elements, sections=bundle.sections
    )
    eligible: list[Element] = []
    for element in bundle.elements:
        info = element_regions.get(element.id)
        if classify_chunk_eligibility(
            element,
            section_by_id=section_by_id,
            region_type=info.region_type if info else None,
            bound_figure_caption_ids=bound_captions,
        ).eligible:
            eligible.append(element)

    unique_spans: dict[str, Any] = {}
    bound_caption_spans = 0
    bound_caption_duplications = 0
    source_provenance_errors = 0
    section_crossings: set[tuple[str, str]] = set()
    for chunk in chunks:
        for span in chunk.spans:
            unique_spans.setdefault(span.id, span)
            if (
                span.provenance_kind == "source"
                and span.element_id in bound_captions
            ):
                bound_caption_spans += 1
                caption = elements_by_id.get(span.element_id)
                if (
                    caption is None
                    or caption.parent_element_id not in chunk.element_ids
                ):
                    bound_caption_duplications += 1
            element = elements_by_id.get(span.element_id)
            if element is None:
                source_provenance_errors += 1
                continue
            if span.provenance_kind == "source":
                try:
                    source = review___source_text(element, span.source_field)
                except (TypeError, ValueError):
                    source_provenance_errors += 1
                else:
                    start = span.character_start_in_element
                    end = span.character_end_in_element
                    if source[start:end] != span.text:
                        source_provenance_errors += 1
            expected_major = resolve_major_section_id(
                element.section_id, section_by_id
            )
            actual_major = chunk.major_section_id or "__unsectioned__"
            if expected_major != actual_major:
                section_crossings.add((chunk.id, span.element_id))

    text_loss = 0
    text_duplication = 0
    figure_lost = 0
    figure_placeholder_count = 0
    figure_placeholder_part_count = 0
    figure_page_or_id_errors = 0
    figure_description_loss = 0
    figure_description_duplication = 0
    figures = [
        element
        for element in eligible
        if element.element_type == ElementType.FIGURE
    ]
    for element in eligible:
        source = element_text(element)
        spans = sorted(
            (
                span
                for span in unique_spans.values()
                if span.element_id == element.id
                and span.provenance_kind == "source"
                and span.source_field == review___element_source_field(element)
            ),
            key=lambda span: (
                span.character_start_in_element,
                span.character_end_in_element,
            ),
        )
        if element.element_type != ElementType.FIGURE:
            cursor = 0
            reconstructed: list[str] = []
            for span in spans:
                start = span.character_start_in_element
                end = span.character_end_in_element
                if start > cursor:
                    text_loss += 1
                if start < cursor:
                    text_duplication += 1
                reconstructed.append(source[start:end])
                cursor = max(cursor, end)
            if cursor < len(source):
                text_loss += 1
            if "".join(reconstructed) != source:
                text_loss += 1
            continue
        markers = [
            span
            for span in unique_spans.values()
            if span.element_id == element.id
            and span.text.startswith("[FIGURE ")
        ]
        figure_placeholder_count += int(bool(markers))
        figure_placeholder_part_count += len(markers)
        if not markers:
            figure_lost += 1
            continue
        for span in markers:
            if (
                span.provenance_kind != "projection"
                or span.source_field is not None
                or span.character_start_in_element != 0
                or span.character_end_in_element != 0
                or span.token_start != 0
                or span.token_end != 0
                or f"id={element.id}" not in span.text
                or not span.text.endswith("[/FIGURE]")
            ):
                figure_page_or_id_errors += 1
            if element.page is not None and f"page={element.page}" not in span.text:
                figure_page_or_id_errors += 1
        for (
            source_element_id,
            source_field,
            expected_start,
            expected_end,
            expected_text,
        ) in review___figure_text_sources(
            element, elements_by_id=elements_by_id
        ):
            textual_spans = sorted(
                (
                    span
                    for span in unique_spans.values()
                    if span.provenance_kind == "source"
                    and span.element_id == source_element_id
                    and span.source_field == source_field
                    and span.character_end_in_element > expected_start
                    and span.character_start_in_element < expected_end
                ),
                key=lambda span: span.character_start_in_element,
            )
            cursor = expected_start
            reconstructed = []
            for span in textual_spans:
                if span.character_start_in_element > cursor:
                    figure_description_loss += 1
                if span.character_start_in_element < cursor:
                    figure_description_duplication += 1
                reconstructed.append(span.text)
                cursor = max(cursor, span.character_end_in_element)
            if cursor < expected_end or "".join(reconstructed) != expected_text:
                figure_description_loss += 1

    fallback_reasons: Counter[str] = Counter()
    for chunk in chunks:
        fallback_reasons.update(chunk.metadata.get("fallback_split_reasons") or {})
    caption_citation_missing = sum(
        1
        for mention in mentions
        if mention.element_id in bound_captions and mention.chunk_id is None
    )
    errors = []
    if figure_lost:
        errors.append(f"figure_lost:{figure_lost}")
    if bound_caption_duplications:
        errors.append(
            f"figure_caption_duplicated:{bound_caption_duplications}"
        )
    if figure_page_or_id_errors:
        errors.append(f"figure_provenance_errors:{figure_page_or_id_errors}")
    if source_provenance_errors:
        errors.append(f"source_provenance_errors:{source_provenance_errors}")
    if caption_citation_missing:
        errors.append(f"caption_citation_missing:{caption_citation_missing}")
    if figure_description_loss or figure_description_duplication:
        errors.append(
            "figure_description_coverage:"
            f"{figure_description_loss}:{figure_description_duplication}"
        )
    if text_loss or text_duplication:
        errors.append(f"text_coverage:{text_loss}:{text_duplication}")
    if section_crossings:
        errors.append(f"section_crossings:{len(section_crossings)}")
    return {
        "figure_input_count": len(figures),
        "figure_placeholder_count": figure_placeholder_count,
        "figure_placeholder_part_count": figure_placeholder_part_count,
        "figure_lost_count": figure_lost,
        "figure_caption_provenance_span_count": bound_caption_spans,
        "figure_caption_duplication_count": bound_caption_duplications,
        "figure_provenance_error_count": figure_page_or_id_errors,
        "source_provenance_error_count": source_provenance_errors,
        "figure_description_loss_count": figure_description_loss,
        "figure_description_duplication_count": figure_description_duplication,
        "table_input_count": sum(
            element.element_type == ElementType.TABLE for element in eligible
        ),
        "equation_input_count": sum(
            element.element_type == ElementType.FORMULA for element in eligible
        ),
        "citation_mention_count": len(mentions),
        "caption_citation_missing_count": caption_citation_missing,
        "text_loss_count": text_loss,
        "text_duplication_count": text_duplication,
        "section_cross_boundary_count": len(section_crossings),
        "fallback_split_count": sum(fallback_reasons.values()),
        "fallback_split_reasons": dict(fallback_reasons),
        "errors": errors,
    }


def review___load_bundle_from_snapshot_dir(snapshot_dir: Path) -> CanonicalBundle:
    warnings_payload = json.loads(
        (snapshot_dir / "warnings.json").read_text(encoding="utf-8")
    )
    return CanonicalBundle(
        snapshot=CanonicalSnapshot.model_validate_json(
            (snapshot_dir / "snapshot.json").read_text(encoding="utf-8")
        ),
        document=Document.model_validate_json(
            (snapshot_dir / "document.json").read_text(encoding="utf-8")
        ),
        sections=[
            Section.model_validate_json(line)
            for line in (snapshot_dir / "sections.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ],
        elements=[
            Element.model_validate_json(line)
            for line in (snapshot_dir / "elements.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ],
        references=[
            ReferenceEntry.model_validate_json(line)
            for line in (snapshot_dir / "references.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ],
        warnings=warnings_payload,
    )


def review___guess_pdf_for_bundle(bundle: CanonicalBundle, corpus_dir: Path) -> Path:
    title_tokens = {
        token
        for token in review___slugify(bundle.document.title).casefold().split("_")
        if len(token) > 2
    }
    manifest_path = corpus_dir.parent / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {}
    )
    title_by_file = {
        Path(str(entry["file"])).name: str(entry.get("title") or "")
        for entry in manifest.values()
    }
    best: tuple[int, Path] | None = None
    for pdf_path in sorted(corpus_dir.glob("*.pdf")):
        candidate = "_".join((pdf_path.stem, title_by_file.get(pdf_path.name, "")))
        pdf_tokens = {
            token
            for token in review___slugify(candidate).casefold().split("_")
            if len(token) > 2
        }
        score = len(title_tokens & pdf_tokens)
        if best is None or score > best[0]:
            best = (score, pdf_path)
    if best is None or best[0] == 0:
        raise RuntimeError(
            f"Unable to map canonical bundle to corpus PDF: {bundle.document.title}"
        )
    return best[1]


def review___process_bundle(
    *,
    bundle: CanonicalBundle,
    pdf_path: Path,
    settings: Any,
    output_dir: Path,
    overlap_tokens: int,
) -> dict[str, Any]:
    target = settings.ingestion.chunk_target_tokens
    hard_max = settings.ingestion.chunk_hard_max_tokens
    result: dict[str, Any] = {
        "pdf": str(pdf_path),
        "status": "pending",
        "snapshot_id": bundle.snapshot.id,
        "document_id": bundle.document.id,
    }
    try:
        tokenizer = review___resolve_tokenizer()
        chunks, mentions = build_chunks(
            document=bundle.document,
            snapshot_id=bundle.snapshot.id,
            sections=bundle.sections,
            elements=bundle.elements,
            references=bundle.references,
            target_tokens=target,
            hard_max_tokens=hard_max,
            overlap_tokens=overlap_tokens,
            tokenizer=tokenizer,
        )
        section_by_id = {section.id: section for section in bundle.sections}
        invariants = review___validate_chunks(
            chunks=chunks, hard_max_tokens=hard_max, section_by_id=section_by_id
        )
        projection_metrics = review___projection_metrics(
            bundle=bundle, chunks=chunks, mentions=mentions
        )
        invariants["errors"].extend(projection_metrics["errors"])
        invariants["pass"] = not invariants["errors"]
        invariants.update(
            {
                key: value
                for key, value in projection_metrics.items()
                if key != "errors"
            }
        )
        token_counts = [chunk.token_count or 0 for chunk in chunks]
        boundaries = Counter(chunk.metadata.get("end_boundary") for chunk in chunks)
        real_emergency_splits = sum(
            int(chunk.metadata.get("real_emergency_splits") or 0) for chunk in chunks
        )
        table_parts = sum(
            int(chunk.metadata.get("table_parts") or 0) for chunk in chunks
        )
        ref_chunks = [
            element
            for element in bundle.elements
            if element.element_type == ElementType.REFERENCE
        ]
        citation_stats = review___citation_metrics(mentions)
        markdown = render_chunk_review_markdown(
            bundle=bundle,
            chunks=chunks,
            mentions=mentions,
            source_pdf=pdf_path,
            target_tokens=target,
            hard_max_tokens=hard_max,
            overlap_tokens=overlap_tokens,
            invariants=invariants,
            citation_stats=citation_stats,
        )
        slug = review___slugify(pdf_path.name)
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / f"""{slug }.chunks.md"""
        md_path.write_text(markdown, encoding="utf-8")
        json_path = output_dir / f"""{slug }.chunks.json"""
        json_path.write_text(
            json.dumps(
                {
                    "pdf": str(pdf_path),
                    "document_id": bundle.document.id,
                    "snapshot_id": bundle.snapshot.id,
                    "chunking_version": CHUNKING_VERSION,
                    "chunk_count": len(chunks),
                    "invariants": invariants,
                    "citation_stats": citation_stats,
                    "task4_metrics": projection_metrics,
                    "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
                    "citation_mentions": [
                        mention.model_dump(mode="json") for mention in mentions
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        result.update(
            {
                "status": "PASS" if invariants["pass"] else "FAIL",
                "chunk_count": len(chunks),
                "min_tokens": min(token_counts) if token_counts else 0,
                "median_tokens": statistics.median(token_counts) if token_counts else 0,
                "max_tokens": max(token_counts) if token_counts else 0,
                "tiny_chunks": sum(
                    1 for count in token_counts if count < TINY_TOKEN_THRESHOLD
                ),
                "emergency_splits": real_emergency_splits,
                "real_emergency_splits": real_emergency_splits,
                "table_parts": table_parts,
                "reference_elements": len(ref_chunks),
                "boundaries": dict(boundaries),
                "markdown_path": str(md_path),
                "json_path": str(json_path),
                "errors": invariants["errors"],
                **{
                    key: value
                    for key, value in projection_metrics.items()
                    if key != "errors"
                },
                **citation_stats,
            }
        )
    except Exception as exc:  # noqa: BLE001 - validation report captures all failures
        result.update(
            {"status": "ERROR", "error": f"""{type (exc ).__name__ }: {exc }"""}
        )
    return result


async def review___process_pdf(
    application: Any,
    pdf_path: Path,
    *,
    settings: Any,
    output_dir: Path,
    overlap_tokens: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"pdf": str(pdf_path), "status": "pending"}
    try:
        canonical = await application.services.ingestion.ingest_pdf_to_canonical(
            pdf_path, dataset=settings.dataset
        )
        return review___process_bundle(
            bundle=canonical.canonical,
            pdf_path=pdf_path,
            settings=settings,
            output_dir=output_dir,
            overlap_tokens=overlap_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - validation report captures all failures
        result.update(
            {"status": "ERROR", "error": f"""{type (exc ).__name__ }: {exc }"""}
        )
    return result


def review___run_rechunk_from_canonical(
    *, run_dir: Path, corpus_dir: Path, settings: Any, overlap_tokens: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    canonical_root = run_dir / "canonical"
    for src_dir in sorted(canonical_root.glob("src_*")):
        snapshot_dirs = sorted(src_dir.glob("snapshot_*"))
        if not snapshot_dirs:
            continue
        bundle = review___load_bundle_from_snapshot_dir(snapshot_dirs[-1])
        pdf_path = review___guess_pdf_for_bundle(bundle, corpus_dir)
        rows.append(
            review___process_bundle(
                bundle=bundle,
                pdf_path=pdf_path,
                settings=settings,
                output_dir=run_dir,
                overlap_tokens=overlap_tokens,
            )
        )
    return rows


async def review__run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    review___assert_validation_run_dir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    configured = load_settings(args.settings)
    settings = configured.model_copy(
        update={
            "data": configured.data.model_copy(
                update={"directory": run_dir, "dataset": args.dataset}
            )
        }
    )
    overlap_tokens = (
        args.overlap_tokens
        if args.overlap_tokens is not None
        else settings.ingestion.chunk_overlap_tokens
    )
    pdfs = review___selected_pdfs(args.corpus_dir, args.papers_config)
    if not pdfs:
        raise RuntimeError(f"""No PDFs found in {args .corpus_dir }""")
    active_slugs = {review___slugify(pdf_path.name) for pdf_path in pdfs}
    review___cleanup_review_artifacts(run_dir, active_slugs=active_slugs)
    if args.rebuild_canonical:
        rebuild_rows = rebuild__rebuild_all_canonical_snapshots(
            run_dir=run_dir, dataset_id=args.dataset
        )
        print(json.dumps({"rebuild_canonical": rebuild_rows}, indent=2))
    if args.rebuild_canonical or args.rechunk_canonical:
        paper_dir = (
            args.corpus_dir
            if args.corpus_dir.name == "papers"
            else args.corpus_dir / "papers"
        )
        rows = review___run_rechunk_from_canonical(
            run_dir=run_dir,
            corpus_dir=paper_dir,
            settings=settings,
            overlap_tokens=overlap_tokens,
        )
    else:
        from paperos_core.application import create_application

        application = create_application(settings)
        application.storage.initialize()
        rows = []
        try:
            for pdf_path in pdfs:
                rows.append(
                    await review___process_pdf(
                        application,
                        pdf_path,
                        settings=settings,
                        output_dir=run_dir,
                        overlap_tokens=overlap_tokens,
                    )
                )
        finally:
            await application.mineru.aclose()
            await application.local_inference_client.aclose()
    summary = {
        "run_dir": str(run_dir),
        "corpus_dir": str(args.corpus_dir.resolve()),
        "dataset": args.dataset,
        "git_commit": review___git_commit(),
        "chunking_version": CHUNKING_VERSION,
        "overlap_tokens": overlap_tokens,
        "chunk_target_tokens": settings.ingestion.chunk_target_tokens,
        "chunk_hard_max_tokens": settings.ingestion.chunk_hard_max_tokens,
        "pdf_count": len(rows),
        "pass_count": sum(1 for row in rows if row.get("status") == "PASS"),
        "structure_only_pass_count": sum(
            1 for row in rows if row.get("status") == "PASS"
        ),
        "overall_status": (
            "STRUCTURE_ONLY_PASS"
            if rows and all(row.get("status") == "PASS" for row in rows)
            else "FAIL"
        ),
        "results": rows,
    }
    metric_keys = (
        "figure_input_count",
        "figure_placeholder_count",
        "figure_placeholder_part_count",
        "figure_lost_count",
        "figure_caption_duplication_count",
        "figure_provenance_error_count",
        "source_provenance_error_count",
        "figure_description_loss_count",
        "figure_description_duplication_count",
        "table_input_count",
        "equation_input_count",
        "citation_mention_count",
        "caption_citation_missing_count",
        "text_loss_count",
        "text_duplication_count",
        "section_cross_boundary_count",
        "fallback_split_count",
    )
    summary.update(
        {
            key: sum(int(row.get(key, 0)) for row in rows)
            for key in metric_keys
        }
    )
    summary["max_chunk_tokens"] = max(
        (int(row.get("max_tokens", 0)) for row in rows), default=0
    )
    summary["hard_max_violation_count"] = sum(
        1
        for row in rows
        for error in row.get("errors", [])
        if str(error).startswith("hard_max_violation:")
    )
    fallback_reasons: Counter[str] = Counter()
    for row in rows:
        fallback_reasons.update(row.get("fallback_split_reasons") or {})
    summary["fallback_split_reasons"] = dict(fallback_reasons)
    report_path = run_dir / "chunk-corpus-review.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["report_path"] = str(report_path)
    markdown_path = run_dir / "chunk-corpus-review.md"
    markdown_lines = [
        "# Chunk Corpus Review",
        "",
        f"Overall status: {summary['overall_status']}",
        f"Papers: {summary['pdf_count']}",
        f"Maximum Chunk tokens: {summary['max_chunk_tokens']}",
        f"Hard-max violations: {summary['hard_max_violation_count']}",
        (
            f"Figure inputs/placeholders/lost: {summary['figure_input_count']}/"
            f"{summary['figure_placeholder_count']}/{summary['figure_lost_count']}"
        ),
        f"Figure caption duplications: {summary['figure_caption_duplication_count']}",
        (
            f"Table/Equation/Citation counts: {summary['table_input_count']}/"
            f"{summary['equation_input_count']}/{summary['citation_mention_count']}"
        ),
        (
            f"Text loss/duplication: {summary['text_loss_count']}/"
            f"{summary['text_duplication_count']}"
        ),
        f"Section cross-boundary count: {summary['section_cross_boundary_count']}",
        (
            f"Fallback splits: {summary['fallback_split_count']} "
            f"{summary['fallback_split_reasons']}"
        ),
        "",
        "## Papers",
        "",
    ]
    for row in rows:
        markdown_lines.extend(
            [
                f"- {Path(str(row['pdf'])).name}: {row.get('status')}",
                (
                    f"  - chunks/max tokens: {row.get('chunk_count', 0)}/"
                    f"{row.get('max_tokens', 0)}"
                ),
                (
                    "  - figures input/placeholders/lost: "
                    f"{row.get('figure_input_count', 0)}/"
                    f"{row.get('figure_placeholder_count', 0)}/"
                    f"{row.get('figure_lost_count', 0)}"
                ),
                (
                    f"  - fallback splits: {row.get('fallback_split_count', 0)} "
                    f"{row.get('fallback_split_reasons', {})}"
                ),
            ]
        )
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    summary["markdown_report_path"] = str(markdown_path)
    return summary


def review__main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir", type=Path, default=Path("data/validation/corpus")
    )
    parser.add_argument(
        "--papers-config",
        type=Path,
        default=Path("data/validation/chunk/config/papers.json"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=review__DEFAULT_RUN_DIR,
        help="Isolated DATA_DIR for this validation run (must be under validation/runs/).",
    )
    parser.add_argument("--dataset", default=review__DEFAULT_DATASET)
    parser.add_argument("--settings", type=Path, default=None)
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=None,
        help="Defaults to production config chunk_overlap_tokens.",
    )
    parser.add_argument(
        "--rebuild-canonical",
        action="store_true",
        help="Re-map canonical from cached parsed MinerU artifacts (no re-OCR).",
    )
    parser.add_argument(
        "--rechunk-canonical",
        action="store_true",
        help="Rebuild chunks from cached canonical snapshots (skip MinerU ingest).",
    )
    args = parser.parse_args()
    summary = asyncio.run(review__run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["pass_count"] != summary["pdf_count"]:
        raise SystemExit(1)


"Validate MinerU source fields → Canonical provenance using Gold v3 anchors."
import re
import sys
from pathlib import Path

source__REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(source__REPOSITORY_ROOT))
from paperos_core.ingestion.normalization import source_evidence_text


def source___git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source__REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source___field_key(record: dict[str, Any]) -> tuple[int, str, int | None]:
    return (
        record["source_item"],
        record["source_domain"],
        record.get("source_subindex"),
    )


def source___span_key(span: Any) -> tuple[int, str, int | None]:
    return (span.item_index, span.source_domain, span.source_subindex)


def source___expected_element_text(element_type: ElementType, source: str) -> str:
    if element_type == ElementType.FORMULA:
        value = source_evidence_text(source)
        value = re.sub("^\\$\\$\\s*", "", value)
        return re.sub("\\s*\\$\\$$", "", value).strip()
    return source_evidence_text(source)


def source___check_element_provenance(
    *,
    paper_key: str,
    bundle: Any,
    fields_by_key: dict[tuple[int, str, int | None], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for element in bundle.elements:
        evidence = element_text(element)
        if not evidence:
            continue
        span = element.source_span
        if span is None:
            failures.append(
                {
                    "failure_type": "CANONICAL_SOURCE_PROVENANCE_MISSING",
                    "paper": paper_key,
                    "element_id": element.id,
                }
            )
            continue
        fields = fields_by_key.get(source___span_key(span), [])
        if len(fields) != 1:
            failures.append(
                {
                    "failure_type": "CANONICAL_SOURCE_FIELD_MISSING",
                    "paper": paper_key,
                    "element_id": element.id,
                    "source_item": span.item_index,
                    "source_domain": span.source_domain,
                    "source_subindex": span.source_subindex,
                    "field_matches": len(fields),
                }
            )
            continue
        source = fields[0]["value"]
        start = span.character_start
        end = span.character_end
        if start is None or end is None or (not 0 <= start <= end <= len(source)):
            failures.append(
                {
                    "failure_type": "CANONICAL_SOURCE_RANGE_INVALID",
                    "paper": paper_key,
                    "element_id": element.id,
                    "source_item": span.item_index,
                    "start": start,
                    "end": end,
                    "source_length": len(source),
                }
            )
            continue
        source_slice = source[start:end]
        expected = (
            source_slice
            if element.element_type == ElementType.TABLE
            else source___expected_element_text(element.element_type, source_slice)
        )
        if evidence != expected:
            failures.append(
                {
                    "failure_type": "CANONICAL_SOURCE_CONTENT_MISMATCH",
                    "paper": paper_key,
                    "element_id": element.id,
                    "source_item": span.item_index,
                    "source_domain": span.source_domain,
                    "canonical_length": len(evidence),
                    "expected_length": len(expected),
                }
            )
    return failures


def source___check_gold_anchors(
    *,
    paper_key: str,
    paper: dict[str, Any],
    bundle: Any,
    fields_by_key: dict[tuple[int, str, int | None], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    elements_by_key: dict[tuple[int, str, int | None], list[Any]] = defaultdict(list)
    for element in bundle.elements:
        if element.source_span is not None:
            elements_by_key[source___span_key(element.source_span)].append(element)
    failures: list[dict[str, Any]] = []
    for occurrence in paper["occurrences"]:
        key = source___field_key(occurrence)
        fields = fields_by_key.get(key, [])
        start = occurrence["start"]
        end = occurrence["end"]
        source_valid = (
            len(fields) == 1
            and 0 <= start <= end <= len(fields[0]["value"])
            and (fields[0]["value"][start:end] == occurrence["surface"])
        )
        if not source_valid:
            failures.append(
                {
                    "failure_type": "GOLD_MINERU_ANCHOR_INVALID",
                    "paper": paper_key,
                    "source_item": occurrence["source_item"],
                    "source_domain": occurrence["source_domain"],
                    "start": start,
                    "end": end,
                    "surface": occurrence["surface"],
                }
            )
            continue
        mapped = False
        for element in elements_by_key.get(key, []):
            span = element.source_span
            if span.character_start is None or span.character_end is None:
                continue
            if not (span.character_start <= start and end <= span.character_end):
                continue
            if gold_builder___flex_occurrences(
                element_text(element), occurrence["surface"]
            ):
                mapped = True
                break
        if not mapped:
            failures.append(
                {
                    "failure_type": "GOLD_CANONICAL_SOURCE_LOSS",
                    "paper": paper_key,
                    "source_item": occurrence["source_item"],
                    "source_domain": occurrence["source_domain"],
                    "source_subindex": occurrence.get("source_subindex"),
                    "start": start,
                    "end": end,
                    "surface": occurrence["surface"],
                }
            )
    return failures


def source__main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("data/validation/chunk/output")
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=Path("data/validation/corpus")
    )
    parser.add_argument(
        "--papers-config",
        type=Path,
        default=Path("data/validation/chunk/config/papers.json"),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/validation/chunk/config/citation_gold.json"),
    )
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    failures: list[dict[str, Any]] = []
    papers_report: dict[str, dict[str, Any]] = {}
    for paper_key, paper in gold["papers"].items():
        snapshot_dirs = sorted(
            (args.run_dir / "canonical" / paper["source_id"]).glob("snapshot_*")
        )
        if len(snapshot_dirs) != 1:
            raise RuntimeError(
                f"""Expected one canonical snapshot for {paper_key }, got {len (snapshot_dirs )}"""
            )
        canonical_dir = snapshot_dirs[0]
        bundle = review___load_bundle_from_snapshot_dir(canonical_dir)
        fields = gold_builder___source_fields(
            gold_builder___mineru_content(
                args.run_dir, paper["source_id"], canonical_dir
            )
        )
        fields_by_key: dict[tuple[int, str, int | None], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        for field in fields:
            fields_by_key[source___field_key(field)].append(field)
        element_failures = source___check_element_provenance(
            paper_key=paper_key, bundle=bundle, fields_by_key=fields_by_key
        )
        gold_failures = source___check_gold_anchors(
            paper_key=paper_key, paper=paper, bundle=bundle, fields_by_key=fields_by_key
        )
        paper_failures = [*element_failures, *gold_failures]
        failures.extend(paper_failures)
        papers_report[paper_key] = {
            "source_id": paper["source_id"],
            "snapshot_id": bundle.snapshot.id,
            "canonical_elements_checked": sum(
                bool(element_text(element)) for element in bundle.elements
            ),
            "gold_occurrences_checked": len(paper["occurrences"]),
            "canonical_source_loss": len(element_failures),
            "gold_canonical_source_loss": len(gold_failures),
            "failures": paper_failures[:20],
            "status": "PASS" if not paper_failures else "FAIL",
        }
    report = {
        "git_commit": source___git_commit(),
        "gold_version": gold.get("gold_version"),
        "canonical_source_loss": sum(
            item["canonical_source_loss"] for item in papers_report.values()
        ),
        "gold_canonical_source_loss": sum(
            item["gold_canonical_source_loss"] for item in papers_report.values()
        ),
        "failure_count": len(failures),
        "papers": papers_report,
        "failures": failures[:100],
        "pass": not failures,
    }
    output = args.run_dir / "canonical-source-survival.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"pass": report["pass"], "failures": len(failures), "report": str(output)},
            indent=2,
        )
    )
    return 0 if report["pass"] else 1


"Validate canonical element text coverage by chunk authoritative spans."
import sys
from pathlib import Path

coverage__REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(coverage__REPOSITORY_ROOT))


def coverage___git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=coverage__REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def coverage__validate_paper(*, bundle, chunks_json: dict) -> dict:
    chunks = chunks_json["chunks"]
    section_by_id = {section.id: section for section in bundle.sections}
    _, element_regions = build_document_regions(
        elements=bundle.elements, sections=bundle.sections
    )
    elements_by_id = {element.id: element for element in bundle.elements}
    bound_figure_captions = figure_caption_element_ids(bundle.elements)
    eligible = []
    excluded = []
    exclusion_stats: Counter[str] = Counter()
    for element in bundle.elements:
        region_info = element_regions.get(element.id)
        eligibility = classify_chunk_eligibility(
            element,
            section_by_id=section_by_id,
            region_type=region_info.region_type if region_info else None,
            bound_figure_caption_ids=bound_figure_captions,
        )
        if not eligibility.eligible:
            excluded.append(
                {
                    "element_id": element.id,
                    "element_type": element.element_type.value,
                    "exclusion_reason": eligibility.reason,
                }
            )
            exclusion_stats[eligibility.reason] += 1
            continue
        eligible.append(element)
    failures = []
    holes = 0
    overlaps = 0
    unique_spans = {
        span["id"]: span
        for chunk in chunks
        for span in chunk.get("spans", [])
    }
    for span in unique_spans.values():
        if span.get("provenance_kind", "source") != "source":
            continue
        element = elements_by_id.get(span["element_id"])
        try:
            source = review___source_text(element, span.get("source_field"))
        except (AttributeError, ValueError):
            failures.append(
                {
                    "element_id": span["element_id"],
                    "span_id": span["id"],
                    "failure_type": "invalid_source_field",
                }
            )
            continue
        start = span["character_start_in_element"]
        end = span["character_end_in_element"]
        if source[start:end] != span["text"]:
            failures.append(
                {
                    "element_id": span["element_id"],
                    "span_id": span["id"],
                    "failure_type": "source_coordinate_mismatch",
                }
            )
    for element in eligible:
        if element.element_type == ElementType.FIGURE:
            markers = [
                span
                for span in unique_spans.values()
                if span["element_id"] == element.id
                and span["text"].startswith("[FIGURE ")
            ]
            if not markers:
                failures.append(
                    {
                        "element_id": element.id,
                        "failure_type": "figure_projection_missing",
                    }
                )
            for span in markers:
                if (
                    span.get("provenance_kind") != "projection"
                    or span.get("source_field") is not None
                    or span["character_start_in_element"] != 0
                    or span["character_end_in_element"] != 0
                    or span["token_start"] != 0
                    or span["token_end"] != 0
                    or f"id={element.id}" not in span["text"]
                    or not span["text"].endswith("[/FIGURE]")
                    or (
                        element.page is not None
                        and f"page={element.page}" not in span["text"]
                    )
                ):
                    failures.append(
                        {
                            "element_id": element.id,
                            "span_id": span["id"],
                            "failure_type": "invalid_figure_projection",
                        }
                    )
            for (
                source_element_id,
                source_field,
                expected_start,
                expected_end,
                expected_text,
            ) in review___figure_text_sources(
                element, elements_by_id=elements_by_id
            ):
                textual_spans = sorted(
                    (
                        span
                        for span in unique_spans.values()
                        if span.get("provenance_kind", "source") == "source"
                        and span["element_id"] == source_element_id
                        and span.get("source_field") == source_field
                        and span["character_end_in_element"] > expected_start
                        and span["character_start_in_element"] < expected_end
                    ),
                    key=lambda item: item["character_start_in_element"],
                )
                cursor = expected_start
                reconstructed_parts = []
                for span in textual_spans:
                    start = span["character_start_in_element"]
                    end = span["character_end_in_element"]
                    if start != cursor:
                        failures.append(
                            {
                                "element_id": source_element_id,
                                "failure_type": "figure_text_gap_or_overlap",
                            }
                        )
                    reconstructed_parts.append(span["text"])
                    cursor = max(cursor, end)
                if cursor != expected_end or "".join(reconstructed_parts) != expected_text:
                    failures.append(
                        {
                            "element_id": source_element_id,
                            "failure_type": "figure_text_source_mismatch",
                        }
                    )
            continue
        source = element_text(element)
        source_field = review___element_source_field(element)
        spans = [
            span
            for span in unique_spans.values()
            if span["element_id"] == element.id
            and span.get("provenance_kind", "source") == "source"
            and span.get("source_field") == source_field
        ]
        spans.sort(key=lambda item: item["character_start_in_element"])
        cursor = 0
        reconstructed_parts = []
        for span in spans:
            start = span["character_start_in_element"]
            end = span["character_end_in_element"]
            if start > cursor:
                holes += 1
                failures.append(
                    {
                        "element_id": element.id,
                        "failure_type": "chunk_source_hole",
                        "hole_start": cursor,
                        "hole_end": start,
                    }
                )
            if start < cursor:
                overlaps += 1
                failures.append(
                    {
                        "element_id": element.id,
                        "failure_type": "chunk_source_overlap",
                        "overlap_start": start,
                        "overlap_end": cursor,
                    }
                )
            reconstructed_parts.append(source[start:end])
            cursor = max(cursor, end)
        if cursor < len(source):
            holes += 1
            failures.append(
                {
                    "element_id": element.id,
                    "failure_type": "chunk_source_hole",
                    "hole_start": cursor,
                    "hole_end": len(source),
                }
            )
        reconstructed = "".join(reconstructed_parts)
        if reconstructed != source:
            failures.append(
                {
                    "element_id": element.id,
                    "failure_type": "chunk_source_mismatch",
                    "source_len": len(source),
                    "reconstructed_len": len(reconstructed),
                }
            )
    return {
        "eligible_elements": len(eligible),
        "excluded_elements": excluded,
        "exclusion_stats": dict(exclusion_stats),
        "chunk_source_holes": holes,
        "chunk_source_overlaps": overlaps,
        "failures": failures,
        "pass": not failures,
    }


def coverage__main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("data/validation/chunk/output")
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=Path("data/validation/corpus")
    )
    parser.add_argument(
        "--papers-config",
        type=Path,
        default=Path("data/validation/chunk/config/papers.json"),
    )
    args = parser.parse_args()
    papers = []
    total_failures = 0
    for src_dir in sorted((args.run_dir / "canonical").glob("src_*")):
        snapshot_dirs = sorted(src_dir.glob("snapshot_*"))
        if not snapshot_dirs:
            continue
        bundle = review___load_bundle_from_snapshot_dir(snapshot_dirs[-1])
        pdf_path = review___guess_pdf_for_bundle(bundle, args.corpus_dir)
        chunk_candidates = list(
            args.run_dir.glob(f"""*{pdf_path .stem }*.chunks.json""")
        )
        if not chunk_candidates:
            chunk_candidates = [
                path
                for path in args.run_dir.glob("*.chunks.json")
                if bundle.snapshot.id in path.read_text(encoding="utf-8")
            ]
        if len(chunk_candidates) != 1:
            raise RuntimeError(
                f"""Unable to locate chunks json for {pdf_path .name }"""
            )
        chunks_json = json.loads(chunk_candidates[0].read_text(encoding="utf-8"))
        result = coverage__validate_paper(bundle=bundle, chunks_json=chunks_json)
        total_failures += len(result["failures"])
        papers.append(
            {"pdf": str(pdf_path), "snapshot_id": bundle.snapshot.id, **result}
        )
    report = {
        "git_commit": coverage___git_commit(),
        "chunk_source_holes": sum(item["chunk_source_holes"] for item in papers),
        "chunk_source_overlaps": sum(
            item["chunk_source_overlaps"] for item in papers
        ),
        "failure_count": total_failures,
        "papers": papers,
        "pass": total_failures == 0,
    }
    output = args.run_dir / "chunk-source-coverage.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {"pass": report["pass"], "failures": total_failures, "report": str(output)},
            indent=2,
        )
    )
    return 0 if report["pass"] else 1


"Validate DocumentRegion boundaries and preassigned CitationNamespace flow."
import sys
from dataclasses import asdict
from pathlib import Path

regions__REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(regions__REPOSITORY_ROOT))
from paperos_core.ingestion.bibliography_scope import (
    FAILURE_NAMESPACE_NOT_ASSIGNED,
    REGION_REFERENCES,
)
from paperos_core.ingestion.document_regions import (
    region_for_element,
)


def regions___git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=regions__REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def regions___chunks_for_snapshot(run_dir: Path, snapshot_id: str) -> dict[str, Any]:
    matches = []
    for path in run_dir.glob("*.chunks.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("snapshot_id") == snapshot_id:
            matches.append(payload)
    if len(matches) != 1:
        raise RuntimeError(
            f"""Expected one chunks file for {snapshot_id }, got {len (matches )}"""
        )
    return matches[0]


def regions__validate_paper(
    *, bundle: Any, chunks_json: dict[str, Any]
) -> dict[str, Any]:
    regions, element_regions = build_document_regions(
        elements=bundle.elements, sections=bundle.sections
    )
    failures: list[dict[str, Any]] = []
    wrong_regions = 0
    wrong_namespaces = 0
    for chunk in chunks_json.get("chunks", []):
        span_infos = [
            element_regions.get(span["element_id"]) for span in chunk.get("spans", [])
        ]
        if not span_infos or any(info is None for info in span_infos):
            wrong_regions += 1
            failures.append(
                {
                    "failure_type": "CHUNK_REGION_ELEMENT_MISSING",
                    "chunk_id": chunk.get("id"),
                }
            )
            continue
        region_ids = {info.region_id for info in span_infos if info is not None}
        namespaces = {
            info.citation_namespace_id for info in span_infos if info is not None
        }
        region_types = {
            region_for_element(info.element_id, element_regions)
            for info in span_infos
            if info is not None
        }
        if (
            len(region_ids) != 1
            or len(region_types) != 1
            or REGION_REFERENCES in region_types
        ):
            wrong_regions += 1
            failures.append(
                {
                    "failure_type": "MIXED_OR_REFERENCE_REGION_CHUNK",
                    "chunk_id": chunk.get("id"),
                    "region_ids": sorted(region_ids),
                    "region_types": sorted(region_types),
                }
            )
        expected_region_id = next(iter(region_ids))
        if chunk.get("metadata", {}).get("region_instance_id") != expected_region_id:
            wrong_regions += 1
            failures.append(
                {
                    "failure_type": "WRONG_CHUNK_REGION_INSTANCE",
                    "chunk_id": chunk.get("id"),
                    "expected": expected_region_id,
                    "actual": chunk.get("metadata", {}).get("region_instance_id"),
                }
            )
        if len(namespaces) != 1 or None in namespaces:
            wrong_namespaces += 1
            failures.append(
                {
                    "failure_type": FAILURE_NAMESPACE_NOT_ASSIGNED,
                    "chunk_id": chunk.get("id"),
                    "namespaces": sorted(value or "<none>" for value in namespaces),
                }
            )
        else:
            expected_namespace = next(iter(namespaces))
            if chunk.get("citation_namespace_id") != expected_namespace:
                wrong_namespaces += 1
                failures.append(
                    {
                        "failure_type": "WRONG_CHUNK_NAMESPACE",
                        "chunk_id": chunk.get("id"),
                        "expected": expected_namespace,
                        "actual": chunk.get("citation_namespace_id"),
                    }
                )
    for mention in chunks_json.get("citation_mentions", []):
        info = element_regions.get(mention.get("element_id"))
        diagnostic = (mention.get("metadata") or {}).get(
            "bibliography_scope_diagnostic"
        )
        if info is None:
            wrong_regions += 1
            failures.append(
                {
                    "failure_type": "MENTION_REGION_ELEMENT_MISSING",
                    "mention_id": mention.get("id"),
                }
            )
            continue
        expected_region = region_for_element(info.element_id, element_regions)
        if (
            mention.get("document_region") != expected_region
            or info.region_type == REGION_REFERENCES
        ):
            wrong_regions += 1
            failures.append(
                {
                    "failure_type": "WRONG_MENTION_REGION",
                    "mention_id": mention.get("id"),
                    "expected": expected_region,
                    "actual": mention.get("document_region"),
                }
            )
        if (
            info.citation_namespace_id is None
            or mention.get("citation_namespace_id") != info.citation_namespace_id
            or mention.get("bibliography_scope_id") != info.citation_namespace_id
            or (diagnostic == FAILURE_NAMESPACE_NOT_ASSIGNED)
        ):
            wrong_namespaces += 1
            failures.append(
                {
                    "failure_type": (
                        FAILURE_NAMESPACE_NOT_ASSIGNED
                        if info.citation_namespace_id is None
                        else "WRONG_MENTION_NAMESPACE"
                    ),
                    "mention_id": mention.get("id"),
                    "surface": mention.get("surface_text"),
                    "expected": info.citation_namespace_id,
                    "actual": mention.get("citation_namespace_id"),
                    "diagnostic": diagnostic,
                }
            )
    return {
        "regions": [asdict(region) for region in regions],
        "wrong_regions": wrong_regions,
        "wrong_namespaces": wrong_namespaces,
        "wrong_bibliography_scopes": wrong_namespaces,
        "failures": failures,
        "pass": not failures,
    }


def regions__main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("data/validation/chunk/output")
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=Path("data/validation/corpus")
    )
    parser.add_argument(
        "--papers-config",
        type=Path,
        default=Path("data/validation/chunk/config/papers.json"),
    )
    args = parser.parse_args()
    papers: list[dict[str, Any]] = []
    for src_dir in sorted((args.run_dir / "canonical").glob("src_*")):
        snapshot_dirs = sorted(src_dir.glob("snapshot_*"))
        if len(snapshot_dirs) != 1:
            raise RuntimeError(
                f"""Expected one canonical snapshot in {src_dir }, got {len (snapshot_dirs )}"""
            )
        bundle = review___load_bundle_from_snapshot_dir(snapshot_dirs[0])
        result = regions__validate_paper(
            bundle=bundle,
            chunks_json=regions___chunks_for_snapshot(args.run_dir, bundle.snapshot.id),
        )
        papers.append(
            {
                "source_id": src_dir.name,
                "title": bundle.document.title,
                "snapshot_id": bundle.snapshot.id,
                **result,
            }
        )
    report = {
        "git_commit": regions___git_commit(),
        "wrong_regions": sum(item["wrong_regions"] for item in papers),
        "wrong_namespaces": sum(item["wrong_namespaces"] for item in papers),
        "wrong_bibliography_scopes": sum(item["wrong_namespaces"] for item in papers),
        "papers": papers,
    }
    report["pass"] = report["wrong_regions"] == 0 and report["wrong_namespaces"] == 0
    output = args.run_dir / "chunk-regions-scopes.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "wrong_regions": report["wrong_regions"],
                "wrong_namespaces": report["wrong_namespaces"],
                "report": str(output),
            },
            indent=2,
        )
    )
    return 0 if report["pass"] else 1


"Validate abstract, formula-cohesion, and table-part chunk contracts."
import sys
from pathlib import Path

boundaries__REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(boundaries__REPOSITORY_ROOT))
from paperos_core.ingestion.chunking import _is_subsection_boundary
from paperos_core.ingestion.document_regions import (
    region_id_for_element,
)
from paperos_core.ingestion.inline_domains import scan_inline_domains
from paperos_core.ingestion.sentence_units import (
    SentenceUnit,
    formula_cohesion_boundary,
    units_for_element,
)


def boundaries___git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=boundaries__REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def boundaries___unit_groups(
    *, bundle: Any, count: Any, hard_max_tokens: int
) -> list[list[SentenceUnit]]:
    section_by_id = {section.id: section for section in bundle.sections}
    _, element_regions = build_document_regions(
        elements=bundle.elements, sections=bundle.sections
    )
    elements_by_id = {element.id: element for element in bundle.elements}
    bound_figure_captions = figure_caption_element_ids(bundle.elements)
    eligible = []
    for element in sorted(bundle.elements, key=lambda item: (item.order, item.id)):
        info = element_regions.get(element.id)
        eligibility = classify_chunk_eligibility(
            element,
            section_by_id=section_by_id,
            region_type=info.region_type if info else None,
            bound_figure_caption_ids=bound_figure_captions,
        )
        if eligibility.eligible:
            eligible.append(element)
    grouped: dict[tuple[str, str], list[Element]] = {}
    for element in eligible:
        major_id = resolve_major_section_id(element.section_id, section_by_id)
        if major_id is None:
            continue
        region_id = (
            region_id_for_element(element.id, element_regions) or "region_main_1"
        )
        grouped.setdefault((major_id, region_id), []).append(element)
    groups: list[list[SentenceUnit]] = []
    for key in sorted(grouped):
        elements = grouped[key]
        units: list[SentenceUnit] = []
        for index, element in enumerate(elements):
            section = section_by_id.get(element.section_id or "")
            units.extend(
                units_for_element(
                    element,
                    count=count,
                    hard_max_tokens=hard_max_tokens,
                    section_id=element.section_id,
                    section_path=section.path if section else None,
                    subsection_end=_is_subsection_boundary(
                        elements, index, section_by_id
                    ),
                    elements_by_id=elements_by_id,
                )
            )
        groups.append(units)
    return groups


def boundaries___paper_contracts(
    *, bundle: Any, chunks_json: dict[str, Any], count: Any, hard_max_tokens: int
) -> dict[str, Any]:
    _, element_regions = build_document_regions(
        elements=bundle.elements, sections=bundle.sections
    )
    abstract_errors: list[dict[str, Any]] = []
    for chunk in chunks_json.get("chunks", []):
        infos = [
            element_regions.get(span.get("element_id"))
            for span in chunk.get("spans", [])
        ]
        if (
            infos
            and all(
                info is not None and info.region_type == "abstract" for info in infos
            )
            and (chunk.get("document_region") != "abstract")
        ):
            abstract_errors.append(
                {"chunk_id": chunk.get("id"), "actual": chunk.get("document_region")}
            )
    span_chunks: dict[str, set[str]] = {}
    for chunk in chunks_json.get("chunks", []):
        for span in chunk.get("spans", []):
            span_chunks.setdefault(span["id"], set()).add(chunk["id"])
    cohesion_cases = 0
    cohesion_breaks: list[dict[str, Any]] = []
    table_misclassifications: list[dict[str, Any]] = []
    unit_by_span: dict[str, SentenceUnit] = {}
    for units in boundaries___unit_groups(
        bundle=bundle, count=count, hard_max_tokens=hard_max_tokens
    ):
        for unit in units:
            unit_by_span[unit.span_id] = unit
            if unit.split_type == "TABLE_PART" and unit.emergency_split:
                table_misclassifications.append(
                    {"span_id": unit.span_id, "split_type": unit.split_type}
                )
        for index in range(len(units) - 1):
            if not formula_cohesion_boundary(units[index], units[index + 1]):
                continue
            group_start = index
            while group_start > 0 and formula_cohesion_boundary(
                units[group_start - 1], units[group_start]
            ):
                group_start -= 1
            group_end = index + 2
            while group_end < len(units) and formula_cohesion_boundary(
                units[group_end - 1], units[group_end]
            ):
                group_end += 1
            group = units[group_start:group_end]
            if sum(unit.tokens for unit in group) > hard_max_tokens:
                continue
            cohesion_cases += 1
            shared_chunks = span_chunks.get(
                units[index].span_id, set()
            ) & span_chunks.get(units[index + 1].span_id, set())
            if not shared_chunks:
                cohesion_breaks.append(
                    {
                        "left_span_id": units[index].span_id,
                        "right_span_id": units[index + 1].span_id,
                        "group_tokens": sum(unit.tokens for unit in group),
                    }
                )
    metadata_errors: list[dict[str, Any]] = []
    table_parts = 0
    real_emergency_splits = 0
    for chunk in chunks_json.get("chunks", []):
        chunk_units = [
            unit_by_span[span["id"]]
            for span in chunk.get("spans", [])
            if span["id"] in unit_by_span
        ]
        expected_table = sum(unit.split_type == "TABLE_PART" for unit in chunk_units)
        expected_emergency = sum(unit.emergency_split for unit in chunk_units)
        table_parts += expected_table
        real_emergency_splits += expected_emergency
        metadata = chunk.get("metadata") or {}
        if int(metadata.get("table_parts") or 0) != expected_table:
            metadata_errors.append(
                {"chunk_id": chunk.get("id"), "failure_type": "TABLE_PART_COUNT"}
            )
        if (
            int(metadata.get("real_emergency_splits") or 0) != expected_emergency
            or int(metadata.get("emergency_oversized_sentence_splits") or 0)
            != expected_emergency
        ):
            metadata_errors.append(
                {"chunk_id": chunk.get("id"), "failure_type": "REAL_EMERGENCY_COUNT"}
            )
    return {
        "abstract_region_errors": len(abstract_errors),
        "abstract_failures": abstract_errors,
        "formula_cohesion_cases": cohesion_cases,
        "avoidable_formula_cohesion_breaks": len(cohesion_breaks),
        "formula_cohesion_failures": cohesion_breaks,
        "table_parts": table_parts,
        "real_emergency_splits": real_emergency_splits,
        "table_part_emergency_misclassification": len(table_misclassifications)
        + len(metadata_errors),
        "table_failures": [*table_misclassifications, *metadata_errors],
    }


class boundaries___CharacterTokenizer:

    def count_tokens(self, text: str) -> int:
        return len(text)


def boundaries__synthetic_multi_part_table_contract() -> dict[str, Any]:
    header = "| Col A | Col B |\n| --- | --- |\n"
    rows = "".join(
        f"""| row{index :02d} value | payload{index :02d} |\n"""
        for index in range(1, 7)
    )
    source = header + rows
    section = Section(
        id="section_table",
        document_id="doc_table",
        canonical_snapshot_id="snapshot_table",
        title="Results",
        level=1,
        order=0,
        path="Results",
    )
    table = Element(
        id="element_table",
        document_id="doc_table",
        canonical_snapshot_id="snapshot_table",
        element_type=ElementType.TABLE,
        order=0,
        section_id=section.id,
        markdown=source,
    )
    tokenizer = boundaries___CharacterTokenizer()
    units = units_for_element(
        table,
        count=tokenizer.count_tokens,
        hard_max_tokens=75,
        section_id=section.id,
        section_path=section.path,
        subsection_end=True,
    )
    failures: list[str] = []
    if len(units) < 3:
        failures.append(f"""table_part_count:{len (units )}""")
    cursor = 0
    reconstructed: list[str] = []
    for unit in units:
        if unit.character_start_in_element != cursor:
            failures.append(
                f"""source_gap_or_overlap:{cursor }:{unit .character_start_in_element }"""
            )
        reconstructed.append(unit.text)
        cursor = unit.character_end_in_element
        if unit.split_type != "TABLE_PART" or unit.emergency_split:
            failures.append(f"""emergency_misclassification:{unit .span_id }""")
    if cursor != len(source) or "".join(reconstructed) != source:
        failures.append("authoritative_source_coverage")
    for unit in units[1:]:
        if not (unit.display_text or "").startswith(header.rstrip("\n")):
            failures.append(f"""display_header_missing:{unit .span_id }""")
        if unit.text.startswith(header.rstrip("\n")):
            failures.append(f"""authoritative_header_repeated:{unit .span_id }""")
    document = Document(
        id="doc_table",
        source_file_id="source_table",
        parse_run_id="parse_table",
        canonical_snapshot_id="snapshot_table",
        language="en",
        title="Synthetic table contract",
    )
    chunks, _ = build_chunks(
        document=document,
        snapshot_id="snapshot_table",
        sections=[section],
        elements=[table],
        references=[],
        target_tokens=60,
        hard_max_tokens=75,
        overlap_tokens=0,
        tokenizer=tokenizer,
    )
    for chunk in chunks:
        starts_after_header = any(
            span.character_start_in_element > 0 for span in chunk.spans
        )
        if starts_after_header:
            if header.rstrip("\n") not in (chunk.retrieval_text or ""):
                failures.append(f"""retrieval_header_missing:{chunk .id }""")
            if chunk.text.startswith(header.rstrip("\n")):
                failures.append(f"""chunk_authoritative_header_repeated:{chunk .id }""")
        if int(chunk.metadata.get("real_emergency_splits") or 0) != 0:
            failures.append(f"""chunk_emergency_misclassification:{chunk .id }""")
    return {
        "multi_part_table_provenance_errors": len(failures),
        "table_part_count": len(units),
        "failures": failures,
    }


def boundaries__synthetic_figure_hard_max_contract() -> dict[str, Any]:
    hard_max = 75
    section = Section(
        id="s",
        document_id="d",
        canonical_snapshot_id="snap",
        title="Results",
        level=1,
        order=0,
        path="Results",
    )
    caption_text = (
        "Figure evidence uses a canonical caption. " * 4
    ).strip()
    figure = Element(
        id="fig",
        document_id="d",
        canonical_snapshot_id="snap",
        element_type=ElementType.FIGURE,
        order=0,
        section_id=section.id,
        page=2,
        caption_element_ids=["cap"],
    )
    caption = Element(
        id="cap",
        document_id="d",
        canonical_snapshot_id="snap",
        element_type=ElementType.CAPTION,
        order=1,
        section_id=section.id,
        parent_element_id=figure.id,
        text=caption_text,
    )
    empty_figure = Element(
        id="emptyfig",
        document_id="d",
        canonical_snapshot_id="snap",
        element_type=ElementType.FIGURE,
        order=2,
        section_id=section.id,
        page=3,
        text="asset OCR is not a caption or alt description",
    )
    alt_text = "  Diagram of the deterministic fallback path.  "
    alt_figure = Element(
        id="altfig",
        document_id="d",
        canonical_snapshot_id="snap",
        element_type=ElementType.FIGURE,
        order=8,
        section_id=section.id,
        page=4,
        metadata={"alt": alt_text},
    )
    punctuation = Element(
        id="punct",
        document_id="d",
        canonical_snapshot_id="snap",
        element_type=ElementType.PARAGRAPH,
        order=3,
        section_id=section.id,
        text=("punctuation boundary, " * 12).strip(),
    )
    whitespace_text = (
        ("word " * 18)
        + "[1-3] "
        + ("tail " * 18)
    ).strip()
    whitespace = Element(
        id="space",
        document_id="d",
        canonical_snapshot_id="snap",
        element_type=ElementType.PARAGRAPH,
        order=4,
        section_id=section.id,
        text=whitespace_text,
    )
    token_safe = Element(
        id="solid",
        document_id="d",
        canonical_snapshot_id="snap",
        element_type=ElementType.PARAGRAPH,
        order=5,
        section_id=section.id,
        text="X" * 190,
    )
    token_safe_citation_text = "X" * 72 + "[1-3]" + "Y" * 100
    token_safe_citation = Element(
        id="solid-citation",
        document_id="d",
        canonical_snapshot_id="snap",
        element_type=ElementType.PARAGRAPH,
        order=9,
        section_id=section.id,
        text=token_safe_citation_text,
    )
    token_safe_math_text = "A" * 70 + r"\(x+y\)" + "B" * 100
    token_safe_math = Element(
        id="solid-math",
        document_id="d",
        canonical_snapshot_id="snap",
        element_type=ElementType.PARAGRAPH,
        order=10,
        section_id=section.id,
        text=token_safe_math_text,
    )
    oversized_domain_text = r"\[" + "Z" * 120 + r"\]"
    oversized_domain = Element(
        id="oversized-domain",
        document_id="d",
        canonical_snapshot_id="snap",
        element_type=ElementType.PARAGRAPH,
        order=11,
        section_id=section.id,
        text=oversized_domain_text,
    )
    formula = Element(
        id="formula",
        document_id="d",
        canonical_snapshot_id="snap",
        element_type=ElementType.FORMULA,
        order=6,
        section_id=section.id,
        latex="z" * 190,
    )
    table_source = (
        "| A | B |\n| - | - |\n"
        + "| long | "
        + ("q" * 170)
        + " |\n"
    )
    table = Element(
        id="table",
        document_id="d",
        canonical_snapshot_id="snap",
        element_type=ElementType.TABLE,
        order=7,
        section_id=section.id,
        markdown=table_source,
    )
    elements = [
        figure,
        caption,
        empty_figure,
        alt_figure,
        punctuation,
        whitespace,
        token_safe,
        token_safe_citation,
        token_safe_math,
        oversized_domain,
        formula,
        table,
    ]
    document = Document(
        id="d",
        source_file_id="src",
        parse_run_id="parse",
        canonical_snapshot_id="snap",
        language="en",
        title="Figure and hard max contract",
    )
    tokenizer = boundaries___CharacterTokenizer()
    chunks, _mentions = build_chunks(
        document=document,
        snapshot_id="snap",
        sections=[section],
        elements=elements,
        references=[],
        target_tokens=55,
        hard_max_tokens=hard_max,
        overlap_tokens=18,
        tokenizer=tokenizer,
    )
    failures: list[str] = []
    if any((chunk.token_count or 0) > hard_max for chunk in chunks):
        failures.append("absolute_hard_max")
    caption_chunks = [chunk for chunk in chunks if "cap" in chunk.element_ids]
    if not caption_chunks or any("fig" not in chunk.element_ids for chunk in caption_chunks):
        failures.append("caption_provenance_without_figure")
    combined = "\n".join(chunk.text for chunk in chunks)
    if "[FIGURE id=emptyfig page=3]\nDescription: \n[/FIGURE]" not in combined:
        failures.append("empty_figure_placeholder")

    unique_spans = {
        span.id: span
        for chunk in chunks
        for span in chunk.spans
    }
    figure_spans = sorted(
        (
            span
            for span in unique_spans.values()
            if span.element_id == figure.id
        ),
        key=lambda span: span.id,
    )
    caption_spans = sorted(
        (
            span
            for span in unique_spans.values()
            if span.element_id == caption.id
        ),
        key=lambda span: span.character_start_in_element,
    )
    if "".join(span.text for span in caption_spans) != caption_text:
        failures.append("caption_source_provenance")
    if not figure_spans or any(
        not (
            span.provenance_kind == "projection"
            and span.source_field is None
            and span.character_start_in_element == 0
            and span.character_end_in_element == 0
            and span.token_start == 0
            and span.token_end == 0
            and span.text.startswith("[FIGURE id=fig page=2")
            and span.text.endswith("[/FIGURE]")
        )
        for span in figure_spans
    ):
        failures.append("figure_placeholder_integrity")
    if any(
        span.provenance_kind == "source"
        and span.element_id == figure.id
        and span.source_field not in {"metadata.alt", "metadata.alt_text", "metadata.description"}
        for span in unique_spans.values()
    ):
        failures.append("figure_false_source_coordinates")
    alt_projection_spans = [
        span
        for span in unique_spans.values()
        if span.element_id == alt_figure.id
        and span.provenance_kind == "projection"
    ]
    alt_source_spans = sorted(
        (
            span
            for span in unique_spans.values()
            if span.element_id == alt_figure.id
            and span.provenance_kind == "source"
        ),
        key=lambda span: span.character_start_in_element,
    )
    if not alt_projection_spans or any(
        span.source_field is not None
        or span.character_start_in_element != 0
        or span.character_end_in_element != 0
        for span in alt_projection_spans
    ):
        failures.append("figure_alt_projection_provenance")
    if (
        not alt_source_spans
        or any(span.source_field != "metadata.alt" for span in alt_source_spans)
        or "".join(span.text for span in alt_source_spans) != alt_text.strip()
        or alt_source_spans[0].token_start != 2
        or alt_source_spans[-1].token_end != len(alt_text) - 2
        or any(
            alt_text[
                span.character_start_in_element : span.character_end_in_element
            ]
            != span.text
            for span in alt_source_spans
        )
    ):
        failures.append("figure_alt_source_provenance")

    by_element = {
        element.id: sorted(
            (
                span
                for span in unique_spans.values()
                if span.element_id == element.id
            ),
            key=lambda span: span.character_start_in_element,
        )
        for element in (
            punctuation,
            whitespace,
            token_safe,
            token_safe_citation,
            token_safe_math,
            oversized_domain,
            formula,
            table,
        )
    }
    for element in (
        punctuation,
        whitespace,
        token_safe,
        token_safe_citation,
        token_safe_math,
        oversized_domain,
        formula,
        table,
    ):
        source = element_text(element)
        reconstructed = "".join(
            source[
                span.character_start_in_element : span.character_end_in_element
            ]
            for span in by_element[element.id]
        )
        if reconstructed != source:
            failures.append(f"source_coverage:{element.id}")
    citation_start = whitespace_text.index("[1-3]")
    citation_end = citation_start + len("[1-3]")
    if any(
        citation_start < boundary < citation_end
        for span in by_element[whitespace.id]
        for boundary in (
            span.character_start_in_element,
            span.character_end_in_element,
        )
    ):
        failures.append("citation_placeholder_split")
    for element, failure_name in (
        (token_safe_citation, "token_safe_citation_domain_split"),
        (token_safe_math, "token_safe_math_domain_split"),
    ):
        domains = scan_inline_domains(element_text(element))
        if not domains:
            failures.append(f"protected_domain_not_detected:{element.id}")
            continue
        if any(
            domain.start < boundary < domain.end
            for domain in domains
            for span in by_element[element.id]
            for boundary in (
                span.character_start_in_element,
                span.character_end_in_element,
            )
        ):
            failures.append(failure_name)

    fallback_reasons: Counter[str] = Counter()
    for chunk in chunks:
        fallback_reasons.update(chunk.metadata.get("fallback_split_reasons") or {})
    for expected in (
        "EMERGENCY_PUNCTUATION",
        "EMERGENCY_WHITESPACE",
        "EMERGENCY_TOKEN_SAFE",
        "EMERGENCY_PROTECTED_DOMAIN",
    ):
        if fallback_reasons.get(expected, 0) == 0:
            failures.append(f"fallback_reason_missing:{expected}")
    return {
        "figure_hard_max_contract_errors": len(failures),
        "figure_input_count": 3,
        "figure_placeholder_parts": (
            len(figure_spans) + len(alt_projection_spans) + 1
        ),
        "synthetic_max_chunk_tokens": max(
            (chunk.token_count or 0 for chunk in chunks), default=0
        ),
        "synthetic_fallback_reasons": dict(fallback_reasons),
        "figure_hard_max_failures": failures,
    }


def boundaries__authoritative_tokenizer_contract() -> dict[str, Any]:
    """Prove production chunk sizing is independent of Cognee's resolver."""

    from paperos_core.adapters.cognee import compat
    from paperos_core.adapters.cognee.pipeline_tasks import academic_chunk_task

    failures: list[str] = []
    tokenizer = review___resolve_tokenizer()
    samples = ("plain ASCII", "论文证据", "math: α+β", "")
    for sample in samples:
        expected = len(sample.encode("utf-8"))
        first = tokenizer.count_tokens(sample)
        second = review___resolve_tokenizer().count_tokens(sample)
        if first != expected or second != expected:
            failures.append(f"nondeterministic_utf8_bound:{sample!r}:{first}:{second}")

    section = Section(
        id="tokenizer_section",
        document_id="tokenizer_document",
        canonical_snapshot_id="tokenizer_snapshot",
        title="Tokenizer",
        level=1,
        order=0,
        path="Tokenizer",
    )
    element = Element(
        id="tokenizer_element",
        document_id="tokenizer_document",
        canonical_snapshot_id="tokenizer_snapshot",
        element_type=ElementType.PARAGRAPH,
        order=0,
        section_id=section.id,
        text="ASCII evidence. 中文证据。" * 20,
    )
    document = Document(
        id="tokenizer_document",
        source_file_id="tokenizer_source",
        parse_run_id="tokenizer_parse",
        canonical_snapshot_id="tokenizer_snapshot",
        language="en",
        title="Authoritative tokenizer contract",
    )

    def forbidden_cognee_resolver() -> Any:
        raise AssertionError("Cognee tokenizer resolver must not size PaperOS chunks")

    original_resolver = compat.resolve_cognee_tokenizer
    try:
        compat.resolve_cognee_tokenizer = forbidden_cognee_resolver
        with tempfile.TemporaryDirectory(prefix="paperos-tokenizer-contract-") as root:
            paths = build_data_paths(Path(root))
            bundle = CanonicalBundle(
                snapshot=CanonicalSnapshot(
                    id="tokenizer_snapshot",
                    source_file_id="tokenizer_source",
                    parse_run_id="tokenizer_parse",
                    document_id=document.id,
                    manifest_path=paths.canonical / "manifest.json",
                ),
                document=document,
                sections=[section],
                elements=[element],
                references=[],
                warnings=[],
            )
            result = asyncio.run(
                academic_chunk_task(
                    [bundle],
                    repository=CanonicalRepository(paths),
                    chunk_target_tokens=55,
                    chunk_hard_max_tokens=75,
                    chunk_overlap_tokens=0,
                )
            )
    except Exception as exc:  # noqa: BLE001 - contract reports exact failures
        failures.append(f"production_pipeline:{type(exc).__name__}:{exc}")
        result = []
    finally:
        compat.resolve_cognee_tokenizer = original_resolver

    chunks = result[0].projection.chunks if result else []
    if not chunks:
        failures.append("production_pipeline:no_chunks")
    for chunk in chunks:
        expected = len(chunk.text.encode("utf-8"))
        if chunk.token_count != expected:
            failures.append(
                f"production_token_count:{chunk.id}:{chunk.token_count}:{expected}"
            )
        if expected > 75:
            failures.append(f"production_hard_max:{chunk.id}:{expected}")
    return {
        "authoritative_tokenizer_contract_errors": len(failures),
        "authoritative_tokenizer_failures": failures,
        "authoritative_tokenizer_chunk_count": len(chunks),
    }


def boundaries__main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("data/validation/chunk/output")
    )
    args = parser.parse_args()
    tokenizer = review___resolve_tokenizer()
    paper_results: list[dict[str, Any]] = []
    for src_dir in sorted((args.run_dir / "canonical").glob("src_*")):
        snapshot_dirs = sorted(src_dir.glob("snapshot_*"))
        if len(snapshot_dirs) != 1:
            raise RuntimeError(
                f"""Expected one canonical snapshot in {src_dir }, got {len (snapshot_dirs )}"""
            )
        bundle = review___load_bundle_from_snapshot_dir(snapshot_dirs[0])
        chunks_json = regions___chunks_for_snapshot(args.run_dir, bundle.snapshot.id)
        hard_max = 1200
        paper_results.append(
            {
                "source_id": src_dir.name,
                "title": bundle.document.title,
                **boundaries___paper_contracts(
                    bundle=bundle,
                    chunks_json=chunks_json,
                    count=tokenizer.count_tokens,
                    hard_max_tokens=hard_max,
                ),
            }
        )
    synthetic = boundaries__synthetic_multi_part_table_contract()
    figure_hard_max = boundaries__synthetic_figure_hard_max_contract()
    authoritative_tokenizer = boundaries__authoritative_tokenizer_contract()
    report = {
        "git_commit": boundaries___git_commit(),
        "paper_count": len(paper_results),
        "abstract_region_errors": sum(
            item["abstract_region_errors"] for item in paper_results
        ),
        "formula_cohesion_cases": sum(
            item["formula_cohesion_cases"] for item in paper_results
        ),
        "avoidable_formula_cohesion_breaks": sum(
            item["avoidable_formula_cohesion_breaks"] for item in paper_results
        ),
        "table_parts": sum(item["table_parts"] for item in paper_results),
        "real_emergency_splits": sum(
            item["real_emergency_splits"] for item in paper_results
        ),
        "table_part_emergency_misclassification": sum(
            item["table_part_emergency_misclassification"] for item in paper_results
        ),
        **synthetic,
        **figure_hard_max,
        **authoritative_tokenizer,
        "papers": paper_results,
    }
    report["pass"] = all(
        report[key] == 0
        for key in (
            "abstract_region_errors",
            "avoidable_formula_cohesion_breaks",
            "table_part_emergency_misclassification",
            "multi_part_table_provenance_errors",
            "figure_hard_max_contract_errors",
            "authoritative_tokenizer_contract_errors",
        )
    )
    output = args.run_dir / "chunk-boundary-contracts.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                **{
                    key: report[key]
                    for key in report
                    if key not in {"papers", "failures"}
                },
                "report": str(output),
            },
            indent=2,
        )
    )
    return 0 if report["pass"] else 1


"Validate production citations against MinerU-anchored Gold v3."
import sys
from pathlib import Path

gold__ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(gold__ROOT))


def gold___key(record: dict[str, Any]) -> tuple[int, str, int | None, int, int]:
    return (
        record["source_item"],
        record["source_domain"],
        record.get("source_subindex"),
        record["start"],
        record["end"],
    )


def gold___chunks_for_snapshot(run_dir: Path, snapshot_id: str) -> dict[str, Any]:
    matches = []
    for path in run_dir.glob("*.chunks.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("snapshot_id") == snapshot_id:
            matches.append(payload)
    if len(matches) != 1:
        raise RuntimeError(
            f"""Expected one chunks file for {snapshot_id }, got {len (matches )}"""
        )
    return matches[0]


def gold___element_domain(element: dict[str, Any]) -> tuple[str, int | None]:
    metadata = element.get("metadata") or {}
    if "caption_index" in metadata:
        return ("caption", metadata["caption_index"])
    if "footnote_index" in metadata:
        return ("footnote", metadata["footnote_index"])
    if element.get("element_type") == "table" and element.get("html"):
        return ("table_body", None)
    return ("text", None)


def gold___actual_source_locator(
    mention: dict[str, Any],
    element: dict[str, Any],
    fields_by_item: dict[int, list[dict[str, Any]]],
    used: set[tuple[int, str, int | None, int, int]],
) -> dict[str, Any] | None:
    source_span = element.get("source_span") or {}
    item_index = source_span.get("item_index")
    if item_index is None:
        return None
    domain, subindex = gold___element_domain(element)
    element_text = (
        element.get("html") if domain == "table_body" else element.get("text")
    )
    element_text = element_text or element.get("markdown") or ""
    left = element_text[
        max(0, mention["character_start"] - 80) : mention["character_start"]
    ]
    right = element_text[mention["character_end"] : mention["character_end"] + 80]
    candidates: list[tuple[float, dict[str, Any], int, int]] = []
    for field in fields_by_item.get(item_index, []):
        if field["source_domain"] != domain or field["source_subindex"] != subindex:
            continue
        for start, end in gold_builder___flex_occurrences(
            field["value"], mention["surface_text"]
        ):
            candidate_key = (item_index, domain, subindex, start, end)
            if candidate_key in used:
                continue
            score = SequenceMatcher(
                None,
                gold_builder___norm(left)[-60:],
                gold_builder___norm(field["value"][max(0, start - 100) : start])[-60:],
            ).ratio()
            score += SequenceMatcher(
                None,
                gold_builder___norm(right)[:60],
                gold_builder___norm(field["value"][end : end + 100])[:60],
            ).ratio()
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


def gold__validate_paper(paper: dict[str, Any], *, run_dir: Path) -> dict[str, Any]:
    canonical_dir = gold_builder___canonical_dir(run_dir, paper["source_id"])
    snapshot = json.loads((canonical_dir / "snapshot.json").read_text(encoding="utf-8"))
    elements = gold_builder___json_lines(canonical_dir / "elements.jsonl")
    references = gold_builder___json_lines(canonical_dir / "references.jsonl")
    element_by_id = {element["id"]: element for element in elements}
    reference_order = {reference["id"]: reference["order"] for reference in references}
    fingerprint_by_order = {
        reference["order"]: reference["fingerprint"]
        for reference in paper["references"]
    }
    fields = gold_builder___source_fields(
        gold_builder___mineru_content(run_dir, paper["source_id"], canonical_dir)
    )
    fields_by_item: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for field in fields:
        fields_by_item[field["source_item"]].append(field)
    chunks = gold___chunks_for_snapshot(run_dir, snapshot["id"])
    mentions_by_span: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mention in chunks.get("citation_mentions", []):
        mentions_by_span[mention["citation_span_id"]].append(mention)
    used: set[tuple[int, str, int | None, int, int]] = set()
    actual: dict[tuple[int, str, int | None, int, int], dict[str, Any]] = {}
    source_mapping_failures: list[dict[str, Any]] = []
    for span_mentions in sorted(
        mentions_by_span.values(),
        key=lambda rows: (
            element_by_id[rows[0]["element_id"]]["order"],
            rows[0]["character_start"],
        ),
    ):
        first = min(span_mentions, key=lambda row: row["group_index"])
        locator = gold___actual_source_locator(
            first, element_by_id[first["element_id"]], fields_by_item, used
        )
        if locator is None:
            source_mapping_failures.append(
                {"surface": first["surface_text"], "element_id": first["element_id"]}
            )
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
        actual[gold___key(locator)] = {
            **locator,
            "targets": targets,
            "citation_namespace_id": first.get("citation_namespace_id")
            or first.get("bibliography_scope_id"),
        }
    expected = {gold___key(item): item for item in paper["occurrences"]}
    missing_keys = sorted(set(expected) - set(actual))
    extra_keys = sorted(set(actual) - set(expected))
    failures: list[dict[str, Any]] = []
    for item_key in missing_keys:
        failures.append(
            {"failure_type": "MISSING_OCCURRENCE", "expected": expected[item_key]}
        )
    for item_key in extra_keys:
        failures.append(
            {"failure_type": "EXTRA_OCCURRENCE", "actual": actual[item_key]}
        )
    wrong_targets = 0
    unresolved_expected = 0
    unattached = 0
    wrong_namespaces = 0
    for item_key in sorted(set(expected) & set(actual)):
        expected_fingerprints = [
            target.get("acceptable_fingerprints") or [target.get("fingerprint")]
            for target in expected[item_key]["targets"]
        ]
        actual_fingerprints = [
            target["fingerprint"] for target in actual[item_key]["targets"]
        ]
        targets_match = len(expected_fingerprints) == len(actual_fingerprints) and all(
            (
                actual_fingerprint in accepted
                for actual_fingerprint, accepted in zip(
                    actual_fingerprints, expected_fingerprints, strict=True
                )
            )
        )
        if not targets_match:
            wrong_targets += 1
            failures.append(
                {
                    "failure_type": "WRONG_TARGETS",
                    "expected": expected[item_key],
                    "actual": actual[item_key],
                }
            )
        unresolved_expected += sum(
            target["resolution_status"] != "resolved"
            for target in actual[item_key]["targets"]
        )
        unattached += sum(
            not target["chunk_id"] for target in actual[item_key]["targets"]
        )
        if (
            expected[item_key]["citation_namespace_id"]
            != actual[item_key]["citation_namespace_id"]
        ):
            wrong_namespaces += 1
            failures.append(
                {
                    "failure_type": "WRONG_NAMESPACE",
                    "expected": expected[item_key],
                    "actual": actual[item_key],
                }
            )
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
        "status": (
            "PASS"
            if not failures
            and (not source_mapping_failures)
            and (not unresolved_expected)
            and (not unattached)
            else "FAIL"
        ),
    }


def gold__main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        type=Path,
        default=gold__ROOT / "data/validation/chunk/config/citation_gold.json",
    )
    parser.add_argument(
        "--run-dir", type=Path, default=gold__ROOT / "data/validation/chunk/output"
    )
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    papers = {
        key: gold__validate_paper(paper, run_dir=args.run_dir)
        for key, paper in gold["papers"].items()
    }
    report = {
        "gold_version": "citation-gold-v3",
        "papers": papers,
        "missing_occurrences": sum(
            item["missing_occurrences"] for item in papers.values()
        ),
        "extra_occurrences": sum(
            item["extra_occurrences"] for item in papers.values()
        ),
        "wrong_targets": sum(item["wrong_targets"] for item in papers.values()),
        "unresolved_expected_targets": sum(
            item["unresolved_expected_targets"] for item in papers.values()
        ),
        "unattached_targets": sum(
            item["unattached_targets"] for item in papers.values()
        ),
        "wrong_namespaces": sum(item["wrong_namespaces"] for item in papers.values()),
        "source_mapping_failures": sum(
            len(item["source_mapping_failures"]) for item in papers.values()
        ),
    }
    report["pass"] = all(item["status"] == "PASS" for item in papers.values())
    output = args.run_dir / "citation-gold-v3-validation.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "report": str(output),
                **{
                    key: report[key]
                    for key in (
                        "missing_occurrences",
                        "extra_occurrences",
                        "wrong_targets",
                        "wrong_namespaces",
                        "source_mapping_failures",
                    )
                },
            },
            indent=2,
        )
    )
    return 0 if report["pass"] else 1


"Validate used ReferenceEntry identities against Gold v3 citation targets."
import sys
from pathlib import Path

references__REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(references__REPOSITORY_ROOT))


def references___git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=references__REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def references___chunks_for_snapshot(run_dir: Path, snapshot_id: str) -> dict[str, Any]:
    matches = []
    for path in run_dir.glob("*.chunks.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("snapshot_id") == snapshot_id:
            matches.append(payload)
    if len(matches) != 1:
        raise RuntimeError(
            f"""Expected one chunks file for {snapshot_id }, got {len (matches )}"""
        )
    return matches[0]


def references__validate_paper(
    *, bundle: Any, chunks_json: dict[str, Any], paper: dict[str, Any]
) -> dict[str, Any]:
    fingerprint_by_order = {
        reference["order"]: reference["fingerprint"]
        for reference in paper["references"]
    }
    order_by_reference_id = {
        reference.id: reference.order for reference in bundle.references
    }
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
            accepted = target.get("acceptable_fingerprints") or [
                target.get("fingerprint")
            ]
            expected_groups.add(tuple(sorted(value for value in accepted if value)))
    expected_union = {fingerprint for group in expected_groups for fingerprint in group}
    unexpected = sorted(actual - expected_union)
    missed = sorted(
        group for group in expected_groups if not actual.intersection(group)
    )
    failures: list[dict[str, Any]] = [
        {"failure_type": "UNEXPECTED_USED_REFERENCE", "fingerprint": fingerprint}
        for fingerprint in unexpected
    ]
    failures.extend(
        {
            "failure_type": "MISSED_EXPECTED_USED_REFERENCE",
            "acceptable_fingerprints": list(group),
        }
        for group in missed
    )
    failures.extend(
        {
            "failure_type": "REFERENCE_ID_NOT_IN_CANONICAL_BIBLIOGRAPHY",
            "reference_entry_id": reference_id,
        }
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


def references__main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("data/validation/chunk/output")
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=Path("data/validation/corpus")
    )
    parser.add_argument(
        "--papers-config",
        type=Path,
        default=Path("data/validation/chunk/config/papers.json"),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/validation/chunk/config/citation_gold.json"),
    )
    args = parser.parse_args()
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    papers: dict[str, dict[str, Any]] = {}
    for paper_key, paper in gold["papers"].items():
        snapshots = sorted(
            (args.run_dir / "canonical" / paper["source_id"]).glob("snapshot_*")
        )
        if len(snapshots) != 1:
            raise RuntimeError(
                f"""Expected one canonical snapshot for {paper_key }, got {len (snapshots )}"""
            )
        bundle = review___load_bundle_from_snapshot_dir(snapshots[0])
        papers[paper_key] = references__validate_paper(
            bundle=bundle,
            chunks_json=references___chunks_for_snapshot(
                args.run_dir, bundle.snapshot.id
            ),
            paper=paper,
        )
    report = {
        "git_commit": references___git_commit(),
        "gold_version": gold.get("gold_version"),
        "missed_used_references": sum(
            item["missed_used_references"] for item in papers.values()
        ),
        "unexpected_used_references": sum(
            item["unexpected_used_references"] for item in papers.values()
        ),
        "papers": papers,
    }
    report["pass"] = (
        report["missed_used_references"] == 0
        and report["unexpected_used_references"] == 0
    )
    output = args.run_dir / "reference-usage.json"
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "missed": report["missed_used_references"],
                "unexpected": report["unexpected_used_references"],
                "report": str(output),
            },
            indent=2,
        )
    )
    return 0 if report["pass"] else 1


"Run the four authoritative six-paper chunk/citation acceptance gates."
import os
import sys
import zipfile
from datetime import UTC
from pathlib import Path

runner__REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
runner__CONTRACT_REPORT_NAME = "chunk-citation-acceptance.json"


def runner___git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=runner__REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runner___load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def runner___sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runner___run_py(command: str, *script_args: str) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(runner__REPOSITORY_ROOT)
    return subprocess.call(
        [sys.executable, "tests/validation/chunk.py", command, *script_args],
        cwd=runner__REPOSITORY_ROOT,
        env=env,
    )


def runner___gate(status: bool, failures: int, **metrics: Any) -> dict[str, Any]:
    return {
        "status": "PASS" if status and failures == 0 else "FAIL",
        "failures": failures,
        **metrics,
    }


def runner___projection_hashes(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(run_dir.glob("*.chunks.json")):
        payload = runner___load_json(path)
        deterministic_projection = {
            "snapshot_id": payload.get("snapshot_id"),
            "chunking_version": payload.get("chunking_version"),
            "chunks": payload.get("chunks", []),
            "citation_mentions": payload.get("citation_mentions", []),
        }
        hashes[path.name] = hashlib.sha256(
            json.dumps(
                deterministic_projection,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    return hashes


def runner___source_anchor_hashes(gold_report: dict[str, Any]) -> dict[str, str | None]:
    return {
        paper: result.get("source_anchor_digest")
        for paper, result in sorted(gold_report.get("papers", {}).items())
    }


def runner___chunk_structure_metrics(run_dir: Path, hard_max: int) -> tuple[int, int]:
    hard_max_violations = 0
    empty_chunks = 0
    for path in run_dir.glob("*.chunks.json"):
        for chunk in runner___load_json(path).get("chunks", []):
            hard_max_violations += int((chunk.get("token_count") or 0) > hard_max)
            empty_chunks += int(not (chunk.get("text") or "").strip())
    return (hard_max_violations, empty_chunks)


def runner___write_result_package(
    *, run_dir: Path, gold: Path, summary: dict[str, Any]
) -> Path:
    candidates = [
        run_dir / "acceptance-summary.json",
        run_dir / "canonical-source-survival.json",
        run_dir / "chunk-source-coverage.json",
        run_dir / "chunk-regions-scopes.json",
        run_dir / "citation-gold-v3-validation.json",
        run_dir / "reference-usage.json",
        run_dir / "chunk-corpus-review.json",
        run_dir / "failure-ledger.json",
        run_dir / "chunk-boundary-contracts.json",
        run_dir / "logs" / "contracts" / runner__CONTRACT_REPORT_NAME,
        gold,
        gold.with_name("gold-v3-audit.json"),
        gold.with_name("gold-v3-audit.md"),
        *sorted(run_dir.glob("*.chunks.json")),
        *sorted(run_dir.glob("*.chunks.md")),
        *sorted((run_dir / "logs").rglob("*")),
    ]
    files = sorted({path.resolve() for path in candidates if path.is_file()})
    manifest_path = run_dir / "result-manifest.json"
    manifest = {
        "git_commit": summary["git_commit"],
        "gold_version": summary["gold_version"],
        "gold_hash": summary["gold_hash"],
        "overall_status": summary["overall_status"],
        "files": [
            {
                "path": (
                    str(path.relative_to(runner__REPOSITORY_ROOT))
                    if path.is_relative_to(runner__REPOSITORY_ROOT)
                    else path.name
                ),
                "sha256": runner___sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    files.append(manifest_path.resolve())
    package = run_dir / "result.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            if path.is_relative_to(run_dir.resolve()):
                arcname = path.relative_to(run_dir.resolve())
            elif path.is_relative_to(runner__REPOSITORY_ROOT):
                arcname = Path("repository") / path.relative_to(runner__REPOSITORY_ROOT)
            else:
                arcname = Path(path.name)
            archive.write(path, arcname.as_posix())
    return package


def runner__main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("data/validation/chunk/output")
    )
    parser.add_argument(
        "--corpus-dir", type=Path, default=Path("data/validation/corpus")
    )
    parser.add_argument(
        "--papers-config",
        type=Path,
        default=Path("data/validation/chunk/config/papers.json"),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/validation/chunk/config/citation_gold.json"),
    )
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--rebuild-canonical", action="store_true")
    parser.add_argument("--skip-determinism", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.corpus_dir = args.corpus_dir.resolve()
    args.gold = args.gold.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    chunk_args = [
        "review",
        "--run-dir",
        str(args.run_dir),
        "--corpus-dir",
        str(args.corpus_dir),
        "--papers-config",
        str(args.papers_config),
        "--overlap-tokens",
        "0",
        "--rechunk-canonical",
    ]
    if args.rebuild_canonical:
        chunk_args.append("--rebuild-canonical")
    structure_code = runner___run_py(*chunk_args)
    paper_dir = (
        args.corpus_dir
        if args.corpus_dir.name == "papers"
        else args.corpus_dir / "papers"
    )
    survival_code = runner___run_py(
        "source",
        "--run-dir",
        str(args.run_dir),
        "--corpus-dir",
        str(paper_dir),
        "--gold",
        str(args.gold),
    )
    coverage_code = runner___run_py(
        "coverage", "--run-dir", str(args.run_dir), "--corpus-dir", str(paper_dir)
    )
    regions_code = runner___run_py(
        "regions", "--run-dir", str(args.run_dir), "--corpus-dir", str(paper_dir)
    )
    boundary_code = runner___run_py("boundaries", "--run-dir", str(args.run_dir))
    gold_code = runner___run_py(
        "gold", "--gold", str(args.gold), "--run-dir", str(args.run_dir)
    )
    reference_code = runner___run_py(
        "references",
        "--gold",
        str(args.gold),
        "--run-dir",
        str(args.run_dir),
        "--corpus-dir",
        str(paper_dir),
    )
    chunk_report = runner___load_json(args.run_dir / "chunk-corpus-review.json")
    survival_report = runner___load_json(
        args.run_dir / "canonical-source-survival.json"
    )
    coverage_report = runner___load_json(args.run_dir / "chunk-source-coverage.json")
    regions_report = runner___load_json(args.run_dir / "chunk-regions-scopes.json")
    boundary_report = runner___load_json(args.run_dir / "chunk-boundary-contracts.json")
    gold_report = runner___load_json(args.run_dir / "citation-gold-v3-validation.json")
    reference_report = runner___load_json(args.run_dir / "reference-usage.json")
    hard_max = int(chunk_report.get("chunk_hard_max_tokens", 0))
    hard_max_violations, empty_chunks = runner___chunk_structure_metrics(
        args.run_dir, hard_max
    )
    structure_failures = int(chunk_report.get("pdf_count", 0)) - int(
        chunk_report.get("pass_count", 0)
    )
    wrong_regions = int(regions_report.get("wrong_regions", 0))
    wrong_namespaces = int(regions_report.get("wrong_namespaces", 0))
    source_failures = int(survival_report.get("failure_count", 0))
    holes = int(coverage_report.get("chunk_source_holes", 0))
    overlaps = int(coverage_report.get("chunk_source_overlaps", 0))
    boundary_metrics = {
        "abstract_region_errors": int(boundary_report.get("abstract_region_errors", 0)),
        "avoidable_formula_cohesion_breaks": int(
            boundary_report.get("avoidable_formula_cohesion_breaks", 0)
        ),
        "table_part_emergency_misclassification": int(
            boundary_report.get("table_part_emergency_misclassification", 0)
        ),
        "multi_part_table_provenance_errors": int(
            boundary_report.get("multi_part_table_provenance_errors", 0)
        ),
        "figure_hard_max_contract_errors": int(
            boundary_report.get("figure_hard_max_contract_errors", 0)
        ),
    }
    citation_metrics = {
        "missing_occurrences": int(gold_report.get("missing_occurrences", 0)),
        "extra_occurrences": int(gold_report.get("extra_occurrences", 0)),
        "wrong_targets": int(gold_report.get("wrong_targets", 0)),
        "unresolved_expected_targets": int(
            gold_report.get("unresolved_expected_targets", 0)
        ),
        "unattached_targets": int(gold_report.get("unattached_targets", 0)),
        "wrong_namespaces": int(gold_report.get("wrong_namespaces", 0)),
        "source_mapping_failures": int(gold_report.get("source_mapping_failures", 0)),
        "unexpected_used_references": int(
            reference_report.get("unexpected_used_references", 0)
        ),
        "missed_expected_used_references": int(
            reference_report.get("missed_used_references", 0)
        ),
    }
    source_failure_count = source_failures + holes + overlaps
    structure_failure_count = (
        structure_failures
        + wrong_regions
        + wrong_namespaces
        + hard_max_violations
        + empty_chunks
        + sum(boundary_metrics.values())
    )
    citation_failure_count = sum(citation_metrics.values())
    gates: dict[str, dict[str, Any]] = {
        "source": runner___gate(
            survival_code == 0 and coverage_code == 0,
            source_failure_count,
            canonical_source_loss=survival_report.get("canonical_source_loss", 0),
            gold_canonical_source_loss=survival_report.get(
                "gold_canonical_source_loss", 0
            ),
            chunk_holes=holes,
            chunk_overlaps=overlaps,
        ),
        "structure": runner___gate(
            structure_code == 0 and regions_code == 0 and (boundary_code == 0),
            structure_failure_count,
            pdf_failures=structure_failures,
            wrong_regions=wrong_regions,
            wrong_namespaces=wrong_namespaces,
            hard_max_violations=hard_max_violations,
            empty_chunks=empty_chunks,
            **boundary_metrics,
        ),
        "citation_gold_v3": runner___gate(
            gold_code == 0 and reference_code == 0,
            citation_failure_count,
            **citation_metrics,
        ),
        "determinism": {"status": "NOT_CHECKED", "failures": 0},
    }
    determinism_failures = 0
    first_projection_hashes = runner___projection_hashes(args.run_dir)
    first_anchor_hashes = runner___source_anchor_hashes(gold_report)
    if not args.skip_determinism and all(
        (
            gate["status"] == "PASS"
            for name, gate in gates.items()
            if name != "determinism"
        )
    ):
        second_structure = runner___run_py(*chunk_args)
        second_gold = runner___run_py(
            "gold", "--gold", str(args.gold), "--run-dir", str(args.run_dir)
        )
        second_gold_report = runner___load_json(
            args.run_dir / "citation-gold-v3-validation.json"
        )
        second_projection_hashes = runner___projection_hashes(args.run_dir)
        second_anchor_hashes = runner___source_anchor_hashes(second_gold_report)
        deterministic = (
            second_structure == 0
            and second_gold == 0
            and (first_projection_hashes == second_projection_hashes)
            and (first_anchor_hashes == second_anchor_hashes)
        )
        determinism_failures = 0 if deterministic else 1
        gates["determinism"] = runner___gate(
            deterministic,
            determinism_failures,
            projection_hashes=second_projection_hashes,
            source_anchor_hashes=second_anchor_hashes,
        )
    overall_pass = all(gate["status"] == "PASS" for gate in gates.values())
    summary = {
        "overall_status": "PASS" if overall_pass else "FAIL",
        "git_commit": runner___git_commit(),
        "gold_version": "citation-gold-v3",
        "gold_hash": runner___sha256(args.gold),
        "pdf_count": chunk_report.get("pdf_count", 0),
        "overlap_tokens": chunk_report.get("overlap_tokens", 0),
        "canonical_source_loss": survival_report.get("canonical_source_loss", 0),
        "gold_canonical_source_loss": survival_report.get(
            "gold_canonical_source_loss", 0
        ),
        "chunk_source_holes": holes,
        "chunk_source_overlaps": overlaps,
        "wrong_regions": wrong_regions,
        "wrong_namespaces": wrong_namespaces,
        "hard_max_violations": hard_max_violations,
        "empty_chunks": empty_chunks,
        **boundary_metrics,
        **citation_metrics,
        "determinism_failures": determinism_failures,
        "gates": gates,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    acceptance_path = args.run_dir / "acceptance-summary.json"
    acceptance_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    ledger_path = args.run_dir / "failure-ledger.json"
    ledger = runner___load_json(ledger_path)
    if not isinstance(ledger, list):
        ledger = []
    ledger.append(
        {
            "iteration": (
                args.iteration if args.iteration is not None else len(ledger) + 1
            ),
            "timestamp": summary["timestamp"],
            "git_commit": summary["git_commit"],
            "overall_status": summary["overall_status"],
            "gates": {name: gate["status"] for name, gate in gates.items()},
        }
    )
    ledger_path.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    contracts_dir = args.run_dir / "logs" / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    contract_path = contracts_dir / runner__CONTRACT_REPORT_NAME
    contract = {
        "overall_status": summary["overall_status"],
        "git_commit": summary["git_commit"],
        "gold_version": summary["gold_version"],
        "gold_hash": summary["gold_hash"],
        "gates": gates,
        "metrics": summary,
        "reports": {
            "source": ["canonical-source-survival.json", "chunk-source-coverage.json"],
            "structure": [
                "chunk-corpus-review.json",
                "chunk-regions-scopes.json",
                "chunk-boundary-contracts.json",
            ],
            "citation_gold_v3": [
                "citation-gold-v3-validation.json",
                "reference-usage.json",
            ],
        },
    }
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    package = runner___write_result_package(
        run_dir=args.run_dir, gold=args.gold, summary=summary
    )
    print(json.dumps(summary, indent=2))
    print(f"""acceptance-summary: {acceptance_path }""")
    print(f"""contract-report: {contract_path }""")
    print(f"""result-package: {package }""")
    return 0 if overall_pass else 1


def _dispatch_main() -> int:
    commands = {
        "run": runner__main,
        "review": review__main,
        "source": source__main,
        "coverage": coverage__main,
        "regions": regions__main,
        "boundaries": boundaries__main,
        "gold": gold__main,
        "references": references__main,
    }
    command = "run"
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        command = sys.argv.pop(1)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", type=Path)
    common.add_argument("--config", type=Path)
    common.add_argument("--output", type=Path)
    common_args, remaining = common.parse_known_args(sys.argv[1:])
    sys.argv[1:] = remaining
    if common_args.output is not None:
        sys.argv.extend(["--run-dir", str(common_args.output)])
    if common_args.corpus is not None and command in {"run", "review"}:
        sys.argv.extend(["--corpus-dir", str(common_args.corpus)])
    if common_args.config is not None and command in {"run", "review"}:
        sys.argv.extend(["--papers-config", str(common_args.config / "papers.json")])
    if common_args.config is not None and command == "run":
        sys.argv.extend(["--gold", str(common_args.config / "citation_gold.json")])
    result = commands[command]()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(_dispatch_main())
