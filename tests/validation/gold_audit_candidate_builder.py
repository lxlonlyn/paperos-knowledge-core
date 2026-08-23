#!/usr/bin/env python3
"""Enumerate citation-like bracket occurrences from MinerU source artifacts.

This script MUST NOT import production citation resolution helpers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

BRACKET_RE = re.compile(r"\[[^\[\]\n]{1,240}\]")
SECTION_REF_RE = re.compile(
    r"\b(?:sec(?:tion)?|cor(?:ollary)?|thm|theorem|fig|figure|eq(?:uation)?)\b",
    re.IGNORECASE,
)
REFERENCE_LINE_RE = re.compile(r"^\s*\[\d+\]\s+\S")


def norm_context(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "").strip()
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    return value


def context_hash(left: str, surface: str, right: str) -> str:
    payload = f"{norm_context(left)}|{norm_context(surface)}|{norm_context(right)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def source_domain_for_item(item: dict[str, Any], field: str) -> str:
    if field == "text":
        if item.get("text_level") is not None:
            return "heading"
        return "text"
    if field.startswith("image_caption"):
        return "image_caption"
    if field.startswith("table_caption"):
        return "table_caption"
    if field.startswith("table_body"):
        return "table_body"
    if field.startswith("image_footnote"):
        return "image_footnote"
    if field.startswith("table_footnote"):
        return "table_footnote"
    return field


def iter_source_texts(item: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    text = item.get("text")
    if isinstance(text, str) and text.strip():
        rows.append(("text", text))
    for key in (
        "image_caption",
        "chart_caption",
        "table_caption",
        "image_footnote",
        "chart_footnote",
        "table_footnote",
    ):
        value = item.get(key)
        if isinstance(value, list):
            for index, entry in enumerate(value):
                if isinstance(entry, str) and entry.strip():
                    rows.append((f"{key}[{index}]", entry))
        elif isinstance(value, str) and value.strip():
            rows.append((key, value))
    table_body = item.get("table_body")
    if isinstance(table_body, str) and table_body.strip():
        rows.append(("table_body", table_body))
    return rows


def is_negative_bracket(inner: str, *, source_domain: str) -> bool:
    if source_domain == "text" and REFERENCE_LINE_RE.match(f"[{inner}]"):
        return True
    if SECTION_REF_RE.search(inner):
        return True
    compact = re.sub(r"\s+", "", inner)
    if compact in {"0,1", "0,t", "-1,1", "1,1"}:
        return True
    return False


def scan_content_list(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item_index, item in enumerate(content):
        page = item.get("page_idx")
        page_num = page + 1 if isinstance(page, int) else None
        for field, text in iter_source_texts(item):
            domain = source_domain_for_item(item, field.split("[", 1)[0])
            for match in BRACKET_RE.finditer(text):
                surface = match.group(0)
                inner = surface[1:-1]
                if is_negative_bracket(inner, source_domain=domain):
                    continue
                left = text[max(0, match.start() - 80) : match.start()]
                right = text[match.end() : match.end() + 80]
                candidates.append(
                    {
                        "item_index": item_index,
                        "page": page_num,
                        "source_domain": domain,
                        "surface_text": surface,
                        "left_context": norm_context(left)[-60:],
                        "right_context": norm_context(right)[:60],
                        "context_hash": context_hash(left, surface, right),
                        "provider_type": item.get("type"),
                    }
                )
    return candidates


def load_content_list(parsed_dir: Path) -> list[dict[str, Any]]:
    artifacts = parsed_dir / "artifacts"
    for path in sorted(artifacts.glob("*content_list*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            isinstance(payload, list)
            and payload
            and all(isinstance(row, dict) for row in payload)
            and any("page_idx" in row for row in payload)
        ):
            return payload
    raise RuntimeError(f"No supported content_list in {parsed_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    content = load_content_list(args.parsed_dir)
    candidates = scan_content_list(content)
    args.output.write_text(
        json.dumps({"candidate_count": len(candidates), "candidates": candidates}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {len(candidates)} candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
