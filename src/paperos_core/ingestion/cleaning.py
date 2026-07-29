"""Deterministic repeated-margin and duplicate-content cleanup."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from paperos_core.ingestion.normalization import normalized_match_text

CLEANING_VERSION = "1"


@dataclass(frozen=True, slots=True)
class MarginText:
    item_index: int
    kind: str
    text: str
    page: int | None


def repeated_margin_indexes(items: Iterable[MarginText]) -> set[int]:
    """Identify headers/footers repeated on at least two distinct pages."""
    rows = list(items)
    pages_by_key: dict[tuple[str, str], set[int]] = {}
    for item in rows:
        key = (item.kind, normalized_match_text(item.text))
        if not key[1] or item.page is None:
            continue
        pages_by_key.setdefault(key, set()).add(item.page)
    repeated = {key for key, pages in pages_by_key.items() if len(pages) >= 2}
    return {
        item.item_index
        for item in rows
        if (item.kind, normalized_match_text(item.text)) in repeated
    }


def adjacent_duplicate_indexes(values: Iterable[tuple[int, str]]) -> set[int]:
    """Remove only adjacent exact normalized duplicates to preserve legitimate reuse."""
    duplicates: set[int] = set()
    previous = ""
    for item_index, value in values:
        normalized = normalized_match_text(value)
        if normalized and normalized == previous:
            duplicates.add(item_index)
        elif normalized:
            previous = normalized
    return duplicates


def duplicate_counts(values: Iterable[str]) -> Counter[str]:
    return Counter(normalized_match_text(value) for value in values if value.strip())
