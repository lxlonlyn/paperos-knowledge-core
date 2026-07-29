"""Deterministic document/section diversification."""

from __future__ import annotations

import re
from collections import Counter

from paperos_core.retrieval.candidates import Candidate


def diversify(
    candidates: list[Candidate],
    *,
    limit: int,
    max_per_document: int,
    max_per_section: int,
    seed_each_document: bool = False,
    aspect_queries: list[str] | None = None,
) -> list[Candidate]:
    document_counts: Counter[str] = Counter()
    section_counts: Counter[tuple[str, str | None]] = Counter()
    selected: list[Candidate] = []
    selected_ids: set[str] = set()

    def can_select(candidate: Candidate) -> bool:
        section_key = (candidate.document_id, candidate.section_id)
        return (
            candidate.id not in selected_ids
            and document_counts[candidate.document_id] < max_per_document
            and section_counts[section_key] < max_per_section
        )

    def add(candidate: Candidate) -> None:
        section_key = (candidate.document_id, candidate.section_id)
        selected.append(candidate)
        selected_ids.add(candidate.id)
        document_counts[candidate.document_id] += 1
        section_counts[section_key] += 1

    # Seed one strongest item from each document before aspect coverage.
    first_per_document: dict[str, Candidate] = {}
    for candidate in candidates:
        first_per_document.setdefault(candidate.document_id, candidate)
    if seed_each_document:
        for candidate in first_per_document.values():
            if can_select(candidate):
                add(candidate)
            if len(selected) == limit:
                return selected

    # Greedily cover query-aspect terms that existing evidence has not covered.
    aspect_terms = {
        token
        for query in aspect_queries or []
        for token in re.findall(r"[a-z0-9]+", query.casefold())
        if len(token) >= 4
    }
    covered = {
        term
        for term in aspect_terms
        if any(term in candidate.text.casefold() for candidate in selected)
    }
    while aspect_terms - covered and len(selected) < limit:
        uncovered = aspect_terms - covered
        choices = [candidate for candidate in candidates if can_select(candidate)]
        if not choices:
            break
        best = max(
            choices,
            key=lambda candidate: (
                sum(term in candidate.text.casefold() for term in uncovered),
                candidate.rerank_score or 0.0,
                -candidates.index(candidate),
            ),
        )
        new_terms = {
            term for term in uncovered if term in best.text.casefold()
        }
        if not new_terms:
            break
        add(best)
        covered.update(new_terms)

    for candidate in candidates:
        if can_select(candidate):
            add(candidate)
        if len(selected) == limit:
            break
    return selected
