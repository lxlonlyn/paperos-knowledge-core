"""Source-only citation candidate detection.

This module knows nothing about bibliographies, references, Works, chunks, or
resolution.  It only returns source surfaces and character coordinates after
inline-domain segmentation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from paperos_core.ingestion.inline_domains import (
    InlineDomainKind,
    bracket_inner,
    iter_bracket_scopes,
    scan_inline_domains,
)


@dataclass(frozen=True, slots=True)
class CitationCandidate:
    surface: str
    start: int
    end: int
    kind: str
    bracket_start: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


_YEAR_ONLY = re.compile(r"^[12]\d{3}[a-d]?$", re.IGNORECASE)
_LEFT_AUTHOR = re.compile(
    r"(?P<author>[A-ZÀ-ÖØ-Þ][\w''\u00C0-\u024F\-]+"
    r"(?:\s+et\s+al\.?)?(?:\s+(?:and|&)\s+"
    r"[A-ZÀ-ÖØ-Þ][\w''\u00C0-\u024F\-]+)?)\s*$"
)
_AUTHOR_YEAR_PAREN = re.compile(
    r"\((?P<author>[A-ZÀ-ÖØ-Þ][\w''\-]+(?:\s+(?:et\s+al\.?|and|&)\s*"
    r"[\w''\-]+)*)\s*,\s*(?P<year>[12]\d{3}[a-d]?)\)"
)
_AUTHOR_YEAR_INLINE = re.compile(
    r"(?<![A-Za-z])(?P<author>[A-ZÀ-ÖØ-Þ][\w''\-]+(?:\s+et\s+al\.?)?"
    r"(?:\s+and\s+[A-ZÀ-ÖØ-Þ][\w''\-]+)?)\s+\((?P<year>[12]\d{3}[a-d]?)\)"
)
_OCR_SYMBOLIC_MATH_CITATION = re.compile(
    r"""
    ^\s*(?:\${1,2}|\\\()\s*
    \\(?:mathrm|text)\s*\{\s*
    \[\s*
    (?P<label>(?:[A-Za-z]\s*){2,12})
    (?P<star>\^\s*\{\s*(?:\\?ast|\*)\s*\})?
    \s*(?:
        \}\s*(?P<year_after_mathrm>(?:\d\s*){2,4}[a-d]?)\s*\]
        |
        (?P<year_inside_mathrm>(?:\d\s*){2,4}[a-d]?)\s*\]\s*\}
    )
    \s*(?:\${1,2}|\\\))\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def detect_citation_candidates(text: str) -> list[CitationCandidate]:
    """Detect balanced bracket and author-year candidates outside math."""
    domains = scan_inline_domains(text)
    candidates: list[CitationCandidate] = []
    masked = text
    for domain in domains:
        if domain.kind != InlineDomainKind.INLINE_MATH:
            continue
        surface = text[domain.start : domain.end]
        label = _ocred_symbolic_math_label(surface)
        if label is None:
            continue
        bracket_offset = surface.find("[")
        candidates.append(
            CitationCandidate(
                surface=surface,
                start=domain.start,
                end=domain.end,
                kind="bracket",
                bracket_start=(
                    domain.start + bracket_offset if bracket_offset >= 0 else domain.start
                ),
                metadata={"inner": label, "source_domain": "inline_math"},
            )
        )
        masked = _mask(masked, domain.start, domain.end)
    for bracket in iter_bracket_scopes(text, domains):
        inner = bracket_inner(text, bracket).strip()
        start = bracket.start
        metadata = {"inner": inner}
        if _YEAR_ONLY.fullmatch(inner):
            author_match = _left_author_match(text, bracket.start)
            if author_match is not None:
                author, start = author_match
                metadata.update({"author": author, "year": inner})
        candidates.append(
            CitationCandidate(
                surface=text[start : bracket.end],
                start=start,
                end=bracket.end,
                kind="bracket",
                bracket_start=bracket.start,
                metadata=metadata,
            )
        )
        masked = _mask(masked, start, bracket.end)

    for match in _AUTHOR_YEAR_PAREN.finditer(masked):
        candidates.append(
            CitationCandidate(
                surface=match.group(0),
                start=match.start(),
                end=match.end(),
                kind="author_year_paren",
                metadata={"author": match.group("author"), "year": match.group("year")},
            )
        )
        masked = _mask(masked, match.start(), match.end())
    for match in _AUTHOR_YEAR_INLINE.finditer(masked):
        candidates.append(
            CitationCandidate(
                surface=match.group(0),
                start=match.start(),
                end=match.end(),
                kind="author_year_inline",
                metadata={"author": match.group("author"), "year": match.group("year")},
            )
        )
    return sorted(candidates, key=lambda item: (item.start, item.end, item.kind))


def _ocred_symbolic_math_label(surface: str) -> str | None:
    match = _OCR_SYMBOLIC_MATH_CITATION.fullmatch(surface)
    if match is None:
        return None
    label = re.sub(r"\s+", "", match.group("label"))
    year = re.sub(
        r"\s+",
        "",
        match.group("year_after_mathrm") or match.group("year_inside_mathrm"),
    )
    star = "*" if match.group("star") else ""
    return f"{label}{star}{year}"


def _left_author_match(text: str, bracket_start: int) -> tuple[str, int] | None:
    window_start = max(0, bracket_start - 120)
    prefix = text[window_start:bracket_start].rstrip()
    match = _LEFT_AUTHOR.search(prefix)
    if match is None:
        return None
    return match.group("author").strip(), window_start + match.start("author")


def _mask(text: str, start: int, end: int) -> str:
    return text[:start] + " " * (end - start) + text[end:]
