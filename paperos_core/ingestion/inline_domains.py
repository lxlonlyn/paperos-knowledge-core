"""Inline domain segmentation for prose elements (math, brackets, parentheses)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class InlineDomainKind(str, Enum):
    TEXT = "text"
    INLINE_MATH = "inline_math"
    BRACKET_SCOPE = "bracket_scope"
    PAREN_SCOPE = "paren_scope"
    BRACE_SCOPE = "brace_scope"


@dataclass(frozen=True, slots=True)
class InlineDomain:
    kind: InlineDomainKind
    start: int
    end: int


_OPENERS = {
    "(": (")", InlineDomainKind.PAREN_SCOPE),
    "[": ("]", InlineDomainKind.BRACKET_SCOPE),
    "{": ("}", InlineDomainKind.BRACE_SCOPE),
    "（": ("）", InlineDomainKind.PAREN_SCOPE),
    "［": ("］", InlineDomainKind.BRACKET_SCOPE),
    "【": ("】", InlineDomainKind.BRACE_SCOPE),
}
_CLOSERS = {close: (open_, kind) for open_, (close, kind) in _OPENERS.items()}


def scan_inline_domains(text: str) -> list[InlineDomain]:
    """Return non-overlapping inline domains sorted by start."""
    domains: list[InlineDomain] = []
    index = 0
    length = len(text)
    while index < length:
        if _starts_display_math(text, index):
            end = _scan_display_math(text, index)
            if end is not None:
                domains.append(InlineDomain(InlineDomainKind.INLINE_MATH, index, end))
                index = end
                continue
        if _starts_inline_math(text, index):
            end = _scan_inline_math(text, index)
            if end is not None:
                domains.append(InlineDomain(InlineDomainKind.INLINE_MATH, index, end))
                index = end
                continue
        if _starts_latex_math_delim(text, index):
            end = _scan_latex_math_delim(text, index)
            if end is not None:
                domains.append(InlineDomain(InlineDomainKind.INLINE_MATH, index, end))
                index = end
                continue
        char = text[index]
        if char in _OPENERS and not _inside_existing(index, domains):
            close, kind = _OPENERS[char]
            end = _scan_balanced(text, index + 1, close)
            if end is not None:
                domains.append(InlineDomain(kind, index, end))
                index = end
                continue
        index += 1
    return sorted(domains, key=lambda item: (item.start, item.end))


def domain_at(position: int, domains: list[InlineDomain]) -> InlineDomainKind:
    for domain in domains:
        if domain.start <= position < domain.end:
            return domain.kind
    return InlineDomainKind.TEXT


def sentence_boundary_allowed(position: int, domains: list[InlineDomain]) -> bool:
    """True when a candidate boundary at ``position`` is outside protected domains."""
    if position <= 0 or position >= 1:
        pass
    # Boundary is the character immediately after punctuation; disallow if inside domain.
    check = position - 1
    for domain in domains:
        if domain.start < check < domain.end:
            return False
    return True


def iter_bracket_scopes(text: str, domains: list[InlineDomain]) -> list[InlineDomain]:
    math_spans = [
        (domain.start, domain.end)
        for domain in domains
        if domain.kind == InlineDomainKind.INLINE_MATH
    ]
    brackets: list[InlineDomain] = []
    index = 0
    length = len(text)
    while index < length:
        inside_math = any(start <= index < end for start, end in math_spans)
        if inside_math:
            index += 1
            continue
        if text[index] != "[":
            index += 1
            continue
        end = _scan_balanced(text, index + 1, "]")
        if end is None:
            index += 1
            continue
        brackets.append(InlineDomain(InlineDomainKind.BRACKET_SCOPE, index, end))
        index = end
    return brackets


_NUMERIC_CITATION_RE = re.compile(r"^\s*\d{1,4}([a-d])?\s*$", re.IGNORECASE)
_CITATION_LIST_RE = re.compile(
    r"^\s*\d{1,4}([a-d])?\s*(?:\s*[,;]\s*\d{1,4}([a-d])?)*\s*(?:\s*[-–−—]\s*\d{1,4}([a-d])?)?\s*$",
    re.IGNORECASE,
)


def _is_prose_numeric_citation(text: str, start: int, end: int) -> bool:
    """True when ``[...]`` inside a math span is a prose numeric citation."""
    inner = text[start + 1 : end - 1]
    if re.search(r"\d\s+\d", inner):
        return False
    compact = re.sub(r"\s+", "", inner)
    if _NUMERIC_CITATION_RE.match(compact):
        return True
    if "," in inner:
        parts = [part.strip() for part in inner.split(",") if part.strip()]
        if parts and all(re.fullmatch(r"\d{1,4}[a-d]?", p, flags=re.I) for p in parts):
            return True
        return False
    if _CITATION_LIST_RE.match(compact):
        return True
    return False


def bracket_inner(text: str, domain: InlineDomain) -> str:
    return text[domain.start + 1 : domain.end - 1]


def _inside_existing(position: int, domains: list[InlineDomain]) -> bool:
    return any(domain.start <= position < domain.end for domain in domains)


def _scan_balanced(text: str, index: int, closer: str) -> int | None:
    depth = 1
    cursor = index
    while cursor < len(text):
        char = text[cursor]
        if char == closer:
            depth -= 1
            if depth == 0:
                return cursor + 1
        elif char in _OPENERS and _OPENERS[char][0] == closer:
            depth += 1
        cursor += 1
    return None


def _starts_display_math(text: str, index: int) -> bool:
    return text.startswith("$$", index)


def _scan_display_math(text: str, index: int) -> int | None:
    if not text.startswith("$$", index):
        return None
    cursor = index + 2
    while cursor < len(text):
        if text.startswith("$$", cursor):
            return cursor + 2
        cursor += 1
    return None


def _starts_inline_math(text: str, index: int) -> bool:
    if text[index] != "$":
        return False
    if index + 1 < len(text) and text[index + 1] == "$":
        return False
    nxt = text[index + 1] if index + 1 < len(text) else ""
    if nxt in {" ", ".", ",", ";", ":"}:
        return False
    if nxt.isalpha() and nxt.islower():
        return False
    return True


def _scan_inline_math(text: str, index: int) -> int | None:
    if text[index] != "$":
        return None
    cursor = index + 1
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 2
            continue
        if text[cursor] == "$":
            return cursor + 1
        cursor += 1
    return None


def _starts_latex_math_delim(text: str, index: int) -> bool:
    return text.startswith(r"\(", index) or text.startswith(r"\[", index)


def _scan_latex_math_delim(text: str, index: int) -> int | None:
    if text.startswith(r"\(", index):
        end = text.find(r"\)", index + 2)
        return end + 2 if end != -1 else None
    if text.startswith(r"\[", index):
        end = text.find(r"\]", index + 2)
        return end + 2 if end != -1 else None
    return None
