"""Provider-neutral text normalization used by the canonical mapper."""

from __future__ import annotations

import html
import re
import unicodedata

NORMALIZATION_VERSION = "1"

_WHITESPACE = re.compile(r"[^\S\n]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_HTML_TAG = re.compile(r"<[^>]+>")
_LATEX_ACCENTS = {
    "a": "à",
    "e": "è",
    "i": "ì",
    "o": "ò",
    "u": "ù",
    "A": "À",
    "E": "È",
    "I": "Ì",
    "O": "Ò",
    "U": "Ù",
}


def normalize_text(value: str) -> str:
    """Normalize Unicode and whitespace without changing mathematical syntax."""
    normalized = unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n"))
    normalized = _WHITESPACE.sub(" ", normalized)
    normalized = "\n".join(line.strip() for line in normalized.splitlines())
    return _BLANK_LINES.sub("\n\n", normalized).strip()


def plain_text(value: str) -> str:
    """Return normalized visible text for metadata and retrieval fields."""
    visible = html.unescape(_HTML_TAG.sub("", value))
    visible = re.sub(r"([AEIOUaeiou])\\?`", _replace_grave, visible)
    return normalize_text(visible)


def normalized_match_text(value: str) -> str:
    value = plain_text(value).casefold()
    value = value.replace("–", "-").replace("—", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", value).strip()


def _replace_grave(match: re.Match[str]) -> str:
    return _LATEX_ACCENTS.get(match.group(1), match.group(1))


def strip_heading_number(value: str) -> str:
    text = plain_text(value)
    return re.sub(r"^\s*(?:\d+(?:\.\d+)*\.?|[A-Z]\.)\s+", "", text).strip()


def normalize_doi(value: str) -> str:
    doi = value.strip().rstrip(".,;)]}")
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.IGNORECASE)
    return doi.lower()
