"""Reference-entry parsing with original text preservation."""

from __future__ import annotations

import re

from paperos_core.domain.canonical import ReferenceEntry
from paperos_core.domain.ids import reference_entry_id
from paperos_core.ingestion.normalization import normalize_doi, plain_text

REFERENCE_PROCESSING_VERSION = "1"

_YEAR = re.compile(r"(?<!\d)((?:18|19|20)\d{2})(?!\d)")
_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s<>\])]+", re.IGNORECASE)
_ARXIV = re.compile(r"\barXiv:\s*([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)", re.IGNORECASE)
_INDEX = re.compile(r"^\s*\[(\d+)]\s*")


def parse_reference_entry(
    *,
    document_id: str,
    snapshot_id: str,
    order: int,
    raw_text: str,
    source_element_id: str | None,
) -> ReferenceEntry:
    normalized = plain_text(raw_text)
    body = _INDEX.sub("", normalized)
    year_match = _YEAR.search(body)
    doi_match = _DOI.search(body)
    url_match = _URL.search(body)
    arxiv_match = _ARXIV.search(body)
    return ReferenceEntry(
        id=reference_entry_id(document_id, order, normalized),
        document_id=document_id,
        canonical_snapshot_id=snapshot_id,
        raw_text=normalized,
        order=order,
        year=int(year_match.group(1)) if year_match else None,
        doi=normalize_doi(doi_match.group(0)) if doi_match else None,
        url=url_match.group(0).rstrip(".,;") if url_match else None,
        arxiv_id=arxiv_match.group(1) if arxiv_match else None,
        source_element_id=source_element_id,
        parsed_fields={
            "reference_number": (
                int(index_match.group(1)) if (index_match := _INDEX.match(normalized)) else None
            )
        },
    )
