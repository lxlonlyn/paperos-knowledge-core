"""Deterministic reference resolution across retained canonical snapshots."""

from __future__ import annotations

import re

from paperos_core.domain.canonical import CanonicalBundle
from paperos_core.domain.provenance import RelationRecord, RelationType

_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def resolve_citations(
    bundle: CanonicalBundle, candidates: list[CanonicalBundle]
) -> list[RelationRecord]:
    """Return only high-confidence citation relations backed by ReferenceEntry."""
    relations: list[RelationRecord] = []
    other_documents = [
        candidate.document
        for candidate in candidates
        if candidate.document.id != bundle.document.id
    ]
    for reference in bundle.references:
        match_id: str | None = None
        if reference.doi:
            doi = reference.doi.casefold()
            matches = [
                document.id
                for document in other_documents
                if document.doi and document.doi.casefold() == doi
            ]
            if len(matches) == 1:
                match_id = matches[0]
        if match_id is None and reference.title:
            title_key = _title_key(reference.title)
            matches = [
                document.id
                for document in other_documents
                if _title_key(document.title) == title_key
            ]
            if title_key and len(matches) == 1:
                match_id = matches[0]
        if match_id is None:
            continue
        evidence_chunks = [
            chunk.id for chunk in bundle.chunks if reference.source_element_id in chunk.element_ids
        ]
        relations.extend(
            [
                RelationRecord(
                    source_id=reference.id,
                    target_id=match_id,
                    relation_type=RelationType.CITES,
                    source_chunk_ids=evidence_chunks,
                    derived_from_ids=[reference.id],
                ),
                RelationRecord(
                    source_id=bundle.document.id,
                    target_id=match_id,
                    relation_type=RelationType.CITES,
                    source_chunk_ids=evidence_chunks,
                    derived_from_ids=[reference.id],
                ),
            ]
        )
    return relations


def _title_key(value: str) -> str:
    return _NON_WORD.sub("", value.casefold())
