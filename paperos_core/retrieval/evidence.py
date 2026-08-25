"""Canonical source-grounded evidence formatting."""

from __future__ import annotations

from paperos_core.retrieval.candidates import Candidate, Evidence
from paperos_core.retrieval.corpus import CorpusView


def format_evidence(candidates: list[Candidate], corpus: CorpusView) -> list[Evidence]:
    """Ignore candidate payload text and rehydrate every field from the corpus."""
    evidence: list[Evidence] = []
    for candidate in candidates:
        chunk = corpus.chunks[candidate.chunk_id]
        bundle = corpus.chunk_bundles[candidate.chunk_id]
        evidence.append(
            Evidence(
                evidence_id=f"evidence:{chunk.id}",
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                source_file_id=bundle.document.source_file_id,
                source_filename=corpus.source_filenames[bundle.document.source_file_id],
                title=bundle.document.title,
                authors=[author.display_name for author in bundle.document.authors],
                year=bundle.document.year,
                section_path=chunk.section_path,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                channels=list(candidate.channels),
                knowledge_kind=candidate.knowledge_kind,
                derived_from_ids=list(candidate.derived_from_ids),
                source_work_id=corpus.work_id_by_document.get(chunk.document_id),
            )
        )
    return evidence
