"""Evidence formatting and provenance completeness."""

from __future__ import annotations

from paperos_core.domain.canonical import CanonicalBundle
from paperos_core.retrieval.candidates import Candidate, Evidence


def format_evidence(
    candidates: list[Candidate], bundles: dict[str, CanonicalBundle]
) -> list[Evidence]:
    evidence: list[Evidence] = []
    for candidate in candidates:
        bundle = bundles[candidate.document_id]
        evidence.append(
            Evidence(
                evidence_id=f"evidence:{candidate.id}",
                chunk_id=candidate.chunk_id,
                document_id=candidate.document_id,
                source_file_id=candidate.source_file_id,
                source_filename=candidate.source_filename,
                title=bundle.document.title,
                section_path=candidate.section_path,
                page_start=candidate.page_start,
                page_end=candidate.page_end,
                text=candidate.text,
                channels=candidate.channels,
                knowledge_kind=candidate.knowledge_kind,
                derived_from_ids=candidate.derived_from_ids,
                source_work_id=candidate.source_work_id,
                subject_work_ids=list(candidate.subject_work_ids),
            )
        )
    return evidence
