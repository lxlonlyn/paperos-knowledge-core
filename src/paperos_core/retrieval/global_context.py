"""Document summary retrieval with source-chunk backtracking."""

from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView


def global_context_retrieve(
    corpus: CorpusView,
    *,
    limit: int,
    document_ids: set[str],
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for document_id, enrichment in corpus.enrichments.items():
        if document_id not in document_ids:
            continue
        for summary in enrichment.summaries:
            for chunk_id in summary.source_chunk_ids[:2]:
                candidates.append(
                    corpus.candidate_for_chunk(
                        chunk_id,
                        channel="global_context",
                        score=1.0,
                        object_id=summary.id,
                        object_type="summary",
                        knowledge_kind="system_inference",
                        derived_from_ids=[summary.id, *summary.derived_from_ids],
                    )
                )
    return candidates[:limit]
