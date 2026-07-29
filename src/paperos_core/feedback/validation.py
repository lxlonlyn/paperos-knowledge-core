"""Feedback validation against retained canonical source evidence."""

from __future__ import annotations

from paperos_core.errors import FeedbackValidationError
from paperos_core.feedback.models import FeedbackRequest, FeedbackType
from paperos_core.ingestion.canonical_repository import CanonicalRepository


def validate_feedback(
    request: FeedbackRequest, canonical_repository: CanonicalRepository
) -> list[str]:
    if (
        request.feedback_type is FeedbackType.CORRECT
        and not (request.replacement_text or "").strip()
    ):
        raise FeedbackValidationError(
            "Correction feedback requires non-empty replacement_text.",
            affected=request.target_id,
        )
    chunk_ids = {
        chunk.id
        for bundle in canonical_repository.list_bundles()
        for chunk in bundle.chunks
    }
    source_chunk_ids: list[str] = []
    for evidence_id in request.evidence_ids:
        prefix = "evidence:"
        chunk_id = evidence_id.removeprefix(prefix)
        if not evidence_id.startswith(prefix) or chunk_id not in chunk_ids:
            raise FeedbackValidationError(
                "Feedback evidence ID does not resolve to a retained canonical chunk.",
                affected=evidence_id,
            )
        source_chunk_ids.append(chunk_id)
    if request.target_id.startswith("chunk_") and request.target_id not in chunk_ids:
        raise FeedbackValidationError(
            "Feedback target chunk does not exist.", affected=request.target_id
        )
    if request.target_id in chunk_ids:
        source_chunk_ids.append(request.target_id)
    return list(dict.fromkeys(source_chunk_ids))
