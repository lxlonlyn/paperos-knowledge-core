"""Ingestion-job state transition policy."""

from __future__ import annotations

from paperos_core.domain.documents import IngestionJob
from paperos_core.domain.enums import IngestionJobStatus
from paperos_core.errors import InvalidJobTransitionError

_ALLOWED_TRANSITIONS: dict[IngestionJobStatus, frozenset[IngestionJobStatus]] = {
    IngestionJobStatus.PENDING: frozenset(
        {
            IngestionJobStatus.VALIDATING,
            IngestionJobStatus.PARSING,
            IngestionJobStatus.FAILED,
            IngestionJobStatus.CANCELLED,
            IngestionJobStatus.INTERRUPTED,
        }
    ),
    IngestionJobStatus.VALIDATING: frozenset(
        {
            IngestionJobStatus.PENDING,
            IngestionJobStatus.PARSING,
            IngestionJobStatus.FAILED,
            IngestionJobStatus.CANCELLED,
            IngestionJobStatus.INTERRUPTED,
        }
    ),
    IngestionJobStatus.PARSING: frozenset(
        {
            IngestionJobStatus.NORMALIZING,
            IngestionJobStatus.FAILED,
            IngestionJobStatus.CANCELLED,
            IngestionJobStatus.INTERRUPTED,
        }
    ),
    IngestionJobStatus.NORMALIZING: frozenset(
        {
            IngestionJobStatus.WRITING,
            IngestionJobStatus.FAILED,
            IngestionJobStatus.CANCELLED,
            IngestionJobStatus.INTERRUPTED,
        }
    ),
    IngestionJobStatus.WRITING: frozenset(
        {
            IngestionJobStatus.INDEXING,
            IngestionJobStatus.FAILED,
            IngestionJobStatus.CANCELLED,
            IngestionJobStatus.INTERRUPTED,
        }
    ),
    IngestionJobStatus.INDEXING: frozenset(
        {
            IngestionJobStatus.POSTPROCESSING,
            IngestionJobStatus.FAILED,
            IngestionJobStatus.CANCELLED,
            IngestionJobStatus.INTERRUPTED,
        }
    ),
    IngestionJobStatus.POSTPROCESSING: frozenset(
        {
            IngestionJobStatus.COMPLETED,
            IngestionJobStatus.FAILED,
            IngestionJobStatus.CANCELLED,
            IngestionJobStatus.INTERRUPTED,
        }
    ),
    IngestionJobStatus.COMPLETED: frozenset(),
    IngestionJobStatus.FAILED: frozenset({IngestionJobStatus.PENDING}),
    IngestionJobStatus.CANCELLED: frozenset(),
    IngestionJobStatus.INTERRUPTED: frozenset(),
}


def validate_transition(current: IngestionJobStatus, target: IngestionJobStatus) -> None:
    if target == current:
        return
    if target not in _ALLOWED_TRANSITIONS[current]:
        raise InvalidJobTransitionError(
            f"Cannot transition ingestion job from '{current.value}' to '{target.value}'.",
            details={"current_status": current.value, "target_status": target.value},
        )


__all__ = ["IngestionJob", "IngestionJobStatus", "validate_transition"]
