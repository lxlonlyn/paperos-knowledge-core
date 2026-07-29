import pytest

from paperos_core.domain.enums import IngestionJobStatus
from paperos_core.errors import InvalidJobTransitionError
from paperos_core.jobs.state import validate_transition


def test_job_state_transitions_are_explicit() -> None:
    validate_transition(IngestionJobStatus.PENDING, IngestionJobStatus.PARSING)
    validate_transition(IngestionJobStatus.PENDING, IngestionJobStatus.FAILED)
    with pytest.raises(InvalidJobTransitionError):
        validate_transition(IngestionJobStatus.PENDING, IngestionJobStatus.COMPLETED)
    with pytest.raises(InvalidJobTransitionError):
        validate_transition(IngestionJobStatus.COMPLETED, IngestionJobStatus.PENDING)
