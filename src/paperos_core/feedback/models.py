"""Versioned feedback, correction, and improvement domain models."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from paperos_core.domain.documents import utc_now

FEEDBACK_SCHEMA_VERSION = "1.0"
FEEDBACK_ID_VERSION = "1"


class FeedbackType(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"
    CORRECT = "correct"
    CONFIRM = "confirm"
    MARK_UNSUPPORTED = "mark_unsupported"


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(min_length=1)
    feedback_type: FeedbackType
    query_id: str | None = None
    answer_id: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    comment: str | None = None
    replacement_text: str | None = None
    created_by: str | None = None


class FeedbackRecord(FeedbackRequest):
    id: str
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = FEEDBACK_SCHEMA_VERSION
    id_version: str = FEEDBACK_ID_VERSION


class Correction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    target_id: str
    replacement_or_correction: str
    status: str = "user_confirmed"
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = FEEDBACK_SCHEMA_VERSION
    id_version: str = FEEDBACK_ID_VERSION
    derived_from_feedback_id: str
    source_chunk_ids: list[str] = Field(default_factory=list)
    supersedes_object_id: str | None = None
    version: int = Field(default=1, gt=0)


class Improvement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    feedback_id: str
    target_id: str
    improvement_type: str
    text: str | None = None
    status: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    derived_from_ids: list[str] = Field(min_length=1)
    correction_id: str | None = None
    version: int = Field(default=1, gt=0)
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = FEEDBACK_SCHEMA_VERSION
    id_version: str = FEEDBACK_ID_VERSION


class ImprovementReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    processed_feedback_ids: list[str]
    corrections: list[Correction]
    improvements: list[Improvement]
