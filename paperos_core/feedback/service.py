"""Immutable SQLite feedback and versioned derived-knowledge service."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from paperos_core.domain.ids import stable_id
from paperos_core.errors import FeedbackStorageError
from paperos_core.feedback.models import (
    FEEDBACK_ID_VERSION,
    Correction,
    FeedbackRecord,
    FeedbackRequest,
    FeedbackType,
    Improvement,
    ImprovementReport,
)
from paperos_core.feedback.validation import validate_feedback
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.paths import DataPaths


class FeedbackService:
    def __init__(
        self, paths: DataPaths, canonical_repository: CanonicalRepository
    ) -> None:
        self.paths = paths
        self.canonical_repository = canonical_repository

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.paths.registry_db, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            with connection:
                yield connection
        finally:
            connection.close()

    def record(self, request: FeedbackRequest) -> FeedbackRecord:
        validate_feedback(request, self.canonical_repository)
        record = FeedbackRecord(
            id=f"feedback_{uuid.uuid4().hex}", **request.model_dump()
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO feedback VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.feedback_type.value,
                        record.target_id,
                        record.query_id,
                        record.answer_id,
                        json.dumps(record.evidence_ids),
                        record.comment,
                        record.replacement_text,
                        record.created_by,
                        record.created_at.isoformat(),
                        record.schema_version,
                        record.id_version,
                    ),
                )
        except sqlite3.Error as exc:
            raise FeedbackStorageError(
                f"Unable to store feedback: {exc}", affected=self.paths.registry_db
            ) from exc
        return record

    def improve(self) -> ImprovementReport:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT f.* FROM feedback f
                LEFT JOIN improvements i ON i.feedback_id = f.id
                WHERE i.id IS NULL ORDER BY f.created_at, f.id
                """
            ).fetchall()
        improvements: list[Improvement] = []
        corrections: list[Correction] = []
        for row in rows:
            feedback = self._feedback_from_row(row)
            source_chunk_ids = validate_feedback(
                FeedbackRequest.model_validate(
                    feedback.model_dump(
                        include={
                            "target_id",
                            "feedback_type",
                            "query_id",
                            "answer_id",
                            "evidence_ids",
                            "comment",
                            "replacement_text",
                            "created_by",
                        }
                    )
                ),
                self.canonical_repository,
            )
            correction: Correction | None = None
            version = self._next_improvement_version(feedback.target_id)
            if feedback.feedback_type is FeedbackType.CORRECT:
                replacement = feedback.replacement_text or ""
                correction_version, supersedes_id = self._next_correction_version(
                    feedback.target_id
                )
                correction = Correction(
                    id=stable_id(
                        "correction",
                        feedback.id,
                        replacement,
                        id_version=FEEDBACK_ID_VERSION,
                    ),
                    target_id=feedback.target_id,
                    replacement_or_correction=replacement,
                    derived_from_feedback_id=feedback.id,
                    source_chunk_ids=source_chunk_ids,
                    supersedes_object_id=supersedes_id or feedback.target_id,
                    version=correction_version,
                )
            status = (
                "user_confirmed"
                if feedback.feedback_type
                in {FeedbackType.ACCEPT, FeedbackType.CONFIRM, FeedbackType.CORRECT}
                else "rejected"
            )
            improvement = Improvement(
                id=stable_id(
                    "improvement",
                    feedback.id,
                    status,
                    id_version=FEEDBACK_ID_VERSION,
                ),
                feedback_id=feedback.id,
                target_id=feedback.target_id,
                improvement_type=feedback.feedback_type.value,
                text=feedback.replacement_text or feedback.comment,
                status=status,
                evidence_ids=feedback.evidence_ids,
                source_chunk_ids=source_chunk_ids,
                derived_from_ids=[feedback.id, *source_chunk_ids],
                correction_id=correction.id if correction else None,
                version=version,
            )
            self._store_derived_feedback(correction, improvement)
            if correction is not None:
                corrections.append(correction)
            improvements.append(improvement)
        return ImprovementReport(
            processed_feedback_ids=[item.feedback_id for item in improvements],
            corrections=corrections,
            improvements=improvements,
        )

    def confirmed_improvements(self) -> list[Improvement]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM improvements WHERE status = 'user_confirmed' "
                "ORDER BY created_at, id"
            ).fetchall()
        return [self._improvement_from_row(row) for row in rows]

    def _next_correction_version(self, target_id: str) -> tuple[int, str | None]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, version FROM corrections
                WHERE target_id = ? ORDER BY version DESC, created_at DESC LIMIT 1
                """,
                (target_id,),
            ).fetchone()
        if row is None:
            return 1, None
        return int(row["version"]) + 1, str(row["id"])

    def _next_improvement_version(self, target_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT MAX(version) FROM improvements WHERE target_id = ?",
                (target_id,),
            ).fetchone()
        return int(row[0] or 0) + 1

    def _store_derived_feedback(
        self,
        correction: Correction | None,
        improvement: Improvement,
    ) -> None:
        with self._connect() as connection:
            if correction is not None:
                connection.execute(
                    "INSERT INTO corrections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        correction.id,
                        correction.target_id,
                        correction.replacement_or_correction,
                        correction.status,
                        correction.created_at.isoformat(),
                        correction.schema_version,
                        correction.id_version,
                        correction.derived_from_feedback_id,
                        json.dumps(correction.source_chunk_ids),
                        correction.supersedes_object_id,
                        correction.version,
                    ),
                )
            connection.execute(
                "INSERT INTO improvements VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    improvement.id,
                    improvement.feedback_id,
                    improvement.target_id,
                    improvement.improvement_type,
                    improvement.text,
                    improvement.status,
                    json.dumps(improvement.evidence_ids),
                    json.dumps(improvement.source_chunk_ids),
                    json.dumps(improvement.derived_from_ids),
                    improvement.correction_id,
                    improvement.version,
                    improvement.created_at.isoformat(),
                    improvement.schema_version,
                    improvement.id_version,
                ),
            )

    @staticmethod
    def _feedback_from_row(row: sqlite3.Row) -> FeedbackRecord:
        return FeedbackRecord(
            id=row["id"],
            feedback_type=row["feedback_type"],
            target_id=row["target_id"],
            query_id=row["query_id"],
            answer_id=row["answer_id"],
            evidence_ids=json.loads(row["evidence_ids"]),
            comment=row["comment"],
            replacement_text=row["replacement_text"],
            created_by=row["created_by"],
            created_at=datetime.fromisoformat(row["created_at"]),
            schema_version=row["schema_version"],
            id_version=row["id_version"],
        )

    @staticmethod
    def _improvement_from_row(row: sqlite3.Row) -> Improvement:
        return Improvement(
            id=row["id"],
            feedback_id=row["feedback_id"],
            target_id=row["target_id"],
            improvement_type=row["improvement_type"],
            text=row["text"],
            status=row["status"],
            evidence_ids=json.loads(row["evidence_ids"]),
            source_chunk_ids=json.loads(row["source_chunk_ids"]),
            derived_from_ids=json.loads(row["derived_from_ids"]),
            correction_id=row["correction_id"],
            version=row["version"],
            created_at=datetime.fromisoformat(row["created_at"]),
            schema_version=row["schema_version"],
            id_version=row["id_version"],
        )
