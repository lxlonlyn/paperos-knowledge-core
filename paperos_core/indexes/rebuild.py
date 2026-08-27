"""Destructive, scoped rebuild of derived knowledge projections."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from paperos_core.adapters.cognee.pipeline import CogneePipelineAdapter
from paperos_core.errors import CogneeStorageError
from paperos_core.indexes.manifest import IndexingReport
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.paths import DataPaths
from paperos_core.storage.initializer import StorageInitializer


class RebuildReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rebuilt_snapshot_ids: list[str]
    deleted_paths: list[Path]
    reports: list[IndexingReport]
    all_snapshot_count: int
    active_snapshot_count: int
    enrichment_existing_count: int
    enrichment_missing_count: int
    enrichment_generated_count: int
    enrichment_reused_count: int
    refresh_enrichment: bool
    llm_enrichment_call_count: int

    def public_dict(self) -> dict[str, object]:
        return {
            "rebuilt_snapshot_ids": self.rebuilt_snapshot_ids,
            "reports": [report.public_dict() for report in self.reports],
            "all_snapshot_count": self.all_snapshot_count,
            "active_snapshot_count": self.active_snapshot_count,
            "enrichment_existing_count": self.enrichment_existing_count,
            "enrichment_missing_count": self.enrichment_missing_count,
            "enrichment_generated_count": self.enrichment_generated_count,
            "enrichment_reused_count": self.enrichment_reused_count,
            "refresh_enrichment": self.refresh_enrichment,
            "llm_enrichment_call_count": self.llm_enrichment_call_count,
        }


class DerivedDataRebuilder:
    def __init__(
        self,
        paths: DataPaths,
        canonical_repository: CanonicalRepository,
        pipeline: CogneePipelineAdapter,
        storage: StorageInitializer,
    ) -> None:
        self.paths = paths
        self.canonical_repository = canonical_repository
        self.pipeline = pipeline
        self.storage = storage

    def select_snapshot_ids(self, snapshot_id: str | None = None) -> list[str]:
        """Select only active revisions for the public rebuild path."""

        active_snapshot_ids = self.canonical_repository.list_active_snapshot_ids()
        if snapshot_id is None:
            return active_snapshot_ids
        if snapshot_id not in active_snapshot_ids:
            raise CogneeStorageError(
                "Only an active canonical snapshot can be rebuilt.",
                affected=snapshot_id,
                details={"reason": "inactive_canonical_snapshot"},
            )
        return [snapshot_id]

    async def rebuild(
        self,
        snapshot_id: str | None = None,
        *,
        refresh_enrichment: bool = False,
    ) -> RebuildReport:
        all_snapshot_ids = self.canonical_repository.list_all_snapshot_ids()
        active_snapshot_ids = self.canonical_repository.list_active_snapshot_ids()
        selected = self.select_snapshot_ids(snapshot_id)
        with sqlite3.connect(self.paths.registry_db) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='document_tombstones'"
            ).fetchone()
            deleted_documents = (
                {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT document_id FROM document_tombstones"
                    ).fetchall()
                }
                if exists
                else set()
            )
        selected = [
            selected_id
            for selected_id in selected
            if self.canonical_repository.get_snapshot(selected_id).document_id
            not in deleted_documents
        ]
        for selected_id in selected:
            self.canonical_repository.verify_snapshot(selected_id)

        enrichment_root = self.paths.cognee / "enrichment"
        existing = [
            selected_id
            for selected_id in selected
            if (enrichment_root / f"{selected_id}.json").is_file()
        ]
        existing_set = set(existing)
        missing = [selected_id for selected_id in selected if selected_id not in existing_set]
        if missing and not refresh_enrichment:
            raise CogneeStorageError(
                "Semantic enrichment is missing for current rebuild snapshots; "
                "run with refresh_enrichment=True to generate only missing artifacts.",
                affected=enrichment_root,
                details={"missing_snapshot_ids": missing},
            )

        deleted: list[Path] = []
        reports: list[IndexingReport] = []
        rebuilt_snapshot_ids: list[str] = []
        for selected_id in selected:
            candidate = self.canonical_repository.create_rebuild_candidate(selected_id)
            candidate_id = candidate.snapshot.id
            reused_enrichment = selected_id in existing_set
            try:
                if reused_enrichment:
                    self.pipeline.reproject_enrichment(selected_id, candidate_id)
                report, _ = await self.pipeline.ingest_bundle(
                    candidate,
                    rebuilt=True,
                    reuse_existing_enrichment=reused_enrichment,
                    generate_enrichment_if_missing=refresh_enrichment,
                )
                previous = self.pipeline.scholarly_registry.publish_candidate(
                    candidate_id,
                    self.canonical_repository,
                    expected_previous_snapshot_id=selected_id,
                )
            except Exception:
                await self.pipeline._cleanup_after_failure(
                    candidate_id,
                    phase="rebuild_candidate",
                )
                raise
            if previous != selected_id:  # Defensive: publication enforces this atomically.
                raise AssertionError("Rebuild publication returned an unexpected revision")
            try:
                deleted.extend(
                    await self.pipeline.cleanup_snapshot_revision(selected_id)
                )
            except Exception as exc:  # noqa: BLE001 - new active remains authoritative.
                self.pipeline._record_cleanup_retry(
                    selected_id,
                    phase="retired_rebuild_revision",
                    exc=exc,
                )
            reports.append(report)
            rebuilt_snapshot_ids.append(candidate_id)
        return RebuildReport(
            rebuilt_snapshot_ids=rebuilt_snapshot_ids,
            deleted_paths=deleted,
            reports=reports,
            all_snapshot_count=len(all_snapshot_ids),
            active_snapshot_count=len(active_snapshot_ids),
            enrichment_existing_count=len(existing),
            enrichment_missing_count=len(missing),
            enrichment_generated_count=len(missing),
            enrichment_reused_count=len(existing),
            refresh_enrichment=refresh_enrichment,
            llm_enrichment_call_count=len(missing),
        )
