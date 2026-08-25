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
    current_snapshot_count: int
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
            "current_snapshot_count": self.current_snapshot_count,
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

    async def rebuild(
        self,
        snapshot_id: str | None = None,
        *,
        refresh_enrichment: bool = False,
        include_history: bool = False,
    ) -> RebuildReport:
        all_snapshot_ids = self.canonical_repository.list_snapshot_ids()
        current_snapshot_ids = self.canonical_repository.list_latest_snapshot_ids()
        if snapshot_id is not None:
            selected = [snapshot_id]
        elif include_history:
            selected = all_snapshot_ids
        else:
            selected = current_snapshot_ids
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

        # Work identity is authoritative, persistent registry state; populate it
        # before deleting and recreating any Cognee/FTS projections.
        self.pipeline.scholarly_registry.backfill(self.canonical_repository)
        deleted = await self._delete_derived_data()
        self.storage.initialize_lexical()
        reports: list[IndexingReport] = []
        for selected_id in selected:
            bundle = self.canonical_repository.get_bundle(selected_id)
            report, _ = await self.pipeline.ingest_bundle(
                bundle,
                rebuilt=True,
                reuse_existing_enrichment=True,
                generate_enrichment_if_missing=refresh_enrichment,
            )
            reports.append(report)
        return RebuildReport(
            rebuilt_snapshot_ids=selected,
            deleted_paths=deleted,
            reports=reports,
            all_snapshot_count=len(all_snapshot_ids),
            current_snapshot_count=len(current_snapshot_ids),
            enrichment_existing_count=len(existing),
            enrichment_missing_count=len(missing),
            enrichment_generated_count=len(missing),
            enrichment_reused_count=len(existing),
            refresh_enrichment=refresh_enrichment,
            llm_enrichment_call_count=len(missing),
        )

    async def _delete_derived_data(self) -> list[Path]:
        from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter

        await CogneeCompatibilityAdapter.prune_derived_data()
        targets = [self.paths.indexes / "lexical.sqlite3"]
        targets.extend((self.paths.indexes / "manifests").glob("*.json"))
        targets.extend((self.paths.cognee / "manifests").glob("*.json"))
        targets.extend((self.paths.cognee / "graphs").glob("*.json"))
        targets.extend((self.paths.cognee / "chunks").glob("*.jsonl"))
        deleted: list[Path] = []
        for target in targets:
            resolved = target.resolve(strict=False)
            self.paths.assert_within_root(resolved)
            if resolved.is_file():
                resolved.unlink()
                deleted.append(resolved)
        return deleted
