"""Destructive, scoped rebuild of derived Gate 4 data."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from paperos_core.adapters.cognee.pipeline import CogneePipeline
from paperos_core.indexes.manifest import IndexingReport
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.paths import DataPaths


class RebuildReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rebuilt_snapshot_ids: list[str]
    deleted_paths: list[Path]
    reports: list[IndexingReport]


class DerivedDataRebuilder:
    def __init__(
        self,
        paths: DataPaths,
        canonical_repository: CanonicalRepository,
        pipeline: CogneePipeline,
    ) -> None:
        self.paths = paths
        self.canonical_repository = canonical_repository
        self.pipeline = pipeline

    async def rebuild(self, snapshot_id: str | None = None) -> RebuildReport:
        selected = (
            [snapshot_id]
            if snapshot_id is not None
            else self.canonical_repository.list_snapshot_ids()
        )
        with sqlite3.connect(self.paths.registry_db) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='document_tombstones'"
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
        deleted = await self._delete_derived_data()
        reports: list[IndexingReport] = []
        for selected_id in selected:
            bundle = self.canonical_repository.get_bundle(selected_id)
            report, _ = await self.pipeline.ingest_bundle(bundle, rebuilt=True)
            reports.append(report)
        return RebuildReport(
            rebuilt_snapshot_ids=selected,
            deleted_paths=deleted,
            reports=reports,
        )

    async def _delete_derived_data(self) -> list[Path]:
        from cognee.modules.data.deletion import (  # type: ignore[import-untyped]
            prune_data,
            prune_system,
        )

        await prune_data()
        await prune_system(graph=True, vector=True, metadata=True, cache=True)
        targets = [
            self.paths.indexes / "lexical.sqlite3",
            # Remove pre-v2 duplicate vector projections during destructive rebuild.
            self.paths.indexes / "vectors.sqlite3",
        ]
        targets.extend((self.paths.indexes / "manifests").glob("*.json"))
        targets.extend((self.paths.cognee / "manifests").glob("*.json"))
        targets.extend((self.paths.cognee / "enrichment").glob("*.json"))
        targets.extend((self.paths.cache / "query").glob("*.json"))
        deleted: list[Path] = []
        for target in targets:
            resolved = target.resolve(strict=False)
            self.paths.assert_within_root(resolved)
            if resolved.is_file():
                resolved.unlink()
                deleted.append(resolved)
        return deleted
