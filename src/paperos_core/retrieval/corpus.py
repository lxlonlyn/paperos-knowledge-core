"""Read-only corpus view over retained canonical source artifacts."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Literal

from paperos_core.domain.canonical import CanonicalBundle, Chunk
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.paths import DataPaths
from paperos_core.retrieval.candidates import Candidate


@dataclass(slots=True)
class CorpusView:
    paths: DataPaths
    bundles: dict[str, CanonicalBundle]
    chunks: dict[str, Chunk]
    chunk_bundles: dict[str, CanonicalBundle]
    source_filenames: dict[str, str]

    @classmethod
    def load(
        cls,
        paths: DataPaths,
        canonical_repository: CanonicalRepository,
        registry: SourceRegistry,
    ) -> CorpusView:
        retained_bundles = canonical_repository.list_bundles()
        bundles = {bundle.document.id: bundle for bundle in retained_bundles}
        with sqlite3.connect(paths.registry_db) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='document_tombstones'"
            ).fetchone()
            deleted = (
                {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT document_id FROM document_tombstones"
                    ).fetchall()
                }
                if exists
                else set()
            )
        bundles = {
            document_id: bundle
            for document_id, bundle in bundles.items()
            if document_id not in deleted
        }
        retained_bundles = [
            bundle
            for bundle in retained_bundles
            if bundle.document.id in bundles
        ]
        chunks = {
            chunk.id: chunk
            for bundle in retained_bundles
            for chunk in bundle.chunks
        }
        chunk_bundles = {
            chunk.id: bundle
            for bundle in retained_bundles
            for chunk in bundle.chunks
        }
        source_filenames = {
            bundle.document.source_file_id: registry.get_source(
                bundle.document.source_file_id
            ).original_filename
            for bundle in bundles.values()
        }
        return cls(
            paths=paths,
            bundles=bundles,
            chunks=chunks,
            chunk_bundles=chunk_bundles,
            source_filenames=source_filenames,
        )

    def candidate_for_chunk(
        self,
        chunk_id: str,
        *,
        channel: str,
        score: float,
        object_id: str | None = None,
        object_type: str = "chunk",
        knowledge_kind: Literal[
            "source_fact",
            "structured_relation",
            "system_inference",
            "user_confirmed",
        ] = "source_fact",
        derived_from_ids: list[str] | None = None,
    ) -> Candidate:
        chunk = self.chunks[chunk_id]
        bundle = self.chunk_bundles[chunk_id]
        return Candidate(
            id=chunk.id,
            object_id=object_id or chunk.id,
            object_type=object_type,
            document_id=chunk.document_id,
            source_file_id=bundle.document.source_file_id,
            source_filename=self.source_filenames[bundle.document.source_file_id],
            canonical_snapshot_id=bundle.snapshot.id,
            chunk_id=chunk.id,
            section_id=chunk.section_id,
            section_path=chunk.section_path,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            text=chunk.text,
            channels=[channel],
            channel_scores={channel: score},
            knowledge_kind=knowledge_kind,
            derived_from_ids=derived_from_ids or [],
        )

    def filtered_document_ids(
        self, requested_document_ids: list[str] | None
    ) -> set[str]:
        return (
            set(requested_document_ids)
            if requested_document_ids is not None
            else set(self.bundles)
        )
