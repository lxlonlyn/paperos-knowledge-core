"""Read-only corpus view over retained canonical source artifacts."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from dataclasses import dataclass, field
from typing import Literal

from paperos_core.domain.canonical import CanonicalBundle, Chunk, RerankSpan
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry
from paperos_core.paths import DataPaths
from paperos_core.retrieval.candidates import Candidate


@dataclass(slots=True)
class CorpusView:
    paths: DataPaths
    active_snapshot_ids: set[str]
    bundles: dict[str, CanonicalBundle]
    chunks: dict[str, Chunk]
    chunk_bundles: dict[str, CanonicalBundle]
    source_filenames: dict[str, str]
    work_id_by_document: dict[str, str] = field(default_factory=dict)
    document_ids_by_work: dict[str, set[str]] = field(default_factory=dict)
    work_titles: dict[str, str] = field(default_factory=dict)
    cited_work_ids_by_chunk: dict[str, set[str]] = field(default_factory=dict)
    rerank_spans_by_chunk: dict[str, list[RerankSpan]] = field(default_factory=dict)
    rerank_projection_versions: set[str] = field(default_factory=set)

    @classmethod
    def load(
        cls,
        paths: DataPaths,
        canonical_repository: CanonicalRepository,
        registry: SourceRegistry,
        scholarly_registry: ScholarlyRegistry | None = None,
    ) -> CorpusView:
        retained_bundles = canonical_repository.list_active_bundles()
        bundles = {bundle.document.id: bundle for bundle in retained_bundles}
        with closing(sqlite3.connect(paths.registry_db)) as connection, connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='document_tombstones'"
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
        retained_bundles = [bundle for bundle in retained_bundles if bundle.document.id in bundles]
        active_snapshot_ids = {
            bundle.snapshot.id for bundle in retained_bundles
        }
        projections = {
            bundle.snapshot.id: canonical_repository.get_chunk_projection(
                bundle.snapshot.id
            )
            for bundle in retained_bundles
        }
        chunks = {
            chunk.id: chunk
            for bundle in retained_bundles
            for chunk in projections[bundle.snapshot.id].chunks
        }
        chunk_bundles = {
            chunk.id: bundle
            for bundle in retained_bundles
            for chunk in projections[bundle.snapshot.id].chunks
        }
        rerank_spans_by_chunk: dict[str, list[RerankSpan]] = {}
        rerank_projection_versions: set[str] = set()
        for projection in projections.values():
            rerank_projection = projection.rerank_projection
            if rerank_projection is None:
                continue
            rerank_projection_versions.add(rerank_projection.projection_version)
            for span in rerank_projection.spans:
                rerank_spans_by_chunk.setdefault(span.parent_chunk_id, []).append(span)
        for spans in rerank_spans_by_chunk.values():
            spans.sort(key=lambda item: item.ordinal)
        source_filenames = {
            bundle.document.source_file_id: registry.get_source(
                bundle.document.source_file_id
            ).original_filename
            for bundle in bundles.values()
        }
        work_id_by_document: dict[str, str] = {}
        document_ids_by_work: dict[str, set[str]] = {}
        work_titles: dict[str, str] = {}
        cited_work_ids_by_chunk: dict[str, set[str]] = {}
        if scholarly_registry is not None:
            for work in scholarly_registry.list_works():
                work_titles[work.id] = work.title
            for chunk in chunks.values():
                for work_id in chunk.citation_work_ids:
                    canonical_work_id = scholarly_registry.canonicalize_work_id(
                        work_id
                    )
                    cited_work_ids_by_chunk.setdefault(chunk.id, set()).add(
                        canonical_work_id
                    )
            for document_id in bundles:
                document_work = scholarly_registry.work_for_document(document_id)
                if document_work is None:
                    continue
                work_id_by_document[document_id] = document_work.id
                document_ids_by_work.setdefault(document_work.id, set()).add(document_id)
                work_titles[document_work.id] = document_work.title
            for redirected_id, canonical_id in scholarly_registry.list_redirects().items():
                resolved_documents = document_ids_by_work.get(canonical_id)
                if resolved_documents:
                    document_ids_by_work[redirected_id] = set(resolved_documents)
        return cls(
            paths=paths,
            active_snapshot_ids=active_snapshot_ids,
            bundles=bundles,
            chunks=chunks,
            chunk_bundles=chunk_bundles,
            source_filenames=source_filenames,
            work_id_by_document=work_id_by_document,
            document_ids_by_work=document_ids_by_work,
            work_titles=work_titles,
            cited_work_ids_by_chunk=cited_work_ids_by_chunk,
            rerank_spans_by_chunk=rerank_spans_by_chunk,
            rerank_projection_versions=rerank_projection_versions,
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
        relation_types: list[str] | None = None,
        source_work_id: str | None = None,
        candidate_id: str | None = None,
    ) -> Candidate:
        chunk = self.chunks[chunk_id]
        bundle = self.chunk_bundles[chunk_id]
        resolved_source_work = source_work_id or self.work_id_by_document.get(chunk.document_id)
        return Candidate(
            id=candidate_id or chunk.id,
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
            relation_types=relation_types or [],
            source_work_id=resolved_source_work,
        )

    def filtered_document_ids(
        self,
        requested_document_ids: list[str] | None,
        dataset_name: str,
    ) -> set[str]:
        dataset_documents = {
            document_id
            for document_id, bundle in self.bundles.items()
            if bundle.snapshot.dataset_id == dataset_name
        }
        if requested_document_ids is None:
            return dataset_documents
        return dataset_documents.intersection(requested_document_ids)

    def snapshot_ids_for_documents(self, document_ids: set[str]) -> set[str]:
        return {
            bundle.snapshot.id
            for document_id, bundle in self.bundles.items()
            if document_id in document_ids
        }

    def document_ids_for_works(self, work_ids: list[str] | set[str]) -> set[str]:
        selected: set[str] = set()
        for work_id in work_ids:
            selected.update(self.document_ids_by_work.get(work_id, set()))
        return selected
