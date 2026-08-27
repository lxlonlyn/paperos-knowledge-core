"""Document listing, inspection, reprocessing, and logical deletion."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.errors import CogneeStorageError, DocumentNotFoundError
from paperos_core.indexes.manager import IndexManager
from paperos_core.indexes.rebuild import DerivedDataRebuilder
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.service import IngestionService
from paperos_core.paths import DataPaths


class DocumentSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    source_file_id: str
    source_filename: str
    canonical_snapshot_id: str
    chunk_count: int
    section_count: int
    deleted: bool = False


class DocumentDetail(DocumentSummary):
    parse_run_id: str
    reference_count: int
    element_count: int
    raw_pdf_path: Path = Field(exclude=True)


class DocumentDeletionReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    status: str = "deleted"
    source_evidence_retained: bool = True
    removed_lexical_objects: int
    removed_vector_objects: int


class DocumentService:
    def __init__(
        self,
        paths: DataPaths,
        canonical_repository: CanonicalRepository,
        ingestion: IngestionService,
        rebuilder: DerivedDataRebuilder,
        indexes: IndexManager,
        cognee: CogneeCompatibilityAdapter,
    ) -> None:
        self.paths = paths
        self.canonical_repository = canonical_repository
        self.ingestion = ingestion
        self.rebuilder = rebuilder
        self.indexes = indexes
        self.cognee = cognee

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.paths.registry_db, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def deleted_document_ids(self) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT document_id FROM document_tombstones"
            ).fetchall()
        return {str(row["document_id"]) for row in rows}

    def list_documents(self, *, include_deleted: bool = False) -> list[DocumentSummary]:
        deleted = self.deleted_document_ids()
        active = {
            bundle.document.id: bundle
            for bundle in self.canonical_repository.list_active_bundles()
        }
        result: list[DocumentSummary] = []
        for document_id, bundle in sorted(
            active.items(), key=lambda item: (item[1].document.title, item[0])
        ):
            is_deleted = document_id in deleted
            if is_deleted and not include_deleted:
                continue
            source = self.ingestion.get_source(bundle.document.source_file_id)
            projection = self.canonical_repository.get_chunk_projection(
                bundle.snapshot.id
            )
            result.append(
                DocumentSummary(
                    document_id=document_id,
                    title=bundle.document.title,
                    source_file_id=source.id,
                    source_filename=source.original_filename,
                    canonical_snapshot_id=bundle.snapshot.id,
                    chunk_count=len(projection.chunks),
                    section_count=len(bundle.sections),
                    deleted=is_deleted,
                )
            )
        return result

    def inspect(self, document_id: str) -> DocumentDetail:
        summaries = {
            item.document_id: item
            for item in self.list_documents(include_deleted=True)
        }
        if document_id not in summaries:
            raise DocumentNotFoundError(
                f"Document '{document_id}' does not exist.", affected=document_id
            )
        snapshot_id = self.canonical_repository.active_snapshot_id(document_id)
        if snapshot_id is None:
            raise DocumentNotFoundError(
                f"Document '{document_id}' has no active canonical snapshot.",
                affected=document_id,
            )
        bundle = self.canonical_repository.get_bundle(snapshot_id)
        source = self.ingestion.get_source(bundle.document.source_file_id)
        return DocumentDetail(
            **summaries[document_id].model_dump(),
            parse_run_id=bundle.snapshot.parse_run_id,
            reference_count=len(bundle.references),
            element_count=len(bundle.elements),
            raw_pdf_path=source.storage_path,
        )

    async def reprocess(self, document_id: str) -> dict[str, object]:
        detail = self.inspect(document_id)
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM document_tombstones WHERE document_id = ?",
                (document_id,),
            )
        result = await self.ingestion.ingest_pdf_to_knowledge(detail.raw_pdf_path)
        return result.public_dict()

    async def delete(self, document_id: str) -> DocumentDeletionReport:
        self.inspect(document_id)
        snapshot_id = self.canonical_repository.active_snapshot_id(document_id)
        if snapshot_id is None:
            raise DocumentNotFoundError(
                f"Document '{document_id}' has no active canonical snapshot.",
                affected=document_id,
            )
        lexical_count = len(self.indexes.lexical.object_ids(snapshot_id))
        bundle = self.canonical_repository.get_bundle(snapshot_id)
        self.canonical_repository.tombstone_active_document(
            document_id,
            expected_snapshot_id=snapshot_id,
        )
        failures: list[Exception] = []
        vector_count = 0
        try:
            vector_count = await self.cognee.delete_document_data(bundle.snapshot.id)
        except Exception as exc:  # noqa: BLE001 - finish hidden local cleanup.
            failures.append(exc)
        try:
            self.indexes.lexical.delete_snapshot(snapshot_id)
        except Exception as exc:  # noqa: BLE001 - finish hidden local cleanup.
            failures.append(exc)
        if failures:
            raise CogneeStorageError(
                "Tombstoned document cleanup is incomplete and must be retried.",
                affected=document_id,
                details={"failure_count": len(failures), "retryable": True},
            ) from failures[0]
        return DocumentDeletionReport(
            document_id=document_id,
            removed_lexical_objects=lexical_count,
            removed_vector_objects=vector_count,
        )
