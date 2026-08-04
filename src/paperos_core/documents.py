"""Document listing, inspection, reprocessing, and logical deletion."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from paperos_core.adapters.cognee.repository import CogneeRepository
from paperos_core.errors import DocumentNotFoundError
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
    snapshot_ids: list[str]
    reference_count: int
    element_count: int
    raw_pdf_path: Path


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
        cognee: CogneeRepository,
    ) -> None:
        self.paths = paths
        self.canonical_repository = canonical_repository
        self.ingestion = ingestion
        self.rebuilder = rebuilder
        self.indexes = indexes
        self.cognee = cognee
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_tombstones (
                    document_id TEXT PRIMARY KEY,
                    deleted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

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
        latest = {}
        for bundle in self.canonical_repository.list_bundles():
            latest[bundle.document.id] = bundle
        result: list[DocumentSummary] = []
        for document_id, bundle in sorted(
            latest.items(), key=lambda item: (item[1].document.title, item[0])
        ):
            is_deleted = document_id in deleted
            if is_deleted and not include_deleted:
                continue
            source = self.ingestion.get_source(bundle.document.source_file_id)
            result.append(
                DocumentSummary(
                    document_id=document_id,
                    title=bundle.document.title,
                    source_file_id=source.id,
                    source_filename=source.original_filename,
                    canonical_snapshot_id=bundle.snapshot.id,
                    chunk_count=len(bundle.chunks),
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
        bundles = [
            bundle
            for bundle in self.canonical_repository.list_bundles()
            if bundle.document.id == document_id
        ]
        bundle = bundles[-1]
        source = self.ingestion.get_source(bundle.document.source_file_id)
        return DocumentDetail(
            **summaries[document_id].model_dump(),
            parse_run_id=bundle.snapshot.parse_run_id,
            snapshot_ids=[item.snapshot.id for item in bundles],
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
        lexical_count = len(self.indexes.lexical.object_ids(document_id))
        bundle = next(
            bundle
            for bundle in reversed(self.canonical_repository.list_bundles())
            if bundle.document.id == document_id
        )
        vector_count = await self.cognee.delete_document_vectors(bundle.snapshot.id)
        with self._connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO document_tombstones(document_id) VALUES (?)",
                (document_id,),
            )
        self.indexes.lexical.delete_document(document_id)
        return DocumentDeletionReport(
            document_id=document_id,
            removed_lexical_objects=lexical_count,
            removed_vector_objects=vector_count,
        )
