"""SQLite FTS5 projection over canonical searchable objects."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from paperos_core.domain.canonical import CanonicalBundle, Chunk
from paperos_core.errors import IndexStorageError
from paperos_core.indexes.manifest import LEXICAL_INDEX_VERSION
from paperos_core.ingestion.retrieval_text import effective_index_text


@dataclass(frozen=True, slots=True)
class LexicalRecord:
    object_id: str
    object_type: str
    document_id: str
    canonical_snapshot_id: str
    schema_version: str
    index_version: str
    field_name: str
    section_id: str | None
    section_path: str | None
    text: str


class LexicalStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            with connection:
                yield connection
        finally:
            connection.close()

    def upsert_bundle(
        self, bundle: CanonicalBundle, *, chunks: list[Chunk]
    ) -> list[str]:
        records = _records_for_bundle(bundle, chunks=chunks)
        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM lexical_records WHERE canonical_snapshot_id = ?",
                    (bundle.snapshot.id,),
                )
                connection.executemany(
                    """
                    INSERT INTO lexical_records (
                        object_id, object_type, document_id, canonical_snapshot_id,
                        schema_version, index_version, field_name, section_id,
                        section_path, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            item.object_id,
                            item.object_type,
                            item.document_id,
                            item.canonical_snapshot_id,
                            item.schema_version,
                            item.index_version,
                            item.field_name,
                            item.section_id,
                            item.section_path,
                            item.text,
                        )
                        for item in records
                    ],
                )
        except sqlite3.Error as exc:
            raise IndexStorageError(
                f"Unable to update SQLite FTS5 index: {exc}",
                affected=self.path,
            ) from exc
        return [item.object_id for item in records]

    def object_ids(self, snapshot_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT object_id FROM lexical_records "
                "WHERE canonical_snapshot_id = ? ORDER BY object_id",
                (snapshot_id,),
            ).fetchall()
        return [str(row["object_id"]) for row in rows]

    def search(
        self,
        query: str,
        *,
        active_snapshot_ids: set[str],
        limit: int = 20,
        allowed_document_ids: set[str] | None = None,
    ) -> list[dict[str, object]]:
        if (
            not query.strip()
            or not active_snapshot_ids
            or (allowed_document_ids is not None and not allowed_document_ids)
        ):
            return []
        selected_snapshots = sorted(active_snapshot_ids)
        snapshot_placeholders = ", ".join("?" for _ in selected_snapshots)
        selected_documents = sorted(allowed_document_ids or set())
        document_placeholders = ", ".join("?" for _ in selected_documents)
        where_document = (
            f"AND r.document_id IN ({document_placeholders})"
            if allowed_document_ids is not None
            else ""
        )
        parameters: tuple[object, ...] = (
            query,
            *selected_snapshots,
            *selected_documents,
            limit,
        )
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    f"""
                    SELECT r.*, bm25(lexical_fts) AS score
                    FROM lexical_fts
                    JOIN lexical_records r ON r.rowid = lexical_fts.rowid
                    WHERE lexical_fts MATCH ?
                      AND r.object_type = 'chunk'
                      AND r.canonical_snapshot_id IN ({snapshot_placeholders})
                      {where_document}
                    ORDER BY score
                    LIMIT ?
                    """,
                    parameters,
                ).fetchall()
        except sqlite3.Error as exc:
            raise IndexStorageError(
                f"SQLite FTS5 search failed: {exc}", affected=self.path
            ) from exc
        return [dict(row) for row in rows]

    def delete_snapshot(self, snapshot_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM lexical_records WHERE canonical_snapshot_id = ?",
                (snapshot_id,),
            )

    def status(self) -> dict[str, object]:
        with self._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM lexical_records").fetchone()[0]
            fts5 = bool(
                connection.execute("SELECT sqlite_compileoption_used('ENABLE_FTS5')").fetchone()[0]
            )
        return {"record_count": count, "fts5": fts5}


def _records_for_bundle(
    bundle: CanonicalBundle, *, chunks: list[Chunk]
) -> list[LexicalRecord]:
    snapshot = bundle.snapshot
    document = bundle.document
    records = [
        LexicalRecord(
            object_id=document.id,
            object_type="document",
            document_id=document.id,
            canonical_snapshot_id=snapshot.id,
            schema_version=document.schema_version,
            index_version=LEXICAL_INDEX_VERSION,
            field_name="title",
            section_id=None,
            section_path=None,
            text=document.title,
        )
    ]
    records.extend(
        LexicalRecord(
            object_id=chunk.id,
            object_type="chunk",
            document_id=document.id,
            canonical_snapshot_id=snapshot.id,
            schema_version=chunk.schema_version,
            index_version=LEXICAL_INDEX_VERSION,
            field_name="text",
            section_id=chunk.section_id,
            section_path=chunk.section_path,
            text=effective_index_text(chunk),
        )
        for chunk in chunks
    )
    records.extend(
        LexicalRecord(
            object_id=reference.id,
            object_type="reference",
            document_id=document.id,
            canonical_snapshot_id=snapshot.id,
            schema_version=reference.schema_version,
            index_version=LEXICAL_INDEX_VERSION,
            field_name="raw_text",
            section_id=None,
            section_path="References",
            text=reference.raw_text,
        )
        for reference in bundle.references
    )
    return records
