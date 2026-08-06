"""SQLite FTS5 projection over canonical searchable objects."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from paperos_core.domain.canonical import CanonicalBundle, Chunk
from paperos_core.errors import IndexStorageError
from paperos_core.indexes.manifest import LEXICAL_INDEX_VERSION


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

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def upsert_bundle(
        self, bundle: CanonicalBundle, *, chunks: list[Chunk]
    ) -> list[str]:
        records = _records_for_bundle(bundle, chunks=chunks)
        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM lexical_records WHERE document_id = ?",
                    (bundle.document.id,),
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

    def object_ids(self, document_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT object_id FROM lexical_records WHERE document_id = ? ORDER BY object_id",
                (document_id,),
            ).fetchall()
        return [str(row["object_id"]) for row in rows]

    def search(
        self, query: str, *, limit: int = 20, document_id: str | None = None
    ) -> list[dict[str, object]]:
        if not query.strip():
            return []
        where_document = "AND r.document_id = ?" if document_id else ""
        parameters: tuple[object, ...] = (
            (query, document_id, limit) if document_id else (query, limit)
        )
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    f"""
                    SELECT r.*, bm25(lexical_fts) AS score
                    FROM lexical_fts
                    JOIN lexical_records r ON r.rowid = lexical_fts.rowid
                    WHERE lexical_fts MATCH ? {where_document}
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

    def delete_document(self, document_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM lexical_records WHERE document_id = ?", (document_id,))

    def status(self) -> dict[str, object]:
        with self._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM lexical_records").fetchone()[0]
            fts5 = bool(
                connection.execute("SELECT sqlite_compileoption_used('ENABLE_FTS5')").fetchone()[0]
            )
        return {"path": str(self.path), "record_count": count, "fts5": fts5}


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
            text=chunk.text,
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
