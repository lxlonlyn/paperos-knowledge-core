"""SQLite FTS5 projection over canonical searchable objects."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from paperos_core.domain.canonical import CanonicalBundle
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS lexical_records (
                        object_id TEXT PRIMARY KEY,
                        object_type TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        canonical_snapshot_id TEXT NOT NULL,
                        schema_version TEXT NOT NULL,
                        index_version TEXT NOT NULL,
                        field_name TEXT NOT NULL,
                        section_id TEXT,
                        section_path TEXT,
                        text TEXT NOT NULL
                    );
                    CREATE VIRTUAL TABLE IF NOT EXISTS lexical_fts USING fts5(
                        object_id UNINDEXED,
                        text,
                        content='lexical_records',
                        content_rowid='rowid',
                        tokenize='unicode61'
                    );
                    CREATE TRIGGER IF NOT EXISTS lexical_records_ai AFTER INSERT ON lexical_records
                    BEGIN
                        INSERT INTO lexical_fts(rowid, object_id, text)
                        VALUES (new.rowid, new.object_id, new.text);
                    END;
                    CREATE TRIGGER IF NOT EXISTS lexical_records_ad AFTER DELETE ON lexical_records
                    BEGIN
                        INSERT INTO lexical_fts(lexical_fts, rowid, object_id, text)
                        VALUES ('delete', old.rowid, old.object_id, old.text);
                    END;
                    CREATE TRIGGER IF NOT EXISTS lexical_records_au AFTER UPDATE ON lexical_records
                    BEGIN
                        INSERT INTO lexical_fts(lexical_fts, rowid, object_id, text)
                        VALUES ('delete', old.rowid, old.object_id, old.text);
                        INSERT INTO lexical_fts(rowid, object_id, text)
                        VALUES (new.rowid, new.object_id, new.text);
                    END;
                    CREATE INDEX IF NOT EXISTS lexical_document_idx
                        ON lexical_records(document_id);
                    """
                )
        except sqlite3.Error as exc:
            raise IndexStorageError(
                f"Unable to initialize SQLite FTS5 index: {exc}",
                affected=self.path,
            ) from exc

    def upsert_bundle(self, bundle: CanonicalBundle) -> list[str]:
        self.initialize()
        records = _records_for_bundle(bundle)
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
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT object_id FROM lexical_records WHERE document_id = ? ORDER BY object_id",
                (document_id,),
            ).fetchall()
        return [str(row["object_id"]) for row in rows]

    def search(
        self, query: str, *, limit: int = 20, document_id: str | None = None
    ) -> list[dict[str, object]]:
        self.initialize()
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
        self.initialize()
        with self._connect() as connection:
            connection.execute("DELETE FROM lexical_records WHERE document_id = ?", (document_id,))

    def status(self) -> dict[str, object]:
        self.initialize()
        with self._connect() as connection:
            count = connection.execute("SELECT COUNT(*) FROM lexical_records").fetchone()[0]
            fts5 = bool(
                connection.execute("SELECT sqlite_compileoption_used('ENABLE_FTS5')").fetchone()[0]
            )
        return {"path": str(self.path), "record_count": count, "fts5": fts5}


def _records_for_bundle(bundle: CanonicalBundle) -> list[LexicalRecord]:
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
        for chunk in bundle.chunks
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
