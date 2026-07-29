"""Local-embedding projection keyed by exact canonical chunk IDs."""

from __future__ import annotations

import math
import sqlite3
import struct
from pathlib import Path

from paperos_core.adapters.models.client import LocalModelGatewayClient
from paperos_core.domain.canonical import CanonicalBundle
from paperos_core.errors import IndexStorageError
from paperos_core.indexes.manifest import VECTOR_INDEX_VERSION


class VectorStore:
    def __init__(
        self,
        path: Path,
        client: LocalModelGatewayClient,
        *,
        model: str,
        dimensions: int,
    ) -> None:
        self.path = path
        self.client = client
        self.model = model
        self.dimensions = dimensions

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
                    CREATE TABLE IF NOT EXISTS vector_records (
                        object_id TEXT PRIMARY KEY,
                        object_type TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        canonical_snapshot_id TEXT NOT NULL,
                        schema_version TEXT NOT NULL,
                        index_version TEXT NOT NULL,
                        field_name TEXT NOT NULL,
                        model TEXT NOT NULL,
                        dimensions INTEGER NOT NULL,
                        text TEXT NOT NULL,
                        embedding BLOB NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS vector_document_idx
                        ON vector_records(document_id);
                    """
                )
        except sqlite3.Error as exc:
            raise IndexStorageError(
                f"Unable to initialize vector projection: {exc}", affected=self.path
            ) from exc

    async def upsert_bundle(self, bundle: CanonicalBundle, *, batch_size: int = 8) -> list[str]:
        self.initialize()
        chunks = bundle.chunks
        embeddings: list[list[float]] = []
        for offset in range(0, len(chunks), batch_size):
            batch = chunks[offset : offset + batch_size]
            embeddings.extend(
                await self.client.embed(
                    [chunk.text for chunk in batch],
                    expected_dimensions=self.dimensions,
                )
            )
        if len(embeddings) != len(chunks):
            raise IndexStorageError(
                "Embedding count does not match canonical chunk count.",
                affected=bundle.snapshot.id,
            )
        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM vector_records WHERE document_id = ?",
                    (bundle.document.id,),
                )
                connection.executemany(
                    """
                    INSERT INTO vector_records (
                        object_id, object_type, document_id, canonical_snapshot_id,
                        schema_version, index_version, field_name, model, dimensions,
                        text, embedding
                    ) VALUES (?, 'chunk', ?, ?, ?, ?, 'text', ?, ?, ?, ?)
                    """,
                    [
                        (
                            chunk.id,
                            bundle.document.id,
                            bundle.snapshot.id,
                            chunk.schema_version,
                            VECTOR_INDEX_VERSION,
                            self.model,
                            self.dimensions,
                            chunk.text,
                            struct.pack(f"<{len(vector)}f", *vector),
                        )
                        for chunk, vector in zip(chunks, embeddings, strict=True)
                    ],
                )
        except (sqlite3.Error, struct.error) as exc:
            raise IndexStorageError(
                f"Unable to update vector projection: {exc}", affected=self.path
            ) from exc
        return [chunk.id for chunk in chunks]

    def object_ids(self, document_id: str) -> list[str]:
        self.initialize()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT object_id FROM vector_records WHERE document_id = ? ORDER BY object_id",
                (document_id,),
            ).fetchall()
        return [str(row["object_id"]) for row in rows]

    def delete_document(self, document_id: str) -> None:
        self.initialize()
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM vector_records WHERE document_id = ?", (document_id,)
            )

    async def search(
        self, query: str, *, limit: int = 20, document_id: str | None = None
    ) -> list[dict[str, object]]:
        query_vector = (await self.client.embed([query], expected_dimensions=self.dimensions))[0]
        self.initialize()
        sql = "SELECT * FROM vector_records"
        parameters: tuple[object, ...] = ()
        if document_id:
            sql += " WHERE document_id = ?"
            parameters = (document_id,)
        with self._connect() as connection:
            rows = connection.execute(sql, parameters).fetchall()
        candidates: list[dict[str, object]] = []
        for row in rows:
            vector = struct.unpack(f"<{row['dimensions']}f", bytes(row["embedding"]))
            candidates.append(
                {
                    "object_id": row["object_id"],
                    "document_id": row["document_id"],
                    "text": row["text"],
                    "score": _cosine(query_vector, vector),
                }
            )

        def score(item: dict[str, object]) -> float:
            value = item["score"]
            return value if isinstance(value, float) else 0.0

        return sorted(candidates, key=score, reverse=True)[:limit]

    def status(self) -> dict[str, object]:
        self.initialize()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*), MIN(dimensions), MAX(dimensions) FROM vector_records"
            ).fetchone()
        return {
            "path": str(self.path),
            "record_count": row[0],
            "minimum_dimensions": row[1],
            "maximum_dimensions": row[2],
            "model": self.model,
        }


def _cosine(left: list[float], right: tuple[float, ...]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return numerator / (left_norm * right_norm) if left_norm and right_norm else 0.0
