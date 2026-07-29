"""Coordinated lexical/vector projections and cross-store validation."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from paperos_core.adapters.models.client import LocalModelGatewayClient
from paperos_core.domain.canonical import CanonicalBundle
from paperos_core.errors import IndexStorageError
from paperos_core.indexes.lexical_store import LexicalStore
from paperos_core.indexes.manifest import IndexManifest
from paperos_core.indexes.vector_store import VectorStore
from paperos_core.paths import DataPaths


class IndexManager:
    def __init__(
        self,
        paths: DataPaths,
        model_client: LocalModelGatewayClient,
        *,
        embedding_model: str,
        embedding_dimensions: int,
    ) -> None:
        self.paths = paths
        self.lexical = LexicalStore(paths.indexes / "lexical.sqlite3")
        self.vector = VectorStore(
            paths.indexes / "vectors.sqlite3",
            model_client,
            model=embedding_model,
            dimensions=embedding_dimensions,
        )
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions

    async def index_bundle(
        self,
        bundle: CanonicalBundle,
        *,
        cognee_manifest: Path,
        cognee_object_ids: list[str],
        relation_count: int,
    ) -> tuple[IndexManifest, Path]:
        lexical_ids = self.lexical.upsert_bundle(bundle)
        vector_ids = await self.vector.upsert_bundle(bundle)
        manifest = IndexManifest(
            canonical_snapshot_id=bundle.snapshot.id,
            document_id=bundle.document.id,
            embedding_model=self.embedding_model,
            embedding_dimensions=self.embedding_dimensions,
            lexical_database=self.lexical.path,
            vector_database=self.vector.path,
            cognee_manifest=cognee_manifest,
            lexical_object_ids=lexical_ids,
            vector_object_ids=vector_ids,
            cognee_object_ids=cognee_object_ids,
            relation_count=relation_count,
        )
        path = self.paths.indexes / "manifests" / f"{bundle.snapshot.id}.json"
        _atomic_json(path, manifest.model_dump(mode="json"))
        self.verify(bundle, manifest)
        return manifest, path

    def verify(self, bundle: CanonicalBundle, manifest: IndexManifest) -> None:
        chunk_ids = {chunk.id for chunk in bundle.chunks}
        lexical_ids = set(self.lexical.object_ids(bundle.document.id))
        vector_ids = set(self.vector.object_ids(bundle.document.id))
        cognee_ids = set(manifest.cognee_object_ids)
        failures: dict[str, object] = {}
        if not chunk_ids.issubset(lexical_ids):
            failures["lexical_missing_chunks"] = sorted(chunk_ids - lexical_ids)
        if vector_ids != chunk_ids:
            failures["vector_chunk_mismatch"] = {
                "missing": sorted(chunk_ids - vector_ids),
                "extra": sorted(vector_ids - chunk_ids),
            }
        if not chunk_ids.issubset(cognee_ids):
            failures["cognee_missing_chunks"] = sorted(chunk_ids - cognee_ids)
        if failures:
            raise IndexStorageError(
                "Canonical ID consistency validation failed across derived stores.",
                affected=bundle.snapshot.id,
                details=failures,
            )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            Path(temporary).unlink()
        except FileNotFoundError:
            pass
