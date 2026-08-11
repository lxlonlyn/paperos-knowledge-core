"""Manage PaperOS FTS and its consistency with the chunk projection."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from paperos_core.domain.canonical import CanonicalBundle, Chunk
from paperos_core.errors import IndexStorageError
from paperos_core.indexes.lexical_store import LexicalStore
from paperos_core.indexes.manifest import IndexManifest
from paperos_core.paths import DataPaths
from paperos_core.storage.path_refs import DataPathCodec


class IndexManager:
    def __init__(
        self,
        paths: DataPaths,
    ) -> None:
        self.paths = paths
        self.path_codec = DataPathCodec(paths.root)
        self.lexical = LexicalStore(paths.indexes / "lexical.sqlite3")

    async def index_bundle(
        self,
        bundle: CanonicalBundle,
        *,
        chunks: list[Chunk],
    ) -> tuple[IndexManifest, Path]:
        lexical_ids = self.lexical.upsert_bundle(bundle, chunks=chunks)
        manifest = IndexManifest(
            canonical_snapshot_id=bundle.snapshot.id,
            document_id=bundle.document.id,
            lexical_database=self.lexical.path,
            lexical_object_ids=lexical_ids,
            chunk_projection_ids=[chunk.id for chunk in chunks],
        )
        path = self.paths.indexes / "manifests" / f"{bundle.snapshot.id}.json"
        payload = manifest.model_dump(mode="json")
        payload["lexical_database"] = self.path_codec.encode(manifest.lexical_database)
        _atomic_json(path, payload)
        self.verify(bundle, manifest, chunks=chunks)
        return manifest, path

    def verify(
        self,
        bundle: CanonicalBundle,
        manifest: IndexManifest,
        *,
        chunks: list[Chunk],
    ) -> None:
        chunk_ids = {chunk.id for chunk in chunks}
        lexical_ids = set(self.lexical.object_ids(bundle.document.id))
        failures: dict[str, object] = {}
        if not chunk_ids.issubset(lexical_ids):
            failures["lexical_missing_chunks"] = sorted(chunk_ids - lexical_ids)
        projected_ids = set(manifest.chunk_projection_ids)
        if projected_ids != chunk_ids:
            failures["chunk_projection_mismatch"] = {
                "missing": sorted(chunk_ids - projected_ids),
                "unexpected": sorted(projected_ids - chunk_ids),
            }
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
