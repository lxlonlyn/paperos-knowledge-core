"""Immutable canonical snapshot persistence and verification."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from paperos_core.domain.canonical import (
    CanonicalBundle,
    CanonicalSnapshot,
    Chunk,
    ChunkProjection,
    Document,
    Element,
    ReferenceEntry,
    Section,
)
from paperos_core.domain.ids import CHUNKING_VERSION, canonical_snapshot_id
from paperos_core.errors import CanonicalStorageError, CanonicalValidationError
from paperos_core.ingestion.validation import calculate_sha256
from paperos_core.paths import DataPaths
from paperos_core.storage.immutable import ImmutableConflictError, write_immutable_bytes
from paperos_core.storage.path_refs import DataPathCodec

_Model = TypeVar("_Model", bound=BaseModel)


class CanonicalRepository:
    def __init__(self, paths: DataPaths) -> None:
        self.path_codec = DataPathCodec(paths.root)
        self.paths = paths

    def chunk_store_path(self, snapshot_id: str) -> Path:
        """Path of the derived chunk store written by the Cognee pipeline."""
        path = (self.paths.cognee / "chunks" / f"{snapshot_id}.jsonl").resolve(
            strict=False
        )
        self.paths.assert_within_root(path)
        return path

    def save_chunks(self, snapshot_id: str, chunks: Sequence[Chunk]) -> Path:
        """Atomically persist chunks produced by the custom pipeline (derived)."""
        path = self.chunk_store_path(snapshot_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._jsonl_bytes(chunks)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".chunks-", dir=path.parent, delete=False
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
        except OSError as exc:
            raise CanonicalStorageError(
                f"Unable to persist derived chunk store: {exc}",
                affected=path,
            ) from exc
        finally:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
        return path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.paths.registry_db, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def snapshot_manifest_path(self, source_file_id: str, parse_run_id: str) -> Path:
        snapshot_id = canonical_snapshot_id(parse_run_id)
        root = (self.paths.canonical / source_file_id / snapshot_id).resolve(strict=False)
        self.paths.assert_within_root(root)
        return root / "manifest.json"

    def save_snapshot(self, bundle: CanonicalBundle) -> CanonicalBundle:
        snapshot = bundle.snapshot
        expected_manifest = self.snapshot_manifest_path(
            snapshot.source_file_id, snapshot.parse_run_id
        )
        if snapshot.manifest_path.resolve(strict=False) != expected_manifest:
            raise CanonicalValidationError(
                "Canonical snapshot manifest path is not the repository path.",
                affected=snapshot.manifest_path,
                details={"expected": str(expected_manifest)},
            )
        self._validate_bundle(bundle, chunks=[])
        root = expected_manifest.parent
        root.mkdir(parents=True, exist_ok=True)
        payloads = {
            "snapshot.json": self._json_bytes(snapshot),
            "document.json": self._json_bytes(bundle.document),
            "sections.jsonl": self._jsonl_bytes(bundle.sections),
            "elements.jsonl": self._jsonl_bytes(bundle.elements),
            "references.jsonl": self._jsonl_bytes(bundle.references),
            "warnings.json": json.dumps(
                bundle.warnings,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ).encode(),
        }
        files: list[dict[str, Any]] = []
        for name, content in payloads.items():
            path = root / name
            self._write_immutable(path, content)
            files.append(
                {
                    "path": name,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "size_bytes": len(content),
                }
            )
        manifest = {
            "schema_version": snapshot.schema_version,
            "id_version": snapshot.id_version,
            "pipeline_version": snapshot.pipeline_version,
            "canonical_snapshot_id": snapshot.id,
            "source_file_id": snapshot.source_file_id,
            "parse_run_id": snapshot.parse_run_id,
            "document_id": snapshot.document_id,
            "dataset_id": snapshot.dataset_id,
            "counts": {
                "sections": len(bundle.sections),
                "elements": len(bundle.elements),
                "references": len(bundle.references),
            },
            "files": files,
        }
        self._write_immutable(
            expected_manifest,
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode(),
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO canonical_snapshots (
                        id, source_file_id, parse_run_id, document_id, manifest_path,
                        created_at, schema_version, id_version, pipeline_version,
                        cleaning_version, classification_version,
                        reference_processing_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot.id,
                        snapshot.source_file_id,
                        snapshot.parse_run_id,
                        snapshot.document_id,
                        self.path_codec.encode(snapshot.manifest_path),
                        snapshot.created_at.isoformat(),
                        snapshot.schema_version,
                        snapshot.id_version,
                        snapshot.pipeline_version,
                        snapshot.cleaning_version,
                        snapshot.classification_version,
                        snapshot.reference_processing_version,
                    ),
                )
        except sqlite3.Error as exc:
            raise CanonicalStorageError(
                f"Unable to register canonical snapshot: {exc}",
                affected=self.paths.registry_db,
            ) from exc
        self.verify_snapshot(snapshot.id)
        return bundle

    def get_snapshot(self, snapshot_id: str) -> CanonicalSnapshot:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM canonical_snapshots WHERE id = ?", (snapshot_id,)
            ).fetchone()
        if row is None:
            raise CanonicalStorageError(
                f"CanonicalSnapshot '{snapshot_id}' does not exist.",
                affected=snapshot_id,
            )
        snapshot_path = self.path_codec.decode(str(row["manifest_path"])).parent / "snapshot.json"
        return self._read_model(snapshot_path, CanonicalSnapshot)

    def get_bundle(self, snapshot_id: str) -> CanonicalBundle:
        snapshot = self.get_snapshot(snapshot_id)
        root = snapshot.manifest_path.parent
        warnings_payload = json.loads((root / "warnings.json").read_text(encoding="utf-8"))
        if not isinstance(warnings_payload, list) or not all(
            isinstance(item, str) for item in warnings_payload
        ):
            raise CanonicalValidationError(
                "Canonical warnings artifact is invalid.",
                affected=root / "warnings.json",
            )
        return CanonicalBundle(
            snapshot=snapshot,
            document=self._read_model(root / "document.json", Document),
            sections=self._read_jsonl(root / "sections.jsonl", Section),
            elements=self._read_jsonl(root / "elements.jsonl", Element),
            references=self._read_jsonl(root / "references.jsonl", ReferenceEntry),
            warnings=warnings_payload,
        )

    def get_chunk_projection(self, snapshot_id: str) -> ChunkProjection:
        """Load the derived chunk projection for one canonical snapshot."""
        self.get_snapshot(snapshot_id)
        derived = self.chunk_store_path(snapshot_id)
        chunks = self._read_jsonl(derived, Chunk) if derived.is_file() else []
        return ChunkProjection(
            snapshot_id=snapshot_id,
            chunking_version=CHUNKING_VERSION,
            chunks=chunks,
        )

    def list_snapshot_ids(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id FROM canonical_snapshots ORDER BY created_at, id"
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def list_latest_snapshot_ids(self) -> list[str]:
        """Return the current canonical snapshot for every Document."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT current.id
                FROM canonical_snapshots AS current
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM canonical_snapshots AS newer
                    WHERE newer.document_id = current.document_id
                      AND (
                          newer.created_at > current.created_at
                          OR (
                              newer.created_at = current.created_at
                              AND newer.id > current.id
                          )
                      )
                )
                ORDER BY current.created_at, current.id
                """
            ).fetchall()
        return [str(row["id"]) for row in rows]

    def list_bundles(self) -> list[CanonicalBundle]:
        return [self.get_bundle(snapshot_id) for snapshot_id in self.list_snapshot_ids()]

    def verify_snapshot(self, snapshot_id: str) -> None:
        snapshot = self.get_snapshot(snapshot_id)
        try:
            manifest = json.loads(snapshot.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CanonicalValidationError(
                f"Unable to read canonical manifest: {exc}",
                affected=snapshot.manifest_path,
            ) from exc
        if manifest.get("canonical_snapshot_id") != snapshot.id:
            raise CanonicalValidationError(
                "Canonical manifest identity does not match its registry record.",
                affected=snapshot.manifest_path,
            )
        for entry in manifest.get("files", []):
            path = snapshot.manifest_path.parent / entry["path"]
            if (
                not path.is_file()
                or path.stat().st_size != entry["size_bytes"]
                or calculate_sha256(path) != entry["sha256"]
            ):
                raise CanonicalValidationError(
                    "Canonical artifact checksum verification failed.",
                    affected=path,
                )
        bundle = self.get_bundle(snapshot_id)
        projection = self.get_chunk_projection(snapshot_id)
        self._validate_bundle(bundle, chunks=projection.chunks)

    def _validate_bundle(
        self, bundle: CanonicalBundle, *, chunks: list[Chunk] | None = None
    ) -> None:
        snapshot = bundle.snapshot
        document = bundle.document
        if document.id != snapshot.document_id:
            raise CanonicalValidationError(
                "Canonical Document ID does not match its snapshot.",
                affected=snapshot.id,
            )
        if (
            document.source_file_id != snapshot.source_file_id
            or document.parse_run_id != snapshot.parse_run_id
            or document.canonical_snapshot_id != snapshot.id
        ):
            raise CanonicalValidationError(
                "Canonical Document provenance does not match its snapshot.",
                affected=snapshot.id,
            )
        section_ids = {section.id for section in bundle.sections}
        element_ids = {element.id for element in bundle.elements}
        if len(section_ids) != len(bundle.sections) or len(element_ids) != len(bundle.elements):
            raise CanonicalValidationError(
                "Canonical object IDs are not unique.", affected=snapshot.id
            )
        for element in bundle.elements:
            if element.section_id is not None and element.section_id not in section_ids:
                raise CanonicalValidationError(
                    "Canonical Element references an unknown Section.",
                    affected=element.id,
                )
            if element.source_span is None:
                raise CanonicalValidationError(
                    "Canonical Element is missing parser provenance.",
                    affected=element.id,
                )
        for chunk in chunks or []:
            if not set(chunk.element_ids) <= element_ids:
                raise CanonicalValidationError(
                    "Canonical Chunk references an unknown Element.",
                    affected=chunk.id,
                )
        for reference in bundle.references:
            if (
                reference.source_element_id is not None
                and reference.source_element_id not in element_ids
            ):
                raise CanonicalValidationError(
                    "ReferenceEntry references an unknown source Element.",
                    affected=reference.id,
                )

    def _write_immutable(self, path: Path, content: bytes) -> None:
        self.paths.assert_within_root(path)
        try:
            write_immutable_bytes(path, content)
        except ImmutableConflictError as exc:
            raise CanonicalStorageError(
                "Immutable canonical artifact already exists with different content.",
                affected=path,
            ) from exc
        except OSError as exc:
            raise CanonicalStorageError(
                f"Unable to persist immutable canonical artifact: {exc}",
                affected=path,
            ) from exc

    def _portable_payload(self, model: BaseModel) -> dict[str, Any]:
        payload = model.model_dump(mode="json")
        if isinstance(model, CanonicalSnapshot):
            payload["manifest_path"] = self.path_codec.encode(model.manifest_path)
        elif isinstance(model, Element) and model.asset_path is not None:
            payload["asset_path"] = self.path_codec.encode(model.asset_path)
        return payload

    def _runtime_payload(
        self, payload: dict[str, Any], model: type[BaseModel]
    ) -> dict[str, Any]:
        if issubclass(model, CanonicalSnapshot):
            payload["manifest_path"] = str(
                self.path_codec.decode(str(payload["manifest_path"]))
            )
        elif issubclass(model, Element) and payload.get("asset_path") is not None:
            payload["asset_path"] = str(
                self.path_codec.decode(str(payload["asset_path"]))
            )
        return payload

    def _json_bytes(self, model: BaseModel) -> bytes:
        return json.dumps(
            self._portable_payload(model),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        ).encode()

    def _jsonl_bytes(self, models: Sequence[BaseModel]) -> bytes:
        return (
            "\n".join(
                json.dumps(
                    self._portable_payload(model),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                for model in models
            )
            + ("\n" if models else "")
        ).encode()

    def _read_model(self, path: Path, model: type[_Model]) -> _Model:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise TypeError("canonical model payload must be an object")
            return model.model_validate(self._runtime_payload(payload, model))
        except (OSError, UnicodeDecodeError, TypeError, ValueError) as exc:
            raise CanonicalValidationError(
                f"Unable to validate canonical artifact: {exc}",
                affected=path,
            ) from exc

    def _read_jsonl(self, path: Path, model: type[_Model]) -> list[_Model]:
        try:
            result: list[_Model] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError("canonical JSONL entry must be an object")
                result.append(model.model_validate(self._runtime_payload(payload, model)))
            return result
        except (OSError, UnicodeDecodeError, TypeError, ValueError) as exc:
            raise CanonicalValidationError(
                f"Unable to validate canonical JSONL artifact: {exc}",
                affected=path,
            ) from exc
