"""Official Cognee structured DataPoint writes, reads, and manifests."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from paperos_core.adapters.cognee.models import DataPointGraph
from paperos_core.domain.datapoints import cognee_uuid
from paperos_core.domain.provenance import RelationRecord
from paperos_core.errors import CogneeStorageError
from paperos_core.paths import DataPaths


class CogneeRepository:
    def __init__(self, paths: DataPaths) -> None:
        self.paths = paths
        self.manifest_root = paths.cognee / "manifests"
        self.manifest_root.mkdir(parents=True, exist_ok=True)

    async def upsert_document_graph(
        self, graph: DataPointGraph, *, snapshot_id: str, document_id: str
    ) -> Path:
        from cognee.tasks.storage.add_data_points import (  # type: ignore[import-untyped]
            add_data_points,
        )

        custom_edges = [_custom_edge(relation, graph.id_mapping) for relation in graph.relations]
        try:
            written = await add_data_points(
                graph.nodes,
                custom_edges=custom_edges,
                embed_triplets=False,
            )
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee failed to write structured DataPoints: {exc}",
                affected=self.paths.cognee,
            ) from exc
        if len(written) != len(graph.nodes):
            raise CogneeStorageError(
                "Cognee returned an unexpected structured write count.",
                affected=snapshot_id,
                details={"expected": len(graph.nodes), "actual": len(written)},
            )
        manifest_path = self.manifest_root / f"{snapshot_id}.json"
        manifest = {
            "mapping_version": "1",
            "canonical_snapshot_id": snapshot_id,
            "document_id": document_id,
            "node_count": len(graph.nodes),
            "relation_count": len(graph.relations),
            "canonical_to_cognee_id": graph.id_mapping,
            "relations": [relation.model_dump(mode="json") for relation in graph.relations],
        }
        _atomic_json(manifest_path, manifest)
        return manifest_path

    async def get_datapoint(self, canonical_id: str) -> dict[str, Any]:
        from cognee.infrastructure.databases.graph.get_graph_engine import (  # type: ignore[import-untyped]
            get_graph_engine,
        )

        engine = await get_graph_engine()
        try:
            result = await engine.get_node(str(cognee_uuid(canonical_id)))
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee failed to read DataPoint: {exc}", affected=canonical_id
            ) from exc
        if result is None:
            raise CogneeStorageError("Cognee DataPoint does not exist.", affected=canonical_id)
        return dict(result)

    async def verify_graph(self, graph: DataPointGraph) -> None:
        failures: list[str] = []
        for node in graph.nodes:
            result = await self.get_datapoint(node.canonical_id)
            properties = _flatten_node(result)
            if properties.get("canonical_id") != node.canonical_id:
                failures.append(node.canonical_id)
        if failures:
            raise CogneeStorageError(
                "Cognee readback did not preserve canonical IDs.",
                affected=self.paths.cognee,
                details={"canonical_ids": failures},
            )

    def read_manifest(self, snapshot_id: str) -> dict[str, Any]:
        path = self.manifest_root / f"{snapshot_id}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise CogneeStorageError(
                f"Unable to read Cognee manifest: {exc}", affected=path
            ) from exc
        if not isinstance(payload, dict):
            raise CogneeStorageError("Invalid Cognee manifest.", affected=path)
        return payload


def _custom_edge(
    relation: RelationRecord, id_mapping: dict[str, str]
) -> tuple[str, str, str, dict[str, Any]]:
    source = id_mapping.get(relation.source_id, str(cognee_uuid(relation.source_id)))
    target = id_mapping.get(relation.target_id, str(cognee_uuid(relation.target_id)))
    return (
        source,
        target,
        relation.relation_type.value,
        {
            "canonical_source_id": relation.source_id,
            "canonical_target_id": relation.target_id,
            "source_chunk_ids": relation.source_chunk_ids,
            "derived_from_ids": relation.derived_from_ids,
        },
    )


def _flatten_node(node: dict[str, Any]) -> dict[str, Any]:
    properties = node.get("properties")
    if isinstance(properties, dict):
        return {**node, **properties}
    return node


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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
