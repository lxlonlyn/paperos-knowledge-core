"""Official Cognee structured DataPoint writes, reads, and manifests."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from paperos_core.adapters.cognee.models import DataPointGraph
from paperos_core.domain.datapoints import cognee_uuid
from paperos_core.domain.provenance import RelationRecord
from paperos_core.errors import CogneeStorageError
from paperos_core.paths import DataPaths

CHUNK_VECTOR_COLLECTIONS = ("ChunkDataPoint_text",)
SUMMARY_VECTOR_COLLECTIONS = ("SummaryDataPoint_text",)
ENTITY_CLAIM_VECTOR_COLLECTIONS = (
    "EntityDataPoint_name",
    "EntityDataPoint_description",
    "ClaimDataPoint_text",
)
SEMANTIC_VECTOR_COLLECTIONS = (
    *CHUNK_VECTOR_COLLECTIONS,
    *ENTITY_CLAIM_VECTOR_COLLECTIONS,
    *SUMMARY_VECTOR_COLLECTIONS,
    "ConceptRelationDataPoint_description",
    "TripletDataPoint_text",
)
GRAPH_SEED_VECTOR_COLLECTIONS = (
    *ENTITY_CLAIM_VECTOR_COLLECTIONS,
    "ConceptRelationDataPoint_description",
    "TripletDataPoint_text",
)


@dataclass(frozen=True, slots=True)
class CogneeVectorHit:
    cognee_id: str
    canonical_id: str
    object_type: str
    score: float
    source_chunk_ids: tuple[str, ...]
    derived_from_ids: tuple[str, ...]
    canonical_snapshot_id: str | None


@dataclass(frozen=True, slots=True)
class CogneeTraversalEvidence:
    source_canonical_id: str
    target_canonical_id: str
    relation_type: str
    source_chunk_ids: tuple[str, ...]
    derived_from_ids: tuple[str, ...]
    score: float


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
            "mapping_version": "2",
            "canonical_snapshot_id": snapshot_id,
            "document_id": document_id,
            "node_count": len(graph.nodes),
            "relation_count": len(graph.relations),
            "canonical_to_cognee_id": graph.id_mapping,
            "vector_collections": _vector_collection_manifest(graph),
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

    async def verify_vector_indexes(self, graph: DataPointGraph) -> list[str]:
        """Read back every declared DataPoint vector from Cognee's vector engine."""
        from cognee.infrastructure.databases.vector import (  # type: ignore[import-untyped]
            get_vector_engine_async,
        )

        engine = await get_vector_engine_async()
        groups = _vector_groups(graph)
        missing: dict[str, list[str]] = {}
        for collection, nodes in groups.items():
            expected = {str(node.id): node.canonical_id for node in nodes}
            try:
                rows = await engine.retrieve(collection, list(expected))
            except Exception as exc:
                raise CogneeStorageError(
                    f"Cognee failed to read vector collection '{collection}': {exc}",
                    affected=self.paths.cognee / "vector",
                ) from exc
            actual = {str(row.id) for row in rows}
            absent = sorted(expected[node_id] for node_id in expected.keys() - actual)
            if absent:
                missing[collection] = absent
        if missing:
            raise CogneeStorageError(
                "Cognee vector readback is missing indexed DataPoints.",
                affected=self.paths.cognee / "vector",
                details={"missing": missing},
            )
        return sorted({node.canonical_id for nodes in groups.values() for node in nodes})

    async def search_vectors(
        self,
        queries: list[str],
        *,
        collections: tuple[str, ...],
        limit: int,
    ) -> list[CogneeVectorHit]:
        """Search Cognee/LanceDB and resolve vector hits through Cognee graph nodes."""
        if not queries or limit <= 0:
            return []
        from cognee.infrastructure.databases.graph.get_graph_engine import (
            get_graph_engine,
        )
        from cognee.infrastructure.databases.vector import (
            get_vector_engine_async,
        )

        vector_engine = await get_vector_engine_async()
        graph_engine = await get_graph_engine()
        try:
            vectors = await vector_engine.embed_data(queries)
            available = {
                collection
                for collection in collections
                if await vector_engine.has_collection(collection)
            }
            raw_hits: list[tuple[str, str, float]] = []
            for query_vector in vectors:
                for collection in sorted(available):
                    rows = await vector_engine.search(
                        collection,
                        query_text=None,
                        query_vector=query_vector,
                        limit=limit,
                        include_payload=False,
                    )
                    raw_hits.extend(
                        (str(row.id), collection, _distance_to_score(float(row.score)))
                        for row in rows
                    )
            nodes = await graph_engine.get_nodes(
                sorted({node_id for node_id, _, _ in raw_hits})
            )
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee vector retrieval failed: {exc}",
                affected=self.paths.cognee / "vector",
            ) from exc
        node_by_id = {
            str(properties.get("id")): properties
            for node in nodes
            if (properties := _flatten_node(dict(node))).get("id") is not None
        }
        best: dict[str, CogneeVectorHit] = {}
        for node_id, collection, score in raw_hits:
            hit_properties = node_by_id.get(node_id)
            if hit_properties is None:
                continue
            canonical_id = hit_properties.get("canonical_id")
            if not isinstance(canonical_id, str) or not canonical_id:
                continue
            hit = CogneeVectorHit(
                cognee_id=node_id,
                canonical_id=canonical_id,
                object_type=str(
                    hit_properties.get("type") or collection.split("_", 1)[0]
                ),
                score=score,
                source_chunk_ids=tuple(
                    _string_list(hit_properties.get("source_chunk_ids"))
                ),
                derived_from_ids=tuple(
                    _string_list(hit_properties.get("derived_from_ids"))
                ),
                canonical_snapshot_id=_optional_string(
                    hit_properties.get("canonical_snapshot_id")
                ),
            )
            existing = best.get(canonical_id)
            if existing is None or hit.score > existing.score:
                best[canonical_id] = hit
        return sorted(best.values(), key=lambda item: (-item.score, item.canonical_id))[
            :limit
        ]

    async def traverse(
        self,
        seeds: list[CogneeVectorHit],
        *,
        depth: int,
        edge_types: set[str],
    ) -> list[CogneeTraversalEvidence]:
        """Execute typed multi-hop traversal in Cognee and return chunk provenance."""
        if not seeds or depth <= 0:
            return []
        from cognee.infrastructure.databases.graph.get_graph_engine import (
            get_graph_engine,
        )

        engine = await get_graph_engine()
        try:
            nodes, edges = await engine.get_neighborhood(
                [seed.cognee_id for seed in seeds],
                depth=depth,
                # Cognee 1.4's Kuzu/Ladybug variable-path implementation cannot
                # bind the edge-type list without triggering a parser assertion.
                # Traverse in the engine, then filter its returned typed edges
                # below. This remains a real graph traversal and is portable to
                # every configured Cognee graph provider.
                edge_types=None,
            )
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee graph traversal failed: {exc}", affected=self.paths.cognee
            ) from exc
        node_properties = {
            str(node_id): _flatten_node(dict(properties))
            for node_id, properties in nodes
        }
        base_score = max(seed.score for seed in seeds)
        evidence: list[CogneeTraversalEvidence] = []
        for source_id, target_id, relation_type, raw_properties in edges:
            relation = str(relation_type)
            if relation not in edge_types:
                continue
            properties = (
                dict(raw_properties) if isinstance(raw_properties, dict) else {}
            )
            source = node_properties.get(str(source_id), {})
            target = node_properties.get(str(target_id), {})
            chunk_ids = list(
                dict.fromkeys(
                    [
                        *_string_list(properties.get("source_chunk_ids")),
                        *_string_list(source.get("source_chunk_ids")),
                        *_string_list(target.get("source_chunk_ids")),
                    ]
                )
            )
            if not chunk_ids:
                continue
            source_canonical_id = str(
                properties.get("canonical_source_id")
                or source.get("canonical_id")
                or source_id
            )
            target_canonical_id = str(
                properties.get("canonical_target_id")
                or target.get("canonical_id")
                or target_id
            )
            evidence.append(
                CogneeTraversalEvidence(
                    source_canonical_id=source_canonical_id,
                    target_canonical_id=target_canonical_id,
                    relation_type=relation,
                    source_chunk_ids=tuple(chunk_ids),
                    derived_from_ids=tuple(
                        dict.fromkeys(
                            [
                                *_string_list(properties.get("derived_from_ids")),
                                source_canonical_id,
                                target_canonical_id,
                            ]
                        )
                    ),
                    score=base_score,
                )
            )
        return evidence

    async def vector_status(self) -> dict[str, object]:
        from cognee.infrastructure.databases.vector import (
            get_vector_engine_async,
        )

        engine = await get_vector_engine_async()
        try:
            connection = await engine.get_connection()
            collections = sorted(await connection.table_names())
            counts: dict[str, int] = {}
            for collection in collections:
                table = await connection.open_table(collection)
                counts[collection] = int(await table.count_rows())
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee vector status failed: {exc}",
                affected=self.paths.cognee / "vector",
            ) from exc
        return {
            "backend": "cognee",
            "path": str(self.paths.cognee / "vector"),
            "collection_count": len(collections),
            "record_count": sum(counts.values()),
            "collections": counts,
            "dimensions": engine.embedding_engine.get_vector_size(),
        }

    async def delete_document_vectors(self, snapshot_id: str) -> int:
        """Delete one document's rebuildable vector projection from Cognee."""
        from cognee.infrastructure.databases.vector import (
            get_vector_engine_async,
        )

        manifest = self.read_manifest(snapshot_id)
        mapping = manifest.get("canonical_to_cognee_id")
        collections = manifest.get("vector_collections")
        if not isinstance(mapping, dict) or not isinstance(collections, dict):
            raise CogneeStorageError(
                "Cognee manifest lacks vector ownership metadata; rebuild derived data first.",
                affected=snapshot_id,
            )
        engine = await get_vector_engine_async()
        deleted_ids: set[str] = set()
        try:
            for collection, canonical_ids in collections.items():
                if not isinstance(collection, str) or not isinstance(canonical_ids, list):
                    continue
                ids = [
                    UUID(str(mapping[canonical_id]))
                    for canonical_id in canonical_ids
                    if canonical_id in mapping
                ]
                await engine.delete_data_points(collection, ids)
                deleted_ids.update(
                    canonical_id
                    for canonical_id in canonical_ids
                    if canonical_id in mapping
                )
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee failed to delete document vectors: {exc}",
                affected=snapshot_id,
            ) from exc
        return len(deleted_ids)

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


def _vector_groups(graph: DataPointGraph) -> dict[str, list[Any]]:
    groups: dict[str, list[Any]] = {}
    for node in graph.nodes:
        for field_name in node.metadata.get("index_fields", []):
            value = getattr(node, field_name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            groups.setdefault(f"{type(node).__name__}_{field_name}", []).append(node)
    return groups


def _vector_collection_manifest(graph: DataPointGraph) -> dict[str, list[str]]:
    return {
        collection: sorted(node.canonical_id for node in nodes)
        for collection, nodes in sorted(_vector_groups(graph).items())
    }


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _distance_to_score(distance: float) -> float:
    return 1.0 / (1.0 + max(distance, 0.0))


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
