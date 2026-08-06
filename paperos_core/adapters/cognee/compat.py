"""Centralized private Cognee API surface, bound to cognee 1.4.0.

PaperOS business code never imports Cognee internals. Everything that touches
Cognee's infrastructure, ORM models, pipeline tasks, or storage engines lives
here and is covered by ``tests/contract/test_cognee_compat.py``. The public
surface used elsewhere is limited to ``cognee.run_custom_pipeline``,
``cognee.search``/``cognee.recall``, and the ``LLMGateway``.

Only three kinds of private access are justified and centralized here:

1. writing structured DataPoints with exact canonical provenance
   (``add_data_points`` needs a registered relational Data item and ctx);
2. verifying and deleting derived projections and closing process-local
   engines, which Cognee does not expose publicly;
3. the narrow typed graph reader used to backtrack search hits to canonical
   source chunks (finite depth, explicit relation types only).
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

# The public ``cognee.run_custom_pipeline`` accepts Task objects built with the
# ``task`` decorator. The decorator lives in Cognee's pipeline internals, so it
# is re-exported here to keep every Cognee-internal import inside this module.
from cognee.modules.pipelines.tasks.task import task  # noqa: F401  # type: ignore[import-untyped]

from paperos_core.adapters.cognee.models import DataPointGraph
from paperos_core.domain.datapoints import cognee_uuid
from paperos_core.domain.documents import SourceFile
from paperos_core.errors import CogneeStorageError
from paperos_core.paths import DataPaths


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


@dataclass(frozen=True, slots=True)
class CogneeDatasetBinding:
    user_id: str
    dataset_id: str
    dataset_name: str
    data_id: str
    data_name: str
    pipeline_id: str
    pipeline_run_id: str
    pipeline_name: str
    provenance_backend: str = "pending"
    provenance_node_count: int = 0
    provenance_edge_count: int = 0


@dataclass(slots=True)
class PipelineItem:
    """Data item handed to ``cognee.run_custom_pipeline``.

    ``id``/``data_id`` point at the registered relational Data row so Cognee's
    provenance stamping and ``add_data_points`` attribute every node and edge
    to the PaperOS PDF. ``bundle`` carries the canonical document for the
    custom tasks.
    """

    id: UUID
    data_id: UUID
    bundle: Any
    source: SourceFile


class CogneeCompatibilityAdapter:
    def __init__(self, paths: DataPaths) -> None:
        self.paths = paths
        self.manifest_root = paths.cognee / "manifests"

    @staticmethod
    def reset_configuration_caches() -> None:
        """Make Cognee observe the environment installed by ``configure_cognee``."""
        from cognee.base_config import get_base_config  # type: ignore[import-untyped]
        from cognee.infrastructure.databases.graph.config import (  # type: ignore[import-untyped]
            get_graph_config,
        )
        from cognee.infrastructure.databases.relational.config import (  # type: ignore[import-untyped]
            get_relational_config,
        )
        from cognee.infrastructure.databases.vector.config import (  # type: ignore[import-untyped]
            get_vectordb_config,
        )
        from cognee.infrastructure.databases.vector.embeddings.config import (  # type: ignore[import-untyped]
            get_embedding_config,
        )
        from cognee.infrastructure.llm.config import (  # type: ignore[import-untyped]
            get_llm_config,
        )

        get_base_config.cache_clear()
        get_graph_config.cache_clear()
        # Cognee caches the first LLMConfig() forever; without clearing it,
        # PaperOS's configure_cognee environment changes are ignored after the
        # first LLM call in the process (health checks would keep using the
        # originally cached provider/model).
        get_llm_config.cache_clear()
        get_relational_config.cache_clear()
        get_vectordb_config.cache_clear()
        get_embedding_config.cache_clear()

    async def aclose(self) -> None:
        """Close Cognee's process-local database engines without deleting data."""
        from cognee.infrastructure.databases.cache.get_cache_engine import (  # type: ignore[import-untyped]
            close_cache_engine,
        )
        from cognee.infrastructure.databases.graph.get_graph_engine import (  # type: ignore[import-untyped]
            _create_graph_engine,
            get_graph_engine,
        )
        from cognee.infrastructure.databases.relational.create_relational_engine import (  # type: ignore[import-untyped]
            create_relational_engine,
        )
        from cognee.infrastructure.databases.relational.get_relational_engine import (  # type: ignore[import-untyped]
            get_relational_engine,
        )
        from cognee.infrastructure.databases.vector.create_vector_engine import (  # type: ignore[import-untyped]
            _create_vector_engine,
        )
        from cognee.infrastructure.databases.vector.get_vector_engine import (  # type: ignore[import-untyped]
            get_vector_engine_async,
        )

        if create_relational_engine.cache_info().currsize:
            relational = get_relational_engine()
            await relational.engine.dispose(close=True)
            create_relational_engine.cache_clear()
        await close_cache_engine()
        subprocess_sessions: list[Any] = []
        if _create_graph_engine.cache_info().currsize:
            graph = await get_graph_engine()
            graph_adapter = graph._engine()
            session = getattr(graph_adapter, "_session", None)
            if session is not None:
                subprocess_sessions.append(session)
            await graph_adapter.close()
            del graph_adapter, graph
        if _create_vector_engine.cache_info().currsize:
            vector = await get_vector_engine_async()
            vector_adapter = vector._engine()
            session = getattr(vector_adapter, "_session", None)
            if session is not None:
                subprocess_sessions.append(session)
            await vector_adapter.close()
            del vector_adapter, vector
        _create_graph_engine.cache_clear()
        _create_vector_engine.cache_clear()
        await _create_graph_engine.cache_await_closed()
        await _create_vector_engine.cache_await_closed()
        for session in subprocess_sessions:
            _close_subprocess_queues(session)
        subprocess_sessions.clear()
        gc.collect()

    async def ensure_dataset(self, dataset_name: str) -> Any:
        """Return the Cognee Dataset, creating it when missing (private path)."""
        from cognee.modules.data.methods import (  # type: ignore[import-untyped]
            load_or_create_datasets,
        )
        from cognee.modules.engine.operations.setup import (  # type: ignore[import-untyped]
            setup,
        )
        from cognee.modules.users.methods import (  # type: ignore[import-untyped]
            get_default_user,
        )

        selected = dataset_name.strip()
        if not selected:
            raise CogneeStorageError("Cognee dataset name must not be empty.")
        try:
            await setup()
            user = await get_default_user()
            datasets = await load_or_create_datasets([selected], [], user)
        except CogneeStorageError:
            raise
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee failed to resolve the dataset context: {exc}",
                affected=selected,
            ) from exc
        if not datasets:
            raise CogneeStorageError(
                "Cognee returned no dataset for the requested name.",
                affected=selected,
            )
        return datasets[0]

    async def register_data_item(
        self,
        *,
        dataset: Any,
        source: SourceFile,
        snapshot_id: str,
        document_id: str,
        title: str,
    ) -> UUID:
        """Register one immutable PaperOS PDF as a Cognee relational Data item."""
        from cognee.infrastructure.databases.relational import (  # type: ignore[import-untyped]
            get_relational_engine,
        )
        from cognee.modules.data.models import (  # type: ignore[import-untyped]
            Data,
            DatasetData,
        )
        from cognee.modules.users.methods import (  # type: ignore[import-untyped]
            get_default_user,
        )

        user = await get_default_user()
        attributes = {
            "name": source.original_filename,
            "label": title,
            "extension": "pdf",
            "mime_type": source.media_type,
            "original_extension": "pdf",
            "original_mime_type": source.media_type,
            "loader_engine": "paperos_canonical",
            "raw_data_location": str(source.storage_path),
            "original_data_location": str(source.storage_path),
            "owner_id": user.id,
            "tenant_id": user.tenant_id,
            "content_hash": source.sha256,
            "raw_content_hash": source.sha256,
            "external_metadata": {
                "paperos": {
                    "source_file_id": source.id,
                    "canonical_snapshot_id": snapshot_id,
                    "document_id": document_id,
                    "source_sha256": source.sha256,
                }
            },
            "pipeline_status": {"paperos_knowledge_ingestion": "registered"},
            "token_count": -1,
            "data_size": source.size_bytes,
            "importance_weight": 0.5,
        }
        from cognee.modules.data.methods import (  # type: ignore[import-untyped]
            get_unique_data_id,
        )

        data_id = await get_unique_data_id(source.sha256, user)
        engine = get_relational_engine()
        try:
            async with engine.get_async_session() as session:
                data_item = await session.get(Data, data_id)
                if data_item is None:
                    data_item = Data(id=data_id, **attributes)
                    session.add(data_item)
                else:
                    for key, value in attributes.items():
                        setattr(data_item, key, value)
                association = await session.get(
                    DatasetData,
                    {"dataset_id": dataset.id, "data_id": data_id},
                )
                if association is None:
                    session.add(DatasetData(dataset_id=dataset.id, data_id=data_id))
                await session.commit()
        except CogneeStorageError:
            raise
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee failed to register the PaperOS Data item: {exc}",
                affected=source.id,
            ) from exc
        return data_id

    async def add_data_points(
        self,
        data_points: list[Any],
        *,
        custom_edges: list[tuple[str, str, str, dict[str, Any]]],
        embed_triplets: bool,
        ctx: Any,
    ) -> list[Any]:
        from cognee.tasks.storage.add_data_points import (  # type: ignore[import-untyped]
            add_data_points,
        )

        try:
            return await add_data_points(
                data_points,
                custom_edges=custom_edges,
                embed_triplets=embed_triplets,
                ctx=ctx,
            )
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee failed to write structured DataPoints: {exc}",
                affected=self.paths.cognee,
            ) from exc

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

    async def vector_status(self) -> dict[str, object]:
        from cognee.infrastructure.databases.vector import (  # type: ignore[import-untyped]
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
        from cognee.infrastructure.databases.vector import (  # type: ignore[import-untyped]
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

    async def provenance_counts(
        self,
        *,
        dataset_id: UUID,
        data_id: UUID,
        pipeline_run_id: UUID,
    ) -> CogneeDatasetBinding:
        """Count provenance-attributed nodes/edges for one PaperOS PDF."""
        from cognee.infrastructure.databases.graph.get_graph_engine import (  # type: ignore[import-untyped]
            get_graph_engine,
        )
        from cognee.infrastructure.databases.provenance import (  # type: ignore[import-untyped]
            make_source_ref_key,
        )
        from cognee.infrastructure.databases.relational import (  # type: ignore[import-untyped]
            get_relational_engine,
        )
        from cognee.modules.graph.models import Edge, Node  # type: ignore[import-untyped]
        from sqlalchemy import func, select

        engine = get_relational_engine()
        ledger_nodes = 0
        ledger_edges = 0
        async with engine.get_async_session() as session:
            ledger_nodes = int(
                await session.scalar(
                    select(func.count()).select_from(Node).where(
                        Node.dataset_id == dataset_id,
                        Node.data_id == data_id,
                    )
                )
                or 0
            )
            ledger_edges = int(
                await session.scalar(
                    select(func.count()).select_from(Edge).where(
                        Edge.dataset_id == dataset_id,
                        Edge.data_id == data_id,
                    )
                )
                or 0
            )
        graph_nodes = 0
        graph_edges = 0
        try:
            graph = await get_graph_engine()
            source_ref = make_source_ref_key(dataset_id, data_id)
            graph_nodes = len(await graph.find_nodes_by_source_ref(source_ref))
            graph_edges = len(await graph.find_edges_by_source_ref(source_ref))
        except Exception:  # noqa: BLE001 - provider without source-ref support.
            graph_nodes = 0
            graph_edges = 0
        backend = "graph" if graph_nodes else "relational"
        node_count = graph_nodes or ledger_nodes
        edge_count = graph_edges or ledger_edges
        return CogneeDatasetBinding(
            user_id="",
            dataset_id=str(dataset_id),
            dataset_name="",
            data_id=str(data_id),
            data_name="",
            pipeline_id="",
            pipeline_run_id=str(pipeline_run_id),
            pipeline_name="paperos_knowledge_ingestion",
            provenance_backend=backend,
            provenance_node_count=node_count,
            provenance_edge_count=edge_count,
        )

    async def resolve_graph_nodes(
        self, cognee_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Read typed node properties for search-hit provenance backtracking."""
        if not cognee_ids:
            return {}
        from cognee.infrastructure.databases.graph.get_graph_engine import (  # type: ignore[import-untyped]
            get_graph_engine,
        )

        engine = await get_graph_engine()
        try:
            nodes = await engine.get_nodes(sorted({str(item) for item in cognee_ids}))
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee graph node readback failed: {exc}",
                affected=self.paths.cognee,
            ) from exc
        resolved: dict[str, dict[str, Any]] = {}
        for node in nodes:
            properties = _flatten_node(dict(node))
            node_id = properties.get("id")
            if node_id is not None:
                resolved[str(node_id)] = properties
        return resolved

    async def typed_traverse(
        self,
        seeds: list[CogneeVectorHit],
        *,
        depth: int,
        edge_types: set[str],
    ) -> list[CogneeTraversalEvidence]:
        """Narrow, finite-depth, typed-edge graph traversal with chunk provenance."""
        if not seeds or depth <= 0:
            return []
        from cognee.infrastructure.databases.graph.get_graph_engine import (  # type: ignore[import-untyped]
            get_graph_engine,
        )

        engine = await get_graph_engine()
        try:
            nodes, edges = await engine.get_neighborhood(
                [seed.cognee_id for seed in seeds],
                depth=depth,
                # Cognee 1.4's Kuzu variable-path implementation cannot bind the
                # edge-type list without triggering a parser assertion; traverse
                # in the engine and filter typed edges here. This remains a real,
                # bounded graph traversal and is portable to configured providers.
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


def _close_subprocess_queues(session: Any) -> None:
    """Release queue transports after Cognee has joined its DB worker."""
    for name in ("_req_q", "_resp_q"):
        queue = getattr(session, name, None)
        if queue is None:
            continue
        try:
            queue.close()
            queue.join_thread()
        except (OSError, ValueError):
            pass


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


def _flatten_node(node: dict[str, Any]) -> dict[str, Any]:
    properties = node.get("properties")
    if isinstance(properties, dict):
        return {**node, **properties}
    return node


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


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
