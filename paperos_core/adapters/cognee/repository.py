"""Official Cognee structured DataPoint writes, reads, and manifests."""

from __future__ import annotations

import gc
import importlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import UUID

from paperos_core.adapters.cognee.models import DataPointGraph
from paperos_core.domain.datapoints import cognee_uuid
from paperos_core.domain.documents import SourceFile
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


class CogneeRepository:
    def __init__(self, paths: DataPaths) -> None:
        self.paths = paths
        self.manifest_root = paths.cognee / "manifests"

    async def aclose(self) -> None:
        """Close Cognee's process-local database engines without deleting data."""

        from cognee.infrastructure.databases.cache.get_cache_engine import (
            close_cache_engine,
        )
        from cognee.infrastructure.databases.graph.get_graph_engine import (
            _create_graph_engine,
            get_graph_engine,
        )
        from cognee.infrastructure.databases.relational.create_relational_engine import (
            create_relational_engine,
        )
        from cognee.infrastructure.databases.relational.get_relational_engine import (
            get_relational_engine,
        )
        from cognee.infrastructure.databases.vector.create_vector_engine import (
            _create_vector_engine,
        )
        from cognee.infrastructure.databases.vector.get_vector_engine import (
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

    async def _create_pipeline_context(
        self,
        *,
        dataset_name: str,
        source: SourceFile,
        snapshot_id: str,
        document_id: str,
        title: str,
    ) -> tuple[Any, CogneeDatasetBinding, Any, Any]:
        """Create Cognee's relational User/Dataset/Data/PipelineRun context."""
        from cognee.modules.data.methods import (  # type: ignore[import-untyped]
            create_authorized_dataset,
            get_authorized_dataset_by_name,
            get_unique_data_id,
        )
        from cognee.modules.engine.operations.setup import (  # type: ignore[import-untyped]
            setup,
        )
        from cognee.modules.pipelines.models import (  # type: ignore[import-untyped]
            PipelineContext,
        )
        from cognee.modules.pipelines.utils import (  # type: ignore[import-untyped]
            generate_pipeline_id,
        )
        from cognee.modules.users.methods import (  # type: ignore[import-untyped]
            get_default_user,
        )

        selected_dataset = dataset_name.strip()
        if not selected_dataset:
            raise CogneeStorageError("Cognee dataset name must not be empty.")
        try:
            await setup()
            user = await get_default_user()
            dataset = await get_authorized_dataset_by_name(
                selected_dataset, user, "write"
            )
            if dataset is None:
                dataset = await create_authorized_dataset(selected_dataset, user)
            data_id = await get_unique_data_id(source.sha256, user)
            data_item = await self._upsert_data_item(
                user=user,
                dataset=dataset,
                data_id=data_id,
                source=source,
                snapshot_id=snapshot_id,
                document_id=document_id,
                title=title,
            )
            pipeline_name = "paperos_knowledge_ingestion"
            pipeline_id = generate_pipeline_id(user.id, dataset.id, pipeline_name)
            pipeline_operations = importlib.import_module(
                "cognee.modules.pipelines.operations"
            )
            pipeline_run = await pipeline_operations.log_pipeline_run_start(
                pipeline_id,
                pipeline_name,
                dataset.id,
                [data_item],
            )
        except CogneeStorageError:
            raise
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee failed to establish Dataset/Data provenance context: {exc}",
                affected=selected_dataset,
            ) from exc
        ctx = PipelineContext(
            user=user,
            dataset=dataset,
            data_item=data_item,
            pipeline_run_id=pipeline_run.pipeline_run_id,
            pipeline_name=pipeline_name,
            extras={
                "paperos_snapshot_id": snapshot_id,
                "paperos_document_id": document_id,
                "paperos_source_file_id": source.id,
            },
        )
        binding = CogneeDatasetBinding(
            user_id=str(user.id),
            dataset_id=str(dataset.id),
            dataset_name=str(dataset.name),
            data_id=str(data_item.id),
            data_name=str(data_item.name),
            pipeline_id=str(pipeline_id),
            pipeline_run_id=str(pipeline_run.pipeline_run_id),
            pipeline_name=pipeline_name,
        )
        return ctx, binding, pipeline_id, data_item

    async def _upsert_data_item(
        self,
        *,
        user: Any,
        dataset: Any,
        data_id: Any,
        source: SourceFile,
        snapshot_id: str,
        document_id: str,
        title: str,
    ) -> Any:
        """Register one immutable PaperOS PDF as a Cognee relational Data item."""
        from cognee.infrastructure.databases.relational import (  # type: ignore[import-untyped]
            get_relational_engine,
        )
        from cognee.modules.data.models import (  # type: ignore[import-untyped]
            Data,
            DatasetData,
        )

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
        engine = get_relational_engine()
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
            await session.refresh(data_item)
            return data_item

    async def _record_pipeline_error(
        self, ctx: Any, pipeline_id: Any, data_item: Any, error: Exception
    ) -> None:
        try:
            pipeline_operations = importlib.import_module(
                "cognee.modules.pipelines.operations"
            )
            await pipeline_operations.log_pipeline_run_error(
                ctx.pipeline_run_id,
                pipeline_id,
                ctx.pipeline_name,
                ctx.dataset.id,
                [data_item],
                error,
            )
        except Exception:  # noqa: BLE001
            # Preserve the structured-write failure as the actionable error.
            return

    async def verify_dataset_binding(
        self, binding: CogneeDatasetBinding
    ) -> CogneeDatasetBinding:
        """Read Dataset/Data/PipelineRun and graph provenance back from Cognee."""
        from cognee.infrastructure.databases.graph.get_graph_engine import (  # type: ignore[import-untyped]
            get_graph_engine,
        )
        from cognee.infrastructure.databases.provenance import (  # type: ignore[import-untyped]
            make_source_ref_key,
        )
        from cognee.infrastructure.databases.relational import (
            get_relational_engine,
        )
        from cognee.modules.data.models import (
            Data,
            Dataset,
            DatasetData,
        )
        from cognee.modules.graph.models import Edge, Node  # type: ignore[import-untyped]
        from cognee.modules.pipelines.models import (
            PipelineRun,
            PipelineRunStatus,
        )
        from sqlalchemy import func, select

        dataset_uuid = UUID(binding.dataset_id)
        data_uuid = UUID(binding.data_id)
        run_uuid = UUID(binding.pipeline_run_id)
        engine = get_relational_engine()
        async with engine.get_async_session() as session:
            dataset = await session.get(Dataset, dataset_uuid)
            data_item = await session.get(Data, data_uuid)
            association = await session.get(
                DatasetData,
                {"dataset_id": dataset_uuid, "data_id": data_uuid},
            )
            completed_run = (
                await session.scalars(
                    select(PipelineRun).where(
                        PipelineRun.pipeline_run_id == run_uuid,
                        PipelineRun.dataset_id == dataset_uuid,
                        PipelineRun.status
                        == PipelineRunStatus.DATASET_PROCESSING_COMPLETED,
                    )
                )
            ).first()
            ledger_nodes = int(
                await session.scalar(
                    select(func.count()).select_from(Node).where(
                        Node.dataset_id == dataset_uuid,
                        Node.data_id == data_uuid,
                    )
                )
                or 0
            )
            ledger_edges = int(
                await session.scalar(
                    select(func.count()).select_from(Edge).where(
                        Edge.dataset_id == dataset_uuid,
                        Edge.data_id == data_uuid,
                    )
                )
                or 0
            )
        failures = []
        if dataset is None or str(dataset.name) != binding.dataset_name:
            failures.append("dataset")
        if data_item is None or str(data_item.name) != binding.data_name:
            failures.append("data_item")
        if association is None:
            failures.append("dataset_data")
        if completed_run is None:
            failures.append("pipeline_run")
        graph_nodes = 0
        graph_edges = 0
        try:
            graph = await get_graph_engine()
            source_ref = make_source_ref_key(dataset_uuid, data_uuid)
            graph_nodes = len(await graph.find_nodes_by_source_ref(source_ref))
            graph_edges = len(await graph.find_edges_by_source_ref(source_ref))
        except Exception:  # noqa: BLE001
            # Older/unmarked providers keep the authoritative provenance ledger
            # in the relational Node/Edge tables instead of graph properties.
            graph_nodes = 0
            graph_edges = 0
        provenance_backend = "graph" if graph_nodes else "relational"
        node_count = graph_nodes or ledger_nodes
        edge_count = graph_edges or ledger_edges
        if node_count == 0 or edge_count == 0:
            failures.append("node_edge_provenance")
        if failures:
            raise CogneeStorageError(
                "Cognee Dataset/Data/PipelineRun provenance readback failed.",
                affected=binding.dataset_name,
                details={"missing": failures},
            )
        return replace(
            binding,
            provenance_backend=provenance_backend,
            provenance_node_count=node_count,
            provenance_edge_count=edge_count,
        )

    async def list_datasets(self) -> list[dict[str, object]]:
        """List datasets visible to Cognee's single-user default principal."""
        from cognee.modules.data.methods import (
            get_dataset_data,
        )
        from cognee.modules.engine.operations.setup import setup
        from cognee.modules.users.methods import (
            get_default_user,
        )
        from cognee.modules.users.permissions.methods import (  # type: ignore[import-untyped]
            get_all_user_permission_datasets,
        )

        await setup()
        user = await get_default_user()
        datasets = await get_all_user_permission_datasets(user, "read")
        result: list[dict[str, object]] = []
        for dataset in sorted(datasets, key=lambda item: (item.name, str(item.id))):
            data_items = await get_dataset_data(dataset.id)
            result.append(
                {
                    "id": str(dataset.id),
                    "name": str(dataset.name),
                    "ownerId": str(dataset.owner_id),
                    "tenantId": (
                        str(dataset.tenant_id) if dataset.tenant_id is not None else None
                    ),
                    "createdAt": (
                        dataset.created_at.isoformat()
                        if dataset.created_at is not None
                        else None
                    ),
                    "updatedAt": (
                        dataset.updated_at.isoformat()
                        if dataset.updated_at is not None
                        else None
                    ),
                    "dataCount": len(data_items),
                }
            )
        return result

    async def upsert_document_graph(
        self,
        graph: DataPointGraph,
        *,
        snapshot_id: str,
        document_id: str,
        dataset_name: str,
        source: SourceFile,
        title: str,
    ) -> tuple[Path, CogneeDatasetBinding]:
        from cognee.tasks.storage.add_data_points import (  # type: ignore[import-untyped]
            add_data_points,
        )

        custom_edges = [_custom_edge(relation, graph.id_mapping) for relation in graph.relations]
        ctx, binding, pipeline_id, data_item = await self._create_pipeline_context(
            dataset_name=dataset_name,
            source=source,
            snapshot_id=snapshot_id,
            document_id=document_id,
            title=title,
        )
        try:
            written = await add_data_points(
                graph.nodes,
                custom_edges=custom_edges,
                embed_triplets=False,
                ctx=ctx,
            )
            if len(written) != len(graph.nodes):
                raise CogneeStorageError(
                    "Cognee returned an unexpected structured write count.",
                    affected=snapshot_id,
                    details={"expected": len(graph.nodes), "actual": len(written)},
                )
            pipeline_operations = importlib.import_module(
                "cognee.modules.pipelines.operations"
            )

            await pipeline_operations.log_pipeline_run_complete(
                ctx.pipeline_run_id,
                pipeline_id,
                ctx.pipeline_name,
                ctx.dataset.id,
                [data_item],
            )
        except Exception as exc:
            await self._record_pipeline_error(ctx, pipeline_id, data_item, exc)
            raise CogneeStorageError(
                f"Cognee failed to write structured DataPoints: {exc}",
                affected=self.paths.cognee,
            ) from exc
        binding = await self.verify_dataset_binding(binding)
        manifest_path = self.manifest_root / f"{snapshot_id}.json"
        manifest = {
            "mapping_version": "3",
            "canonical_snapshot_id": snapshot_id,
            "document_id": document_id,
            "dataset": {
                "id": binding.dataset_id,
                "name": binding.dataset_name,
                "owner_id": binding.user_id,
            },
            "data_item": {
                "id": binding.data_id,
                "name": binding.data_name,
                "source_file_id": source.id,
                "source_sha256": source.sha256,
            },
            "pipeline": {
                "id": binding.pipeline_id,
                "run_id": binding.pipeline_run_id,
                "name": binding.pipeline_name,
            },
            "provenance": {
                "backend": binding.provenance_backend,
                "node_count": binding.provenance_node_count,
                "edge_count": binding.provenance_edge_count,
            },
            "node_count": len(graph.nodes),
            "relation_count": len(graph.relations),
            "canonical_to_cognee_id": graph.id_mapping,
            "vector_collections": _vector_collection_manifest(graph),
            "relations": [relation.model_dump(mode="json") for relation in graph.relations],
        }
        _atomic_json(manifest_path, manifest)
        return manifest_path, binding

    async def get_datapoint(self, canonical_id: str) -> dict[str, Any]:
        from cognee.infrastructure.databases.graph.get_graph_engine import (
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
