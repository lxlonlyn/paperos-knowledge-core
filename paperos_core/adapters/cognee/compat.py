"""Centralized private Cognee API surface, bound to cognee 1.4.0.

PaperOS business code never imports Cognee internals. Everything that touches
Cognee's infrastructure, ORM models, pipeline tasks, or storage engines lives
here and is checked by the real-case acceptance entry. The public
surface used elsewhere is limited to ``cognee.run_custom_pipeline``,
``cognee.search``/``cognee.recall``, and the ``LLMGateway``.

Only narrowly version-locked private access is justified and centralized here:

1. writing structured DataPoints with exact canonical provenance
   (``add_data_points`` needs a registered relational Data item and ctx);
2. verifying and deleting derived projections and closing process-local
   engines, which Cognee does not expose publicly;
3. Cognee 1.4.0 retrieval fallbacks for custom DataPoint collection selection,
   canonical provenance readback, and finite-depth typed graph traversal.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, UUID, uuid5

from cognee.infrastructure.engine import (  # type: ignore[import-untyped]
    DataPoint as DataPoint,  # noqa: PLC0414
)

# The public ``cognee.run_custom_pipeline`` accepts Task objects built with the
# ``task`` decorator. The decorator lives in Cognee's pipeline internals, so it
# is re-exported here to keep every Cognee-internal import inside this module.
from cognee.modules.pipelines.tasks.task import (  # type: ignore[import-untyped]
    task as task,  # noqa: PLC0414
)

from paperos_core.domain.documents import SourceFile
from paperos_core.domain.provenance import SEMANTIC_RELATION_TYPES
from paperos_core.errors import CogneeStorageError
from paperos_core.paths import DataPaths

if TYPE_CHECKING:
    from paperos_core.adapters.cognee.models import DataPointGraph


# Cognee 1.4.0 public search cannot select custom-pipeline DataPoint
# collections. Collection names remain private to this compatibility boundary.
_COGNEE_1_4_RETRIEVAL_COLLECTIONS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "PAPEROS_CHUNKS": (("ChunkDataPoint_text", "text", "ChunkDataPoint"),),
}


def cognee_uuid(canonical_id: str, *, mapping_version: str = "1") -> UUID:
    return uuid5(NAMESPACE_URL, f"paperos:cognee:{mapping_version}:{canonical_id}")


def resolve_cognee_tokenizer() -> Any:
    """Resolve the tokenizer Cognee would use for the configured embedding model."""
    from cognee.infrastructure.databases.vector.embeddings.config import (  # type: ignore[import-untyped]
        get_embedding_config,
    )
    from cognee.infrastructure.llm.tokenizer.resolver import (  # type: ignore[import-untyped]
        resolve_embedding_tokenizer,
    )

    config = get_embedding_config()
    return resolve_embedding_tokenizer(
        provider=config.embedding_provider,
        model=config.embedding_model,
        max_completion_tokens=config.embedding_max_completion_tokens,
        huggingface_tokenizer=config.huggingface_tokenizer,
    )


@dataclass(frozen=True, slots=True)
class CogneeVectorHit:
    cognee_id: str
    canonical_id: str
    object_type: str
    text: str
    score: float
    source_chunk_ids: tuple[str, ...]
    derived_from_ids: tuple[str, ...]
    canonical_snapshot_id: str | None


@dataclass(frozen=True, slots=True)
class CogneeSemanticRelation:
    source_canonical_id: str
    target_canonical_id: str
    relation_type: str
    grounded_object_ids: tuple[str, ...]
    source_chunk_ids: tuple[str, ...]
    derived_from_ids: tuple[str, ...]
    score: float


@dataclass(frozen=True, slots=True)
class CogneeIncomingRelation:
    """One bounded incoming typed edge onto an exact canonical target."""

    source_canonical_id: str
    target_canonical_id: str
    relation_type: str
    source_chunk_ids: tuple[str, ...]
    derived_from_ids: tuple[str, ...]
    roles: tuple[str, ...]
    source_work_id: str | None
    source_document_id: str | None
    text: str
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

    id: UUID | None = None
    data_id: UUID | None = None
    bundle: Any = None
    source: SourceFile | None = None


class CogneeCompatibilityAdapter:
    def __init__(self, paths: DataPaths) -> None:
        self.paths = paths
        self.manifest_root = paths.cognee / "manifests"
        self.retrieval_fallback_types_used: set[str] = set()
        self.last_semantic_relation_trace: dict[str, int] = {
            "raw_neighbor_nodes": 0,
            "filtered_grounded_entities": 0,
            "filtered_semantic_relations": 0,
            "missing_relation_source_provenance": 0,
        }

    async def _dataset_scope(self, dataset_name: str) -> Any:
        """Return Cognee's dataset/user context manager for scoped readback."""
        from cognee.context_global_variables import (  # type: ignore[import-untyped]
            set_database_global_context_variables,
        )
        from cognee.modules.data.methods import (  # type: ignore[import-untyped]
            get_authorized_existing_datasets,
        )
        from cognee.modules.engine.operations.setup import setup  # type: ignore[import-untyped]
        from cognee.modules.users.methods import get_default_user  # type: ignore[import-untyped]

        await setup()
        user = await get_default_user()
        datasets = await get_authorized_existing_datasets([dataset_name], "read", user)
        if len(datasets) != 1:
            raise CogneeStorageError(
                "Cognee dataset does not resolve uniquely for scoped readback.",
                affected=dataset_name,
            )
        dataset = datasets[0]
        return set_database_global_context_variables(dataset.id, dataset.owner_id)

    @staticmethod
    def reset_configuration_caches() -> None:
        """Clear Cognee-owned settings caches (mainly for isolated tests)."""
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
        from cognee.infrastructure.databases.vector.embeddings.config import (
            get_embedding_config,
        )
        from cognee.infrastructure.llm.config import (  # type: ignore[import-untyped]
            get_llm_config,
        )

        get_base_config.cache_clear()
        get_graph_config.cache_clear()
        get_llm_config.cache_clear()
        get_relational_config.cache_clear()
        get_vectordb_config.cache_clear()
        get_embedding_config.cache_clear()

    @staticmethod
    def runtime_config_snapshot() -> dict[str, Any]:
        """Read Cognee's resolved settings without returning any credentials."""
        from cognee.infrastructure.databases.graph.config import (
            get_graph_config,
        )
        from cognee.infrastructure.databases.relational.config import (
            get_relational_config,
        )
        from cognee.infrastructure.databases.vector.config import (
            get_vectordb_config,
        )
        from cognee.infrastructure.databases.vector.embeddings.config import (
            get_embedding_config,
        )
        from cognee.infrastructure.llm.config import (
            get_llm_config,
        )

        llm = get_llm_config()
        embedding = get_embedding_config()
        relational = get_relational_config()
        vector = get_vectordb_config()
        graph = get_graph_config()
        return {
            "llm_provider": llm.llm_provider,
            "llm_model": llm.llm_model,
            "llm_endpoint": llm.llm_endpoint,
            "embedding_provider": embedding.embedding_provider,
            "embedding_model": embedding.embedding_model,
            "embedding_endpoint": embedding.embedding_endpoint,
            "embedding_dimensions": embedding.embedding_dimensions,
            "embedding_max_tokens": embedding.embedding_max_completion_tokens,
            "db_provider": relational.db_provider,
            "db_path": relational.db_path,
            "vector_db_provider": vector.vector_db_provider,
            "vector_db_url": vector.vector_db_url,
            "graph_database_provider": graph.graph_database_provider,
            "graph_file_path": graph.graph_file_path,
        }

    @staticmethod
    async def test_llm_connection() -> None:
        """Delegate provider-specific validation to Cognee."""
        from cognee.infrastructure.llm.utils import (  # type: ignore[import-untyped]
            test_llm_connection,
        )

        await test_llm_connection()

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
        # Cognee telemetry owns a process-wide aiohttp session but exposes no
        # public shutdown hook in 1.4.0. Closing it here prevents connector
        # leaks when PaperOS exits after either a successful run or a failed
        # real-case assertion. This private lifecycle access is intentionally
        # kept inside the compatibility boundary.
        from cognee.shared import utils as cognee_utils  # type: ignore[import-untyped]

        telemetry_session = getattr(cognee_utils, "_telemetry_session", None)
        if telemetry_session is not None and not telemetry_session.closed:
            await telemetry_session.close()
        cognee_utils._telemetry_session = None
        cognee_utils._telemetry_session_loop = None
        gc.collect()

    async def ensure_dataset(self, dataset_name: str) -> Any:
        """Return the Cognee Dataset, creating it when missing (private path)."""
        from cognee.modules.data.methods import (
            load_or_create_datasets,
        )
        from cognee.modules.engine.operations.setup import (
            setup,
        )
        from cognee.modules.users.methods import (
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
        from cognee.modules.users.methods import (
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
        from cognee.modules.data.methods import (
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
        return UUID(str(data_id))

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
            result = await add_data_points(
                data_points,
                custom_edges=custom_edges,
                embed_triplets=embed_triplets,
                ctx=ctx,
            )
            return list(result)
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

    async def get_datapoint(
        self,
        canonical_id: str,
        *,
        dataset_name: str | None = None,
    ) -> dict[str, Any]:
        if dataset_name is not None:
            async with await self._dataset_scope(dataset_name):
                return await self.get_datapoint(canonical_id)
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

    async def vector_status(self, *, dataset_name: str | None = None) -> dict[str, object]:
        if dataset_name is not None:
            async with await self._dataset_scope(dataset_name):
                return await self.vector_status()
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
            "collection_count": len(collections),
            "record_count": sum(counts.values()),
            "collections": counts,
            "dimensions": engine.embedding_engine.get_vector_size(),
        }

    async def search_datapoint_vectors(
        self,
        query: str,
        *,
        dataset_name: str,
        search_type: str,
        canonical_ids: dict[str, str],
        top_k: int,
    ) -> list[CogneeVectorHit]:
        """Search PaperOS DataPoint collections in one Cognee dataset.

        Cognee 1.4's public ``CHUNKS`` retriever is bound to its built-in
        ``DocumentChunk_text`` collection.  A custom pipeline correctly creates
        collections such as ``ChunkDataPoint_text`` instead, but the public
        search API cannot name them.  Keep this version-specific vector-engine
        access here and return only normalized, provenance-bearing hits.

        Each collection tuple contains ``(collection_name, text_field,
        object_type)``.  ``ScoredResult.score`` is a cosine distance, so the
        normalized score is monotonic with lower distance being better.
        """
        if top_k <= 0:
            return []
        collections = _COGNEE_1_4_RETRIEVAL_COLLECTIONS.get(search_type)
        if collections is None:
            raise CogneeStorageError(
                f"Unsupported PaperOS compatibility search type: {search_type}",
                affected=dataset_name,
            )
        self.retrieval_fallback_types_used.add("custom_datapoint_vector_search")
        from cognee.context_global_variables import (
            set_database_global_context_variables,
        )
        from cognee.infrastructure.databases.vector import (
            get_vector_engine_async,
        )
        from cognee.modules.data.methods import (
            get_authorized_existing_datasets,
        )
        from cognee.modules.engine.operations.setup import (
            setup,
        )
        from cognee.modules.users.methods import (
            get_default_user,
        )

        try:
            await setup()
            user = await get_default_user()
            datasets = await get_authorized_existing_datasets([dataset_name], "read", user)
            if len(datasets) != 1:
                raise CogneeStorageError(
                    "Cognee dataset does not resolve uniquely for vector search.",
                    affected=dataset_name,
                )
            dataset = datasets[0]
            best: dict[str, CogneeVectorHit] = {}
            allowed_canonical_ids = set(canonical_ids.values())
            async with set_database_global_context_variables(dataset.id, dataset.owner_id):
                engine = await get_vector_engine_async()
                for collection, text_field, object_type in collections:
                    if not await engine.has_collection(collection):
                        continue
                    results = await engine.search(
                        collection,
                        query_text=query,
                        query_vector=None,
                        limit=top_k,
                        include_payload=True,
                    )
                    for result in results:
                        payload = result.payload
                        if not isinstance(payload, dict):
                            continue
                        cognee_id = str(payload.get("id") or result.id)
                        canonical_id = canonical_ids.get(cognee_id)
                        payload_canonical_id = payload.get("canonical_id")
                        if (
                            canonical_id is None
                            and payload_canonical_id is not None
                            and str(payload_canonical_id) in allowed_canonical_ids
                        ):
                            canonical_id = str(payload_canonical_id)
                        if not canonical_id:
                            continue
                        text = payload.get(text_field)
                        if not isinstance(text, str) or not text.strip():
                            continue
                        distance = max(float(result.score), 0.0)
                        hit = CogneeVectorHit(
                            cognee_id=cognee_id,
                            canonical_id=canonical_id,
                            object_type=object_type,
                            text=text,
                            score=1.0 / (1.0 + distance),
                            source_chunk_ids=tuple(_string_list(payload.get("source_chunk_ids"))),
                            derived_from_ids=tuple(_string_list(payload.get("derived_from_ids"))),
                            canonical_snapshot_id=(
                                str(payload["canonical_snapshot_id"])
                                if payload.get("canonical_snapshot_id")
                                else None
                            ),
                        )
                        previous = best.get(cognee_id)
                        if previous is None or hit.score > previous.score:
                            best[cognee_id] = hit
            return sorted(best.values(), key=lambda item: (-item.score, item.canonical_id))[:top_k]
        except CogneeStorageError:
            raise
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee DataPoint vector search failed: {exc}",
                affected=dataset_name,
            ) from exc

    async def delete_document_data(self, snapshot_id: str) -> int:
        """Delete one registered document through Cognee's public dataset API."""
        from cognee import datasets  # type: ignore[import-untyped]

        manifest = self.read_manifest(snapshot_id)
        dataset = manifest.get("dataset")
        data_item = manifest.get("data_item")
        if not isinstance(dataset, dict) or not isinstance(data_item, dict):
            raise CogneeStorageError(
                "Cognee manifest lacks dataset/data-item provenance.",
                affected=snapshot_id,
            )
        try:
            dataset_id = UUID(str(dataset["id"]))
            data_id = UUID(str(data_item["id"]))
            await datasets.delete_data(dataset_id=dataset_id, data_id=data_id)
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee document deletion failed: {exc}",
                affected=snapshot_id,
            ) from exc
        node_count = manifest.get("node_count", 0)
        return int(node_count) if isinstance(node_count, int) else 0

    async def provenance_counts(
        self,
        *,
        dataset_id: UUID,
        data_id: UUID | None,
        pipeline_run_id: UUID,
    ) -> CogneeDatasetBinding:
        """Count provenance-attributed nodes/edges for one PaperOS PDF."""
        if data_id is None:
            return CogneeDatasetBinding(
                user_id="",
                dataset_id=str(dataset_id),
                dataset_name="",
                data_id="",
                data_name="",
                pipeline_id="",
                pipeline_run_id=str(pipeline_run_id),
                pipeline_name="paperos_knowledge_ingestion",
                provenance_backend="none",
                provenance_node_count=0,
                provenance_edge_count=0,
            )
        from cognee.infrastructure.databases.graph.get_graph_engine import (
            get_graph_engine,
        )
        from cognee.infrastructure.databases.provenance import (  # type: ignore[import-untyped]
            make_source_ref_key,
        )
        from cognee.infrastructure.databases.relational import (
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
                    select(func.count())
                    .select_from(Node)
                    .where(
                        Node.dataset_id == dataset_id,
                        Node.data_id == data_id,
                    )
                )
                or 0
            )
            ledger_edges = int(
                await session.scalar(
                    select(func.count())
                    .select_from(Edge)
                    .where(
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

    async def resolve_graph_nodes(self, cognee_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Read Cognee 1.4.0 node identity/provenance absent from public results.

        Live contract capability: graph_node_provenance_readback.
        """
        if not cognee_ids:
            return {}
        self.retrieval_fallback_types_used.add("graph_node_provenance_readback")
        from cognee.infrastructure.databases.graph.get_graph_engine import (
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

    async def read_graph_records(
        self,
        cognee_ids: list[str],
        *,
        dataset_name: str,
        depth: int = 1,
    ) -> dict[str, list[dict[str, Any]]]:
        """Read unfiltered nodes and typed edges for live graph contracts.

        Cognee 1.4.0 has no public structured graph readback API. This
        contract-only read preserves edges without chunk provenance (notably
        ``REPRESENTS_WORK``) and does not infer evidence. All private
        graph-engine access remains isolated here.
        """
        if not cognee_ids or depth <= 0:
            return {"nodes": [], "edges": []}
        self.retrieval_fallback_types_used.add("structured_graph_readback")
        from cognee.infrastructure.databases.graph.get_graph_engine import (
            get_graph_engine,
        )

        seeds = sorted({str(item) for item in cognee_ids})
        async with await self._dataset_scope(dataset_name):
            engine = await get_graph_engine()
            try:
                nodes, edges = await engine.get_neighborhood(
                    seeds,
                    depth=depth,
                    edge_types=None,
                )
            except Exception as exc:
                raise CogneeStorageError(
                    f"Cognee structured graph readback failed: {exc}",
                    affected=self.paths.cognee,
                ) from exc

        node_records = []
        for node_id, raw_properties in nodes:
            properties = (
                _flatten_node(dict(raw_properties)) if isinstance(raw_properties, dict) else {}
            )
            properties.setdefault("id", str(node_id))
            node_records.append(properties)

        edge_records = []
        for source_id, target_id, relation_type, raw_properties in edges:
            properties = dict(raw_properties) if isinstance(raw_properties, dict) else {}
            edge_records.append(
                {
                    "source_id": str(source_id),
                    "target_id": str(target_id),
                    "relation_type": str(relation_type),
                    **properties,
                }
            )
        return {
            "nodes": sorted(
                node_records,
                key=lambda item: str(item.get("id") or item.get("canonical_id") or ""),
            ),
            "edges": sorted(
                edge_records,
                key=lambda item: (
                    str(item["source_id"]),
                    str(item["relation_type"]),
                    str(item["target_id"]),
                ),
            ),
        }

    async def semantic_relations_for_chunks(
        self,
        chunk_ids: list[str],
        *,
        relation_types: set[str],
        dataset_name: str,
        limit: int = 200,
    ) -> list[CogneeSemanticRelation]:
        """Return one-hop relations on semantic objects grounded in seed Chunks.

        The fixed operator is Chunk provenance -> grounded Entity -> one direct
        semantic edge. Layout, citation, ABOUT, Summary, and arbitrary connector
        paths are never traversed. Private graph access remains dataset-scoped
        inside this compatibility boundary.
        """
        self.last_semantic_relation_trace = {
            "raw_neighbor_nodes": 0,
            "filtered_grounded_entities": 0,
            "filtered_semantic_relations": 0,
            "missing_relation_source_provenance": 0,
        }
        if not chunk_ids or not relation_types or limit <= 0:
            return []
        central_types = {item.value for item in SEMANTIC_RELATION_TYPES}
        unsupported = relation_types - central_types
        if unsupported:
            raise CogneeStorageError(
                "Semantic expansion requested non-semantic relation types.",
                affected=dataset_name,
                details={"unsupported_relation_types": sorted(unsupported)},
            )
        self.retrieval_fallback_types_used.add("direct_semantic_relations")
        from cognee.infrastructure.databases.graph.get_graph_engine import (
            get_graph_engine,
        )

        seeds = list(dict.fromkeys(str(item) for item in chunk_ids))
        seed_cognee_ids = [str(cognee_uuid(item)) for item in seeds]
        async with await self._dataset_scope(dataset_name):
            engine = await get_graph_engine()
            try:
                nodes, edges = await engine.get_neighborhood(
                    seed_cognee_ids,
                    # Exactly enough to observe Entity -> Chunk provenance and
                    # one direct Entity -> Entity semantic edge.
                    depth=2,
                    edge_types=None,
                )
            except Exception as exc:
                raise CogneeStorageError(
                    f"Cognee direct semantic relation read failed: {exc}",
                    affected=self.paths.cognee,
                ) from exc
        return _direct_semantic_relations(
            nodes,
            edges,
            seed_chunk_ids=set(seeds),
            relation_types=relation_types,
            limit=limit,
            diagnostics=self.last_semantic_relation_trace,
        )

    async def incoming_typed_relations(
        self,
        target_canonical_ids: list[str],
        *,
        dataset_name: str,
        relation_type: str,
        limit: int = 200,
    ) -> list[CogneeIncomingRelation]:
        """Read bounded incoming typed edges for exact canonical targets.

        Production-compatible Cognee 1.4.0 path used by scholarly navigation.
        Private graph access stays here: dataset-scoped, fixed depth=1,
        relation-filtered, result-bounded, provenance-preserving.
        """
        if not target_canonical_ids or limit <= 0:
            return []
        self.retrieval_fallback_types_used.add("incoming_typed_relations")
        from cognee.infrastructure.databases.graph.get_graph_engine import (
            get_graph_engine,
        )

        target_ids = list(dict.fromkeys(str(item) for item in target_canonical_ids))
        seed_cognee_ids = [str(cognee_uuid(item)) for item in target_ids]
        seed_by_cognee = {str(cognee_uuid(item)): item for item in target_ids}
        async with await self._dataset_scope(dataset_name):
            engine = await get_graph_engine()
            try:
                nodes, edges = await engine.get_neighborhood(
                    seed_cognee_ids,
                    depth=1,
                    # Cognee 1.4 Kuzu cannot bind the edge-type list reliably;
                    # filter the single-hop relation after the scoped read.
                    edge_types=None,
                )
            except Exception as exc:
                raise CogneeStorageError(
                    f"Cognee incoming typed relation read failed: {exc}",
                    affected=self.paths.cognee,
                ) from exc

        node_properties = {
            str(node_id): _flatten_node(dict(properties)) for node_id, properties in nodes
        }
        results: list[CogneeIncomingRelation] = []
        for source_id, target_id, edge_relation, raw_properties in edges:
            relation = str(edge_relation)
            if relation != relation_type:
                continue
            target_key = str(target_id)
            if target_key not in seed_by_cognee:
                continue
            properties = dict(raw_properties) if isinstance(raw_properties, dict) else {}
            source = node_properties.get(str(source_id), {})
            source_canonical_id = str(
                properties.get("canonical_source_id") or source.get("canonical_id") or source_id
            )
            target_canonical_id = seed_by_cognee[target_key]
            chunk_ids = list(
                dict.fromkeys(
                    [
                        *_string_list(properties.get("source_chunk_ids")),
                        *_node_source_chunk_ids(source),
                    ]
                )
            )
            if not chunk_ids:
                continue
            roles_value = properties.get("roles")
            roles = (
                tuple(str(item) for item in roles_value if item is not None)
                if isinstance(roles_value, list)
                else ()
            )
            text = str(source.get("text") or "").strip()
            source_work_id = source.get("source_work_id")
            source_document_id = source.get("source_document_id")
            results.append(
                CogneeIncomingRelation(
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
                    roles=roles,
                    source_work_id=(str(source_work_id) if source_work_id else None),
                    source_document_id=(str(source_document_id) if source_document_id else None),
                    text=text,
                    score=1.0,
                )
            )
        results.sort(
            key=lambda item: (
                item.target_canonical_id,
                item.source_canonical_id,
                item.source_chunk_ids,
            )
        )
        return results[:limit]

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

    @staticmethod
    async def prune_derived_data() -> None:
        """Destructively prune Cognee's derived graph/vector/metadata stores."""
        from cognee.modules.data.deletion import (  # type: ignore[import-untyped]
            prune_data,
            prune_system,
        )

        await prune_data()
        await prune_system(graph=True, vector=True, metadata=True, cache=True)


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


def _flatten_node(node: dict[str, Any]) -> dict[str, Any]:
    properties = node.get("properties")
    if isinstance(properties, dict):
        return {**node, **properties}
    return node


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _node_source_chunk_ids(node: dict[str, Any]) -> list[str]:
    return _string_list(node.get("source_chunk_ids"))


def _direct_semantic_relations(
    nodes: list[Any],
    edges: list[Any],
    *,
    seed_chunk_ids: set[str],
    relation_types: set[str],
    limit: int,
    diagnostics: dict[str, int] | None = None,
) -> list[CogneeSemanticRelation]:
    """Apply the fixed grounded-Entity/direct-semantic-edge operator."""
    node_properties = {
        str(node_id): _flatten_node(dict(properties)) for node_id, properties in nodes
    }
    semantic_node_ids = {
        node_id
        for node_id, properties in node_properties.items()
        if str(
            properties.get("type")
            or properties.get("object_type")
            or properties.get("__type__")
            or ""
        )
        == "EntityDataPoint"
    }
    seed_cognee_ids = {str(cognee_uuid(chunk_id)) for chunk_id in seed_chunk_ids}
    grounded_node_ids = {
        node_id
        for node_id in semantic_node_ids
        if seed_chunk_ids.intersection(_node_source_chunk_ids(node_properties[node_id]))
    }
    for source_id, target_id, relation_type, _properties in edges:
        source_key = str(source_id)
        target_key = str(target_id)
        if (
            str(relation_type) == "DERIVED_FROM"
            and source_key in semantic_node_ids
            and target_key in seed_cognee_ids
        ):
            grounded_node_ids.add(source_key)

    if diagnostics is not None:
        diagnostics["raw_neighbor_nodes"] = len(node_properties)
        diagnostics["filtered_grounded_entities"] = len(grounded_node_ids)

    results: list[CogneeSemanticRelation] = []
    seen: set[tuple[str, str, str]] = set()
    for source_id, target_id, relation_type, raw_properties in edges:
        relation = str(relation_type)
        source_key = str(source_id)
        target_key = str(target_id)
        if (
            relation not in relation_types
            or source_key not in semantic_node_ids
            or target_key not in semantic_node_ids
            or not {source_key, target_key}.intersection(grounded_node_ids)
        ):
            continue
        properties = dict(raw_properties) if isinstance(raw_properties, dict) else {}
        source = node_properties[source_key]
        target = node_properties[target_key]
        source_canonical_id = str(
            properties.get("canonical_source_id") or source.get("canonical_id") or source_key
        )
        target_canonical_id = str(
            properties.get("canonical_target_id") or target.get("canonical_id") or target_key
        )
        key = (source_canonical_id, relation, target_canonical_id)
        if key in seen:
            continue
        seen.add(key)
        chunk_ids = tuple(dict.fromkeys(_string_list(properties.get("source_chunk_ids"))))
        if not chunk_ids:
            if diagnostics is not None:
                diagnostics["missing_relation_source_provenance"] += 1
            continue
        grounded_object_ids = tuple(
            canonical_id
            for node_id, canonical_id in (
                (source_key, source_canonical_id),
                (target_key, target_canonical_id),
            )
            if node_id in grounded_node_ids
        )
        results.append(
            CogneeSemanticRelation(
                source_canonical_id=source_canonical_id,
                target_canonical_id=target_canonical_id,
                relation_type=relation,
                grounded_object_ids=grounded_object_ids,
                source_chunk_ids=chunk_ids,
                derived_from_ids=tuple(
                    dict.fromkeys(
                        [
                            *grounded_object_ids,
                            source_canonical_id,
                            target_canonical_id,
                            *_string_list(properties.get("derived_from_ids")),
                        ]
                    )
                ),
                score=1.0,
            )
        )
    results.sort(
        key=lambda item: (
            item.source_canonical_id,
            item.relation_type,
            item.target_canonical_id,
        )
    )
    if diagnostics is not None:
        diagnostics["filtered_semantic_relations"] = len(results)
    return results[:limit]


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
