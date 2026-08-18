"""CogneePipelineAdapter: run PaperOS ingestion through Cognee's custom pipeline.

The adapter owns the ingestion orchestration boundary only. Cognee's public
``run_custom_pipeline`` executes the five PaperOS tasks (AcademicChunkTask,
ScholarlyIdentityTask, SemanticEnrichmentTask, DataPointMappingTask, add_data_points), Cognee owns the
dataset/pipeline lifecycle, and every Cognee-internal call is centralized in
``compat``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from paperos_core.adapters.cognee.compat import (
    CogneeCompatibilityAdapter,
    CogneeDatasetBinding,
    PipelineItem,
)
from paperos_core.adapters.cognee.llm import LLMClient
from paperos_core.adapters.cognee.models import (
    DataPointGraph,
)
from paperos_core.adapters.cognee.pipeline_tasks import configure_pipeline_tasks
from paperos_core.config import IngestionSettings
from paperos_core.domain.canonical import CanonicalBundle, CanonicalIngestionResult
from paperos_core.domain.knowledge import SemanticEnrichment
from paperos_core.errors import CogneeStorageError
from paperos_core.indexes.manager import IndexManager
from paperos_core.indexes.manifest import IndexingReport
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry
from paperos_core.paths import DataPaths


class KnowledgeIngestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_result: CanonicalIngestionResult
    indexing: IndexingReport
    enrichment_path: Path

    def public_dict(self) -> dict[str, object]:
        payload = self.canonical_result.public_dict()
        payload["knowledge"] = self.indexing.public_dict()
        return payload


class CogneePipelineAdapter:
    def __init__(
        self,
        paths: DataPaths,
        canonical_repository: CanonicalRepository,
        source_registry: SourceRegistry,
        scholarly_registry: ScholarlyRegistry,
        compat: CogneeCompatibilityAdapter,
        index_manager: IndexManager,
        llm: LLMClient,
        ingestion: IngestionSettings,
    ) -> None:
        self.paths = paths
        self.canonical_repository = canonical_repository
        self.source_registry = source_registry
        self.scholarly_registry = scholarly_registry
        self.compat = compat
        self.index_manager = index_manager
        self.llm = llm
        self.ingestion = ingestion

    async def ingest_canonical_snapshot(
        self, canonical_result: CanonicalIngestionResult, *, rebuilt: bool = False
    ) -> KnowledgeIngestionResult:
        bundle = canonical_result.canonical
        report, enrichment_path = await self.ingest_bundle(bundle, rebuilt=rebuilt)
        fresh_bundle = self.canonical_repository.get_bundle(bundle.snapshot.id)
        projection = self.canonical_repository.get_chunk_projection(bundle.snapshot.id)
        return KnowledgeIngestionResult(
            canonical_result=CanonicalIngestionResult(
                parsed=canonical_result.parsed,
                canonical=fresh_bundle,
                chunk_count=len(projection.chunks),
            ),
            indexing=report,
            enrichment_path=enrichment_path,
        )

    async def ingest_bundle(
        self,
        bundle: CanonicalBundle,
        *,
        rebuilt: bool = False,
        reuse_existing_enrichment: bool = False,
        generate_enrichment_if_missing: bool = True,
    ) -> tuple[IndexingReport, Path]:
        """Run one canonical/enrichment pair through the Cognee custom pipeline."""
        self.canonical_repository.verify_snapshot(bundle.snapshot.id)
        dataset_name = bundle.snapshot.dataset_id
        dataset = await self.compat.ensure_dataset(dataset_name)
        source = self.source_registry.get_source(bundle.snapshot.source_file_id)
        data_id = await self.compat.register_data_item(
            dataset=dataset,
            source=source,
            snapshot_id=bundle.snapshot.id,
            document_id=bundle.document.id,
            title=bundle.document.title,
        )
        graph_results: list[DataPointGraph] = []
        tasks = configure_pipeline_tasks(
            repository=self.canonical_repository,
            scholarly_registry=self.scholarly_registry,
            compat=self.compat,
            llm=self.llm,
            enrichment_root=self.paths.cognee / "enrichment",
            graph_root=self.paths.cognee / "graphs",
            chunk_target_tokens=self.ingestion.chunk_target_tokens,
            chunk_overlap_tokens=self.ingestion.chunk_overlap_tokens,
            graph_results=graph_results,
            reuse_existing_enrichment=reuse_existing_enrichment,
            generate_enrichment_if_missing=generate_enrichment_if_missing,
        )
        item = PipelineItem(
            id=data_id,
            data_id=data_id,
            bundle=bundle,
            source=source,
        )
        run_infos = await self._run_pipeline(
            tasks=tasks,
            item=item,
            dataset_name=str(dataset.name),
        )
        run_info = _single_run_info(run_infos, dataset_name)
        run_id = UUID(str(run_info.pipeline_run_id))
        fresh_bundle = self.canonical_repository.get_bundle(bundle.snapshot.id)
        projection = self.canonical_repository.get_chunk_projection(bundle.snapshot.id)
        enrichment = self._load_enrichment(bundle.snapshot.id)
        _validate_semantic_provenance(projection.chunks, enrichment)
        if len(graph_results) != 1:
            raise CogneeStorageError(
                "Cognee pipeline did not return exactly one mapped DataPointGraph.",
                affected=bundle.snapshot.id,
                details={"graph_count": len(graph_results)},
            )
        graph = graph_results[0]
        binding = await self.compat.provenance_counts(
            dataset_id=UUID(str(dataset.id)),
            data_id=data_id,
            pipeline_run_id=run_id,
        )
        binding = CogneeDatasetBinding(
            user_id=binding.user_id,
            dataset_id=str(dataset.id),
            dataset_name=str(dataset.name),
            data_id=str(data_id),
            data_name=source.original_filename,
            pipeline_id="",
            pipeline_run_id=str(run_id),
            pipeline_name="paperos_knowledge_ingestion",
            provenance_backend=binding.provenance_backend,
            provenance_node_count=binding.provenance_node_count,
            provenance_edge_count=binding.provenance_edge_count,
        )
        await self.compat.verify_graph(graph)
        cognee_vector_ids = await self.compat.verify_vector_indexes(graph)
        cognee_manifest = self._persist_manifest(
            snapshot_id=bundle.snapshot.id,
            document_id=bundle.document.id,
            source=source,
            binding=binding,
            graph=graph,
        )
        index_manifest, index_manifest_path = await self.index_manager.index_bundle(
            fresh_bundle,
            chunks=projection.chunks,
        )
        runtime_config = self.llm.runtime_config.read()
        report = IndexingReport(
            canonical_snapshot_id=bundle.snapshot.id,
            document_id=bundle.document.id,
            manifest_path=index_manifest_path,
            cognee_manifest_path=cognee_manifest,
            lexical_database=index_manifest.lexical_database,
            vector_database=runtime_config.vector_db_url,
            dataset_name=binding.dataset_name,
            cognee_dataset_id=binding.dataset_id,
            cognee_data_id=binding.data_id,
            cognee_pipeline_run_id=binding.pipeline_run_id,
            cognee_provenance_backend=binding.provenance_backend,
            cognee_object_count=len(graph.nodes),
            relation_count=len(graph.relations),
            lexical_object_count=len(index_manifest.lexical_object_ids),
            vector_object_count=len(cognee_vector_ids),
            embedding_dimensions=runtime_config.embedding_dimensions,
            semantic_entity_count=len(enrichment.entities),
            semantic_claim_count=len(enrichment.claims),
            semantic_relation_count=len(enrichment.relations),
            summary_count=len(enrichment.summaries),
            consistency_valid=True,
            rebuilt=rebuilt,
        )
        return report, self.paths.cognee / "enrichment" / f"{bundle.snapshot.id}.json"

    async def _run_pipeline(
        self,
        *,
        tasks: list[Any],
        item: PipelineItem,
        dataset_name: str,
    ) -> dict[str, Any]:
        import cognee  # type: ignore[import-untyped]

        try:
            result = await cognee.run_custom_pipeline(
                tasks=tasks,
                data=item,
                dataset=dataset_name,
                pipeline_name="paperos_knowledge_ingestion",
            )
            return dict(result)
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee custom pipeline failed: {exc}",
                affected=dataset_name,
            ) from exc

    def _load_enrichment(self, snapshot_id: str) -> SemanticEnrichment:
        path = self.paths.cognee / "enrichment" / f"{snapshot_id}.json"
        try:
            return SemanticEnrichment.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise CogneeStorageError(
                f"Unable to read semantic enrichment artifact: {exc}",
                affected=path,
            ) from exc

    def _persist_manifest(
        self,
        *,
        snapshot_id: str,
        document_id: str,
        source: Any,
        binding: CogneeDatasetBinding,
        graph: DataPointGraph,
    ) -> Path:
        manifest_path = self.paths.cognee / "manifests" / f"{snapshot_id}.json"
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
            "relations": [
                relation.model_dump(mode="json") for relation in graph.relations
            ],
        }
        _atomic_json(manifest_path, manifest)
        return manifest_path


def _single_run_info(run_infos: dict[str, Any], dataset_name: str) -> Any:
    if isinstance(run_infos, dict):
        if len(run_infos) == 1:
            return next(iter(run_infos.values()))
        dataset_id = next(
            (key for key in run_infos if str(key) == dataset_name),
            None,
        )
        if dataset_id is not None:
            return run_infos[dataset_id]
    return run_infos


def _validate_semantic_provenance(
    chunks: list[Any], enrichment: SemanticEnrichment
) -> None:
    chunk_ids = {chunk.id for chunk in chunks}
    for entity in enrichment.entities:
        _validate_provenance(entity.id, entity.source_chunk_ids, chunk_ids)
    for claim in enrichment.claims:
        _validate_provenance(claim.id, claim.source_chunk_ids, chunk_ids)
        for about in claim.about:
            _validate_provenance(
                f"{claim.id}:ABOUT:{about.work_id}",
                about.source_chunk_ids,
                chunk_ids,
            )
    for relation in enrichment.relations:
        _validate_provenance(relation.id, relation.source_chunk_ids, chunk_ids)
    for summary in enrichment.summaries:
        _validate_provenance(summary.id, summary.source_chunk_ids, chunk_ids)


def _validate_provenance(
    object_id: str, source_chunk_ids: list[str], valid_chunk_ids: set[str]
) -> None:
    if not source_chunk_ids or not set(source_chunk_ids).issubset(valid_chunk_ids):
        raise CogneeStorageError(
            "Semantic object lacks valid canonical source provenance.",
            affected=object_id,
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
