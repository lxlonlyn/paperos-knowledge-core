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
from paperos_core.domain.ids import semantic_object_id
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
    enrichment_path: Path | None

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
        snapshot_id = bundle.snapshot.id
        try:
            report, enrichment_path = await self.ingest_bundle(bundle, rebuilt=rebuilt)
            previous_by_snapshot = await self._publish_candidate_family(bundle)
        except Exception:
            await self._cleanup_after_failure(snapshot_id, phase="candidate")
            raise

        published_snapshot_ids = set(previous_by_snapshot)
        retired_snapshot_ids = list(
            dict.fromkeys(
                previous
                for previous in previous_by_snapshot.values()
                if previous is not None and previous not in published_snapshot_ids
            )
        )
        for previous_snapshot_id in retired_snapshot_ids:
            try:
                await self.cleanup_snapshot_revision(previous_snapshot_id)
            except Exception as exc:  # noqa: BLE001 - activation must not roll back.
                self._record_cleanup_retry(
                    previous_snapshot_id,
                    phase="retired_revision",
                    exc=exc,
                )
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

    async def _publish_candidate_family(
        self,
        primary_bundle: CanonicalBundle,
    ) -> dict[str, str | None]:
        """Build and atomically publish every active projection affected by Work changes."""

        owner_snapshot_id = primary_bundle.snapshot.id
        dependent_candidates = await self._prepare_affected_active_reprojections(
            owner_snapshot_id,
            exclude_document_ids={primary_bundle.document.id},
        )
        snapshot_ids = [owner_snapshot_id, *dependent_candidates]
        try:
            return self.scholarly_registry.publish_candidate_set(
                owner_snapshot_id,
                self.canonical_repository,
                snapshot_ids=snapshot_ids,
                expected_previous_snapshot_ids={
                    candidate_id: active_snapshot_id
                    for candidate_id, active_snapshot_id in dependent_candidates.items()
                },
            )
        except Exception:
            await self._cleanup_candidate_set(
                list(dependent_candidates),
                phase="reconciliation_publication",
            )
            raise

    async def _prepare_affected_active_reprojections(
        self,
        owner_snapshot_id: str,
        *,
        exclude_document_ids: set[str],
    ) -> dict[str, str]:
        """Fully index isolated replacements for active documents changed in staging."""

        affected = self.scholarly_registry.affected_active_snapshot_ids(
            owner_snapshot_id,
            self.canonical_repository,
            exclude_document_ids=exclude_document_ids,
        )
        prepared: dict[str, str] = {}
        current_candidate_id: str | None = None
        try:
            for active_snapshot_id in affected:
                candidate = self.canonical_repository.create_rebuild_candidate(
                    active_snapshot_id
                )
                current_candidate_id = candidate.snapshot.id
                if self.ingestion.semantic_enrichment_enabled:
                    self.reproject_enrichment(active_snapshot_id, current_candidate_id)
                await self.ingest_bundle(
                    candidate,
                    rebuilt=True,
                    reuse_existing_enrichment=True,
                    generate_enrichment_if_missing=False,
                    scholarly_candidate_snapshot_id=owner_snapshot_id,
                )
                prepared[current_candidate_id] = active_snapshot_id
                current_candidate_id = None
        except Exception:
            cleanup_ids = list(prepared)
            if current_candidate_id is not None:
                cleanup_ids.append(current_candidate_id)
            await self._cleanup_candidate_set(
                cleanup_ids,
                phase="scholarly_reconciliation",
            )
            raise
        return prepared

    async def _cleanup_candidate_set(
        self,
        snapshot_ids: list[str],
        *,
        phase: str,
    ) -> None:
        for snapshot_id in reversed(snapshot_ids):
            await self._cleanup_after_failure(snapshot_id, phase=phase)

    async def ingest_bundle(
        self,
        bundle: CanonicalBundle,
        *,
        rebuilt: bool = False,
        reuse_existing_enrichment: bool = False,
        generate_enrichment_if_missing: bool = True,
        scholarly_candidate_snapshot_id: str | None = None,
    ) -> tuple[IndexingReport, Path | None]:
        """Run one canonical graph projection with optional semantic enrichment."""
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
        self._persist_candidate_manifest(
            bundle=bundle,
            source=source,
            dataset=dataset,
            data_id=data_id,
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
            chunk_hard_max_tokens=self.ingestion.chunk_hard_max_tokens,
            chunk_overlap_tokens=self.ingestion.chunk_overlap_tokens,
            graph_results=graph_results,
            scholarly_candidate_snapshot_id=scholarly_candidate_snapshot_id,
            reuse_existing_enrichment=reuse_existing_enrichment,
            generate_enrichment_if_missing=generate_enrichment_if_missing,
            semantic_enrichment_enabled=self.ingestion.semantic_enrichment_enabled,
            claim_enrichment_enabled=self.ingestion.claim_enrichment_enabled,
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
        if projection.rerank_projection is None:
            raise CogneeStorageError(
                "Structured rerank projection is missing after ChunkProjection build.",
                affected=bundle.snapshot.id,
            )
        enrichment: SemanticEnrichment | None = None
        if self.ingestion.semantic_enrichment_enabled:
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
            semantic_enrichment_enabled=self.ingestion.semantic_enrichment_enabled,
            semantic_entity_count=len(enrichment.entities) if enrichment else 0,
            semantic_claim_count=len(enrichment.claims) if enrichment else 0,
            semantic_relation_count=len(enrichment.relations) if enrichment else 0,
            consistency_valid=True,
            rebuilt=rebuilt,
        )
        enrichment_path = (
            self.paths.cognee / "enrichment" / f"{bundle.snapshot.id}.json"
            if self.ingestion.semantic_enrichment_enabled
            else None
        )
        return report, enrichment_path

    async def cleanup_snapshot_derived(
        self,
        snapshot_id: str,
        *,
        preserve_enrichment: bool = False,
    ) -> list[Path]:
        """Idempotently remove only one snapshot's derived projections."""

        deleted: list[Path] = []
        failures: list[Exception] = []
        cognee_manifest = self.paths.cognee / "manifests" / f"{snapshot_id}.json"
        if cognee_manifest.is_file():
            try:
                await self.compat.delete_document_data(snapshot_id)
            except Exception as exc:  # noqa: BLE001 - finish safe local cleanup.
                failures.append(exc)
            else:
                cognee_manifest.unlink(missing_ok=True)
                deleted.append(cognee_manifest)
        try:
            self.index_manager.lexical.delete_snapshot(snapshot_id)
        except Exception as exc:  # noqa: BLE001 - collect retryable cleanup failures.
            failures.append(exc)
        targets = [
            self.paths.indexes / "manifests" / f"{snapshot_id}.json",
            self.paths.cognee / "graphs" / f"{snapshot_id}.json",
            self.paths.cognee / "chunks" / f"{snapshot_id}.jsonl",
            self.paths.cognee / "citation_mentions" / f"{snapshot_id}.jsonl",
            self.paths.cognee / "rerank_projections" / f"{snapshot_id}.json",
        ]
        if not preserve_enrichment:
            targets.append(self.paths.cognee / "enrichment" / f"{snapshot_id}.json")
        for target in targets:
            resolved = target.resolve(strict=False)
            self.paths.assert_within_root(resolved)
            if resolved.is_file():
                resolved.unlink()
                deleted.append(resolved)
        if failures:
            raise CogneeStorageError(
                "Snapshot-derived cleanup is incomplete and must be retried.",
                affected=snapshot_id,
                details={"failure_count": len(failures), "retryable": True},
            ) from failures[0]
        self._cleanup_retry_path(snapshot_id).unlink(missing_ok=True)
        return deleted

    async def cleanup_snapshot_revision(self, snapshot_id: str) -> list[Path]:
        """Remove one inactive derived projection and immutable canonical revision."""

        if self.canonical_repository.is_active_snapshot(snapshot_id):
            raise CogneeStorageError(
                "The active canonical snapshot cannot be cleaned up.",
                affected=snapshot_id,
            )
        deleted = await self.cleanup_snapshot_derived(snapshot_id)
        self.scholarly_registry.discard_candidate(snapshot_id)
        self.canonical_repository.cleanup_snapshot(snapshot_id)
        return deleted

    def reproject_enrichment(
        self,
        source_snapshot_id: str,
        target_snapshot_id: str,
    ) -> Path:
        """Copy immutable semantic facts under a new canonical revision identity."""

        source = self._load_enrichment(source_snapshot_id)
        entity_ids: dict[str, str] = {}
        entities = []
        for entity in source.entities:
            entity_id = semantic_object_id(
                "entity",
                target_snapshot_id,
                f"{entity.entity_type}:{entity.name}",
                entity.source_chunk_ids,
            )
            entity_ids[entity.id] = entity_id
            entities.append(
                entity.model_copy(
                    update={
                        "id": entity_id,
                        "canonical_snapshot_id": target_snapshot_id,
                    }
                )
            )
        claims = [
            claim.model_copy(
                update={
                    "id": semantic_object_id(
                        "claim",
                        target_snapshot_id,
                        claim.text,
                        claim.source_chunk_ids,
                    ),
                    "canonical_snapshot_id": target_snapshot_id,
                }
            )
            for claim in source.claims
        ]
        relations = []
        for relation in source.relations:
            source_object_id = entity_ids.get(
                relation.source_object_id,
                relation.source_object_id,
            )
            target_object_id = entity_ids.get(
                relation.target_object_id,
                relation.target_object_id,
            )
            relations.append(
                relation.model_copy(
                    update={
                        "id": semantic_object_id(
                            "relation",
                            target_snapshot_id,
                            f"{source_object_id}:{relation.relation_type}:"
                            f"{target_object_id}",
                            relation.source_chunk_ids,
                        ),
                        "canonical_snapshot_id": target_snapshot_id,
                        "source_object_id": source_object_id,
                        "target_object_id": target_object_id,
                    }
                )
            )
        target = self.paths.cognee / "enrichment" / f"{target_snapshot_id}.json"
        _atomic_json(
            target,
            source.model_copy(
                update={
                    "entities": entities,
                    "claims": claims,
                    "relations": relations,
                }
            ).model_dump(mode="json"),
        )
        return target

    async def _cleanup_after_failure(self, snapshot_id: str, *, phase: str) -> None:
        failures: list[Exception] = []
        try:
            await self.cleanup_snapshot_derived(snapshot_id)
        except Exception as exc:  # noqa: BLE001 - preserve original failure.
            failures.append(exc)
        self.scholarly_registry.discard_candidate(snapshot_id)
        try:
            self.canonical_repository.cleanup_snapshot(snapshot_id)
        except Exception as exc:  # noqa: BLE001 - preserve original failure.
            failures.append(exc)
        if failures:
            self._record_cleanup_retry(snapshot_id, phase=phase, exc=failures[0])

    def _cleanup_retry_path(self, snapshot_id: str) -> Path:
        return self.paths.jobs / "cleanup" / f"{snapshot_id}.json"

    def _record_cleanup_retry(
        self,
        snapshot_id: str,
        *,
        phase: str,
        exc: Exception,
    ) -> None:
        _atomic_json(
            self._cleanup_retry_path(snapshot_id),
            {
                "snapshot_id": snapshot_id,
                "phase": phase,
                "status": "pending",
                "retryable": True,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )

    def _persist_candidate_manifest(
        self,
        *,
        bundle: CanonicalBundle,
        source: Any,
        dataset: Any,
        data_id: UUID,
    ) -> Path:
        path = self.paths.cognee / "manifests" / f"{bundle.snapshot.id}.json"
        _atomic_json(
            path,
            {
                "mapping_version": "4",
                "status": "candidate",
                "canonical_snapshot_id": bundle.snapshot.id,
                "document_id": bundle.document.id,
                "dataset": {
                    "id": str(dataset.id),
                    "name": str(dataset.name),
                    "owner_id": str(dataset.owner_id),
                },
                "data_item": {
                    "id": str(data_id),
                    "name": source.original_filename,
                    "source_file_id": source.id,
                    "source_sha256": source.sha256,
                },
            },
        )
        return path

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
            "mapping_version": "4",
            "status": "complete",
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
            "semantic_enrichment_enabled": self.ingestion.semantic_enrichment_enabled,
            "canonical_to_cognee_id": graph.id_mapping,
            "relations": [relation.model_dump(mode="json") for relation in graph.relations],
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


def _validate_semantic_provenance(chunks: list[Any], enrichment: SemanticEnrichment) -> None:
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
