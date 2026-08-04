"""Gate 4 canonical-to-knowledge pipeline."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from paperos_core.adapters.cognee.models import canonical_to_datapoints
from paperos_core.adapters.cognee.reference_resolution import resolve_citations
from paperos_core.adapters.cognee.repository import CogneeRepository
from paperos_core.adapters.llm import DeepSeekClient
from paperos_core.domain.canonical import CanonicalBundle, CanonicalIngestionResult
from paperos_core.domain.knowledge import SemanticEnrichment
from paperos_core.errors import CogneeStorageError
from paperos_core.indexes.manager import IndexManager
from paperos_core.indexes.manifest import IndexingReport
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.paths import DataPaths


class KnowledgeIngestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_result: CanonicalIngestionResult
    indexing: IndexingReport
    enrichment_path: Path

    def public_dict(self) -> dict[str, object]:
        payload = self.canonical_result.public_dict()
        payload["knowledge"] = self.indexing.model_dump(mode="json")
        payload["knowledge"]["enrichment_path"] = str(self.enrichment_path)
        return payload


class CogneePipeline:
    def __init__(
        self,
        paths: DataPaths,
        canonical_repository: CanonicalRepository,
        source_registry: SourceRegistry,
        cognee_repository: CogneeRepository,
        index_manager: IndexManager,
        deepseek: DeepSeekClient,
    ) -> None:
        self.paths = paths
        self.canonical_repository = canonical_repository
        self.source_registry = source_registry
        self.cognee_repository = cognee_repository
        self.index_manager = index_manager
        self.deepseek = deepseek

    async def ingest_canonical_snapshot(
        self, canonical_result: CanonicalIngestionResult, *, rebuilt: bool = False
    ) -> KnowledgeIngestionResult:
        bundle = canonical_result.canonical
        report, enrichment_path = await self.ingest_bundle(bundle, rebuilt=rebuilt)
        return KnowledgeIngestionResult(
            canonical_result=canonical_result,
            indexing=report,
            enrichment_path=enrichment_path,
        )

    async def ingest_bundle(
        self, bundle: CanonicalBundle, *, rebuilt: bool = False
    ) -> tuple[IndexingReport, Path]:
        """Index one repository-loaded canonical bundle without reconstructing intake."""
        self.canonical_repository.verify_snapshot(bundle.snapshot.id)
        await self.deepseek.health_check()
        enrichment = await self.deepseek.enrich(bundle)
        _validate_semantic_provenance(bundle, enrichment)
        enrichment_path = self._persist_enrichment(bundle, enrichment)
        graph = canonical_to_datapoints(bundle, enrichment)
        graph.relations.extend(resolve_citations(bundle, self.canonical_repository.list_bundles()))
        source = self.source_registry.get_source(bundle.snapshot.source_file_id)
        (
            cognee_manifest,
            dataset_binding,
        ) = await self.cognee_repository.upsert_document_graph(
            graph,
            snapshot_id=bundle.snapshot.id,
            document_id=bundle.document.id,
            dataset_name=bundle.snapshot.dataset_id,
            source=source,
            title=bundle.document.title,
        )
        await self.cognee_repository.verify_graph(graph)
        cognee_vector_ids = await self.cognee_repository.verify_vector_indexes(graph)
        index_manifest, index_manifest_path = await self.index_manager.index_bundle(
            bundle,
            cognee_manifest=cognee_manifest,
            cognee_object_ids=sorted(graph.id_mapping),
            cognee_vector_object_ids=cognee_vector_ids,
            relation_count=len(graph.relations),
            dataset_binding=dataset_binding,
        )
        report = IndexingReport(
            canonical_snapshot_id=bundle.snapshot.id,
            document_id=bundle.document.id,
            manifest_path=index_manifest_path,
            cognee_manifest_path=cognee_manifest,
            lexical_database=index_manifest.lexical_database,
            vector_database=index_manifest.vector_database,
            dataset_name=dataset_binding.dataset_name,
            cognee_dataset_id=dataset_binding.dataset_id,
            cognee_data_id=dataset_binding.data_id,
            cognee_pipeline_run_id=dataset_binding.pipeline_run_id,
            cognee_provenance_backend=dataset_binding.provenance_backend,
            cognee_object_count=len(graph.nodes),
            relation_count=len(graph.relations),
            lexical_object_count=len(index_manifest.lexical_object_ids),
            vector_object_count=len(index_manifest.vector_object_ids),
            embedding_dimensions=index_manifest.embedding_dimensions,
            semantic_entity_count=len(enrichment.entities),
            semantic_claim_count=len(enrichment.claims),
            semantic_relation_count=len(enrichment.relations),
            summary_count=len(enrichment.summaries),
            consistency_valid=True,
            rebuilt=rebuilt,
        )
        return report, enrichment_path

    def _persist_enrichment(self, bundle: CanonicalBundle, enrichment: SemanticEnrichment) -> Path:
        path = self.paths.cognee / "enrichment" / f"{bundle.snapshot.id}.json"
        _atomic_json(path, enrichment.model_dump(mode="json"))
        return path


def _validate_semantic_provenance(bundle: CanonicalBundle, enrichment: SemanticEnrichment) -> None:
    chunk_ids = {chunk.id for chunk in bundle.chunks}
    for entity in enrichment.entities:
        _validate_provenance(entity.id, entity.source_chunk_ids, chunk_ids)
    for claim in enrichment.claims:
        _validate_provenance(claim.id, claim.source_chunk_ids, chunk_ids)
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
