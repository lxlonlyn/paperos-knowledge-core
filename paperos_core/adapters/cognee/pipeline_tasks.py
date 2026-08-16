"""Cognee custom pipeline tasks for PaperOS academic knowledge ingestion.

The custom pipeline runs inside ``cognee.run_custom_pipeline``:

    AcademicChunkTask -> ScholarlyIdentityTask -> SemanticEnrichmentTask
    -> DataPointMappingTask
    -> add_data_points

PaperOS decides the academic chunking rules and the enrichment schema; Cognee
executes the pipeline, provides the tokenizer and token limits, and owns the
final DataPoint write.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperos_core.adapters.cognee.compat import (
    CogneeCompatibilityAdapter,
    cognee_uuid,
    resolve_cognee_tokenizer,
    task,
)
from paperos_core.adapters.cognee.llm import LLMClient
from paperos_core.adapters.cognee.models import (
    DataPointGraph,
    canonical_to_datapoints,
)
from paperos_core.domain.canonical import CanonicalBundle, ChunkProjection
from paperos_core.domain.knowledge import SemanticEnrichment
from paperos_core.domain.scholarly import ScholarlyContext
from paperos_core.errors import CogneeStorageError
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.chunking import build_chunks
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry


@dataclass(slots=True)
class ChunkedBundle:
    bundle: CanonicalBundle
    projection: ChunkProjection


@dataclass(slots=True)
class IdentityBoundBundle:
    bundle: CanonicalBundle
    projection: ChunkProjection
    scholarly: ScholarlyContext


@dataclass(slots=True)
class EnrichedBundle:
    bundle: CanonicalBundle
    projection: ChunkProjection
    enrichment: SemanticEnrichment
    scholarly: ScholarlyContext


async def academic_chunk_task(
    data: list[Any],
    ctx: Any = None,
    *,
    repository: CanonicalRepository,
    chunk_target_tokens: int,
    chunk_overlap_tokens: int,
) -> list[ChunkedBundle]:
    """Produce canonical chunks from sections/elements and persist them."""
    tokenizer = resolve_cognee_tokenizer()
    results: list[ChunkedBundle] = []
    for item in data:
        bundle = getattr(item, "bundle", item)
        chunks = build_chunks(
            document_id=bundle.document.id,
            snapshot_id=bundle.snapshot.id,
            sections=bundle.sections,
            elements=bundle.elements,
            target_tokens=chunk_target_tokens,
            overlap_tokens=chunk_overlap_tokens,
            tokenizer=tokenizer,
        )
        repository.save_chunks(bundle.snapshot.id, chunks)
        results.append(
            ChunkedBundle(
                bundle=bundle,
                projection=ChunkProjection(
                    snapshot_id=bundle.snapshot.id,
                    chunks=chunks,
                ),
            )
        )
    return results


async def scholarly_identity_task(
    data: list[ChunkedBundle],
    ctx: Any = None,
    *,
    scholarly_registry: ScholarlyRegistry,
) -> list[IdentityBoundBundle]:
    """Resolve and persist Work identities before any semantic or graph task."""
    return [
        IdentityBoundBundle(
            bundle=item.bundle,
            projection=item.projection,
            scholarly=scholarly_registry.resolve_bundle(
                item.bundle, item.projection.chunks
            ),
        )
        for item in data
    ]


async def semantic_enrichment_task(
    data: list[IdentityBoundBundle],
    ctx: Any = None,
    *,
    llm: LLMClient,
    enrichment_root: Path,
    reuse_existing: bool = False,
    generate_if_missing: bool = True,
) -> list[EnrichedBundle]:
    """Reuse validated enrichment or generate it only when explicitly allowed."""
    results: list[EnrichedBundle] = []
    for chunked in data:
        enrichment_path = enrichment_root / f"{chunked.bundle.snapshot.id}.json"
        if reuse_existing and enrichment_path.is_file():
            enrichment = _load_enrichment(enrichment_path)
        elif not generate_if_missing:
            raise CogneeStorageError(
                "Semantic enrichment artifact is missing and generation is disabled.",
                affected=enrichment_path,
            )
        else:
            enrichment = await llm.enrich(
                chunked.bundle, chunked.projection.chunks
            )
            _persist_enrichment(
                enrichment_root, chunked.bundle.snapshot.id, enrichment
            )
        _validate_semantic_provenance(chunked.projection.chunks, enrichment)
        results.append(
            EnrichedBundle(
                bundle=chunked.bundle,
                projection=chunked.projection,
                enrichment=enrichment,
                scholarly=chunked.scholarly,
            )
        )
    return results


async def datapoint_mapping_task(
    data: list[EnrichedBundle],
    ctx: Any = None,
    *,
    graph_root: Path,
    graph_results: list[DataPointGraph] | None = None,
) -> list[DataPointGraph]:
    """Map one canonical/enrichment pair to a Cognee DataPoint graph."""
    results: list[DataPointGraph] = []
    for enriched in data:
        graph = canonical_to_datapoints(
            enriched.bundle,
            enriched.projection.chunks,
            enriched.enrichment,
            enriched.scholarly,
        )
        _persist_graph(graph_root, enriched.bundle.snapshot.id, graph)
        if graph_results is not None:
            graph_results.append(graph)
        results.append(graph)
    return results


async def store_datapoints_task(
    data: list[DataPointGraph],
    ctx: Any = None,
    *,
    compat: CogneeCompatibilityAdapter,
) -> list[DataPointGraph]:
    """Write mapped DataPoints and typed edges through Cognee's add_data_points."""
    results: list[DataPointGraph] = []
    for graph in data:
        custom_edges = [
            _custom_edge(relation, graph.id_mapping) for relation in graph.relations
        ]
        # Single triplet representation: PaperOS TripletDataPoint carries
        # canonical_id, source_chunk_ids, and derived_from_ids. Cognee's
        # embed_triplets=True produces Triplet(text, from_node_id, to_node_id)
        # without stable canonical IDs or chunk provenance, so enabling it
        # would store a second, lower-fidelity triplet set. Keep it disabled
        # until a compatibility experiment proves Cognee's Triplet preserves
        # provenance.
        await compat.add_data_points(
            graph.nodes,
            custom_edges=custom_edges,
            embed_triplets=False,
            ctx=ctx,
        )
        results.append(graph)
    return results


def configure_pipeline_tasks(
    *,
    repository: CanonicalRepository,
    scholarly_registry: ScholarlyRegistry,
    compat: CogneeCompatibilityAdapter,
    llm: LLMClient,
    enrichment_root: Path,
    graph_root: Path,
    chunk_target_tokens: int,
    chunk_overlap_tokens: int,
    graph_results: list[DataPointGraph],
    reuse_existing_enrichment: bool,
    generate_enrichment_if_missing: bool,
) -> list[Any]:
    """Bind per-run dependencies and return the Cognee Task list."""
    return [
        task(
            academic_chunk_task,
            batch_size=1,
            repository=repository,
            chunk_target_tokens=chunk_target_tokens,
            chunk_overlap_tokens=chunk_overlap_tokens,
        ).task,
        task(
            scholarly_identity_task,
            batch_size=1,
            scholarly_registry=scholarly_registry,
        ).task,
        task(
            semantic_enrichment_task,
            batch_size=1,
            llm=llm,
            enrichment_root=enrichment_root,
            reuse_existing=reuse_existing_enrichment,
            generate_if_missing=generate_enrichment_if_missing,
        ).task,
        task(
            datapoint_mapping_task,
            batch_size=1,
            graph_root=graph_root,
            graph_results=graph_results,
        ).task,
        task(
            store_datapoints_task,
            batch_size=1,
            compat=compat,
        ).task,
    ]


def _custom_edge(
    relation: Any, id_mapping: dict[str, str]
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


def _validate_semantic_provenance(
    chunks: list[Any], enrichment: SemanticEnrichment
) -> None:
    chunk_ids = {chunk.id for chunk in chunks}
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


def _persist_enrichment(
    root: Path, snapshot_id: str, enrichment: SemanticEnrichment
) -> Path:
    path = root / f"{snapshot_id}.json"
    _atomic_json(path, enrichment.model_dump(mode="json"))
    return path


def _load_enrichment(path: Path) -> SemanticEnrichment:
    try:
        return SemanticEnrichment.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise CogneeStorageError(
            f"Unable to read semantic enrichment artifact: {exc}",
            affected=path,
        ) from exc


def _persist_graph(
    root: Path, snapshot_id: str, graph: DataPointGraph
) -> Path:
    path = root / f"{snapshot_id}.json"
    _atomic_json(path, graph.to_json())
    return path


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
