"""Cognee custom pipeline tasks for PaperOS academic knowledge ingestion.

The custom pipeline runs inside ``cognee.run_custom_pipeline``:

    AcademicChunkTask -> SemanticEnrichmentTask -> DataPointMappingTask
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

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter, task
from paperos_core.adapters.cognee.models import (
    DataPointGraph,
    canonical_to_datapoints,
)
from paperos_core.adapters.cognee.reference_resolution import resolve_citations
from paperos_core.adapters.llm import LLMClient
from paperos_core.domain.canonical import CanonicalBundle
from paperos_core.domain.datapoints import cognee_uuid
from paperos_core.domain.knowledge import SemanticEnrichment
from paperos_core.errors import CogneeStorageError
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.chunking import build_chunks, resolve_cognee_tokenizer


@dataclass(slots=True)
class EnrichedBundle:
    bundle: CanonicalBundle
    enrichment: SemanticEnrichment


async def academic_chunk_task(
    data: list[Any],
    ctx: Any = None,
    *,
    repository: CanonicalRepository,
    chunk_target_tokens: int,
    chunk_overlap_tokens: int,
) -> list[CanonicalBundle]:
    """Produce canonical chunks from sections/elements and persist them."""
    tokenizer = resolve_cognee_tokenizer()
    results: list[CanonicalBundle] = []
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
        results.append(bundle.model_copy(update={"chunks": chunks}))
    return results


async def semantic_enrichment_task(
    data: list[CanonicalBundle],
    ctx: Any = None,
    *,
    llm: LLMClient,
    enrichment_root: Path,
) -> list[EnrichedBundle]:
    """Run PaperOS's enrichment schema through Cognee's LLMGateway."""
    results: list[EnrichedBundle] = []
    for bundle in data:
        await llm.health_check()
        enrichment = await llm.enrich(bundle)
        _validate_semantic_provenance(bundle, enrichment)
        _persist_enrichment(enrichment_root, bundle.snapshot.id, enrichment)
        results.append(EnrichedBundle(bundle=bundle, enrichment=enrichment))
    return results


async def datapoint_mapping_task(
    data: list[EnrichedBundle],
    ctx: Any = None,
    *,
    repository: CanonicalRepository,
) -> list[DataPointGraph]:
    """Map one canonical/enrichment pair to a Cognee DataPoint graph."""
    results: list[DataPointGraph] = []
    for enriched in data:
        graph = canonical_to_datapoints(enriched.bundle, enriched.enrichment)
        graph.relations.extend(
            resolve_citations(enriched.bundle, repository.list_bundles())
        )
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
        # until a contract test proves Cognee's Triplet preserves provenance.
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
    compat: CogneeCompatibilityAdapter,
    llm: LLMClient,
    enrichment_root: Path,
    chunk_target_tokens: int,
    chunk_overlap_tokens: int,
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
            semantic_enrichment_task,
            batch_size=1,
            llm=llm,
            enrichment_root=enrichment_root,
        ).task,
        task(
            datapoint_mapping_task,
            batch_size=1,
            repository=repository,
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
    bundle: CanonicalBundle, enrichment: SemanticEnrichment
) -> None:
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


def _persist_enrichment(
    root: Path, snapshot_id: str, enrichment: SemanticEnrichment
) -> Path:
    path = root / f"{snapshot_id}.json"
    _atomic_json(path, enrichment.model_dump(mode="json"))
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
