"""Custom pipeline task contract tests (Cognee external calls are faked)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

from paperos_core.adapters.cognee.models import DataPointGraph
from paperos_core.adapters.cognee.pipeline_tasks import (
    ChunkedBundle,
    EnrichedBundle,
    _custom_edge,
    academic_chunk_task,
    datapoint_mapping_task,
    semantic_enrichment_task,
    store_datapoints_task,
)
from paperos_core.domain.canonical import (
    CanonicalBundle,
    CanonicalSnapshot,
    ChunkProjection,
    Document,
    Element,
    Section,
    SourceSpan,
)
from paperos_core.domain.enums import ElementType
from paperos_core.domain.knowledge import SemanticEnrichment
from paperos_core.domain.provenance import RelationRecord


class _Tokenizer:
    def count_tokens(self, text: str) -> int:
        return max(1, len(text.split()))


def _bundle(snapshot_id: str = "snapshot_test") -> CanonicalBundle:
    snapshot = CanonicalSnapshot(
        id=snapshot_id,
        source_file_id="src_test",
        parse_run_id="parse_test",
        document_id="doc_test",
        manifest_path=Path("/tmp/manifest.json"),
        dataset_id="papers",
    )
    document = Document(
        id="doc_test",
        source_file_id="src_test",
        parse_run_id="parse_test",
        canonical_snapshot_id=snapshot_id,
        language="en",
        title="Test paper",
    )
    section = Section(
        id="section_intro",
        document_id="doc_test",
        canonical_snapshot_id=snapshot_id,
        title="Introduction",
        level=1,
        order=0,
        path="/Introduction",
    )
    elements = [
        Element(
            id=f"element_{index}",
            document_id="doc_test",
            canonical_snapshot_id=snapshot_id,
            element_type=ElementType.PARAGRAPH,
            order=index,
            section_id=section.id,
            text=("word " * 60).strip(),
            page=1,
            source_span=SourceSpan(artifact_id="artifact", item_index=index, page=1),
        )
        for index in range(3)
    ]
    return CanonicalBundle(
        snapshot=snapshot,
        document=document,
        sections=[section],
        elements=elements,
        references=[],
        warnings=[],
    )


def _enrichment(snapshot_id: str) -> SemanticEnrichment:
    return SemanticEnrichment(
        entities=[],
        claims=[],
        relations=[],
        summaries=[],
        model="example-model",
        model_version="example-model",
        prompt_name="semantic_enrichment",
        prompt_version="1",
        prompt_sha256="0" * 64,
    )


async def test_academic_chunk_task_builds_and_persists_chunks(monkeypatch) -> None:
    saved: list[tuple[str, list[object]]] = []

    class _Repository:
        def save_chunks(self, snapshot_id: str, chunks: list[object]) -> Path:
            saved.append((snapshot_id, chunks))
            return Path("/tmp/chunks.jsonl")

    monkeypatch.setattr(
        "paperos_core.adapters.cognee.pipeline_tasks.resolve_cognee_tokenizer",
        lambda: _Tokenizer(),
    )
    bundle = _bundle()
    results = await academic_chunk_task(
        [bundle],
        repository=_Repository(),
        chunk_target_tokens=20,
        chunk_overlap_tokens=4,
    )
    assert len(results) == 1
    chunks = results[0].projection.chunks
    assert chunks
    assert saved and saved[0][0] == bundle.snapshot.id
    assert saved[0][1] == chunks
    assert all(chunk.token_count and chunk.token_count <= 20 for chunk in chunks)
    assert all(chunk.section_id == "section_intro" for chunk in chunks)


async def test_semantic_enrichment_task_persists_through_llm_client(
    tmp_path: Path,
) -> None:
    llm = AsyncMock()
    bundle = _bundle()
    enrichment = _enrichment(bundle.snapshot.id)
    llm.enrich = AsyncMock(return_value=enrichment)
    chunked = ChunkedBundle(
        bundle=bundle,
        projection=ChunkProjection(snapshot_id=bundle.snapshot.id, chunks=[]),
    )
    results = await semantic_enrichment_task(
        [chunked],
        llm=llm,
        enrichment_root=tmp_path,
    )
    assert len(results) == 1
    assert results[0].enrichment == enrichment
    payload = json.loads(
        (tmp_path / f"{bundle.snapshot.id}.json").read_text(encoding="utf-8")
    )
    assert payload["model"] == "example-model"
    llm.enrich.assert_awaited_once_with(bundle, [])


async def test_datapoint_mapping_task_builds_graph_without_external_calls(
    tmp_path: Path,
) -> None:
    bundle = _bundle()
    projection = ChunkProjection(snapshot_id=bundle.snapshot.id, chunks=[])
    enriched = EnrichedBundle(
        bundle=bundle,
        projection=projection,
        enrichment=_enrichment(bundle.snapshot.id),
    )

    class _Repository:
        def list_bundles(self):
            return [bundle]

    results = await datapoint_mapping_task(
        [enriched],
        repository=_Repository(),
        graph_root=tmp_path,
    )
    assert len(results) == 1
    graph = results[0]
    assert graph.nodes
    assert graph.id_mapping
    assert graph.relations
    assert (tmp_path / f"{bundle.snapshot.id}.json").is_file()


async def test_store_datapoints_task_writes_single_triplet_representation() -> None:
    compat = AsyncMock()
    compat.add_data_points = AsyncMock(return_value=[])
    graph_obj = DataPointGraph(
        nodes=[],
        relations=[
            RelationRecord(
                source_id="source_a",
                target_id="target_b",
                relation_type="HAS_SECTION",
            )
        ],
    )
    results = await store_datapoints_task([graph_obj], compat=compat)
    assert results == [graph_obj]
    compat.add_data_points.assert_awaited_once()
    _, kwargs = compat.add_data_points.await_args
    assert kwargs["embed_triplets"] is False
    assert kwargs["custom_edges"] == [
        _custom_edge(graph_obj.relations[0], graph_obj.id_mapping)
    ]
