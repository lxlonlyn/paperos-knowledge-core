"""Section-aware enrichment grouping, batching, and merge behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from paperos_core.adapters.llm import (
    _chunk_batches,
    _chunks_by_section,
    _ClaimExtraction,
    _EntityExtraction,
    _merge_section_extractions,
    _RelationExtraction,
    _SectionExtraction,
)
from paperos_core.domain.canonical import (
    CanonicalBundle,
    CanonicalSnapshot,
    Chunk,
    Document,
    Section,
)
from paperos_core.errors import SemanticEnrichmentError


def _bundle() -> CanonicalBundle:
    snapshot = CanonicalSnapshot(
        id="snapshot_test",
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
        canonical_snapshot_id=snapshot.id,
        language="en",
        title="Test",
    )
    first = Section(
        id="section_first",
        document_id=document.id,
        canonical_snapshot_id=snapshot.id,
        title="First",
        level=1,
        order=0,
        path="/First",
    )
    second = Section(
        id="section_second",
        document_id=document.id,
        canonical_snapshot_id=snapshot.id,
        title="Second",
        level=1,
        order=1,
        path="/Second",
    )

    def chunk(chunk_id: str, section_id: str | None, text: str) -> Chunk:
        return Chunk(
            id=chunk_id,
            document_id=document.id,
            canonical_snapshot_id=snapshot.id,
            text=text,
            order=0,
            element_ids=[f"element_{chunk_id}"],
            element_span_ids=[f"element_{chunk_id}:0"],
            section_id=section_id,
            section_path=section_id or None,
            token_count=max(1, len(text.split())),
        )

    chunks = [
        chunk("chunk_front", None, "front matter text"),
        chunk("chunk_a", first.id, "first section content alpha"),
        chunk("chunk_b", first.id, "first section content beta"),
        chunk("chunk_c", second.id, "second section content gamma"),
    ]
    return CanonicalBundle(
        snapshot=snapshot,
        document=document,
        sections=[first, second],
        elements=[],
        references=[],
        warnings=[],
    ), chunks


def test_chunks_are_grouped_by_section_in_document_order() -> None:
    bundle, chunks = _bundle()
    groups = _chunks_by_section(chunks, bundle.sections)
    assert [section_id for section_id, _chunks in groups] == [
        None,
        "section_first",
        "section_second",
    ]
    by_id = {chunk.id: section_id for section_id, chunks in groups for chunk in chunks}
    assert by_id == {
        "chunk_front": None,
        "chunk_a": "section_first",
        "chunk_b": "section_first",
        "chunk_c": "section_second",
    }


def test_batches_cover_every_chunk_exactly_once() -> None:
    _, chunks = _bundle()
    batches = _chunk_batches(chunks, character_budget=120)
    assert len(batches) > 1
    seen = [chunk.id for batch in batches for chunk in batch]
    assert len(seen) == len(set(seen)) == len(chunks)


def test_merge_deduplicates_entities_claims_and_resolves_relations() -> None:
    bundle, chunks = _bundle()
    first_batch = _SectionExtraction(
        entities=[
            _EntityExtraction(
                key="e1",
                name="Graph",
                entity_type="method",
                source_chunk_ids=["chunk_a"],
            ),
            _EntityExtraction(
                key="e2",
                name="Loss",
                entity_type="concept",
                source_chunk_ids=["chunk_a"],
            ),
        ],
        claims=[
            _ClaimExtraction(
                key="c1",
                text="Graph uses loss.",
                source_chunk_ids=["chunk_a"],
            )
        ],
        relations=[
            _RelationExtraction(
                source_key="e1",
                target_key="e2",
                relation_type="USES",
                source_chunk_ids=["chunk_a"],
            )
        ],
    )
    second_batch = _SectionExtraction(
        entities=[
            _EntityExtraction(
                key="x",
                name="Graph",
                entity_type="method",
                source_chunk_ids=["chunk_b"],
            ),
            _EntityExtraction(
                key="y",
                name="Data",
                entity_type="concept",
                source_chunk_ids=["chunk_b"],
            ),
        ],
        claims=[
            _ClaimExtraction(
                key="c2",
                text="Graph uses loss.",
                source_chunk_ids=["chunk_b"],
            )
        ],
        relations=[
            _RelationExtraction(
                source_key="x",
                target_key="y",
                relation_type="RELATED_TO",
                source_chunk_ids=["chunk_b"],
            )
        ],
    )
    entities, claims, relations = _merge_section_extractions(
        bundle,
        [
            ("section_first", [chunks[1]], first_batch),
            ("section_second", [chunks[2]], second_batch),
        ],
        model="test-model",
    )
    graph = next(entity for entity in entities if entity.name == "Graph")
    assert graph.source_chunk_ids == ["chunk_a", "chunk_b"]
    assert graph.model == "test-model"
    assert len(entities) == 3
    assert len(claims) == 1
    assert claims[0].source_chunk_ids == ["chunk_a", "chunk_b"]
    assert len(relations) == 2
    uses = next(relation for relation in relations if relation.relation_type == "USES")
    assert uses.source_object_id == graph.id


def test_merge_rejects_chunk_ids_outside_the_batch() -> None:
    bundle, chunks = _bundle()
    extraction = _SectionExtraction(
        entities=[
            _EntityExtraction(
                key="e1",
                name="Ghost",
                entity_type="concept",
                source_chunk_ids=["chunk_c"],  # not in this batch
            )
        ],
        claims=[],
        relations=[],
    )
    with pytest.raises(SemanticEnrichmentError, match="outside the supplied evidence"):
        _merge_section_extractions(
            bundle,
            [("section_first", [chunks[1]], extraction)],
            model="test-model",
        )
