"""Permanent contracts for the single Chunk-first search architecture."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from paperos_core.adapters.cognee import datapoints
from paperos_core.adapters.cognee.compat import (
    CogneeSemanticRelation,
    _direct_semantic_relations,
    _vector_groups,
    cognee_uuid,
)
from paperos_core.adapters.cognee.llm import _SectionExtractionWithoutClaims
from paperos_core.domain.canonical import Chunk
from paperos_core.domain.provenance import SEMANTIC_RELATION_TYPES, RelationType
from paperos_core.errors import ConfigurationError
from paperos_core.retrieval.candidates import Candidate, QueryRequest
from paperos_core.retrieval.evidence import format_evidence
from paperos_core.retrieval.expansion import (
    local_neighbor_expand,
    semantic_post_hit_expand,
)
from paperos_core.retrieval.fusion import weighted_rrf
from paperos_core.retrieval.service import RetrievalService


def _candidate(chunk_id: str, channel: str, *, candidate_id: str) -> Candidate:
    return Candidate(
        id=candidate_id,
        object_id=candidate_id,
        object_type="chunk",
        document_id="document_1",
        source_file_id="source_1",
        source_filename="paper.pdf",
        canonical_snapshot_id="snapshot_1",
        chunk_id=chunk_id,
        text="candidate payload",
        channels=[channel],
        channel_scores={channel: 1.0},
    )


def _chunk(
    chunk_id: str,
    order: int,
    *,
    region: str = "MAIN",
    major: str = "section_1",
    previous: str | None = None,
    next_: str | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        document_id="document_1",
        canonical_snapshot_id="snapshot_1",
        text=f"canonical {chunk_id}",
        order=order,
        element_ids=[f"element_{order}"],
        document_region=region,
        major_section_id=major,
        previous_chunk_id=previous,
        next_chunk_id=next_,
    )


def test_query_request_rejects_removed_routing_fields() -> None:
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"query": "compare limitations", "profile": "truth"})
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"query": "paper title", "scope": {}})
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"query": "paper title", "source_work_ids": ["work_1"]})
    with pytest.raises(ValidationError):
        QueryRequest.model_validate({"query": "paper title", "subject_work_ids": ["work_1"]})
    request = QueryRequest(
        query="paper title comparison limitations",
        document_ids=["document_1"],
        expand_context=True,
        expand_graph=True,
    )
    assert request.document_ids == ["document_1"]
    assert set(QueryRequest.model_fields) == {
        "query",
        "dataset",
        "top_k",
        "document_ids",
        "work_ids",
        "expand_context",
        "expand_graph",
    }


def test_semantic_relation_grammar_is_central_and_excludes_infrastructure() -> None:
    assert SEMANTIC_RELATION_TYPES == {
        RelationType.MENTIONS,
        RelationType.SUPPORTS,
        RelationType.CONTRADICTS,
        RelationType.USES,
        RelationType.EXTENDS,
        RelationType.COMPARES_WITH,
        RelationType.EVALUATES_ON,
        RelationType.PROPOSES,
        RelationType.RELATED_TO,
    }
    assert not SEMANTIC_RELATION_TYPES.intersection(
        {
            RelationType.CITES,
            RelationType.ABOUT,
            RelationType.DERIVED_FROM,
            RelationType.HAS_SECTION,
            RelationType.HAS_CHUNK,
            RelationType.HAS_ELEMENT,
            RelationType.HAS_REFERENCE,
            RelationType.RESOLVES_TO,
        }
    )


def test_direct_semantic_operator_rejects_indirect_and_infrastructure_paths() -> None:
    seed_id = "chunk_seed"
    seed_graph_id = str(cognee_uuid(seed_id))
    entity_a = "entity-a"
    entity_b = "entity-b"
    entity_c = "entity-c"
    layout = "section-layout"
    work_a = "work-a"
    work_b = "work-b"
    nodes = [
        (seed_graph_id, {"type": "ChunkDataPoint", "canonical_id": seed_id}),
        (
            entity_a,
            {
                "type": "EntityDataPoint",
                "canonical_id": "entity_a",
                "source_chunk_ids": [seed_id, "source_entity_mention"],
            },
        ),
        (
            entity_b,
            {
                "type": "EntityDataPoint",
                "canonical_id": "entity_b",
                "source_chunk_ids": ["target_entity_mention"],
            },
        ),
        (
            entity_c,
            {
                "type": "EntityDataPoint",
                "canonical_id": "entity_c",
                "source_chunk_ids": ["chunk_indirect"],
            },
        ),
        (layout, {"type": "SectionDataPoint", "canonical_id": "section_1"}),
        (work_a, {"type": "ScholarlyWorkDataPoint", "canonical_id": "work_a"}),
        (work_b, {"type": "ScholarlyWorkDataPoint", "canonical_id": "work_b"}),
    ]
    edges = [
        (entity_a, seed_graph_id, "DERIVED_FROM", {}),
        (
            entity_a,
            entity_b,
            "USES",
            {"source_chunk_ids": [seed_id, "chunk_target"]},
        ),
        (entity_b, entity_c, "EXTENDS", {"source_chunk_ids": ["chunk_indirect"]}),
        (entity_a, layout, "MENTIONS", {"source_chunk_ids": [seed_id]}),
        (layout, entity_c, "RELATED_TO", {"source_chunk_ids": ["chunk_indirect"]}),
        (work_a, work_b, "CITES", {"source_chunk_ids": [seed_id]}),
    ]
    relations = _direct_semantic_relations(
        nodes,
        edges,
        seed_chunk_ids={seed_id},
        relation_types={item.value for item in SEMANTIC_RELATION_TYPES},
        limit=20,
    )
    assert [
        (item.source_canonical_id, item.relation_type, item.target_canonical_id)
        for item in relations
    ] == [("entity_a", "USES", "entity_b")]
    assert relations[0].source_chunk_ids == (seed_id, "chunk_target")


def test_direct_semantic_operator_requires_relation_level_source_provenance() -> None:
    seed_id = "chunk_seed"
    seed_graph_id = str(cognee_uuid(seed_id))
    nodes = [
        (seed_graph_id, {"type": "ChunkDataPoint", "canonical_id": seed_id}),
        (
            "entity-a",
            {
                "type": "EntityDataPoint",
                "canonical_id": "entity_a",
                "source_chunk_ids": [seed_id, "source_entity_mention"],
            },
        ),
        (
            "entity-b",
            {
                "type": "EntityDataPoint",
                "canonical_id": "entity_b",
                "source_chunk_ids": ["target_entity_mention"],
            },
        ),
    ]
    diagnostics = {
        "raw_neighbor_nodes": 0,
        "filtered_grounded_entities": 0,
        "filtered_semantic_relations": 0,
        "missing_relation_source_provenance": 0,
    }
    relations = _direct_semantic_relations(
        nodes,
        [("entity-a", "entity-b", "USES", {})],
        seed_chunk_ids={seed_id},
        relation_types={"USES"},
        limit=20,
        diagnostics=diagnostics,
    )
    assert relations == []
    assert diagnostics == {
        "raw_neighbor_nodes": 3,
        "filtered_grounded_entities": 1,
        "filtered_semantic_relations": 0,
        "missing_relation_source_provenance": 1,
    }


def test_semantic_expansion_is_dataset_scoped_and_rehydrates_canonical_chunks() -> None:
    target = _chunk("chunk_target", 2)
    chunks = {target.id: target}

    class Corpus:
        def __init__(self) -> None:
            self.chunks = chunks

        def candidate_for_chunk(self, chunk_id: str, **kwargs: object) -> Candidate:
            candidate = _candidate(
                chunk_id,
                str(kwargs["channel"]),
                candidate_id=chunk_id,
            )
            candidate.text = self.chunks[chunk_id].text
            candidate.derived_from_ids = list(kwargs["derived_from_ids"])
            candidate.relation_types = list(kwargs["relation_types"])
            return candidate

    class Compat:
        def __init__(self) -> None:
            self.dataset_name: str | None = None
            self.relation_types: set[str] = set()

        async def semantic_relations_for_chunks(
            self,
            chunk_ids: list[str],
            *,
            dataset_name: str,
            relation_types: set[str],
            limit: int,
        ) -> list[CogneeSemanticRelation]:
            assert chunk_ids == ["chunk_seed"]
            assert limit == 10
            self.dataset_name = dataset_name
            self.relation_types = relation_types
            return [
                CogneeSemanticRelation(
                    source_canonical_id="entity_a",
                    target_canonical_id="entity_b",
                    relation_type="USES",
                    grounded_object_ids=("entity_a",),
                    source_chunk_ids=("missing_chunk", "chunk_target"),
                    derived_from_ids=("entity_a", "entity_b"),
                    score=1.0,
                )
            ]

    compat = Compat()
    expanded = asyncio.run(
        semantic_post_hit_expand(
            compat,  # type: ignore[arg-type]
            Corpus(),  # type: ignore[arg-type]
            [_candidate("chunk_seed", "vector", candidate_id="chunk_seed")],
            dataset_name="dataset_exact",
            document_ids={"document_1"},
            limit=10,
        )
    )
    assert compat.dataset_name == "dataset_exact"
    assert compat.relation_types == {item.value for item in SEMANTIC_RELATION_TYPES}
    assert [item.chunk_id for item in expanded] == ["chunk_target"]
    assert expanded[0].text == target.text
    assert expanded[0].relation_types == ["USES"]


def test_rrf_deduplicates_by_canonical_chunk_id() -> None:
    fused = weighted_rrf(
        {
            "lexical": [_candidate("chunk_1", "lexical", candidate_id="lex_1")],
            "vector": [_candidate("chunk_1", "vector", candidate_id="vec_1")],
        }
    )
    assert len(fused) == 1
    assert fused[0].id == "chunk_1"
    assert fused[0].channels == ["lexical", "vector"]
    assert fused[0].channel_ranks == {"lexical": 1, "vector": 1}


def test_local_expansion_respects_region_and_major_section() -> None:
    anchor = _chunk("chunk_2", 2, previous="chunk_1", next_="chunk_3")
    chunks = {
        "chunk_1": _chunk("chunk_1", 1, region="REFERENCES", next_="chunk_2"),
        "chunk_2": anchor,
        "chunk_3": _chunk("chunk_3", 3, previous="chunk_2"),
    }

    class Corpus:
        def __init__(self) -> None:
            self.chunks = chunks

        def candidate_for_chunk(self, chunk_id: str, **kwargs: object) -> Candidate:
            return _candidate(
                chunk_id,
                str(kwargs["channel"]),
                candidate_id=chunk_id,
            )

    expanded = local_neighbor_expand(
        Corpus(),
        [_candidate("chunk_2", "vector", candidate_id="chunk_2")],
        document_ids={"document_1"},
    )
    assert [item.chunk_id for item in expanded] == ["chunk_3"]


def test_claim_disabled_schema_has_no_claim_output_field() -> None:
    properties = _SectionExtractionWithoutClaims.model_json_schema()["properties"]
    assert set(properties) == {"entities", "relations"}


def test_evidence_rehydrates_canonical_chunk_text() -> None:
    chunk = _chunk("chunk_1", 1)
    document = SimpleNamespace(
        source_file_id="source_1",
        title="Canonical Paper",
        authors=[SimpleNamespace(display_name="Ada Author")],
        year=2025,
    )
    bundle = SimpleNamespace(document=document)
    corpus = SimpleNamespace(
        chunks={chunk.id: chunk},
        chunk_bundles={chunk.id: bundle},
        source_filenames={"source_1": "paper.pdf"},
        work_id_by_document={"document_1": "work_1"},
    )
    candidate = _candidate("chunk_1", "vector", candidate_id="derived_1")
    candidate.text = "untrusted derived text"
    evidence = format_evidence([candidate], corpus)
    assert evidence[0].text == chunk.text
    assert evidence[0].document_id == chunk.document_id
    assert evidence[0].authors == ["Ada Author"]
    assert evidence[0].year == 2025


def _retrieval_service(*, rerank_enabled: bool) -> RetrievalService:
    service = object.__new__(RetrievalService)
    service.config = SimpleNamespace(
        dataset="dataset_1",
        retrieval=SimpleNamespace(
            top_k=2,
            candidate_pool_size=4,
            rerank_enabled=rerank_enabled,
            synthesis_max_input_tokens=48_000,
        ),
    )
    service.paths = SimpleNamespace()
    service.canonical_repository = object()
    service.registry = object()
    service.scholarly_registry = object()
    service.search = object()
    service.compat = object()
    service.index_manager = SimpleNamespace(lexical=object())
    service.model_client = object()
    service.llm = SimpleNamespace(model="test-llm")
    return service


@pytest.mark.parametrize(
    ("expand_context", "expand_graph"),
    [(True, False), (False, True)],
)
def test_expansion_is_rejected_when_reranker_is_disabled(
    expand_context: bool, expand_graph: bool
) -> None:
    service = _retrieval_service(rerank_enabled=False)
    with pytest.raises(ConfigurationError, match="requires retrieval.rerank_enabled=true"):
        asyncio.run(
            service.query(
                QueryRequest(
                    query="test",
                    expand_context=expand_context,
                    expand_graph=expand_graph,
                )
            )
        )


def _pipeline_corpus() -> SimpleNamespace:
    return SimpleNamespace(
        chunks={
            "chunk_seed": _chunk("chunk_seed", 1, next_="chunk_new"),
            "chunk_new": _chunk("chunk_new", 2, previous="chunk_seed"),
        },
        filtered_document_ids=lambda _ids, _dataset: {"document_1"},
        document_ids_for_works=lambda _ids: {"document_1"},
    )


def _run_expansion_pipeline(
    monkeypatch: pytest.MonkeyPatch, expanded: list[Candidate]
) -> tuple[object, list[list[str]], list[str]]:
    import paperos_core.retrieval.service as service_module

    seed = _candidate("chunk_seed", "lexical", candidate_id="chunk_seed")
    service = _retrieval_service(rerank_enabled=True)
    rerank_inputs: list[list[str]] = []
    synthesis_prompts: list[str] = []

    async def rerank(
        _query: str, candidates: list[Candidate], *, limit: int
    ) -> list[Candidate]:
        rerank_inputs.append([item.chunk_id for item in candidates])
        return candidates[:limit]

    async def no_vector(*_args: object, **_kwargs: object) -> list[Candidate]:
        return []

    async def answer(*_args: object, **kwargs: object) -> str:
        synthesis_prompts.append(str(kwargs["prompt"]))
        return "grounded answer"

    service._rerank = rerank  # type: ignore[method-assign]
    monkeypatch.setattr(service_module.CorpusView, "load", lambda *_args: _pipeline_corpus())
    monkeypatch.setattr(service_module, "lexical_retrieve", lambda *_args, **_kwargs: [seed])
    monkeypatch.setattr(service_module, "semantic_retrieve", no_vector)
    monkeypatch.setattr(
        service_module, "local_neighbor_expand", lambda *_args, **_kwargs: expanded
    )
    monkeypatch.setattr(service_module, "format_evidence", lambda *_args: [])
    monkeypatch.setattr(service_module, "synthesize_answer", answer)
    response = asyncio.run(
        service.query(QueryRequest(query="test", expand_context=True, top_k=2))
    )
    return response, rerank_inputs, synthesis_prompts


def test_new_expanded_chunk_enters_second_rerank_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expanded = [_candidate("chunk_new", "local_expansion", candidate_id="chunk_new")]
    response, rerank_inputs, synthesis_prompts = _run_expansion_pipeline(
        monkeypatch, expanded
    )
    assert rerank_inputs == [["chunk_seed"], ["chunk_seed", "chunk_new"]]
    assert response.trace.local_new_chunk_ids == ["chunk_new"]
    assert response.trace.second_rerank_candidate_ids == ["chunk_seed", "chunk_new"]
    assert "second_rerank" in response.stages
    assert response.replay.original_query == "test"
    assert response.replay.replay_text == synthesis_prompts[0]


def test_duplicate_only_expansion_skips_second_rerank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = [_candidate("chunk_seed", "local_expansion", candidate_id="duplicate")]
    response, rerank_inputs, _synthesis_prompts = _run_expansion_pipeline(
        monkeypatch, duplicate
    )
    assert rerank_inputs == [["chunk_seed"]]
    assert response.trace.local_new_chunk_ids == []
    assert response.trace.second_rerank_candidate_ids == []
    assert "second_rerank" not in response.stages


def test_only_chunk_datapoints_declare_vector_indexes() -> None:
    indexed = {
        name: model.model_fields["metadata"].default["index_fields"]
        for name, model in vars(datapoints).items()
        if isinstance(model, type)
        and issubclass(model, datapoints.PaperOSGraphDataPoint)
        and "metadata" in model.model_fields
        and model.model_fields["metadata"].default["index_fields"]
    }
    assert indexed == {"ChunkDataPoint": ["text"]}
    assert not hasattr(datapoints, "TripletDataPoint")
    assert "TRIPLET_SOURCE" not in RelationType.__members__
    assert "TRIPLET_TARGET" not in RelationType.__members__

    chunk_node_type = type("ChunkDataPoint", (), {})
    chunk_node = chunk_node_type()
    chunk_node.metadata = {"index_fields": ["text"]}
    chunk_node.text = "canonical chunk"
    graph = SimpleNamespace(nodes=[chunk_node])
    # The production collection contract remains the Chunk text collection.
    assert set(_vector_groups(graph)) == {"ChunkDataPoint_text"}
