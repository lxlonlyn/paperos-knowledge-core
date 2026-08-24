"""Permanent contracts for the single Chunk-first search architecture."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from paperos_core.adapters.cognee.llm import _SectionExtractionWithoutClaims
from paperos_core.domain.canonical import Chunk
from paperos_core.retrieval.candidates import Candidate, QueryRequest
from paperos_core.retrieval.evidence import format_evidence
from paperos_core.retrieval.expansion import local_neighbor_expand
from paperos_core.retrieval.fusion import weighted_rrf


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
    request = QueryRequest(
        query="paper title comparison limitations",
        document_ids=["document_1"],
        expand_context=True,
        expand_graph=True,
    )
    assert request.document_ids == ["document_1"]


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
    document = SimpleNamespace(source_file_id="source_1", title="Canonical Paper")
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
