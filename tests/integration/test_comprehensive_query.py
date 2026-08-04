"""Live Gate 1-5 acceptance from the genuine four-paper corpus."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path

from fastapi.testclient import TestClient
from paperos_core.api.app import create_app
from paperos_core.application import application_from_config
from paperos_core.config import load_settings
from paperos_core.retrieval.candidates import QueryRequest, QueryResponse


def _file_hashes(roots: list[Path]) -> dict[str, str]:
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }


def _load_query_cases(query_dir: Path) -> list[dict]:
    cases: list[dict] = []
    for name in ("truth.jsonl", "associative.jsonl", "comprehensive.jsonl"):
        cases.extend(
            json.loads(line)
            for line in (query_dir / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    return cases


def _contains_concept(searchable: str, concept: str) -> bool:
    normalized = concept.casefold()
    if normalized in searchable:
        return True
    tokens = re.findall(r"[a-z0-9]+", normalized)
    if not tokens:
        return False
    long_tokens = [token for token in tokens if len(token) >= 4]
    return bool(long_tokens) and all(token[:5] in searchable for token in long_tokens)


def _assert_case(case: dict, response: QueryResponse) -> None:
    assert response.profile.value == case["profile"]
    assert response.expansion.raw_output
    assert set(case.get("required_channels", [])) <= set(response.channels_used)
    assert set(case.get("required_stages", [])) <= set(response.stages)
    assert response.provenance_complete is True
    assert len(response.evidence) == len(response.candidates) > 0
    assert all(candidate.rerank_score is not None for candidate in response.candidates)
    assert all(evidence.chunk_id for evidence in response.evidence)
    assert any(
        evidence.evidence_id in response.answer for evidence in response.evidence
    )
    filenames = {evidence.source_filename for evidence in response.evidence}
    assert set(case["expected_documents"]) <= filenames
    assert response.distinct_documents >= case["minimum_distinct_documents"]
    if case.get("requires_page"):
        assert all(evidence.page_start is not None for evidence in response.evidence)
    if case.get("requires_graph_relation"):
        assert "graph" in response.channels_used
        assert any(
            "graph" in candidate.channels
            or candidate.knowledge_kind == "structured_relation"
            for candidate in response.candidates
        )
    if case.get("requires_inference_labels"):
        assert any(
            evidence.knowledge_kind in {"structured_relation", "system_inference"}
            for evidence in response.evidence
        )
    searchable = " ".join(
        [
            response.answer,
            *(evidence.text for evidence in response.evidence),
            *response.expansion.lexical_queries,
            *response.expansion.semantic_queries,
            *response.expansion.entity_queries,
            *response.expansion.relation_queries,
        ]
    ).casefold()
    for group in case.get("required_evidence_groups", []):
        assert any(term.casefold() in searchable for term in group["any_of"]), group
    for concept in case.get("required_concepts", []):
        assert _contains_concept(searchable, concept), concept


async def _run_live_gate5(
    run_root: Path,
    pdf_dir: Path,
    query_dir: Path,
    papers: list[dict],
    logs: Path,
) -> list[QueryResponse]:
    application = application_from_config(data_dir=run_root)
    try:
        reuse_ingestion = os.getenv("PAPEROS_GATE5_REUSE_INGESTION") == "true"
        if reuse_ingestion:
            bundles = application.canonical_repository.list_bundles()
            assert len(bundles) == len(papers)
            registered = {
                source.original_filename: source
                for source in (
                    application.registry.get_source(bundle.document.source_file_id)
                    for bundle in bundles
                )
            }
            for paper in papers:
                source = registered[paper["pdf_file"]]
                assert source.sha256 == paper["sha256"]
                assert source.storage_path.read_bytes() == (
                    pdf_dir / paper["pdf_file"]
                ).read_bytes()
        else:
            for paper in papers:
                result = await application.ingestion.ingest_pdf_to_knowledge(
                    pdf_dir / paper["pdf_file"]
                )
                (logs / f"ingest-{paper['case_id']}.json").write_text(
                    json.dumps(result.public_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
        protected_roots = [
            application.paths.raw,
            application.paths.parsed,
            application.paths.canonical,
        ]
        before = _file_hashes(protected_roots)
        responses: list[QueryResponse] = []
        for case in _load_query_cases(query_dir):
            output_path = logs / f"query-{case['case_id']}.json"
            if (
                os.getenv("PAPEROS_GATE5_REUSE_QUERIES") == "true"
                and output_path.is_file()
            ):
                stored_response = QueryResponse.model_validate_json(
                    output_path.read_text(encoding="utf-8")
                )
                try:
                    _assert_case(case, stored_response)
                    response = stored_response
                except AssertionError:
                    response = await application.retrieval.query(
                        QueryRequest(
                            query=case["query"],
                            profile=case["profile"],
                        )
                    )
                if response is not stored_response:
                    output_path.write_text(
                        response.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
            else:
                response = await application.retrieval.query(
                    QueryRequest(
                        query=case["query"],
                        profile=case["profile"],
                    )
                )
                output_path.write_text(
                    response.model_dump_json(indent=2),
                    encoding="utf-8",
                )
            _assert_case(case, response)
            responses.append(response)
        assert _file_hashes(protected_roots) == before
        return responses
    finally:
        await application.aclose()


def test_gate5_live_corpus_query_http(
    gate1_run_dir: Path,
    configured_data_dir: Path,
    corpus_manifest: dict,
) -> None:
    run_root = gate1_run_dir / "gate5-live"
    logs = gate1_run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pdf_dir = configured_data_dir / "test-corpus" / "pdfs"
    query_dir = configured_data_dir / "test-corpus" / "queries"
    papers = corpus_manifest["papers"]

    responses = asyncio.run(
        _run_live_gate5(run_root, pdf_dir, query_dir, papers, logs)
    )
    assert len(responses) == 22
    assert {response.profile.value for response in responses} == {
        "truth",
        "associative",
        "comprehensive",
    }

    settings = load_settings(
        environ={**os.environ, "PAPEROS_DATA_DIR": str(run_root)}
    )
    with TestClient(create_app(settings)) as client:
        comprehensive_response = client.post(
            "/api/v1/query",
            json={
                "query": "What are the four papers' main geometric representations?",
                "profile": "comprehensive",
            },
        )
        assert comprehensive_response.status_code == 200, comprehensive_response.text
        comprehensive_payload = comprehensive_response.json()
        assert comprehensive_payload["provenance_complete"] is True
        assert comprehensive_payload["distinct_documents"] == 4
        http_response = client.post(
            "/api/v1/query",
            json={
                "query": "How do the papers represent and evolve geometry?",
                "profile": "associative",
            },
        )
    assert http_response.status_code == 200, http_response.text
    http_payload = http_response.json()
    assert http_payload["provenance_complete"] is True
    assert http_payload["distinct_documents"] >= 2
    (logs / "http-comprehensive-query.json").write_text(
        json.dumps(comprehensive_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (logs / "http-query.json").write_text(
        json.dumps(http_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
