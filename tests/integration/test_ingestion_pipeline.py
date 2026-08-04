from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from paperos_core.adapters.cognee.repository import (
    GRAPH_SEED_VECTOR_COLLECTIONS,
    SEMANTIC_VECTOR_COLLECTIONS,
)
from paperos_core.bootstrap import build_application
from paperos_core.cli import app
from paperos_core.domain.provenance import RelationType
from paperos_core.ingestion.expected_validation import (
    expected_path_for_source,
    validate_expected_case,
)

runner = CliRunner()


def _json_output(result) -> dict:
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_gate4_real_pdf_through_live_cumulative_cli_and_rebuild(
    real_pdf_case, gate1_run_dir: Path, configured_data_dir: Path
) -> None:
    pdf_path, case = real_pdf_case
    run_root = gate1_run_dir / "gate4-live"

    result = _json_output(
        runner.invoke(app, ["ingest", str(pdf_path), "--data-dir", str(run_root)])
    )

    assert result["duplicate"] is False
    assert result["status"] == "completed"
    assert result["parse_run"]["provider"] == "mineru_cloud"
    assert result["parse_run"]["provider_task_id"]
    assert result["parse_run"]["source_file_id"] == result["source_file_id"]
    assert result["canonical_snapshot"]["parse_run_id"] == result["parse_run"]["id"]
    assert result["canonical_snapshot"]["source_file_id"] == result["source_file_id"]
    assert result["document"]["title"] == case["title"]
    assert result["counts"]["sections"] >= 8
    assert result["counts"]["chunks"] >= 12
    assert result["counts"]["references"] >= 25
    knowledge = result["knowledge"]
    assert knowledge["consistency_valid"] is True
    assert knowledge["vector_backend"] == "cognee"
    assert knowledge["cognee_object_count"] > result["counts"]["chunks"]
    assert knowledge["vector_object_count"] > result["counts"]["chunks"]
    assert knowledge["lexical_object_count"] >= result["counts"]["chunks"]
    assert knowledge["embedding_dimensions"] == 768
    assert knowledge["semantic_entity_count"] > 0
    assert knowledge["semantic_claim_count"] > 0
    assert knowledge["semantic_relation_count"] > 0
    assert knowledge["summary_count"] == 1
    artifact_types = {artifact["artifact_type"] for artifact in result["artifacts"]}
    assert {
        "archive",
        "provider_response",
        "task_metadata",
        "markdown",
        "content_list",
        "model_output",
        "asset",
    } <= artifact_types

    application = build_application(data_dir=run_root)
    source = application.ingestion.get_source(result["source_file_id"])
    stored = source.storage_path
    assert stored.is_relative_to(run_root.resolve())
    assert stored.read_bytes() == pdf_path.read_bytes()
    assert hashlib.sha256(stored.read_bytes()).hexdigest() == case["sha256"]
    job = application.ingestion.get_job(result["job_id"])
    assert job.source_file_id == source.id
    assert job.status.value == "completed"
    assert job.current_operation == "completed"
    parse_run = application.parser_artifacts.get_parse_run(result["parse_run"]["id"])
    assert parse_run.status.value == "completed"
    assert parse_run.artifact_manifest_path.is_file()
    application.parser_artifacts.verify_artifact_checksums(parse_run.id)
    provider_response = next(
        artifact
        for artifact in application.parser_artifacts.list_artifacts(parse_run.id)
        if artifact.artifact_type.value == "provider_response"
    )
    provider_metadata = json.loads(provider_response.storage_path.read_text())
    assert provider_metadata["submission"]["data"]["batch_id"] == parse_run.provider_task_id
    assert provider_metadata["status"]["data"]["extract_result"][0]["state"] == "done"
    assert application.paths.registry_db.is_file()

    snapshot_id = result["canonical_snapshot"]["id"]
    application.canonical_repository.verify_snapshot(snapshot_id)
    canonical = application.canonical_repository.get_bundle(snapshot_id)
    assert canonical.document.source_file_id == source.id
    assert canonical.document.parse_run_id == parse_run.id
    assert canonical.document.canonical_snapshot_id == snapshot_id
    assert all(element.source_span is not None for element in canonical.elements)
    assert all(chunk.element_ids for chunk in canonical.chunks)
    rebuilt = application.canonical_mapper.build_canonical_snapshot(
        source=source,
        parse_run=parse_run,
        artifacts=application.parser_artifacts.list_artifacts(parse_run.id),
        manifest_path=canonical.snapshot.manifest_path,
    )
    assert rebuilt.snapshot.id == canonical.snapshot.id
    assert [item.id for item in rebuilt.sections] == [item.id for item in canonical.sections]
    assert [item.id for item in rebuilt.elements] == [item.id for item in canonical.elements]
    assert [item.id for item in rebuilt.chunks] == [item.id for item in canonical.chunks]
    assert [item.id for item in rebuilt.references] == [item.id for item in canonical.references]
    expected = expected_path_for_source(configured_data_dir / "test-corpus" / "expected", source)
    report = validate_expected_case(bundle=canonical, source=source, expected_path=expected)
    assert report["passed"] is True

    index_manifest_path = Path(knowledge["manifest_path"])
    cognee_manifest_path = Path(knowledge["cognee_manifest_path"])
    enrichment_path = Path(knowledge["enrichment_path"])
    assert index_manifest_path.is_relative_to(run_root.resolve())
    assert cognee_manifest_path.is_relative_to(run_root.resolve())
    assert enrichment_path.is_relative_to(run_root.resolve())
    index_manifest = json.loads(index_manifest_path.read_text())
    cognee_manifest = json.loads(cognee_manifest_path.read_text())
    enrichment = json.loads(enrichment_path.read_text())
    canonical_chunk_ids = {chunk.id for chunk in canonical.chunks}
    assert index_manifest["vector_backend"] == "cognee"
    assert canonical_chunk_ids <= set(index_manifest["vector_object_ids"])
    assert canonical_chunk_ids <= set(index_manifest["lexical_object_ids"])
    assert canonical_chunk_ids <= set(index_manifest["cognee_object_ids"])
    assert set(cognee_manifest["canonical_to_cognee_id"]) == set(
        index_manifest["cognee_object_ids"]
    )
    semantic_objects = [
        *enrichment["entities"],
        *enrichment["claims"],
        *enrichment["relations"],
        *enrichment["summaries"],
    ]
    assert semantic_objects
    assert all(
        item["source_chunk_ids"] and set(item["source_chunk_ids"]) <= canonical_chunk_ids
        for item in semantic_objects
    )
    semantic_ids = {item["id"] for item in semantic_objects}
    vector_semantic_ids = {
        item["id"]
        for group in (
            enrichment["entities"],
            enrichment["claims"],
            enrichment["summaries"],
        )
        for item in group
    }
    assert vector_semantic_ids <= set(index_manifest["vector_object_ids"])
    triplet_ids = {
        canonical_id
        for canonical_id in cognee_manifest["canonical_to_cognee_id"]
        if canonical_id.startswith("triplet_")
    }
    assert triplet_ids
    assert triplet_ids <= set(index_manifest["vector_object_ids"])
    assert "TripletDataPoint_text" in cognee_manifest["vector_collections"]
    assert all(
        relation["source_chunk_ids"]
        for relation in cognee_manifest["relations"]
        if relation["source_id"] in semantic_ids
    )
    document_node = asyncio.run(
        application.knowledge_pipeline.cognee_repository.get_datapoint(canonical.document.id)
    )
    document_properties = document_node.get("properties", document_node)
    assert document_properties["canonical_id"] == canonical.document.id
    with sqlite3.connect(knowledge["lexical_database"]) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM lexical_records WHERE document_id = ?",
                (canonical.document.id,),
            ).fetchone()[0]
            == knowledge["lexical_object_count"]
        )
    vector_status = asyncio.run(
        application.knowledge_pipeline.cognee_repository.vector_status()
    )
    assert vector_status["backend"] == "cognee"
    assert vector_status["dimensions"] == 768
    assert vector_status["collections"]["ChunkDataPoint_text"] == len(
        canonical.chunks
    )
    assert vector_status["collections"]["TripletDataPoint_text"] == len(triplet_ids)
    assert Path(knowledge["vector_database"]) == application.paths.cognee / "vector"
    assert not (application.paths.indexes / "vectors.sqlite3").exists()

    async def verify_cognee_query_backbone() -> None:
        await application.model_process.start()
        try:
            repository = application.knowledge_pipeline.cognee_repository
            semantic_hits = await repository.search_vectors(
                [canonical.chunks[0].text[:200]],
                collections=SEMANTIC_VECTOR_COLLECTIONS,
                limit=10,
            )
            assert semantic_hits
            assert any(hit.object_type == "ChunkDataPoint" for hit in semantic_hits)
            entity_id = enrichment["entities"][0]["id"]
            entity_hits = await repository.search_vectors(
                [enrichment["entities"][0]["name"]],
                collections=GRAPH_SEED_VECTOR_COLLECTIONS,
                limit=10,
            )
            assert entity_id in {hit.canonical_id for hit in entity_hits}
            traversed = await repository.traverse(
                entity_hits,
                depth=2,
                edge_types={relation.value for relation in RelationType},
            )
            assert traversed
            assert all(item.source_chunk_ids for item in traversed)
        finally:
            await application.aclose()

    asyncio.run(verify_cognee_query_backbone())
    process_record = json.loads((application.paths.jobs / "model-gateway-process.json").read_text())
    assert process_record["status"] == "stopped"
    assert Path(process_record["log_path"]).is_relative_to(run_root.resolve())

    protected_roots = [
        application.paths.raw,
        application.paths.parsed,
        application.paths.canonical,
    ]
    protected_hashes = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in protected_roots
        for path in root.rglob("*")
        if path.is_file()
    }
    rebuild = _json_output(
        runner.invoke(
            app,
            [
                "rebuild",
                "--snapshot-id",
                snapshot_id,
                "--data-dir",
                str(run_root),
            ],
        )
    )
    assert rebuild["rebuilt_snapshot_ids"] == [snapshot_id]
    assert rebuild["reports"][0]["rebuilt"] is True
    assert rebuild["reports"][0]["consistency_valid"] is True
    assert rebuild["reports"][0]["vector_object_count"] > len(canonical.chunks)
    assert {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in protected_roots
        for path in root.rglob("*")
        if path.is_file()
    } == protected_hashes

    status = _json_output(runner.invoke(app, ["status", "--data-dir", str(run_root)]))
    assert status["source_file_count"] == 1
    assert status["ingestion_job_count"] == 1
    assert status["jobs_by_status"] == {"completed": 1}

    job_status = _json_output(
        runner.invoke(
            app,
            ["status", "--job-id", result["job_id"], "--data-dir", str(run_root)],
        )
    )
    assert job_status["job"]["source_file_id"] == source.id


def test_gate1_cli_reports_invalid_pdf(gate1_run_dir: Path, configured_data_dir: Path) -> None:
    non_pdf = configured_data_dir / "test-corpus" / "manifest.json"
    run_root = gate1_run_dir / "invalid-input"
    result = runner.invoke(app, ["ingest", str(non_pdf), "--data-dir", str(run_root)])
    assert result.exit_code == 2
    payload = json.loads(result.stderr)
    assert payload["error"]["code"] == "invalid_pdf"
    assert str(non_pdf) == payload["error"]["affected"]
