from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from paperos_core.bootstrap import build_application
from paperos_core.cli import app
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
    assert knowledge["cognee_object_count"] > result["counts"]["chunks"]
    assert knowledge["vector_object_count"] == result["counts"]["chunks"]
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
    assert set(index_manifest["vector_object_ids"]) == canonical_chunk_ids
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
    with sqlite3.connect(knowledge["vector_database"]) as connection:
        vector_rows = connection.execute(
            "SELECT object_id, dimensions FROM vector_records WHERE document_id = ?",
            (canonical.document.id,),
        ).fetchall()
    assert {row[0] for row in vector_rows} == canonical_chunk_ids
    assert {row[1] for row in vector_rows} == {768}
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
    asyncio.run(application.aclose())

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
    assert rebuild["reports"][0]["vector_object_count"] == len(canonical.chunks)
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
