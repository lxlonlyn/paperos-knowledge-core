"""Live Gate 1-6 acceptance over the retained genuine four-paper run."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient
from paperos_core.api.app import create_app
from paperos_core.application import application_from_config
from paperos_core.config import load_settings
from paperos_core.feedback.models import FeedbackRequest, FeedbackType
from paperos_core.retrieval.candidates import QueryRequest, RetrievalProfile

def _hash_files(roots: list[Path]) -> dict[str, str]:
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }


def _assert_retained_unchanged(expected: dict[str, str]) -> None:
    assert all(
        Path(path).is_file()
        and hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
        for path, digest in expected.items()
    )


async def _run_gate6(
    run_root: Path,
    configured_data_dir: Path,
    corpus_manifest: dict,
    logs: Path,
) -> dict[str, object]:
    application = application_from_config(data_dir=run_root)
    protected_roots = [
        application.paths.raw,
        application.paths.parsed,
        application.paths.canonical,
    ]
    protected_before = _hash_files(protected_roots)
    try:
        documents = application.documents.list_documents()
        assert len(documents) == len(corpus_manifest["papers"]) == 4
        by_filename = {item.source_filename: item for item in documents}
        for paper in corpus_manifest["papers"]:
            source = application.ingestion.get_source(
                by_filename[paper["pdf_file"]].source_file_id
            )
            original = configured_data_dir / "test-corpus" / "pdfs" / paper["pdf_file"]
            assert source.sha256 == paper["sha256"]
            assert source.storage_path.read_bytes() == original.read_bytes()

        gaussian = by_filename[
            "3d_gaussian_splatting_for_real_time_radiance_field_rendering.pdf"
        ]
        initial = await application.retrieval.query(
            QueryRequest(
                query=(
                    "What are the storage and memory limitations of 3D Gaussian "
                    "Splatting for real-time radiance field rendering?"
                ),
                profile=RetrievalProfile.TRUTH,
                document_ids=[gaussian.document_id],
                top_k=6,
            )
        )
        assert initial.provenance_complete and initial.evidence
        target = next(
            (
                item
                for item in initial.evidence
                if "memor" in item.text.casefold()
                or "storage" in item.text.casefold()
            ),
            initial.evidence[0],
        )
        correction_text = (
            "User-confirmed knowledge: 3D Gaussian Splatting can require a "
            "substantial memory and storage footprint because many Gaussian "
            "primitives are retained."
        )
        records = [
            application.feedback.record(
                FeedbackRequest(
                    target_id=target.chunk_id,
                    feedback_type=FeedbackType.CONFIRM,
                    query_id=initial.id,
                    answer_id=initial.id,
                    evidence_ids=[target.evidence_id],
                    comment="The retained paper evidence supports this passage.",
                    created_by="gate6-live-test",
                )
            ),
            application.feedback.record(
                FeedbackRequest(
                    target_id=target.chunk_id,
                    feedback_type=FeedbackType.CORRECT,
                    query_id=initial.id,
                    answer_id=initial.id,
                    evidence_ids=[target.evidence_id],
                    replacement_text=correction_text,
                    created_by="gate6-live-test",
                )
            ),
            application.feedback.record(
                FeedbackRequest(
                    target_id=initial.id,
                    feedback_type=FeedbackType.REJECT,
                    query_id=initial.id,
                    answer_id=initial.id,
                    evidence_ids=[target.evidence_id],
                    comment="Reject unsupported interpretations beyond this evidence.",
                    created_by="gate6-live-test",
                )
            ),
        ]
        queued = application.queue.enqueue("improve")
        assert application.worker is not None
        completed = await application.worker.run_once()
        assert completed is not None
        assert completed.id == queued.id and completed.status == "completed"
        assert completed.result is not None
        improvements = application.feedback.confirmed_improvements()
        correction = next(
            item
            for item in improvements
            if item.improvement_type == FeedbackType.CORRECT.value
            and item.text == correction_text
            and target.chunk_id in item.source_chunk_ids
        )
        assert correction.status == "user_confirmed"
        assert correction.source_chunk_ids == [target.chunk_id]
        assert target.chunk_id in correction.derived_from_ids

        iga = by_filename[
            "isogeometric_analysis_of_geometric_partial_differential_equations.pdf"
        ]
        snapshots_before = set(application.canonical_repository.list_snapshot_ids())
        if os.environ.get("PAPEROS_GATE6_REUSE_REPROCESS") == "true":
            assert len(application.documents.inspect(iga.document_id).snapshot_ids) >= 2
            snapshots_after = snapshots_before
            new_snapshot_id = application.documents.inspect(
                iga.document_id
            ).snapshot_ids[-1]
        else:
            reprocessed = await application.documents.reprocess(iga.document_id)
            snapshots_after = set(application.canonical_repository.list_snapshot_ids())
            assert len(snapshots_after - snapshots_before) == 1
            assert reprocessed["duplicate"] is True
            assert reprocessed["status"] == "completed"
            new_snapshot_id = next(iter(snapshots_after - snapshots_before))

        if os.environ.get("PAPEROS_GATE6_REUSE_REBUILD") == "true":
            for snapshot_id in snapshots_after:
                application.canonical_repository.verify_snapshot(snapshot_id)
                assert (
                    application.paths.indexes / "manifests" / f"{snapshot_id}.json"
                ).is_file()
                assert (
                    application.paths.cognee / "enrichment" / f"{snapshot_id}.json"
                ).is_file()
            rebuild_payload: dict[str, object] = {
                "reused_verified_rebuild": True,
                "rebuilt_snapshot_ids": sorted(snapshots_after),
            }
        else:
            rebuild = await application.rebuilder.rebuild()
            assert set(rebuild.rebuilt_snapshot_ids) == snapshots_after
            assert all(report.consistency_valid for report in rebuild.reports)
            rebuild_payload = rebuild.model_dump(mode="json")
        _assert_retained_unchanged(protected_before)

        repeated_request = QueryRequest(
            query=(
                "What user-confirmed knowledge is available about the memory and "
                "storage footprint of 3D Gaussian Splatting, and how should it be "
                "distinguished from source facts or system inferences?"
            ),
            profile=RetrievalProfile.COMPREHENSIVE,
            document_ids=[gaussian.document_id],
            top_k=8,
        )
        first = await application.retrieval.query(repeated_request)
        second = await application.retrieval.query(repeated_request)
        for response in (first, second):
            assert response.provenance_complete
            assert "confirmed_knowledge" in response.channels_used
            assert any(
                item.knowledge_kind == "user_confirmed"
                and correction_text in item.text
                for item in response.evidence
            )
            assert any(item.evidence_id in response.answer for item in response.evidence)
        first_confirmed = {
            item.evidence_id
            for item in first.evidence
            if item.knowledge_kind == "user_confirmed"
        }
        second_confirmed = {
            item.evidence_id
            for item in second.evidence
            if item.knowledge_kind == "user_confirmed"
        }
        assert first_confirmed == second_confirmed == {target.evidence_id}
        assert first.id == second.id
        observed_kinds = {
            item.knowledge_kind
            for response in (initial, first, second)
            for item in response.evidence
        }
        assert {"source_fact", "user_confirmed"} <= observed_kinds
        assert observed_kinds & {"structured_relation", "system_inference"}

        detail = application.documents.inspect(iga.document_id)
        assert len(detail.snapshot_ids) >= 2
        health = await application.health.report()
        assert health["status"] == "healthy"
        worker_record = json.loads(
            (application.paths.jobs / "worker-process.json").read_text(encoding="utf-8")
        )
        assert worker_record["pid"] == os.getpid()
        assert worker_record["status"] == "completed"

        (logs / "initial-query.json").write_text(
            initial.model_dump_json(indent=2), encoding="utf-8"
        )
        (logs / "repeated-query-1.json").write_text(
            first.model_dump_json(indent=2), encoding="utf-8"
        )
        (logs / "repeated-query-2.json").write_text(
            second.model_dump_json(indent=2), encoding="utf-8"
        )
        (logs / "feedback-records.json").write_text(
            json.dumps(
                [record.model_dump(mode="json") for record in records],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (logs / "rebuild.json").write_text(
            json.dumps(rebuild_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (logs / "health.json").write_text(
            json.dumps(health, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "target_id": target.chunk_id,
            "evidence_id": target.evidence_id,
            "document_id": gaussian.document_id,
            "health": health,
            "protected_file_count": len(protected_before),
            "new_snapshot_id": new_snapshot_id,
        }
    finally:
        await application.aclose()


def test_gate6_live_feedback_improve_rebuild_and_operations(
    configured_data_dir: Path,
    corpus_manifest: dict,
) -> None:
    requested_root = os.environ.get("PAPEROS_GATE6_RUN_ROOT")
    assert requested_root, "PAPEROS_GATE6_RUN_ROOT must name the retained live Gate 5 run"
    run_root = Path(requested_root).resolve()
    assert run_root.is_dir()
    logs = run_root / "logs" / "gate6"
    logs.mkdir(parents=True, exist_ok=True)

    result = asyncio.run(
        _run_gate6(run_root, configured_data_dir, corpus_manifest, logs)
    )

    settings = load_settings(
        environ={**os.environ, "PAPEROS_DATA_DIR": str(run_root)}
    )
    with TestClient(create_app(settings)) as client:
        documents = client.get("/api/v1/documents")
        assert documents.status_code == 200
        inspected = client.get(
            f"/api/v1/documents/{result['document_id']}"
        )
        assert inspected.status_code == 200
        api_feedback = client.post(
            "/api/v1/feedback",
            json={
                "target_id": result["target_id"],
                "feedback_type": "accept",
                "evidence_ids": [result["evidence_id"]],
                "comment": "HTTP API acceptance with canonical provenance.",
            },
        )
        assert api_feedback.status_code == 200, api_feedback.text
        api_improve = client.post("/api/v1/improve")
        assert api_improve.status_code == 200, api_improve.text
        api_health = client.get("/api/v1/health")
        assert api_health.status_code == 200
        routes = set(client.app.openapi()["paths"])
        assert {
            "/api/v1/ingest",
            "/api/v1/ingest/{job_id}",
            "/api/v1/query",
            "/api/v1/documents",
            "/api/v1/documents/{document_id}",
            "/api/v1/documents/{document_id}/reprocess",
            "/api/v1/feedback",
            "/api/v1/improve",
            "/api/v1/health",
        } <= routes
        deletion = client.delete(f"/api/v1/documents/{result['document_id']}")
        assert deletion.status_code == 200, deletion.text
    (logs / "api-feedback.json").write_text(
        json.dumps(api_feedback.json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (logs / "api-health.json").write_text(
        json.dumps(api_health.json(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    deletion_payload = deletion.json()
    assert deletion_payload["status"] == "deleted"
    assert deletion_payload["source_evidence_retained"] is True
    (logs / "api-delete.json").write_text(
        json.dumps(deletion_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    deleted_application = application_from_config(data_dir=run_root)
    try:
        active_ids = {
            item.document_id for item in deleted_application.documents.list_documents()
        }
        assert result["document_id"] not in active_ids
        detail = deleted_application.documents.inspect(str(result["document_id"]))
        assert detail.deleted is True
        assert detail.raw_pdf_path.is_file()
        assert result["target_id"] in {
            chunk.id
            for bundle in deleted_application.canonical_repository.list_bundles()
            for chunk in bundle.chunks
        }
    finally:
        asyncio.run(deleted_application.aclose())
