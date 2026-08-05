from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[2]


def _port_is_open(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def _wait_for_health(process: subprocess.Popen[bytes], timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    with httpx.Client(trust_env=False, timeout=30) as client:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise AssertionError(
                    "python server.py exited during startup with code "
                    f"{process.returncode}"
                )
            try:
                response = client.get("http://127.0.0.1:8000/api/v1/health")
                if response.status_code == 200:
                    return response.json()
            except (httpx.HTTPError, OSError) as exc:
                last_error = exc
            time.sleep(1)
    raise AssertionError(f"PaperOS health did not become available: {last_error}")


def _wait_for_port_closed(port: int, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _port_is_open(port):
            return
        time.sleep(0.2)
    raise AssertionError(f"127.0.0.1:{port} remained open after server shutdown")


def _wait_for_job(
    client: httpx.Client, job_id: str, *, timeout: float
) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["status"] in {"completed", "failed"}:
            return job
        time.sleep(2)
    raise AssertionError(f"Operational job {job_id} did not finish within {timeout}s")


def _shutdown_server(
    process: subprocess.Popen[bytes] | None, local_pid: int | None
) -> None:
    if process is not None and process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=40)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    _wait_for_port_closed(8000)
    _wait_for_port_closed(8081)
    if local_pid is not None:
        with pytest.raises(ProcessLookupError):
            os.kill(local_pid, 0)


@pytest.mark.skipif(
    os.getenv("PAPEROS_RUN_SERVER_LIFECYCLE") != "true",
    reason="requires the prepared real local models",
)
def test_python_server_owns_one_runtime_and_worker(gate1_run_dir: Path) -> None:
    assert not _port_is_open(8000), "PaperOS API port is already occupied"
    assert not _port_is_open(8081), "private local inference port is already occupied"
    run_root = gate1_run_dir / "server-lifecycle"
    run_root.mkdir(parents=True)
    log_path = run_root / "logs" / "server.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "PAPEROS_DATA_DIR": str(run_root),
        "NODE_LLAMA_CPP_SKIP_DOWNLOAD": "true",
        "MINERU_API_KEY": "",
        "LLM_API_KEY": "",
    }
    process: subprocess.Popen[bytes] | None = None
    local_pid: int | None = None
    with log_path.open("wb") as log_stream:
        try:
            process = subprocess.Popen(
                [sys.executable, "server.py"],
                cwd=ROOT,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            health = _wait_for_health(process, 240)
            assert process.poll() is None
            assert health["status"] == "degraded"
            assert health["components"]["local_models"]["status"] == "healthy"
            assert health["components"]["mineru"]["status"] == "unavailable"
            assert health["components"]["llm"]["status"] == "unavailable"

            local_record_path = run_root / "jobs" / "local-inference-process.json"
            worker_record_path = run_root / "jobs" / "worker-process.json"
            local_record = json.loads(local_record_path.read_text(encoding="utf-8"))
            worker_record = json.loads(worker_record_path.read_text(encoding="utf-8"))
            local_pid = int(local_record["pid"])
            assert local_record["status"] == "running"
            assert worker_record["status"] in {"running", "idle"}
            assert worker_record["pid"] == process.pid
            assert local_pid != process.pid
            assert _port_is_open(8000)
            assert _port_is_open(8081)

            process.send_signal(signal.SIGTERM)
            # Uvicorn performs lifespan shutdown and then re-raises the captured
            # signal, so POSIX may report the initiating signal as the exit code.
            assert process.wait(timeout=40) in {0, -signal.SIGTERM}
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)

    _wait_for_port_closed(8000)
    _wait_for_port_closed(8081)
    stopped_local = json.loads(
        (run_root / "jobs" / "local-inference-process.json").read_text(
            encoding="utf-8"
        )
    )
    stopped_worker = json.loads(
        (run_root / "jobs" / "worker-process.json").read_text(encoding="utf-8")
    )
    assert stopped_local["status"] == "stopped"
    assert stopped_local["exit_code"] == 0
    assert stopped_worker["status"] == "stopped"
    if local_pid is not None:
        with pytest.raises(ProcessLookupError):
            os.kill(local_pid, 0)


@pytest.mark.skipif(
    os.getenv("PAPEROS_RUN_LIVE_CUMULATIVE") != "true",
    reason="requires live MinerU, LLM, Cognee, and prepared local models",
)
def test_real_pdf_cumulative_pipeline_uses_only_http(
    gate1_run_dir: Path,
    real_pdf_case: tuple[Path, dict],
    configured_data_dir: Path,
) -> None:
    assert os.getenv("MINERU_API_KEY"), "live MinerU key is required"
    assert os.getenv("LLM_API_KEY"), "live LLM key is required"
    assert not _port_is_open(8000), "PaperOS API port is already occupied"
    assert not _port_is_open(8081), "private local inference port is already occupied"
    pdf_path, paper = real_pdf_case
    run_root = gate1_run_dir / "server-cumulative"
    run_root.mkdir(parents=True)
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    server_log = logs / "server.log"
    environment = {
        **os.environ,
        "PAPEROS_DATA_DIR": str(run_root),
        "NODE_LLAMA_CPP_SKIP_DOWNLOAD": "true",
    }
    process: subprocess.Popen[bytes] | None = None
    local_pid: int | None = None
    with server_log.open("wb") as log_stream:
        try:
            process = subprocess.Popen(
                [sys.executable, "server.py"],
                cwd=ROOT,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            startup_health = _wait_for_health(process, 240)
            assert process.poll() is None
            assert startup_health["components"]["local_models"]["status"] == "healthy"
            local_record = json.loads(
                (run_root / "jobs" / "local-inference-process.json").read_text(
                    encoding="utf-8"
                )
            )
            local_pid = int(local_record["pid"])

            with httpx.Client(
                base_url="http://127.0.0.1:8000", trust_env=False, timeout=180
            ) as client, pdf_path.open("rb") as stream:
                accepted = client.post(
                    "/api/v1/ingest",
                    params={"dataset": "papers"},
                    files={"file": (pdf_path.name, stream, "application/pdf")},
                )
                assert accepted.status_code == 202, accepted.text
                job = _wait_for_job(
                    client, accepted.json()["job_id"], timeout=2_400
                )
                (logs / "ingestion-job.json").write_text(
                    json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                assert job["status"] == "completed", job.get("error")
                result = job["result"]
                assert isinstance(result, dict)

                query_case = next(
                    json.loads(line)
                    for line in (
                        configured_data_dir / "test-corpus" / "queries" / "truth.jsonl"
                    )
                    .read_text(encoding="utf-8")
                    .splitlines()
                    if paper["pdf_file"] in json.loads(line)["expected_documents"]
                )
                query_response = client.post(
                    "/api/v1/query",
                    json={
                        "query": query_case["query"],
                        "profile": query_case["profile"],
                        "document_ids": [result["document"]["id"]],
                    },
                )
                assert query_response.status_code == 200, query_response.text
                query = query_response.json()
                (logs / "query.json").write_text(
                    json.dumps(query, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                documents = client.get("/api/v1/documents")
                assert documents.status_code == 200, documents.text
                assert len(documents.json()) == 1
                health_response = client.get("/api/v1/health")
                assert health_response.status_code == 200, health_response.text
                (logs / "health.json").write_text(
                    json.dumps(health_response.json(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            assert result["status"] == "completed"
            assert result["parse_run"]["provider"] == "mineru_cloud"
            assert result["canonical_snapshot"]["parse_run_id"] == result["parse_run"]["id"]
            assert result["document"]["title"] == paper["title"]
            assert result["counts"]["chunks"] > 0
            assert result["knowledge"]["vector_backend"] == "cognee"
            assert result["knowledge"]["consistency_valid"] is True
            assert result["knowledge"]["dataset_name"] == "papers"

            stored_pdf = run_root / "raw" / result["source_file_id"] / "source.pdf"
            assert stored_pdf.read_bytes() == pdf_path.read_bytes()
            assert hashlib.sha256(stored_pdf.read_bytes()).hexdigest() == paper["sha256"]
            for artifact in result["artifacts"]:
                artifact_path = Path(artifact["storage_path"])
                assert artifact_path.is_relative_to(run_root.resolve())
                assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact["sha256"]
            assert Path(result["canonical_snapshot"]["manifest_path"]).is_relative_to(
                run_root.resolve()
            )
            assert query["provenance_complete"] is True
            assert query["evidence"]
            assert paper["pdf_file"] in {
                evidence["source_filename"] for evidence in query["evidence"]
            }
            assert {"lexical", "semantic"} <= set(query["channels_used"])
            assert {"lexical", "semantic", "fusion", "synthesis"} <= set(
                query["stages"]
            )

            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=40) in {0, -signal.SIGTERM}
        finally:
            _shutdown_server(process, local_pid)

    stopped_local = json.loads(
        (run_root / "jobs" / "local-inference-process.json").read_text(
            encoding="utf-8"
        )
    )
    stopped_worker = json.loads(
        (run_root / "jobs" / "worker-process.json").read_text(encoding="utf-8")
    )
    assert stopped_local["status"] == "stopped"
    assert stopped_worker["status"] == "stopped"


@pytest.mark.skipif(
    not os.getenv("PAPEROS_MAINTENANCE_RUN_ROOT"),
    reason="requires a retained live cumulative HTTP run",
)
def test_real_http_maintenance_routes_preserve_source_evidence() -> None:
    assert os.getenv("MINERU_API_KEY"), "live MinerU key is required"
    assert os.getenv("LLM_API_KEY"), "live LLM key is required"
    assert not _port_is_open(8000), "PaperOS API port is already occupied"
    assert not _port_is_open(8081), "private local inference port is already occupied"
    run_root = Path(os.environ["PAPEROS_MAINTENANCE_RUN_ROOT"]).resolve()
    assert run_root.is_dir()
    logs = run_root / "logs"
    server_log = logs / "maintenance-server.log"
    protected_roots = [run_root / "raw", run_root / "parsed", run_root / "canonical"]
    protected_before = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in protected_roots
        for path in root.rglob("*")
        if path.is_file()
    }
    process: subprocess.Popen[bytes] | None = None
    local_pid: int | None = None
    environment = {
        **os.environ,
        "PAPEROS_DATA_DIR": str(run_root),
        "NODE_LLAMA_CPP_SKIP_DOWNLOAD": "true",
    }
    with server_log.open("wb") as log_stream:
        try:
            process = subprocess.Popen(
                [sys.executable, "server.py"],
                cwd=ROOT,
                env=environment,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _wait_for_health(process, 300)
            local_pid = int(
                json.loads(
                    (run_root / "jobs" / "local-inference-process.json").read_text(
                        encoding="utf-8"
                    )
                )["pid"]
            )
            previous_query = json.loads(
                (logs / "query.json").read_text(encoding="utf-8")
            )
            with httpx.Client(
                base_url="http://127.0.0.1:8000", trust_env=False, timeout=180
            ) as client:
                documents = client.get("/api/v1/documents")
                assert documents.status_code == 200, documents.text
                assert len(documents.json()) == 1
                document_id = documents.json()[0]["document_id"]
                inspected = client.get(f"/api/v1/documents/{document_id}")
                assert inspected.status_code == 200, inspected.text
                original_snapshot_ids = inspected.json()["snapshot_ids"]

                evidence = previous_query["evidence"][0]
                feedback = client.post(
                    "/api/v1/feedback",
                    json={
                        "target_id": evidence["chunk_id"],
                        "feedback_type": "confirm",
                        "query_id": previous_query["id"],
                        "answer_id": previous_query["id"],
                        "evidence_ids": [evidence["evidence_id"]],
                        "comment": "Real HTTP maintenance acceptance.",
                    },
                )
                assert feedback.status_code == 200, feedback.text

                improve = client.post("/api/v1/improve")
                assert improve.status_code == 202, improve.text
                improve_job = _wait_for_job(
                    client, improve.json()["job_id"], timeout=120
                )
                assert improve_job["status"] == "completed", improve_job["error"]

                rebuild = client.post("/api/v1/rebuild")
                assert rebuild.status_code == 202, rebuild.text
                rebuild_job = _wait_for_job(
                    client, rebuild.json()["job_id"], timeout=2_400
                )
                assert rebuild_job["status"] == "completed", rebuild_job["error"]
                assert set(rebuild_job["result"]["rebuilt_snapshot_ids"]) == set(
                    original_snapshot_ids
                )

                reprocess = client.post(
                    f"/api/v1/documents/{document_id}/reprocess"
                )
                assert reprocess.status_code == 202, reprocess.text
                reprocess_job = _wait_for_job(
                    client, reprocess.json()["job_id"], timeout=2_400
                )
                assert reprocess_job["status"] == "completed", reprocess_job["error"]
                assert reprocess_job["result"]["duplicate"] is True
                inspected_again = client.get(f"/api/v1/documents/{document_id}")
                assert inspected_again.status_code == 200, inspected_again.text
                assert len(inspected_again.json()["snapshot_ids"]) == (
                    len(original_snapshot_ids) + 1
                )

                request = {
                    "query": previous_query["query"],
                    "profile": previous_query["profile"],
                    "document_ids": [document_id],
                }
                first = client.post("/api/v1/query", json=request)
                second = client.post("/api/v1/query", json=request)
                assert first.status_code == second.status_code == 200
                assert first.json()["id"] == second.json()["id"]
                assert first.json()["provenance_complete"] is True
                assert [item["evidence_id"] for item in first.json()["evidence"]] == [
                    item["evidence_id"] for item in second.json()["evidence"]
                ]

                deletion = client.delete(f"/api/v1/documents/{document_id}")
                assert deletion.status_code == 200, deletion.text
                assert deletion.json()["source_evidence_retained"] is True
                assert client.get("/api/v1/documents").json() == []

                (logs / "maintenance.json").write_text(
                    json.dumps(
                        {
                            "feedback": feedback.json(),
                            "improve": improve_job,
                            "rebuild": rebuild_job,
                            "reprocess": reprocess_job,
                            "repeated_query_id": first.json()["id"],
                            "delete": deletion.json(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=60) in {0, -signal.SIGTERM}
        finally:
            _shutdown_server(process, local_pid)

    for path, digest in protected_before.items():
        retained = Path(path)
        assert retained.is_file()
        assert hashlib.sha256(retained.read_bytes()).hexdigest() == digest
    assert json.loads(
        (run_root / "jobs" / "local-inference-process.json").read_text(
            encoding="utf-8"
        )
    )["status"] == "stopped"
    assert json.loads(
        (run_root / "jobs" / "worker-process.json").read_text(encoding="utf-8")
    )["status"] == "stopped"
