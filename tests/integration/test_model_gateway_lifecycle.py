"""Real foreground lifecycle acceptance for the repository local-model gateway."""

from __future__ import annotations

import json
import math
import os
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

import httpx


def _port_is_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((host, port)) == 0


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_model_gateway_cli_stays_alive_and_cleans_up(
    gate1_run_dir: Path,
) -> None:
    host = "127.0.0.1"
    port = 8081
    endpoint = f"http://{host}:{port}"
    assert not _port_is_open(host, port), f"Required test port is already occupied: {endpoint}"
    executable = shutil.which("paperos")
    assert executable is not None, "Editable-install paperos CLI is not available"
    run_root = gate1_run_dir / "model-gateway-lifecycle"
    logs = run_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            executable,
            "model-gateway",
            "--host",
            host,
            "--port",
            str(port),
            "--data-dir",
            str(run_root),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    output = ""
    child_pid: int | None = None
    health_payload: dict[str, object] | None = None
    models_payload: dict[str, object] | None = None
    embedding_summary: dict[str, object] | None = None
    rerank_payload: dict[str, object] | None = None
    expansion_payload: dict[str, object] | None = None
    try:
        deadline = time.monotonic() + 180
        with httpx.Client(base_url=endpoint, timeout=120, trust_env=False) as client:
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    output, _ = process.communicate(timeout=5)
                    raise AssertionError(
                        f"model-gateway exited before becoming healthy: {output}"
                    )
                try:
                    response = client.get("/health")
                    if response.status_code == 200:
                        health_payload = response.json()
                        if health_payload.get("status") == "healthy":
                            break
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
            assert health_payload is not None and health_payload["status"] == "healthy"
            assert process.poll() is None, "CLI exited immediately after health became ready"
            assert health_payload["embedding"]["dimensions"] == 768
            assert health_payload["reranker"]["loaded"] is True
            assert health_payload["query_expansion"]["loaded"] is True
            assert all(
                "path" not in key.casefold()
                for component in health_payload.values()
                if isinstance(component, dict)
                for key in component
            )

            models_response = client.get("/v1/models")
            models_response.raise_for_status()
            models_payload = models_response.json()
            assert len(models_payload["data"]) == 3
            capabilities = {
                capability
                for model in models_payload["data"]
                for capability in model["capabilities"]
            }
            assert capabilities == {"embeddings", "rerank", "query-expansion"}

            embedding_response = client.post(
                "/v1/embeddings",
                json={"input": ["PaperOS local model gateway lifecycle verification."]},
            )
            embedding_response.raise_for_status()
            embedding_payload = embedding_response.json()
            vector = embedding_payload["data"][0]["embedding"]
            assert len(vector) == 768
            vector_norm = math.sqrt(sum(float(value) ** 2 for value in vector))
            assert vector_norm > 0
            embedding_summary = {
                "model": embedding_payload["model"],
                "count": len(embedding_payload["data"]),
                "dimensions": len(vector),
                "l2_norm": vector_norm,
            }

            rerank_response = client.post(
                "/v1/rerank",
                json={
                    "query": "local academic paper retrieval",
                    "candidate_ids": ["candidate_a", "candidate_b"],
                    "texts": [
                        "Local semantic retrieval over academic paper evidence.",
                        "A recipe for baking bread at home.",
                    ],
                    "limit": 2,
                },
            )
            rerank_response.raise_for_status()
            rerank_payload = rerank_response.json()
            assert len(rerank_payload["results"]) == 2
            assert {item["candidate_id"] for item in rerank_payload["results"]} == {
                "candidate_a",
                "candidate_b",
            }
            assert all(
                0 <= item["relevance_score"] <= 1
                for item in rerank_payload["results"]
            )

            expansion_response = client.post(
                "/v1/query-expansion",
                json={
                    "query": "How does 3D Gaussian Splatting balance rendering quality and memory?",
                    "profile": "comprehensive",
                },
            )
            expansion_response.raise_for_status()
            expansion_payload = expansion_response.json()
            for field in (
                "lexical_queries",
                "semantic_queries",
                "entity_queries",
                "relation_queries",
            ):
                assert expansion_payload[field]
            assert expansion_payload["hyde_text"].strip()
            assert expansion_payload["raw_output"].strip()

        record_path = run_root / "jobs" / "model-gateway-process.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        child_pid = int(record["pid"])
        assert record["status"] == "running"
        assert record["endpoint"] == endpoint
        assert _pid_exists(child_pid)

        conflict = subprocess.run(
            [
                executable,
                "model-gateway",
                "--host",
                host,
                "--port",
                str(port),
                "--data-dir",
                str(run_root),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
        assert conflict.returncode == 2
        assert "already in use" in (conflict.stdout + conflict.stderr)
        assert process.poll() is None
        assert _pid_exists(child_pid)
        assert json.loads(record_path.read_text(encoding="utf-8"))["status"] == "running"

        process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=30)
        assert process.returncode == 0, output
        assert f"Model gateway listening on {endpoint}" in output
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and _port_is_open(host, port):
            time.sleep(0.2)
        assert not _port_is_open(host, port)
        assert not _pid_exists(child_pid)
        stopped_record = json.loads(record_path.read_text(encoding="utf-8"))
        assert stopped_record["status"] == "stopped"
        assert stopped_record["exit_code"] == 0
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                output, _ = process.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate(timeout=10)
        if child_pid is not None and _pid_exists(child_pid):
            os.kill(child_pid, signal.SIGTERM)
        (logs / "gateway-stdout.log").write_text(output, encoding="utf-8")
        (logs / "lifecycle-results.json").write_text(
            json.dumps(
                {
                    "command": [
                        "paperos",
                        "model-gateway",
                        "--host",
                        host,
                        "--port",
                        str(port),
                        "--data-dir",
                        str(run_root),
                    ],
                    "health": health_payload,
                    "models": models_payload,
                    "embedding": embedding_summary,
                    "rerank": rerank_payload,
                    "query_expansion": expansion_payload,
                    "cli_returncode": process.returncode,
                    "child_pid": child_pid,
                    "port_closed": not _port_is_open(host, port),
                    "child_stopped": child_pid is None or not _pid_exists(child_pid),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
