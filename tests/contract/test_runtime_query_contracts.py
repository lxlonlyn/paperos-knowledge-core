"""Direct runtime contracts for configuration, empty retrieval, health, and API errors.

Run from the repository root:

    python tests/contract/test_runtime_query_contracts.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.config import RuntimeSettings, load_settings
from paperos_core.errors import ConfigurationError, PaperOSError
from paperos_core.health import HealthService, local_model_enablement
from paperos_core.indexes.manager import IndexManager
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry
from paperos_core.jobs.queue import JobQueue
from paperos_core.paths import build_data_paths
from paperos_core.retrieval.candidates import QueryRequest, QueryResponse
from paperos_core.retrieval.service import (
    NO_EVIDENCE_ANSWER,
    NO_EVIDENCE_MODEL,
    RetrievalService,
    effective_candidate_pool_size,
)
from paperos_core.runtime.local_inference.runtime import LocalRuntimeUsage
from paperos_core.storage.initializer import StorageInitializer


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class _ForbiddenDependency:
    """Tripwire proving an external boundary was not touched; it returns no values."""

    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"Empty retrieval touched forbidden dependency: {name}")


def _write_config(path: Path, *, embedding_endpoint: str, rerank_enabled: bool) -> None:
    path.write_text(
        "\n".join(
            [
                "[cognee.embedding]",
                f'endpoint = "{embedding_endpoint}"',
                "",
                "[local_inference]",
                "enabled = false",
                'host = "127.0.0.1"',
                "port = 8081",
                "",
                "[retrieval]",
                f"rerank_enabled = {str(rerank_enabled).lower()}",
            ]
        ),
        encoding="utf-8",
    )


def _require_configuration_error(path: Path, message_fragment: str) -> None:
    try:
        load_settings(path)
    except ConfigurationError as exc:
        _require(
            message_fragment in exc.message,
            f"Unexpected configuration error: {exc.message}",
        )
    else:
        raise RuntimeError(f"Configuration unexpectedly loaded: {path}")


def configuration_contract(root: Path) -> dict[str, object]:
    local_embedding = root / "local-embedding.toml"
    _write_config(
        local_embedding,
        embedding_endpoint="http://localhost:8081/v1",
        rerank_enabled=False,
    )
    _require_configuration_error(
        local_embedding,
        "local_inference.enabled must be true",
    )

    local_reranker = root / "local-reranker.toml"
    _write_config(
        local_reranker,
        embedding_endpoint="https://embedding.example/v1",
        rerank_enabled=True,
    )
    _require_configuration_error(
        local_reranker,
        "no remote reranker is configured",
    )

    example = load_settings(REPOSITORY_ROOT / "config" / "paperos.example.toml")
    _require(example.local_inference.enabled, "Example local inference must be enabled")
    _require(example.retrieval.rerank_enabled, "Example reranker must be enabled")
    return {
        "status": "passed",
        "invalid_local_embedding": True,
        "invalid_local_reranker": True,
        "example_consistent": True,
    }


def _assert_no_evidence_response(
    response: QueryResponse,
    *,
    forbidden_stages: set[str],
) -> None:
    _require(response.answer == NO_EVIDENCE_ANSWER, "No-evidence answer is unstable")
    _require(response.answer_model == NO_EVIDENCE_MODEL, "No-evidence model is misleading")
    _require(response.replay.replay_text == "", "No-LLM Replay must be empty")
    _require(response.candidates == [], "No-evidence candidates must be empty")
    _require(response.evidence == [], "No-evidence Evidence must be empty")
    _require(response.channels_used == [], "No-evidence channels must be empty")
    _require(response.distinct_documents == 0, "No-evidence document count must be zero")
    _require(response.provenance_complete is False, "Empty provenance must be incomplete")
    _require("no_evidence" in response.stages, "no_evidence stage is missing")
    _require(
        not forbidden_stages.intersection(response.stages),
        f"Unexecuted stages were reported: {response.stages}",
    )
    trace = response.trace.model_dump(mode="json")
    _require(trace["applied_document_ids"] == [], "Empty query applied documents")
    _require(
        all(value == [] for key, value in trace.items() if key != "applied_document_ids"),
        f"Empty query trace is not empty: {trace}",
    )


async def empty_retrieval_contract(root: Path) -> dict[str, object]:
    settings = RuntimeSettings.model_validate(
        {
            "data": {"directory": root / "empty-data", "dataset": "empty-contract"},
            "local_inference": {"enabled": True},
            "retrieval": {
                "top_k": 12,
                "candidate_pool_size": 40,
                "rerank_enabled": True,
            },
        }
    )
    paths = build_data_paths(settings.data_dir)
    StorageInitializer(paths).initialize()
    forbidden_dependency = _ForbiddenDependency()
    service = RetrievalService(
        settings,
        paths,
        CanonicalRepository(paths),
        SourceRegistry(paths),
        ScholarlyRegistry(paths),
        forbidden_dependency,  # type: ignore[arg-type]
        forbidden_dependency,  # type: ignore[arg-type]
        IndexManager(paths),
        forbidden_dependency,  # type: ignore[arg-type]
        forbidden_dependency,  # type: ignore[arg-type]
    )
    forbidden_common = {
        "vector_chunk_retrieval",
        "first_rerank",
        "second_rerank",
        "synthesis",
    }
    cases = (
        ("default", QueryRequest(query="empty library"), set()),
        (
            "expand_context",
            QueryRequest(query="empty local expansion", expand_context=True),
            {"local_post_hit_expansion"},
        ),
        (
            "expand_graph",
            QueryRequest(query="empty graph expansion", expand_graph=True),
            {"semantic_relation_expansion"},
        ),
    )
    executed: list[str] = []
    for name, request, case_forbidden in cases:
        response = await service.query(request)
        _assert_no_evidence_response(
            response,
            forbidden_stages=forbidden_common | case_forbidden,
        )
        executed.append(name)

    reranked = await service._rerank(
        "documents exist but channels returned no candidates",
        [],
        limit=40,
    )
    _require(reranked == [], "Zero candidates unexpectedly invoked reranking")

    service_source = (
        REPOSITORY_ROOT / "paperos_core" / "retrieval" / "service.py"
    ).read_text(encoding="utf-8")
    _require(
        "if request.expand_context and seeds:" in service_source,
        "Local post-hit expansion is not guarded by real seeds",
    )
    _require(
        "if request.expand_graph and seeds:" in service_source,
        "Graph post-hit expansion is not guarded by real seeds",
    )
    _require(
        "if fused else []" in service_source,
        "First reranking is not guarded by actual candidates",
    )
    _require(
        effective_candidate_pool_size(40, 100) == 100,
        "top_k is still truncated by candidate_pool_size",
    )
    return {
        "status": "passed",
        "runtime_cases": executed,
        "zero_candidate_guards": True,
        "top_k_pool": 100,
    }


def api_error_contract() -> dict[str, object]:
    internal_message = (
        "backend failed at /srv/paperos/private.db, "
        r"C:\Users\alice\secret.toml and file:///opt/paperos/token"
    )
    error = ConfigurationError(
        internal_message,
        affected=Path("/home/alice/private/config.toml"),
        details={
            "reason": "synthesis_context_too_small",
            "configured_tokens": 50,
            "required_estimated_tokens": 100,
            "candidate_count": 0,
            "retry_enabled": False,
            "evidence_id": "evidence:stable123",
            "local_path": "/home/alice/private/paper.pdf",
            "windows_path": r"C:\Users\alice\paper.pdf",
            "file_uri": "file:///tmp/private.json",
            "last_error": "sqlite failed at /var/lib/paperos/state.db",
            "status": "failed at /mnt/private/index",
            "nested": {
                "safe_count": 2,
                "safe_flag": True,
                "message": r"failure in C:\private\state.db",
            },
        },
    )
    payload = error.as_api_dict()
    _require(error.message == internal_message, "Internal error message was not retained")
    _require(
        payload["error"]["message"] == "PaperOS configuration is invalid.",
        "API did not use the stable public configuration message",
    )
    _require("affected" not in payload["error"], "API exposed affected")
    details = payload["error"].get("details")
    _require(isinstance(details, dict), "Safe API details are missing")
    assert isinstance(details, dict)
    _require(details["reason"] == "synthesis_context_too_small", "Safe reason was lost")
    _require(details["configured_tokens"] == 50, "Configured tokens were lost")
    _require(details["required_estimated_tokens"] == 100, "Required tokens were lost")
    _require(details["candidate_count"] == 0, "Count detail was lost")
    _require(details["retry_enabled"] is False, "Boolean detail was lost")
    _require(details["evidence_id"] == "evidence:stable123", "Stable ID was lost")
    _require(details["nested"] == {"safe_count": 2, "safe_flag": True}, "Nested safety failed")
    rendered = json.dumps(payload, ensure_ascii=False)
    for forbidden in ("/srv/", "/home/", "/var/", "/mnt/", "C:\\\\Users", "file://"):
        _require(forbidden not in rendered, f"API leaked local reference: {forbidden}")

    class UnknownError(PaperOSError):
        code = "unregistered_internal_error"

    unknown = UnknownError("secret backend exception /private/state.db").as_api_dict()
    _require(
        unknown["error"]["message"] == "The request could not be completed.",
        "Unknown errors do not use the safe fallback message",
    )
    return {
        "status": "passed",
        "public_message": payload["error"]["message"],
        "safe_detail_keys": sorted(details),
        "unknown_fallback": True,
    }


_PRIVATE_POSIX_PATH = "/home/user/private.db"
_PRIVATE_WINDOWS_PATH = r"C:\Users\user\secret.toml"
_PRIVATE_FILE_URI = "file:///tmp/token"
_PRIVATE_ENDPOINT = "https://internal.example.local:9443/private"
_PRIVATE_SECRET = "task1c-test-secret"
_PRIVATE_DIAGNOSTIC = (
    f"{_PRIVATE_POSIX_PATH} | {_PRIVATE_WINDOWS_PATH} | {_PRIVATE_FILE_URI} | "
    f"{_PRIVATE_ENDPOINT} | {_PRIVATE_SECRET}"
)


def _assert_public_json_safe(payload: object, label: str) -> None:
    rendered = json.dumps(payload, ensure_ascii=False)
    normalized = rendered.replace("\\\\", "\\")
    for forbidden in (
        _PRIVATE_POSIX_PATH,
        _PRIVATE_WINDOWS_PATH,
        _PRIVATE_FILE_URI,
        _PRIVATE_ENDPOINT,
        _PRIVATE_SECRET,
        _PRIVATE_DIAGNOSTIC,
    ):
        _require(forbidden not in normalized, f"{label} leaked {forbidden}")


class _Probe:
    def __init__(self, result: dict[str, object] | None = None, *, fail: bool = False) -> None:
        self.result = result or {}
        self.fail = fail

    async def health_check(self) -> dict[str, object]:
        if self.fail:
            raise RuntimeError(_PRIVATE_DIAGNOSTIC)
        return self.result

    async def health(self) -> dict[str, object]:
        if self.fail:
            raise RuntimeError(_PRIVATE_DIAGNOSTIC)
        return self.result


class _RuntimeConfig:
    embedding_dimensions = 768

    def embedding_targets(self, host: str, port: int) -> bool:
        return True


class _RuntimeConfigReader:
    def read(self) -> _RuntimeConfig:
        return _RuntimeConfig()


class _CogneeProbe:
    def __init__(self, *, fail: bool) -> None:
        self.fail = fail

    def read_manifest(self, snapshot_id: str) -> dict[str, object]:
        return {"dataset": {"name": "health-contract"}}

    async def vector_status(self, *, dataset_name: str | None = None) -> dict[str, object]:
        if self.fail:
            raise RuntimeError(_PRIVATE_DIAGNOSTIC)
        return {
            "backend": "cognee",
            "collection_count": 2,
            "record_count": 7,
            "dimensions": 768,
            "collections": {"private": 7},
            "endpoint": _PRIVATE_ENDPOINT,
            "path": _PRIVATE_POSIX_PATH,
            "secret": _PRIVATE_SECRET,
        }

    async def get_datapoint(
        self,
        document_id: str,
        *,
        dataset_name: str | None = None,
        snapshot_id: str | None = None,
    ) -> object:
        if self.fail:
            raise RuntimeError(_PRIVATE_DIAGNOSTIC)
        return {"secret": _PRIVATE_SECRET}


def _health_service(*, fail: bool) -> HealthService:
    settings = RuntimeSettings()
    local_result = {
        "status": "healthy",
        "cuda_visible_devices": _PRIVATE_SECRET,
        "endpoint": _PRIVATE_ENDPOINT,
        "path": _PRIVATE_POSIX_PATH,
        "secret": _PRIVATE_SECRET,
        "embedding": {
            "model": "embedding-contract",
            "dimensions": 768,
            "loaded": True,
            "path": _PRIVATE_POSIX_PATH,
            "secret": _PRIVATE_SECRET,
        },
        "reranker": {
            "model": "reranker-contract",
            "loaded": True,
            "path": _PRIVATE_WINDOWS_PATH,
            "secret": _PRIVATE_SECRET,
        },
    }
    mineru_result = {
        "provider": "mineru-contract",
        "configured": True,
        "reachable": True,
        "endpoint": _PRIVATE_ENDPOINT,
        "path": _PRIVATE_POSIX_PATH,
        "secret": _PRIVATE_SECRET,
    }
    llm_result = {
        "provider": "llm-contract",
        "model": "model-contract",
        "endpoint": _PRIVATE_ENDPOINT,
        "path": _PRIVATE_WINDOWS_PATH,
        "secret": _PRIVATE_SECRET,
    }
    bundle = SimpleNamespace(
        snapshot=SimpleNamespace(id="snapshot:health-contract"),
        document=SimpleNamespace(id="document:health-contract"),
    )
    return HealthService(
        paths=SimpleNamespace(),
        registry=SimpleNamespace(status=lambda: {"ingestion_job_count": 0}),
        canonical_repository=SimpleNamespace(list_active_bundles=lambda: [bundle]),
        mineru=SimpleNamespace(provider=_Probe(mineru_result, fail=fail)),
        llm=SimpleNamespace(
            health_check=_Probe(llm_result, fail=fail).health_check,
        ),
        local_inference=SimpleNamespace(
            settings=settings,
            cognee_config=_RuntimeConfigReader(),
            client=_Probe(local_result, fail=fail),
        ),
        cognee=_CogneeProbe(fail=fail),
        indexes=SimpleNamespace(
            lexical=SimpleNamespace(
                status=lambda: {"record_count": 0, "fts5": True}
            )
        ),
        queue=SimpleNamespace(list_jobs=list),
    )


async def health_contract() -> dict[str, object]:
    enablement = local_model_enablement(
        LocalRuntimeUsage(embedding=False, reranker=True)
    )
    _require(
        enablement == {"embedding_enabled": False, "reranker_enabled": True},
        f"Health conflated embedding and reranker: {enablement}",
    )
    failed = await _health_service(fail=True).report()
    expected_errors = {
        "mineru": (
            "unavailable",
            "mineru_unavailable",
            "The document parser is unavailable.",
        ),
        "llm": (
            "unavailable",
            "llm_unavailable",
            "The language model is unavailable.",
        ),
        "local_models": (
            "unavailable",
            "local_models_unavailable",
            "Local inference is unavailable.",
        ),
        "vector": (
            "degraded",
            "vector_unavailable",
            "The vector index is unavailable.",
        ),
        "cognee_graph": (
            "degraded",
            "cognee_graph_unavailable",
            "The knowledge graph is unavailable.",
        ),
    }
    for component, (status, code, message) in expected_errors.items():
        actual = failed["components"][component]
        _require(actual["status"] == status, f"Unexpected {component} status")
        _require(actual["error"]["code"] == code, f"Unexpected {component} code")
        _require(
            actual["error"]["message"] == message,
            f"Unexpected {component} message",
        )
        _require(
            set(actual["error"]) == {"code", "message"},
            f"{component} error is not a stable public diagnostic",
        )
    _assert_public_json_safe(failed, "failed health response")

    healthy = await _health_service(fail=False).report()
    components = healthy["components"]
    _require(
        set(components["mineru"])
        == {"status", "provider", "configured", "reachable"},
        "MinerU health allowlist changed",
    )
    _require(
        set(components["llm"]) == {"status", "provider", "model"},
        "LLM health allowlist changed",
    )
    _require(
        set(components["local_models"])
        == {
            "status",
            "embedding_enabled",
            "reranker_enabled",
            "embedding",
            "reranker",
        },
        "Local model health allowlist changed",
    )
    _require(
        set(components["local_models"]["embedding"])
        == {"model", "dimensions", "loaded"},
        "Embedding health allowlist changed",
    )
    _require(
        set(components["local_models"]["reranker"]) == {"model", "loaded"},
        "Reranker health allowlist changed",
    )
    _require(
        set(components["vector"])
        == {
            "status",
            "backend",
            "collection_count",
            "record_count",
            "dimensions",
        },
        "Vector health allowlist changed",
    )
    _require(
        components["cognee_graph"]
        == {"status": "healthy", "document_count": 1},
        "Graph health response changed",
    )
    _assert_public_json_safe(healthy, "healthy health response")
    return {
        "status": "passed",
        **enablement,
        "failure_codes": sorted(code for _, code, _ in expected_errors.values()),
        "healthy_allowlists": True,
    }


def job_status_contract(root: Path) -> dict[str, object]:
    paths = build_data_paths(root / "job-data")
    StorageInitializer(paths).initialize()
    queue = JobQueue(paths)
    pending = queue.enqueue(
        "ingest",
        {"path": paths.tmp / "staging.pdf", "source": "contract"},
    )
    pending_public = queue.public_dict(pending)
    _require(pending_public["error"] is None, "Pending job has a public error")
    _require("path" not in pending_public["payload"], "Staging path was exposed")
    _assert_public_json_safe(pending_public, "pending job response")

    running = queue.claim_next()
    _require(running is not None, "Pending job was not claimed")
    assert running is not None
    running_public = queue.public_dict(running)
    _require(running_public["error"] is None, "Running job has a public error")
    _require("path" not in running_public["payload"], "Running staging path exposed")
    _assert_public_json_safe(running_public, "running job response")

    failed = queue.fail(running.id, f"RuntimeError: {_PRIVATE_DIAGNOSTIC}")
    internal = queue.get(failed.id)
    _require(
        internal.error == f"RuntimeError: {_PRIVATE_DIAGNOSTIC}",
        "Internal job error was not retained",
    )
    public = queue.public_dict(internal)
    _require(
        public["error"]
        == {
            "code": "operational_job_failed",
            "message": "The operation could not be completed.",
        },
        "Failed job did not use the fixed public diagnostic",
    )
    _require("path" not in public["payload"], "Failed staging path was exposed")
    _assert_public_json_safe(public, "failed job response")

    completed = queue.enqueue("export", {"format": "json"})
    result = {"export_id": "export:stable123", "count": 3}
    completed = queue.complete(completed.id, result)
    completed_public = queue.public_dict(completed)
    _require(completed_public["error"] is None, "Completed job has a public error")
    _require(completed_public["result"] == result, "Job result fields changed")
    _assert_public_json_safe(completed_public, "completed job response")
    return {
        "status": "passed",
        "internal_error_retained": True,
        "public_error": public["error"],
        "staging_path_hidden": True,
    }


async def run_contract() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="paperos-task1b-") as temporary:
        root = Path(temporary)
        return {
            "configuration": configuration_contract(root),
            "retrieval": await empty_retrieval_contract(root),
            "api_errors": api_error_contract(),
            "health": await health_contract(),
            "jobs": job_status_contract(root),
        }


def main() -> None:
    report = asyncio.run(run_contract())
    print(json.dumps({"status": "passed", **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
