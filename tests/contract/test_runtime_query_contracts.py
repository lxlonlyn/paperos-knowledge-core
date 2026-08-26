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
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.config import RuntimeSettings, load_settings
from paperos_core.errors import ConfigurationError, PaperOSError
from paperos_core.health import local_model_enablement
from paperos_core.indexes.manager import IndexManager
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry
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


def health_contract() -> dict[str, object]:
    enablement = local_model_enablement(
        LocalRuntimeUsage(embedding=False, reranker=True)
    )
    _require(
        enablement == {"embedding_enabled": False, "reranker_enabled": True},
        f"Health conflated embedding and reranker: {enablement}",
    )
    return {"status": "passed", **enablement}


async def run_contract() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="paperos-task1b-") as temporary:
        root = Path(temporary)
        return {
            "configuration": configuration_contract(root),
            "retrieval": await empty_retrieval_contract(root),
            "api_errors": api_error_contract(),
            "health": health_contract(),
        }


def main() -> None:
    report = asyncio.run(run_contract())
    print(json.dumps({"status": "passed", **report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
