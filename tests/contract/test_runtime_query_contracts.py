"""Runtime contracts for configuration, empty retrieval, health, and API errors.

Run from the repository root:

    python -m pytest tests/contract/test_runtime_query_contracts.py
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import subprocess
import sys
import tempfile
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.adapters.cognee.configurator import CogneeConfigurator
from paperos_core.adapters.cognee.runtime_config import CogneeRuntimeConfigReader
from paperos_core.application import Application
from paperos_core.config import RuntimeSettings, load_settings
from paperos_core.documents import DocumentService
from paperos_core.domain.enums import IngestionJobStatus, ParseRunStatus
from paperos_core.errors import (
    ConfigurationError,
    LocalInferenceRuntimeIncompatibleError,
    PaperOSError,
)
from paperos_core.feedback.models import FeedbackRequest, FeedbackType
from paperos_core.feedback.service import FeedbackService
from paperos_core.health import HealthService, local_model_enablement
from paperos_core.indexes.manager import IndexManager
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.parser_artifacts import ParserArtifactRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry
from paperos_core.ingestion.validation import validate_pdf
from paperos_core.jobs.queue import JobQueue
from paperos_core.jobs.worker import BackgroundWorker
from paperos_core.paths import DataPaths, build_data_paths
from paperos_core.retrieval.candidates import QueryRequest, QueryResponse
from paperos_core.retrieval.service import (
    NO_EVIDENCE_ANSWER,
    NO_EVIDENCE_MODEL,
    RetrievalService,
    effective_candidate_pool_size,
)
from paperos_core.runtime.local_inference.runtime import (
    LocalInferenceRuntime,
    LocalRuntimeUsage,
)
from paperos_core.storage.initializer import (
    LEXICAL_SCHEMA_VERSION,
    REGISTRY_SCHEMA_VERSION,
    StorageInitializer,
)


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

    claim_without_semantic = root / "claim-without-semantic.toml"
    claim_without_semantic.write_text(
        "[cognee.embedding]\n"
        "endpoint = \"https://embedding.example/v1\"\n"
        "\n"
        "[local_inference]\n"
        "enabled = false\n"
        "\n"
        "[retrieval]\n"
        "rerank_enabled = false\n"
        "\n"
        "[ingestion]\n"
        "semantic_enrichment_enabled = false\n"
        "claim_enrichment_enabled = true\n",
        encoding="utf-8",
    )
    _require_configuration_error(
        claim_without_semantic,
        "semantic_enrichment_enabled=true",
    )

    example = load_settings(REPOSITORY_ROOT / "config" / "paperos.example.toml")
    _require(example.local_inference.enabled, "Example local inference must be enabled")
    _require(example.retrieval.rerank_enabled, "Example reranker must be enabled")
    _require(
        not example.ingestion.semantic_enrichment_enabled,
        "Example semantic enrichment must default to disabled",
    )
    defaults = RuntimeSettings()
    _require(
        not defaults.ingestion.semantic_enrichment_enabled
        and not defaults.ingestion.claim_enrichment_enabled,
        "Semantic and Claim enrichment defaults must both be disabled",
    )
    return {
        "status": "passed",
        "invalid_local_embedding": True,
        "invalid_local_reranker": True,
        "invalid_claim_without_semantic": True,
        "semantic_enrichment_default": False,
        "example_consistent": True,
    }


def _database_user_version(path: Path) -> int:
    with closing(sqlite3.connect(path)) as connection, connection:
        return int(connection.execute("PRAGMA user_version").fetchone()[0])


def _require_storage_failure(initializer: StorageInitializer, label: str) -> None:
    try:
        initializer.initialize()
    except ConfigurationError:
        return
    raise RuntimeError(f"{label} database unexpectedly initialized")


def storage_baseline_contract(root: Path) -> dict[str, object]:
    fresh_paths = build_data_paths(root / "storage-fresh")
    fresh = StorageInitializer(fresh_paths)
    fresh.initialize()
    _require(
        _database_user_version(fresh_paths.registry_db) == REGISTRY_SCHEMA_VERSION,
        "Fresh registry schema version is not 1",
    )
    _require(
        _database_user_version(fresh_paths.lexical_db) == LEXICAL_SCHEMA_VERSION,
        "Fresh lexical schema version is not 1",
    )
    fresh.initialize()
    _require(fresh.validate().valid, "Version 1 databases did not reopen")

    source_registry = SourceRegistry(fresh_paths)
    with source_registry._connect() as owned_connection:
        owned_connection.execute("SELECT 1").fetchone()
    try:
        owned_connection.execute("SELECT 1")
    except sqlite3.ProgrammingError:
        pass
    else:
        raise RuntimeError("Repository-owned SQLite connection remained open")

    scholarly_registry = ScholarlyRegistry(fresh_paths)
    with closing(
        sqlite3.connect(fresh_paths.registry_db)
    ) as external_connection, external_connection:
        external_connection.row_factory = sqlite3.Row
        scholarly_registry.canonicalize_work_id(
            "work_missing_connection_contract",
            external_connection,
        )
        external_connection.execute("SELECT 1").fetchone()

    legacy_registry_paths = build_data_paths(root / "storage-legacy-registry")
    legacy_registry_paths.initialize()
    with closing(
        sqlite3.connect(legacy_registry_paths.registry_db)
    ) as connection, connection:
        connection.execute("CREATE TABLE source_files (id TEXT PRIMARY KEY)")
    _require_storage_failure(
        StorageInitializer(legacy_registry_paths),
        "Pre-1.0 registry",
    )

    legacy_lexical_paths = build_data_paths(root / "storage-legacy-lexical")
    legacy_lexical_paths.initialize()
    with closing(
        sqlite3.connect(legacy_lexical_paths.lexical_db)
    ) as connection, connection:
        connection.execute("CREATE TABLE lexical_records (object_id TEXT PRIMARY KEY)")
    _require_storage_failure(
        StorageInitializer(legacy_lexical_paths),
        "Pre-1.0 lexical",
    )

    future_registry_paths = build_data_paths(root / "storage-future-registry")
    future_registry_paths.initialize()
    with closing(
        sqlite3.connect(future_registry_paths.registry_db)
    ) as connection, connection:
        connection.execute(f"PRAGMA user_version = {REGISTRY_SCHEMA_VERSION + 1}")
    _require_storage_failure(
        StorageInitializer(future_registry_paths),
        "Future registry",
    )

    future_lexical_paths = build_data_paths(root / "storage-future-lexical")
    future_lexical = StorageInitializer(future_lexical_paths)
    future_lexical.initialize()
    with closing(
        sqlite3.connect(future_lexical_paths.lexical_db)
    ) as connection, connection:
        connection.execute(f"PRAGMA user_version = {LEXICAL_SCHEMA_VERSION + 1}")
    _require_storage_failure(future_lexical, "Future lexical")

    custom_config = root / "custom-storage.toml"
    custom_config.write_text(
        "[data]\n"
        'directory = "custom-storage-data"\n'
        "\n"
        "[storage]\n"
        'registry_filename = "paperos-registry.db"\n'
        'lexical_filename = "paperos-search.db"\n'
        "\n"
        "[cognee.storage]\n"
        'database_name = "paperos_cognee"\n',
        encoding="utf-8",
    )
    custom = load_settings(custom_config)
    custom_paths = build_data_paths(
        custom.data_dir,
        registry_filename=custom.storage.registry_filename,
        lexical_filename=custom.storage.lexical_filename,
    )
    StorageInitializer(custom_paths).initialize()
    _require(custom_paths.registry_db.is_file(), "Custom registry filename was not used")
    _require(custom_paths.lexical_db.is_file(), "Custom lexical filename was not used")
    _require(
        not (custom_paths.jobs / "registry.sqlite3").exists(),
        "Default registry was created beside the configured database",
    )
    _require(
        not (custom_paths.indexes / "lexical.sqlite3").exists(),
        "Default lexical database was created beside the configured database",
    )
    CogneeConfigurator().apply(custom, custom_paths)
    _require(
        CogneeRuntimeConfigReader().read().db_name == "paperos_cognee",
        "Custom Cognee database name was not applied",
    )

    unsafe_configs = {
        "parent": '[storage]\nregistry_filename = "../registry.sqlite3"\n',
        "absolute": '[storage]\nlexical_filename = "/tmp/lexical.sqlite3"\n',
        "nested": '[cognee.storage]\ndatabase_name = "nested/name"\n',
    }
    for name, content in unsafe_configs.items():
        path = root / f"unsafe-storage-{name}.toml"
        path.write_text(content, encoding="utf-8")
        _require_configuration_error(path, "storage names must be safe relative names")

    backfill_config = root / "backfill-storage.toml"
    backfill_config.write_text(
        "[data]\n"
        'directory = "backfill-data"\n'
        "\n"
        "[storage]\n"
        'registry_filename = "custom-registry.sqlite3"\n'
        'lexical_filename = "custom-lexical.sqlite3"\n',
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.backfill_scholarly_works",
            "--config",
            str(backfill_config),
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    _require(completed.returncode == 0, f"Scholarly backfill failed: {completed.stderr}")
    backfill_data = root / "backfill-data"
    _require(
        (backfill_data / "jobs" / "custom-registry.sqlite3").is_file(),
        "Scholarly backfill ignored the configured registry filename",
    )
    _require(
        not (backfill_data / "jobs" / "registry.sqlite3").exists(),
        "Scholarly backfill created the default registry",
    )
    _require(
        (backfill_data / "indexes" / "custom-lexical.sqlite3").is_file(),
        "Scholarly backfill ignored the configured lexical filename",
    )
    _require(
        not (backfill_data / "indexes" / "lexical.sqlite3").exists(),
        "Scholarly backfill created the default lexical database",
    )

    return {
        "status": "passed",
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "lexical_schema_version": LEXICAL_SCHEMA_VERSION,
        "version_1_reopen": True,
        "owned_connection_closed": True,
        "external_connection_retained": True,
        "legacy_version_0_rejected": ["registry", "lexical"],
        "unsupported_version_rejected": ["registry", "lexical"],
        "custom_registry": custom_paths.registry_db.name,
        "custom_lexical": custom_paths.lexical_db.name,
        "custom_cognee": "paperos_cognee",
        "unsafe_names_rejected": sorted(unsafe_configs),
        "backfill_registry": "custom-registry.sqlite3",
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
        all(not value for value in trace.values()),
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
    )
    executed: list[str] = []
    for name, request, case_forbidden in cases:
        response = await service.query(request)
        _assert_no_evidence_response(
            response,
            forbidden_stages=forbidden_common | case_forbidden,
        )
        executed.append(name)

    try:
        await service.query(
            QueryRequest(query="disabled graph expansion", expand_graph=True)
        )
    except ConfigurationError as exc:
        _require(
            exc.details.get("reason") == "semantic_enrichment_disabled",
            f"Unexpected semantic expansion guard: {exc.details}",
        )
        public = exc.as_api_dict()
        public_error = public.get("error", {})
        public_details = (
            public_error.get("details", {})
            if isinstance(public_error, dict)
            else {}
        )
        _require(
            isinstance(public_details, dict)
            and public_details.get("reason") == "semantic_enrichment_disabled",
            f"Semantic expansion reason was not safe for the API: {public}",
        )
    else:
        raise RuntimeError("Disabled semantic expansion did not fail explicitly.")

    reranked = await service._rerank(
        "documents exist but channels returned no candidates",
        [],
        corpus=object(),  # type: ignore[arg-type]
        limit=40,
    )
    _require(
        reranked.candidates == []
        and reranked.projection_version is None
        and reranked.span_count == 0,
        "Zero candidates unexpectedly invoked reranking",
    )

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
    _require("if fused" in service_source, "First reranking lacks an empty guard")
    _require(
        effective_candidate_pool_size(40, 100) == 100,
        "top_k is still truncated by candidate_pool_size",
    )
    return {
        "status": "passed",
        "runtime_cases": executed,
        "semantic_expansion_guard": "semantic_enrichment_disabled",
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
    def __init__(
        self,
        *,
        embedding_model: str = "embedding-contract",
        embedding_dimensions: int = 768,
        embedding_max_tokens: int = 2048,
        embedding_enabled: bool = True,
    ) -> None:
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self.embedding_max_tokens = embedding_max_tokens
        self.embedding_enabled = embedding_enabled

    def embedding_targets(self, host: str, port: int) -> bool:
        return self.embedding_enabled


class _RuntimeConfigReader:
    def __init__(self, config: _RuntimeConfig | None = None) -> None:
        self.config = config or _RuntimeConfig()

    def read(self) -> _RuntimeConfig:
        return self.config


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


def _health_service(*, fail: bool, worker_running: bool = True) -> HealthService:
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
        worker=SimpleNamespace(running=worker_running),
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
    _require(
        components["worker"] == {"status": "healthy", "running": True},
        "Running worker health response changed",
    )
    _require(healthy["status"] == "healthy", "Live worker degraded overall health")
    _assert_public_json_safe(healthy, "healthy health response")

    dead = await _health_service(fail=False, worker_running=False).report()
    _require(dead["status"] == "degraded", "Dead worker did not degrade health")
    _require(
        dead["components"]["worker"]
        == {
            "status": "unavailable",
            "error": {
                "code": "worker_unavailable",
                "message": "The operational worker is unavailable.",
            },
            "running": False,
        },
        "Dead worker public diagnostic changed",
    )
    _assert_public_json_safe(dead, "dead worker health response")
    return {
        "status": "passed",
        **enablement,
        "failure_codes": sorted(code for _, code, _ in expected_errors.values()),
        "healthy_allowlists": True,
        "worker_liveness": True,
    }


def local_runtime_identity_contract(root: Path) -> dict[str, object]:
    embedding_path = root / "embedding-contract.gguf"
    reranker_path = root / "reranker-contract.gguf"
    embedding_path.write_bytes(b"embedding-contract")
    reranker_path.write_bytes(b"reranker-contract")
    settings = RuntimeSettings.model_validate(
        {
            "local_inference": {
                "embedding_model_path": embedding_path,
                "reranker_model_path": reranker_path,
                "cuda_devices": [2, 5],
            },
            "retrieval": {"rerank_enabled": True},
        }
    )
    runtime = LocalInferenceRuntime(
        settings,
        build_data_paths(root / "runtime-identity-data"),
        _Probe(),
        _RuntimeConfigReader(),
    )
    expected = runtime._expected_runtime_identity()
    expected_embedding_file = expected["embedding"]["model"]["file"]
    _require(
        expected_embedding_file
        == {
            "resolved_path": str(embedding_path.resolve()),
            "file_size": str(embedding_path.stat().st_size),
            "mtime_ns": str(embedding_path.stat().st_mtime_ns),
        },
        "Embedding file identity is not deterministic",
    )
    runtime._validate_reuse_identity(
        {"status": "healthy", "runtime_identity": expected}
    )
    _require(
        expected["embedding"]["max_tokens"] == 2048
        and expected["reranker"]["max_tokens"] == 4096,
        "Runtime token limits are missing or use inconsistent types",
    )

    rejected: list[str] = []

    def changed_identity() -> dict[str, Any]:
        return json.loads(json.dumps(expected))

    def require_rejected(label: str, identity: dict[str, Any]) -> None:
        try:
            runtime._validate_reuse_identity(
                {"status": "healthy", "runtime_identity": identity}
            )
        except LocalInferenceRuntimeIncompatibleError as exc:
            _require(
                exc.code == "local_runtime_incompatible" and not exc.retryable,
                f"{label} used the wrong compatibility error",
            )
            _require(
                exc.details == {"reason": "runtime_identity_mismatch"},
                f"{label} exposed unstable mismatch details",
            )
            rejected.append(label)
        else:
            raise RuntimeError(f"{label} unexpectedly reused the local runtime")

    changed = changed_identity()
    changed["embedding"]["model"]["name"] = "embedding-contract-v2"
    require_rejected("embedding_model", changed)

    changed = changed_identity()
    changed["embedding"]["dimensions"] = 1024
    require_rejected("embedding_dimensions", changed)

    changed = changed_identity()
    changed["embedding"]["max_tokens"] = 4096
    require_rejected("embedding_max_tokens", changed)

    changed = changed_identity()
    changed["reranker"]["enabled"] = False
    require_rejected("reranker_config", changed)

    changed = changed_identity()
    changed["reranker"]["max_tokens"] = 8192
    require_rejected("reranker_max_tokens", changed)

    changed = changed_identity()
    changed["cuda_visible_devices"] = "5"
    require_rejected("cuda_visible_devices", changed)

    changed = changed_identity()
    changed["protocol_version"] = expected["protocol_version"] + 1
    require_rejected("protocol_version", changed)

    _require(len(rejected) == 7, "Not every incompatible runtime was rejected")
    return {
        "status": "passed",
        "same_config_reused": True,
        "rejected": rejected,
    }


class _ReplayCanonicalProbe:
    document_id = "document:replay-correctness"
    chunk_id = "chunk_replay_correctness"

    def __init__(self, dataset_id: str) -> None:
        self._revision = 1
        self._set_active(dataset_id)

    def _set_active(self, dataset_id: str | None) -> None:
        snapshot_id = f"snapshot:replay-correctness:{self._revision}"
        self.bundle = SimpleNamespace(
            snapshot=SimpleNamespace(
                id=snapshot_id,
                parse_run_id=f"parse:replay-correctness:{self._revision}",
                dataset_id=dataset_id,
            ),
            document=SimpleNamespace(
                id=self.document_id,
                title="Replay correctness",
                source_file_id="source:replay-correctness",
            ),
            sections=[],
            references=[],
            elements=[],
        )

    def activate_reprocessed(self, dataset_id: str | None) -> None:
        self._revision += 1
        self._set_active(dataset_id)

    def list_active_bundles(self) -> list[object]:
        return [self.bundle]

    def active_snapshot_id(self, document_id: str) -> str | None:
        return self.bundle.snapshot.id if document_id == self.document_id else None

    def get_bundle(self, snapshot_id: str) -> object:
        _require(snapshot_id == self.bundle.snapshot.id, "Replay probe requested stale bundle")
        return self.bundle

    def get_chunk_projection(self, snapshot_id: str) -> object:
        _require(snapshot_id == self.bundle.snapshot.id, "Replay probe requested stale projection")
        return SimpleNamespace(chunks=[SimpleNamespace(id=self.chunk_id)])


class _ReplayIngestionProbe:
    def __init__(self, repository: _ReplayCanonicalProbe, source_path: Path) -> None:
        self.repository = repository
        self.source_path = source_path
        self.reprocess_dataset: str | None = None

    def get_source(self, source_file_id: str) -> object:
        return SimpleNamespace(
            id=source_file_id,
            original_filename="replay-correctness.pdf",
            storage_path=self.source_path,
        )

    async def ingest_pdf_to_knowledge(
        self,
        path: Path,
        *,
        dataset: str | None = None,
        operation_id: str | None = None,
    ) -> object:
        _require(path == self.source_path, "Reprocess changed the retained source path")
        _require(operation_id is None, "Direct reprocess unexpectedly set an operation ID")
        self.reprocess_dataset = dataset
        self.repository.activate_reprocessed(dataset)
        return SimpleNamespace(public_dict=lambda: {"dataset_id": dataset})


async def replay_correctness_contract(root: Path) -> dict[str, object]:
    paths = build_data_paths(root / "replay-correctness-data")
    StorageInitializer(paths).initialize()
    dataset_id = "dataset-replay-preserved"
    repository = _ReplayCanonicalProbe(dataset_id)
    ingestion = _ReplayIngestionProbe(
        repository,
        paths.raw / "source:replay-correctness" / "source.pdf",
    )
    documents = DocumentService(
        paths,
        repository,  # type: ignore[arg-type]
        ingestion,  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(),  # type: ignore[arg-type]
    )
    old_snapshot_id = repository.bundle.snapshot.id
    reprocessed = await documents.reprocess(repository.document_id)
    _require(
        repository.bundle.snapshot.id != old_snapshot_id,
        "Reprocess probe did not activate a new revision",
    )
    _require(
        ingestion.reprocess_dataset == dataset_id
        and repository.bundle.snapshot.dataset_id == dataset_id
        and reprocessed["dataset_id"] == dataset_id,
        "Reprocess did not preserve the active snapshot dataset",
    )

    feedback = FeedbackService(
        paths,
        repository,  # type: ignore[arg-type]
    )
    recorded = feedback.record(
        FeedbackRequest(
            target_id=repository.chunk_id,
            feedback_type=FeedbackType.CORRECT,
            evidence_ids=[f"evidence:{repository.chunk_id}"],
            replacement_text="Corrected replay-safe text",
        )
    )
    with closing(sqlite3.connect(paths.registry_db)) as connection, connection:
        connection.execute(
            """
            CREATE TRIGGER fail_replay_improvement
            BEFORE INSERT ON improvements
            BEGIN
                SELECT RAISE(ABORT, 'injected improvement failure');
            END;
            """
        )
    try:
        feedback.improve()
    except sqlite3.IntegrityError:
        pass
    else:
        raise RuntimeError("Injected Improvement write failure was swallowed")

    with closing(sqlite3.connect(paths.registry_db)) as connection, connection:
        partial_counts = (
            int(connection.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]),
            int(connection.execute("SELECT COUNT(*) FROM improvements").fetchone()[0]),
        )
        connection.execute("DROP TRIGGER fail_replay_improvement")
    _require(
        partial_counts == (0, 0),
        f"Improve left partial derived state after failure: {partial_counts}",
    )

    retry = feedback.improve()
    _require(
        retry.processed_feedback_ids == [recorded.id]
        and len(retry.corrections) == 1
        and len(retry.improvements) == 1
        and retry.improvements[0].correction_id == retry.corrections[0].id,
        "Improve retry did not create one complete derived pair",
    )
    repeated = feedback.improve()
    _require(
        repeated.processed_feedback_ids == []
        and repeated.corrections == []
        and repeated.improvements == [],
        "Complete Improvement replay was not safely skipped",
    )
    with closing(sqlite3.connect(paths.registry_db)) as connection, connection:
        final_counts = (
            int(connection.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]),
            int(connection.execute("SELECT COUNT(*) FROM improvements").fetchone()[0]),
        )
    _require(final_counts == (1, 1), f"Improve replay created duplicates: {final_counts}")
    return {
        "status": "passed",
        "reprocess_dataset": repository.bundle.snapshot.dataset_id,
        "improve_partial_counts": partial_counts,
        "improve_final_counts": final_counts,
        "complete_replay_skipped": True,
    }


class _LifecycleProbe:
    required = False

    def __init__(self, *, fail_start: bool = False) -> None:
        self.fail_start = fail_start

    def cleanup_stale_record(self) -> None:
        return None

    async def start(self) -> None:
        if self.fail_start:
            raise RuntimeError("injected application startup failure")

    async def stop(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class _RecoveryObservingWorker:
    def __init__(
        self,
        queue: JobQueue,
        operational_job_id: str,
        registry: SourceRegistry,
        ingestion_job_id: str,
        parser_artifacts: ParserArtifactRepository,
        parse_run_id: str,
    ) -> None:
        self.queue = queue
        self.operational_job_id = operational_job_id
        self.registry = registry
        self.ingestion_job_id = ingestion_job_id
        self.parser_artifacts = parser_artifacts
        self.parse_run_id = parse_run_id
        self.running = False
        self.operational_status_at_start: str | None = None
        self.ingestion_status_at_start: IngestionJobStatus | None = None
        self.parse_status_at_start: ParseRunStatus | None = None

    def cleanup_stale_record(self) -> None:
        return None

    async def start(self) -> None:
        self.operational_status_at_start = self.queue.get(self.operational_job_id).status
        self.ingestion_status_at_start = self.registry.get_job(self.ingestion_job_id).status
        self.parse_status_at_start = self.parser_artifacts.get_parse_run(self.parse_run_id).status
        self.running = True

    async def stop(self) -> None:
        self.running = False


def _contract_application(
    paths: DataPaths,
    *,
    queue: JobQueue | None = None,
    worker: Any | None = None,
) -> Application:
    local_inference = _LifecycleProbe()
    selected_queue = queue or JobQueue(paths)
    selected_worker = worker or _LifecycleProbe()
    return Application(
        settings=RuntimeSettings(),
        services=SimpleNamespace(),  # type: ignore[arg-type]
        runtime=SimpleNamespace(
            local_inference=local_inference,
            worker=selected_worker,
        ),  # type: ignore[arg-type]
        paths=paths,
        registry=SourceRegistry(paths),
        scholarly_registry=SimpleNamespace(),  # type: ignore[arg-type]
        parser_artifacts=ParserArtifactRepository(paths),
        canonical_repository=SimpleNamespace(),  # type: ignore[arg-type]
        canonical_mapper=SimpleNamespace(),  # type: ignore[arg-type]
        mineru=_LifecycleProbe(),  # type: ignore[arg-type]
        local_inference_client=_LifecycleProbe(),  # type: ignore[arg-type]
        llm=SimpleNamespace(),  # type: ignore[arg-type]
        knowledge_pipeline=SimpleNamespace(compat=_LifecycleProbe()),  # type: ignore[arg-type]
        queue=selected_queue,
        storage=StorageInitializer(paths),
    )


async def application_singleton_contract(root: Path) -> dict[str, object]:
    shared_paths = build_data_paths(root / "singleton-shared-data")
    other_paths = build_data_paths(root / "singleton-other-data")
    first = _contract_application(shared_paths)
    second = _contract_application(shared_paths)
    other = _contract_application(other_paths)

    await first.start()
    try:
        try:
            await second.start()
        except RuntimeError as exc:
            _require(
                "already active" in str(exc),
                f"Unexpected same-root singleton error: {exc}",
            )
        else:
            raise RuntimeError("A second Application acquired the same data-root lock")

        await other.start()
        await other.aclose()
    finally:
        await first.aclose()

    await second.start()
    await second.aclose()

    failure_paths = build_data_paths(root / "singleton-startup-failure-data")
    failing = _contract_application(
        failure_paths,
        worker=_LifecycleProbe(fail_start=True),
    )
    try:
        await failing.start()
    except RuntimeError as exc:
        _require(
            "injected application startup failure" in str(exc),
            f"Unexpected injected startup error: {exc}",
        )
    else:
        raise RuntimeError("Injected Application startup failure did not occur")

    successor = _contract_application(failure_paths)
    await successor.start()
    await successor.aclose()
    return {
        "status": "passed",
        "same_root_rejected": True,
        "released_on_close": True,
        "different_roots_allowed": True,
        "released_on_startup_failure": True,
    }


async def application_start_recovery_contract(root: Path) -> dict[str, object]:
    paths = build_data_paths(root / "application-recovery-data")
    storage = StorageInitializer(paths)
    storage.initialize()

    registry = SourceRegistry(paths)
    source_path = root / "application-recovery-source.pdf"
    source_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    source, _ = registry.register_source(
        validate_pdf(source_path, max_file_mb=1),
        dataset_id="recovery-contract",
    )
    ingestion_job = registry.create_job(source.id, dataset_id="recovery-contract")
    ingestion_job = registry.update_job(
        ingestion_job.id,
        status=IngestionJobStatus.PARSING,
        current_operation="waiting_for_mineru",
    )

    parser_artifacts = ParserArtifactRepository(paths)
    parse_run = parser_artifacts.create_parse_run(
        source,
        provider="contract",
        backend="contract",
        request_options={},
    )
    parse_run = parser_artifacts.update_parse_run(
        parse_run.id,
        status=ParseRunStatus.RUNNING,
        provider_task_id="stale-mineru-task",
    )

    queue = JobQueue(paths)
    pending = queue.enqueue("rebuild")
    interrupted = queue.claim_next()
    _require(
        interrupted is not None and interrupted.id == pending.id,
        "Application recovery fixture did not create a running operational job",
    )
    assert interrupted is not None
    worker = _RecoveryObservingWorker(
        queue,
        interrupted.id,
        registry,
        ingestion_job.id,
        parser_artifacts,
        parse_run.id,
    )
    application = _contract_application(paths, queue=queue, worker=worker)
    await application.start()
    try:
        recovered_operational = queue.get(interrupted.id)
        recovered_ingestion = registry.get_job(ingestion_job.id)
        recovered_parse = parser_artifacts.get_parse_run(parse_run.id)
        _require(
            worker.operational_status_at_start == "pending",
            "Application started worker before operational-job recovery",
        )
        _require(
            worker.ingestion_status_at_start == IngestionJobStatus.INTERRUPTED,
            "Application started worker before ingestion-attempt recovery",
        )
        _require(
            worker.parse_status_at_start == ParseRunStatus.INTERRUPTED,
            "Application started worker before parse-attempt recovery",
        )
        _require(
            recovered_operational.error == "worker_interrupted",
            "Application startup recovery reason changed",
        )
        _require(
            recovered_ingestion.status == IngestionJobStatus.INTERRUPTED,
            "Non-terminal IngestionJob was not marked interrupted",
        )
        _require(
            recovered_parse.status == ParseRunStatus.INTERRUPTED,
            "Running ParseRun was not marked interrupted",
        )

        ingestion_before_repeat = recovered_ingestion.model_dump(mode="json")
        parse_before_repeat = recovered_parse.model_dump(mode="json")
        _require(
            registry.recover_interrupted_jobs() == 0,
            "Repeated ingestion-attempt recovery was not idempotent",
        )
        _require(
            parser_artifacts.recover_interrupted_runs() == 0,
            "Repeated parse-attempt recovery was not idempotent",
        )
        _require(
            registry.get_job(ingestion_job.id).model_dump(mode="json")
            == ingestion_before_repeat,
            "Repeated recovery modified the interrupted IngestionJob",
        )
        _require(
            parser_artifacts.get_parse_run(parse_run.id).model_dump(mode="json")
            == parse_before_repeat,
            "Repeated recovery modified the interrupted ParseRun",
        )
    finally:
        await application.aclose()
    return {
        "status": "passed",
        "recovered_before_worker_start": True,
        "ingestion_attempt_interrupted": True,
        "parse_attempt_interrupted": True,
        "attempt_recovery_idempotent": True,
        "operational_job_requeued": True,
    }


class _TransientClaimQueue:
    def __init__(self, jobs: Path) -> None:
        self.paths = SimpleNamespace(jobs=jobs)
        self.claim_calls = 0

    def claim_next(self) -> None:
        self.claim_calls += 1
        if self.claim_calls == 1:
            raise RuntimeError("transient claim failure")


async def worker_loop_contract(root: Path) -> dict[str, object]:
    jobs = root / "worker-loop"
    jobs.mkdir(parents=True)
    queue = _TransientClaimQueue(jobs)
    dependency = SimpleNamespace()
    worker = BackgroundWorker(
        queue,  # type: ignore[arg-type]
        dependency,  # type: ignore[arg-type]
        dependency,  # type: ignore[arg-type]
        dependency,  # type: ignore[arg-type]
        dependency,  # type: ignore[arg-type]
        poll_interval_seconds=0.01,
    )
    await worker.start()
    try:
        for _ in range(100):
            if queue.claim_calls >= 2:
                break
            await asyncio.sleep(0.005)
        _require(
            queue.claim_calls >= 2,
            "Worker did not continue after a transient claim_next failure",
        )
        _require(worker.running, "Worker task exited after a queue-level exception")
    finally:
        await worker.stop()
    _require(not worker.running, "Stopped worker still reports running")
    return {
        "status": "passed",
        "claim_calls": queue.claim_calls,
        "continued_after_error": True,
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

    recovery_paths = build_data_paths(root / "job-recovery-data")
    StorageInitializer(recovery_paths).initialize()
    recovery_queue = JobQueue(recovery_paths)
    interrupted_pending = recovery_queue.enqueue(
        "ingest",
        {"path": recovery_paths.tmp / "interrupted.pdf"},
    )
    interrupted = recovery_queue.claim_next()
    _require(
        interrupted is not None and interrupted.id == interrupted_pending.id,
        "Recovery fixture did not create a running job",
    )
    assert interrupted is not None
    untouched_pending = recovery_queue.enqueue("rebuild")
    untouched_completed = recovery_queue.complete(
        recovery_queue.enqueue("export").id,
        {"count": 1},
    )
    untouched_failed = recovery_queue.fail(
        recovery_queue.enqueue("improve").id,
        "already_failed",
    )
    untouched_before = {
        job.id: job.model_dump(mode="json")
        for job in (untouched_pending, untouched_completed, untouched_failed)
    }

    recovered_count = recovery_queue.recover_interrupted_jobs()
    _require(recovered_count == 1, "Recovery did not update exactly one running job")
    recovered = recovery_queue.get(interrupted.id)
    _require(recovered.status == "pending", "Interrupted job was not requeued")
    _require(recovered.error == "worker_interrupted", "Recovery reason changed")
    _require(
        recovered.updated_at > interrupted.updated_at,
        "Recovery did not update the job timestamp",
    )
    for job_id, before in untouched_before.items():
        _require(
            recovery_queue.get(job_id).model_dump(mode="json") == before,
            f"Recovery modified non-running job: {job_id}",
        )

    recovered_before_repeat = recovered.model_dump(mode="json")
    _require(
        recovery_queue.recover_interrupted_jobs() == 0,
        "Repeated recovery was not idempotent",
    )
    _require(
        recovery_queue.get(recovered.id).model_dump(mode="json")
        == recovered_before_repeat,
        "Repeated recovery modified the requeued job",
    )
    recovered_public = recovery_queue.public_dict(recovered)
    _require(
        recovered_public["error"] is None,
        "Recovered pending job exposed a public error",
    )
    _require(
        "worker_interrupted" not in json.dumps(recovered_public),
        "Internal recovery reason leaked through the public job payload",
    )
    _assert_public_json_safe(recovered_public, "recovered job response")
    return {
        "status": "passed",
        "internal_error_retained": True,
        "public_error": public["error"],
        "staging_path_hidden": True,
        "startup_recovery": {
            "recovered_count": recovered_count,
            "internal_reason_retained": True,
            "idempotent": True,
            "non_running_unchanged": True,
        },
    }


async def run_contract() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="paperos-task1b-") as temporary:
        root = Path(temporary)
        return {
            "configuration": configuration_contract(root),
            "storage": storage_baseline_contract(root),
            "retrieval": await empty_retrieval_contract(root),
            "api_errors": api_error_contract(),
            "health": await health_contract(),
            "local_runtime_identity": local_runtime_identity_contract(root),
            "replay_correctness": await replay_correctness_contract(root),
            "jobs": job_status_contract(root),
            "worker": await worker_loop_contract(root),
            "singleton": await application_singleton_contract(root),
            "application_startup": await application_start_recovery_contract(root),
        }


def main() -> None:
    report = asyncio.run(run_contract())
    print(json.dumps({"status": "passed", **report}, ensure_ascii=False, indent=2))


def test_runtime_query_contracts() -> None:
    main()


if __name__ == "__main__":
    main()
