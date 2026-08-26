"""Runtime configuration, empty retrieval, and safe API error contracts."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from paperos_core.config import RuntimeSettings, load_settings
from paperos_core.errors import ConfigurationError
from paperos_core.health import HealthService
from paperos_core.retrieval.candidates import QueryRequest
from paperos_core.retrieval.service import NO_EVIDENCE_ANSWER, RetrievalService
from paperos_core.retrieval.synthesis import FinalSynthesisContext, render_synthesis_prompt


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


def test_local_embedding_endpoint_requires_enabled_runtime(tmp_path: Path) -> None:
    config_path = tmp_path / "paperos.toml"
    _write_config(
        config_path,
        embedding_endpoint="http://localhost:8081/v1",
        rerank_enabled=False,
    )

    with pytest.raises(ConfigurationError, match="local_inference.enabled must be true"):
        load_settings(config_path)


def test_local_reranker_requires_enabled_runtime(tmp_path: Path) -> None:
    config_path = tmp_path / "paperos.toml"
    _write_config(
        config_path,
        embedding_endpoint="https://embedding.example/v1",
        rerank_enabled=True,
    )

    with pytest.raises(ConfigurationError, match="no remote reranker is configured"):
        load_settings(config_path)


def test_example_configuration_is_internally_runnable() -> None:
    settings = load_settings(Path("config/paperos.example.toml"))

    assert settings.local_inference.enabled is True
    assert settings.retrieval.rerank_enabled is True


def _retrieval_service(*, pool_size: int = 40) -> RetrievalService:
    service = object.__new__(RetrievalService)
    service.config = SimpleNamespace(
        dataset="papers",
        retrieval=SimpleNamespace(
            top_k=12,
            candidate_pool_size=pool_size,
            rerank_enabled=True,
            synthesis_max_input_tokens=1,
        ),
    )
    service.paths = SimpleNamespace()
    service.canonical_repository = object()
    service.registry = object()
    service.scholarly_registry = object()
    service.search = object()
    service.compat = object()
    service.index_manager = SimpleNamespace(lexical=object())
    service.model_client = object()
    service.llm = SimpleNamespace(model="must-not-run")
    return service


def _corpus(document_ids: set[str]) -> SimpleNamespace:
    return SimpleNamespace(
        chunks={},
        filtered_document_ids=lambda _ids, _dataset: set(document_ids),
        document_ids_for_works=lambda _ids: set(document_ids),
    )


def test_empty_library_skips_vector_reranker_and_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    import paperos_core.retrieval.service as service_module

    service = _retrieval_service()
    calls = {"lexical": 0, "vector": 0, "reranker": 0, "llm": 0}

    def lexical(*_args: object, **_kwargs: object) -> list[object]:
        calls["lexical"] += 1
        return []

    async def vector(*_args: object, **_kwargs: object) -> list[object]:
        calls["vector"] += 1
        return []

    async def reranker(*_args: object, **_kwargs: object) -> list[object]:
        calls["reranker"] += 1
        return []

    async def llm(*_args: object, **_kwargs: object) -> str:
        calls["llm"] += 1
        return "unexpected"

    service._rerank = reranker  # type: ignore[method-assign]
    monkeypatch.setattr(service_module.CorpusView, "load", lambda *_args: _corpus(set()))
    monkeypatch.setattr(service_module, "lexical_retrieve", lexical)
    monkeypatch.setattr(service_module, "semantic_retrieve", vector)
    monkeypatch.setattr(service_module, "synthesize_answer", llm)

    response = asyncio.run(service.query(QueryRequest(query="empty library")))

    assert calls == {"lexical": 1, "vector": 0, "reranker": 0, "llm": 0}
    assert response.answer == NO_EVIDENCE_ANSWER
    assert response.evidence == []
    assert response.candidates == []
    assert response.provenance_complete is False
    assert "no_evidence" in response.stages
    assert response.replay.replay_text == render_synthesis_prompt(
        FinalSynthesisContext(original_query="empty library", evidence=[])
    )


def test_top_k_expands_actual_candidate_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    import paperos_core.retrieval.service as service_module

    service = _retrieval_service(pool_size=40)
    observed: dict[str, int] = {}

    def lexical(*_args: object, **kwargs: object) -> list[object]:
        observed["lexical_limit"] = int(kwargs["limit"])
        return []

    async def vector(*_args: object, **kwargs: object) -> list[object]:
        observed["vector_limit"] = int(kwargs["limit"])
        return []

    monkeypatch.setattr(service_module.CorpusView, "load", lambda *_args: _corpus({"doc"}))
    monkeypatch.setattr(service_module, "lexical_retrieve", lexical)
    monkeypatch.setattr(service_module, "semantic_retrieve", vector)

    response = asyncio.run(service.query(QueryRequest(query="wide", top_k=100)))

    assert observed == {"lexical_limit": 100, "vector_limit": 100}
    assert response.answer == NO_EVIDENCE_ANSWER


def test_api_error_details_keep_reason_and_remove_local_paths() -> None:
    error = ConfigurationError(
        "Budget is too small.",
        affected=Path("/home/user/private/config.toml"),
        details={
            "reason": "synthesis_context_too_small",
            "configured_tokens": 50,
            "required_estimated_tokens": 100,
            "local_path": "/home/user/private/paper.pdf",
            "nested": {"artifact": Path("/tmp/private.json"), "safe": True},
        },
    )

    assert error.as_api_dict() == {
        "error": {
            "code": "configuration_error",
            "message": "Budget is too small.",
            "retryable": False,
            "details": {
                "reason": "synthesis_context_too_small",
                "configured_tokens": 50,
                "required_estimated_tokens": 100,
                "nested": {"safe": True},
            },
        }
    }


def test_health_reports_embedding_and_reranker_enablement_separately() -> None:
    settings = RuntimeSettings.model_validate(
        {
            "cognee": {"embedding": {"endpoint": "https://embedding.example/v1"}},
            "local_inference": {"enabled": True},
            "retrieval": {"rerank_enabled": True},
        }
    )
    cognee_config = SimpleNamespace(
        read=lambda: SimpleNamespace(
            embedding_targets=lambda _host, _port: False,
            embedding_dimensions=768,
        )
    )
    local = SimpleNamespace(
        settings=settings,
        cognee_config=cognee_config,
        client=SimpleNamespace(
            health=lambda: _async_value(
                {"status": "healthy", "reranker": {"loaded": True}}
            )
        ),
    )
    service = HealthService(
        SimpleNamespace(),
        SimpleNamespace(
            status=lambda: {"ingestion_job_count": 0},
        ),
        SimpleNamespace(list_bundles=list),
        SimpleNamespace(
            provider=SimpleNamespace(health_check=lambda: _async_value({}))
        ),
        SimpleNamespace(
            health_check=lambda: _async_value({"provider": "test", "model": "test"}),
            runtime_config=cognee_config,
        ),
        local,
        SimpleNamespace(),
        SimpleNamespace(lexical=SimpleNamespace(status=dict)),
        SimpleNamespace(list_jobs=list),
    )

    report = asyncio.run(service.report())

    assert report["components"]["local_models"]["embedding_enabled"] is False
    assert report["components"]["local_models"]["reranker_enabled"] is True


async def _async_value(value: Any) -> Any:
    return value
