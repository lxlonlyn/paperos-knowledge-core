"""Fast contracts for upload, API bind, and local CUDA defaults."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from paperos_core.api.app import create_app
from paperos_core.config import RuntimeSettings
from paperos_core.paths import build_data_paths
from paperos_core.runtime.local_inference.runtime import LocalInferenceRuntime


class _QueueTripwire:
    def __init__(self) -> None:
        self.enqueue_calls = 0

    def enqueue(
        self,
        job_type: str,
        payload: dict[str, object] | None = None,
    ) -> None:
        self.enqueue_calls += 1
        raise AssertionError(f"Oversized upload enqueued {job_type}: {payload}")


class _ResolvedCogneeConfig:
    embedding_model = "embedding-contract"
    embedding_dimensions = 768
    embedding_max_tokens = 2048

    @staticmethod
    def embedding_targets(host: str, port: int) -> bool:
        return bool(host and port)


class _RuntimeConfigReader:
    @staticmethod
    def read() -> _ResolvedCogneeConfig:
        return _ResolvedCogneeConfig()


def _local_runtime(settings: RuntimeSettings, root: Path) -> LocalInferenceRuntime:
    return LocalInferenceRuntime(
        settings,
        build_data_paths(root),
        SimpleNamespace(),  # type: ignore[arg-type]
        _RuntimeConfigReader(),  # type: ignore[arg-type]
    )


def test_oversized_upload_is_rejected_before_enqueue(tmp_path: Path) -> None:
    async def scenario() -> None:
        settings = RuntimeSettings.model_validate(
            {
                "data": {"directory": tmp_path / "data"},
                "ingestion": {"max_file_mb": 1},
            }
        )
        paths = build_data_paths(settings.data_dir)
        paths.initialize()
        queue = _QueueTripwire()
        api = create_app(settings)
        api.state.paperos = SimpleNamespace(
            paths=paths,
            settings=settings,
            queue=queue,
        )
        transport = httpx.ASGITransport(app=api)
        oversized = b"%PDF-1.7\n" + b"x" * (1024 * 1024)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://paperos.test",
        ) as client:
            response = await client.post(
                "/api/v1/ingest",
                files={"file": ("oversized.pdf", oversized, "application/pdf")},
            )

        assert response.status_code == 413
        assert response.json()["error"]["code"] == "pdf_too_large"
        assert queue.enqueue_calls == 0
        upload_root = paths.tmp / "uploads"
        assert upload_root.is_dir()
        assert tuple(upload_root.iterdir()) == ()

    asyncio.run(scenario())


def test_safe_api_and_cuda_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding = tmp_path / "embedding.gguf"
    reranker = tmp_path / "reranker.gguf"
    embedding.write_bytes(b"embedding")
    reranker.write_bytes(b"reranker")
    settings = RuntimeSettings.model_validate(
        {
            "local_inference": {
                "embedding_model_path": embedding,
                "reranker_model_path": reranker,
            }
        }
    )
    assert settings.api.host == "127.0.0.1"
    assert settings.local_inference.cuda_devices == []

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    default_runtime = _local_runtime(settings, tmp_path / "default-runtime")
    assert "CUDA_VISIBLE_DEVICES" not in default_runtime._process_environment()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5")
    assert default_runtime._process_environment()["CUDA_VISIBLE_DEVICES"] == "4,5"
    assert (
        default_runtime._expected_runtime_identity()["cuda_visible_devices"]
        == "4,5"
    )

    explicit_settings = settings.model_copy(
        update={
            "local_inference": settings.local_inference.model_copy(
                update={"cuda_devices": [6, 7]}
            )
        }
    )
    explicit_runtime = _local_runtime(explicit_settings, tmp_path / "explicit-runtime")
    assert explicit_runtime._process_environment()["CUDA_VISIBLE_DEVICES"] == "6,7"
    assert (
        explicit_runtime._expected_runtime_identity()["cuda_visible_devices"]
        == "6,7"
    )
