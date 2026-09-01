"""Fast contract for early upload-size rejection."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx

from paperos_core.api.app import create_app
from paperos_core.config import RuntimeSettings
from paperos_core.paths import build_data_paths


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
