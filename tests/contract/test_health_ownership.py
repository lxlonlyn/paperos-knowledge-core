from __future__ import annotations

import os

import pytest

from paperos_core.application import create_application
from paperos_core.config import load_settings


@pytest.mark.asyncio
async def test_health_degrades_without_starting_or_restarting_resources(
    gate1_run_dir,
) -> None:
    settings = load_settings(
        environ={
            **os.environ,
            "PAPEROS_DATA_DIR": str(gate1_run_dir / "health-ownership"),
            "MINERU_API_KEY": "invalid-health-test-key",
            "DEEPSEEK_API_KEY": "invalid-health-test-key",
        }
    )
    settings = settings.model_copy(
        update={
            "mineru": settings.mineru.model_copy(
                update={"endpoint": "http://127.0.0.1:1"}
            ),
            "deepseek": settings.deepseek.model_copy(
                update={"endpoint": "http://127.0.0.1:1", "model": "unavailable"}
            ),
        }
    )
    application = create_application(settings)
    application.storage.initialize()
    try:
        assert application.runtime.local_inference.running is False
        assert application.runtime.worker.running is False
        report = await application.services.health.report()
        assert report["status"] == "degraded"
        assert report["components"]["mineru"]["status"] == "unavailable"
        assert report["components"]["deepseek"]["status"] == "unavailable"
        assert application.runtime.local_inference.running is False
        assert application.runtime.worker.running is False
    finally:
        await application.aclose()
