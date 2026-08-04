"""Provider-neutral asynchronous MinerU polling client."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from paperos_core.adapters.mineru.providers import MinerUProvider
from paperos_core.adapters.mineru.schemas import MinerUParseResult
from paperos_core.config import MinerUSettings
from paperos_core.domain.documents import SourceFile
from paperos_core.errors import MinerUParseError, MinerUTimeoutError


class MinerUClient:
    def __init__(self, provider: MinerUProvider, config: MinerUSettings) -> None:
        self.provider = provider
        self.config = config

    async def parse_pdf(
        self, source: SourceFile, *, request_options: dict[str, Any] | None = None
    ) -> MinerUParseResult:
        task = await self.provider.submit_pdf(source, request_options=request_options or {})
        started = time.monotonic()
        history: list[dict[str, Any]] = [
            {
                "state": task.state,
                "task_id": task.task_id,
                "provider_metadata": task.raw_metadata,
            }
        ]
        while True:
            if time.monotonic() - started > self.config.timeout_seconds:
                raise MinerUTimeoutError(
                    f"MinerU task exceeded {self.config.timeout_seconds} seconds.",
                    affected=task.task_id,
                )
            await asyncio.sleep(self.config.poll_interval_seconds)
            task = await self.provider.get_task_status(task)
            history.append(
                {
                    "state": task.state,
                    "progress": task.progress,
                    "error_code": task.error_code,
                    "error_message": task.error_message,
                    "provider_metadata": task.raw_metadata,
                }
            )
            if task.state == "done":
                return await self.provider.fetch_result(task, poll_history=history)
            if task.state == "failed":
                raise MinerUParseError(
                    f"MinerU task failed: {task.error_message or 'unknown failure'}",
                    affected=task.task_id,
                    details={"provider_error_code": task.error_code},
                )

    async def aclose(self) -> None:
        await self.provider.aclose()
