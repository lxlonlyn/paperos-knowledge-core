"""Managed single-user worker lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from paperos_core.jobs.locking import WorkerLifecycleLock
from paperos_core.jobs.queue import JobQueue, OperationalJob

if TYPE_CHECKING:
    from paperos_core.application import Application


class Worker:
    def __init__(self, application: Application, queue: JobQueue) -> None:
        self.application = application
        self.queue = queue
        self.record_path = application.paths.jobs / "worker-process.json"
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def lifecycle_lock(self) -> WorkerLifecycleLock:
        return WorkerLifecycleLock(self.application.paths)

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self.run(), name="paperos-worker")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            await self._task
            self._task = None
        self._record("stopped")

    async def run(self) -> None:
        while not self._stop_event.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.application.settings.worker.poll_interval_seconds,
                )
            except TimeoutError:
                pass

    async def run_once(self) -> OperationalJob | None:
        self._record("running")
        job = self.queue.claim_next()
        if job is None:
            self._record("idle")
            return None
        try:
            if job.job_type == "improve":
                result = self.application.feedback.improve().model_dump(mode="json")
            elif job.job_type == "rebuild":
                result = (
                    await self.application.rebuilder.rebuild(
                        job.payload.get("snapshot_id")
                    )
                ).model_dump(mode="json")
            elif job.job_type == "reprocess":
                result = await self.application.documents.reprocess(
                    str(job.payload["document_id"])
                )
            else:
                raise ValueError(f"Unsupported operational job type: {job.job_type}")
            completed = self.queue.complete(job.id, result)
            self._record("completed", job_id=job.id)
            return completed
        except Exception as exc:  # noqa: BLE001 - worker persists arbitrary job failures.
            failed = self.queue.fail(job.id, f"{type(exc).__name__}: {exc}")
            self._record("failed", job_id=job.id)
            return failed

    def _record(self, status: str, *, job_id: str | None = None) -> None:
        payload = {
            "pid": os.getpid(),
            "status": status,
            "job_id": job_id,
            "path": str(Path(__file__).resolve()),
        }
        self.record_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
