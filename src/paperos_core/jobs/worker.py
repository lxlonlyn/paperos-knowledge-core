"""Managed single-user worker lifecycle."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

from paperos_core.jobs.locking import WorkerLifecycleLock
from paperos_core.jobs.queue import JobQueue, OperationalJob

if TYPE_CHECKING:
    from paperos_core.bootstrap import Application


class Worker:
    def __init__(self, application: Application, queue: JobQueue) -> None:
        self.application = application
        self.queue = queue
        self.record_path = application.paths.jobs / "worker-process.json"

    def lifecycle_lock(self) -> WorkerLifecycleLock:
        return WorkerLifecycleLock(self.application.paths)

    def stop(self) -> None:
        self._record("stopped")

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
