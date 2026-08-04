"""The single application-owned background job consumer."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from paperos_core.documents import DocumentService
from paperos_core.feedback.service import FeedbackService
from paperos_core.indexes.rebuild import DerivedDataRebuilder
from paperos_core.ingestion.service import IngestionService
from paperos_core.jobs.queue import JobQueue, OperationalJob


class BackgroundWorker:
    def __init__(
        self,
        queue: JobQueue,
        ingestion: IngestionService,
        rebuilder: DerivedDataRebuilder,
        documents: DocumentService,
        feedback: FeedbackService,
        *,
        poll_interval_seconds: float,
    ) -> None:
        self.queue = queue
        self.ingestion = ingestion
        self.rebuilder = rebuilder
        self.documents = documents
        self.feedback = feedback
        self.poll_interval_seconds = poll_interval_seconds
        self.record_path = queue.paths.jobs / "worker-process.json"
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
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
        self._record("running")
        while not self._stop_event.is_set():
            await self.run_once()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.poll_interval_seconds
                )
            except TimeoutError:
                pass

    async def run_once(self) -> OperationalJob | None:
        job = self.queue.claim_next()
        if job is None:
            self._record("idle")
            return None
        try:
            if job.job_type == "ingest":
                result = await self.ingestion.ingest_pdf_to_knowledge(
                    Path(str(job.payload["path"])),
                    dataset=job.payload.get("dataset"),
                    metadata=job.payload.get("metadata"),
                )
                payload = result.public_dict()
            elif job.job_type == "improve":
                payload = self.feedback.improve().model_dump(mode="json")
            elif job.job_type == "rebuild":
                payload = (
                    await self.rebuilder.rebuild(job.payload.get("snapshot_id"))
                ).model_dump(mode="json")
            elif job.job_type == "reprocess":
                payload = await self.documents.reprocess(
                    str(job.payload["document_id"])
                )
            else:
                raise ValueError(f"Unsupported operational job type: {job.job_type}")
            completed = self.queue.complete(job.id, payload)
            self._record("completed", job_id=job.id)
            return completed
        except Exception as exc:  # noqa: BLE001 - consumer must persist job failures.
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
