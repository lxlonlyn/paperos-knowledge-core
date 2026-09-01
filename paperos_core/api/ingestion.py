"""Asynchronous PDF ingestion HTTP route."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, UploadFile, status

from paperos_core.api.dependencies import ApplicationDep
from paperos_core.errors import FileTooLargeError

router = APIRouter(prefix="/api/v1", tags=["ingestion"])
_UPLOAD_CHUNK_SIZE = 1024 * 1024


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest(
    application: ApplicationDep,
    file: Annotated[UploadFile, File()],
    dataset: str | None = None,
) -> dict[str, object]:
    filename = Path(file.filename or "upload.pdf").name
    staging_root = application.paths.tmp / "uploads" / uuid.uuid4().hex
    staging_root.mkdir(parents=True)
    staged = staging_root / filename
    max_bytes = application.settings.ingestion.max_file_mb * 1024 * 1024
    received_bytes = 0
    try:
        with staged.open("wb") as stream:
            while chunk := await file.read(_UPLOAD_CHUNK_SIZE):
                received_bytes += len(chunk)
                if received_bytes > max_bytes:
                    raise FileTooLargeError(
                        "Uploaded PDF exceeds the configured ingestion size limit.",
                        affected=filename,
                        details={
                            "size_bytes": received_bytes,
                            "max_bytes": max_bytes,
                        },
                    )
                stream.write(chunk)
        job = application.queue.enqueue(
            "ingest",
            {"path": str(staged), "dataset": dataset or application.settings.dataset},
        )
    except BaseException:
        staged.unlink(missing_ok=True)
        if staging_root.exists():
            staging_root.rmdir()
        raise
    finally:
        await file.close()
    return {"job_id": job.id, "status": job.status}
