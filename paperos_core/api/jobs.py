"""Operational job status and maintenance queue routes."""

from typing import Annotated

from fastapi import APIRouter, Query, status

from paperos_core.api.dependencies import ApplicationDep

router = APIRouter(prefix="/api/v1", tags=["jobs"])


@router.get("/jobs")
async def list_jobs(
    application: ApplicationDep,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[dict[str, object]]:
    return [
        application.queue.public_dict(job)
        for job in application.queue.list_jobs(limit=limit)
    ]


@router.get("/jobs/{job_id}")
async def job_status(job_id: str, application: ApplicationDep) -> dict[str, object]:
    job = application.queue.get(job_id)
    return application.queue.public_dict(job)


@router.post("/rebuild", status_code=status.HTTP_202_ACCEPTED)
async def rebuild(application: ApplicationDep) -> dict[str, object]:
    job = application.queue.enqueue("rebuild")
    return {"id": job.id, "status": job.status}
