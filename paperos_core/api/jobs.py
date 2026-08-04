"""Operational job status and maintenance queue routes."""

from fastapi import APIRouter, status

from paperos_core.api.dependencies import ApplicationDep

router = APIRouter(prefix="/api/v1", tags=["jobs"])


@router.get("/jobs/{job_id}")
async def job_status(job_id: str, application: ApplicationDep) -> dict[str, object]:
    return application.queue.get(job_id).model_dump(mode="json")


@router.post("/rebuild", status_code=status.HTTP_202_ACCEPTED)
async def rebuild(application: ApplicationDep) -> dict[str, object]:
    job = application.queue.enqueue("rebuild")
    return {"job_id": job.id, "status": job.status}
