"""Feedback and queued improvement HTTP routes."""

from fastapi import APIRouter, status

from paperos_core.api.dependencies import ApplicationDep
from paperos_core.feedback.models import FeedbackRequest

router = APIRouter(prefix="/api/v1", tags=["feedback"])


@router.post("/feedback")
async def feedback(
    application: ApplicationDep, body: FeedbackRequest
) -> dict[str, object]:
    return application.services.feedback.record(body).model_dump(mode="json")


@router.post("/improve", status_code=status.HTTP_202_ACCEPTED)
async def improve(application: ApplicationDep) -> dict[str, object]:
    job = application.queue.enqueue("improve")
    return {"job_id": job.id, "status": job.status}
