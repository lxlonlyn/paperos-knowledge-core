"""Document inspection and maintenance HTTP routes."""

from fastapi import APIRouter, status

from paperos_core.api.dependencies import ApplicationDep

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])


@router.get("")
async def list_documents(application: ApplicationDep) -> list[dict[str, object]]:
    return [
        item.model_dump(mode="json")
        for item in application.services.documents.list_documents()
    ]


@router.get("/{document_id}")
async def inspect_document(
    document_id: str, application: ApplicationDep
) -> dict[str, object]:
    return application.services.documents.inspect(document_id).model_dump(mode="json")


@router.delete("/{document_id}")
async def delete_document(
    document_id: str, application: ApplicationDep
) -> dict[str, object]:
    report = await application.services.documents.delete(document_id)
    return report.model_dump(mode="json")


@router.post("/{document_id}/reprocess", status_code=status.HTTP_202_ACCEPTED)
async def reprocess_document(
    document_id: str, application: ApplicationDep
) -> dict[str, object]:
    application.services.documents.inspect(document_id)
    job = application.queue.enqueue("reprocess", {"document_id": document_id})
    return {"id": job.id, "status": job.status}
