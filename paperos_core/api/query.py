"""Research query HTTP route."""

from fastapi import APIRouter

from paperos_core.api.dependencies import ApplicationDep
from paperos_core.retrieval.candidates import QueryRequest, QueryResponse

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query(
    application: ApplicationDep, body: QueryRequest
) -> QueryResponse:
    return await application.services.retrieval.query(body)
