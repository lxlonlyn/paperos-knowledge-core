"""Complete FastAPI application backed by shared application services."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse

from paperos_core.bootstrap import Application, build_application
from paperos_core.errors import PaperOSError
from paperos_core.feedback.models import FeedbackRequest
from paperos_core.retrieval.candidates import QueryRequest, QueryResponse


def create_app(
    *,
    config_path: Path | None = None,
    data_dir: Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        application = build_application(config_path=config_path, data_dir=data_dir)
        app.state.paperos = application
        try:
            yield
        finally:
            await application.aclose()

    api = FastAPI(title="PaperOS Knowledge Core", lifespan=lifespan)

    @api.exception_handler(PaperOSError)
    async def paperos_error(
        _request: Request, error: PaperOSError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=503 if error.retryable else 400,
            content=error.as_dict(),
        )

    @api.post("/api/v1/query", response_model=QueryResponse)
    async def query(request: Request, body: QueryRequest) -> QueryResponse:
        application: Application = request.app.state.paperos
        return await application.retrieval.query(body)

    @api.post("/api/v1/ingest")
    async def ingest(
        request: Request, file: Annotated[UploadFile, File()]
    ) -> dict[str, object]:
        application: Application = request.app.state.paperos
        filename = Path(file.filename or "upload.pdf").name
        temporary_root = application.paths.tmp / f"api-{uuid.uuid4().hex}"
        temporary_root.mkdir(parents=True)
        temporary = temporary_root / filename
        try:
            with temporary.open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    stream.write(chunk)
            result = await application.ingestion.ingest_pdf_to_knowledge(temporary)
            return result.public_dict()
        finally:
            temporary.unlink(missing_ok=True)
            if temporary_root.exists():
                temporary_root.rmdir()

    @api.get("/api/v1/ingest/{job_id}")
    async def ingestion_status(request: Request, job_id: str) -> dict[str, object]:
        application: Application = request.app.state.paperos
        return application.ingestion.get_job(job_id).model_dump(mode="json")

    @api.get("/api/v1/documents")
    async def list_documents(request: Request) -> list[dict[str, object]]:
        application: Application = request.app.state.paperos
        return [
            item.model_dump(mode="json")
            for item in application.documents.list_documents()
        ]

    @api.get("/api/v1/documents/{document_id}")
    async def inspect_document(
        request: Request, document_id: str
    ) -> dict[str, object]:
        application: Application = request.app.state.paperos
        return application.documents.inspect(document_id).model_dump(mode="json")

    @api.delete("/api/v1/documents/{document_id}")
    async def delete_document(
        request: Request, document_id: str
    ) -> dict[str, object]:
        application: Application = request.app.state.paperos
        return (
            await application.documents.delete(document_id)
        ).model_dump(mode="json")

    @api.post("/api/v1/documents/{document_id}/reprocess")
    async def reprocess_document(
        request: Request, document_id: str
    ) -> dict[str, object]:
        application: Application = request.app.state.paperos
        return await application.documents.reprocess(document_id)

    @api.post("/api/v1/feedback")
    async def feedback(
        request: Request, body: FeedbackRequest
    ) -> dict[str, object]:
        application: Application = request.app.state.paperos
        return application.feedback.record(body).model_dump(mode="json")

    @api.post("/api/v1/improve")
    async def improve(request: Request) -> dict[str, object]:
        application: Application = request.app.state.paperos
        return application.feedback.improve().model_dump(mode="json")

    @api.get("/api/v1/health")
    @api.get("/health", include_in_schema=False)
    async def health(request: Request) -> dict[str, object]:
        application: Application = request.app.state.paperos
        return await application.health.report()

    return api
