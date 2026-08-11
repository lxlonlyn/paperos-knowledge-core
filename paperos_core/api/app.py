"""FastAPI construction, lifespan, errors, and router registration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from paperos_core.api.documents import router as documents_router
from paperos_core.api.feedback import router as feedback_router
from paperos_core.api.health import router as health_router
from paperos_core.api.ingestion import router as ingestion_router
from paperos_core.api.jobs import router as jobs_router
from paperos_core.api.query import router as query_router
from paperos_core.api.visualize import router as visualize_router
from paperos_core.application import Application, create_application
from paperos_core.config import RuntimeSettings
from paperos_core.errors import PaperOSError


def create_app(settings: RuntimeSettings) -> FastAPI:
    application: Application | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal application
        if application is not None:
            raise RuntimeError("PaperOS Application was already constructed for this server.")
        application = create_application(settings)
        app.state.paperos = application
        try:
            await application.start()
            yield
        finally:
            await application.aclose()

    api = FastAPI(title="PaperOS Knowledge Core", lifespan=lifespan)

    @api.exception_handler(PaperOSError)
    async def paperos_error(_request: Request, error: PaperOSError) -> JSONResponse:
        return JSONResponse(
            status_code=503 if error.retryable else 400,
            content={
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "retryable": error.retryable,
                }
            },
        )

    for router in (
        ingestion_router,
        query_router,
        documents_router,
        jobs_router,
        feedback_router,
        health_router,
        visualize_router,
    ):
        api.include_router(router)
    return api
