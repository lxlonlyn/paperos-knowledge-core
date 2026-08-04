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
        _include_cognee_routers(app)
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
            content=error.as_dict(),
        )

    for router in (
        ingestion_router,
        query_router,
        documents_router,
        jobs_router,
        feedback_router,
        health_router,
    ):
        api.include_router(router)
    return api


def _include_cognee_routers(app: FastAPI) -> None:
    if getattr(app.state, "cognee_routes_attached", False):
        return
    from cognee.api.v1.datasets.routers import (  # type: ignore[import-untyped]
        get_datasets_router,
    )
    from cognee.api.v1.users.routers import (  # type: ignore[import-untyped]
        get_visualize_router,
    )
    from cognee.modules.users.methods import (  # type: ignore[import-untyped]
        get_authenticated_user,
        get_default_user,
    )

    app.dependency_overrides[get_authenticated_user] = get_default_user
    app.include_router(
        get_datasets_router(), prefix="/api/v1/datasets", tags=["cognee-datasets"]
    )
    app.include_router(
        get_visualize_router(), prefix="/api/v1/visualize", tags=["cognee-visualize"]
    )
    app.state.cognee_routes_attached = True
