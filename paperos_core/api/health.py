"""Dependency-aware health HTTP routes."""

from fastapi import APIRouter

from paperos_core.api.dependencies import ApplicationDep

router = APIRouter(tags=["health"])


@router.get("/api/v1/health")
@router.get("/health", include_in_schema=False)
async def health(application: ApplicationDep) -> dict[str, object]:
    return await application.services.health.report()
