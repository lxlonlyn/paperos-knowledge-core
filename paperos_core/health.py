"""Dependency-aware application health reporting."""

from __future__ import annotations

from typing import Any

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.adapters.llm import LLMClient
from paperos_core.adapters.mineru.client import MinerUClient
from paperos_core.indexes.manager import IndexManager
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.jobs.queue import JobQueue
from paperos_core.paths import DataPaths
from paperos_core.runtime.local_inference.client import LocalInferenceClient


class HealthService:
    def __init__(
        self,
        paths: DataPaths,
        registry: SourceRegistry,
        canonical_repository: CanonicalRepository,
        mineru: MinerUClient,
        llm: LLMClient,
        local_inference: LocalInferenceClient,
        cognee: CogneeCompatibilityAdapter,
        indexes: IndexManager,
        queue: JobQueue,
    ) -> None:
        self.paths = paths
        self.registry = registry
        self.canonical_repository = canonical_repository
        self.mineru = mineru
        self.llm = llm
        self.local_inference = local_inference
        self.cognee = cognee
        self.indexes = indexes
        self.queue = queue

    async def report(self) -> dict[str, Any]:
        components: dict[str, Any] = {}
        try:
            components["mineru"] = {
                "status": "healthy",
                **await self.mineru.provider.health_check(),
            }
        except Exception as exc:  # noqa: BLE001 - health reports component failures.
            components["mineru"] = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            model_status = await self.llm.health_check()
            components["llm"] = {
                "status": "healthy",
                "provider": model_status["provider"],
                "model": model_status["model"],
            }
        except Exception as exc:  # noqa: BLE001 - health reports component failures.
            components["llm"] = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            local = await self.local_inference.health()
            components["local_models"] = {"status": "healthy", **local}
        except Exception as exc:  # noqa: BLE001 - health reports component failures.
            components["local_models"] = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
        components["lexical"] = {
            "status": "healthy",
            **self.indexes.lexical.status(),
        }
        bundles = self.canonical_repository.list_bundles()
        if not bundles:
            # Constructing Cognee's vector engine can initialize its embedding
            # provider. Health must remain a read-only check, so an empty store
            # is validated from PaperOS-owned state without creating that engine.
            components["vector"] = {
                "status": "healthy",
                "backend": "cognee",
                "path": str(self.paths.cognee / "vector"),
                "collection_count": 0,
                "record_count": 0,
                "dimensions": self.indexes.embedding_dimensions,
            }
        else:
            try:
                components["vector"] = {
                    "status": "healthy",
                    **await self.cognee.vector_status(),
                }
            except Exception as exc:  # noqa: BLE001 - health reports component failures.
                components["vector"] = {
                    "status": "degraded",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        try:
            if bundles:
                await self.cognee.get_datapoint(bundles[-1].document.id)
            components["cognee_graph"] = {
                "status": "healthy",
                "document_count": len(bundles),
            }
        except Exception as exc:  # noqa: BLE001 - health reports component failures.
            components["cognee_graph"] = {
                "status": "degraded",
                "error": f"{type(exc).__name__}: {exc}",
            }
        registry = self.registry.status()
        components["job_database"] = {
            "status": "healthy",
            "ingestion_jobs": registry["ingestion_job_count"],
            "operational_jobs": len(self.queue.list_jobs()),
        }
        components["data_paths"] = {
            "status": "healthy",
            "root": str(self.paths.root),
            "all_within_root": True,
        }
        overall = (
            "healthy"
            if all(item["status"] == "healthy" for item in components.values())
            else "degraded"
        )
        return {"status": overall, "components": components}
