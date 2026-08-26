"""Dependency-aware application health reporting."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from paperos_core.errors import public_diagnostic
from paperos_core.runtime.local_inference.runtime import (
    LocalRuntimeUsage,
    local_runtime_usage,
)

if TYPE_CHECKING:
    from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
    from paperos_core.adapters.cognee.llm import LLMClient
    from paperos_core.adapters.mineru.client import MinerUClient
    from paperos_core.indexes.manager import IndexManager
    from paperos_core.ingestion.canonical_repository import CanonicalRepository
    from paperos_core.ingestion.registry import SourceRegistry
    from paperos_core.jobs.queue import JobQueue
    from paperos_core.paths import DataPaths
    from paperos_core.runtime.local_inference.runtime import LocalInferenceRuntime


logger = logging.getLogger(__name__)


def _component_failure(code: str) -> dict[str, str | dict[str, str]]:
    return {"status": "unavailable", "error": public_diagnostic(code)}


def _log_component_failure(component: str, exc: Exception) -> None:
    """Record failure class internally without logging exception-controlled text."""

    logger.warning(
        "Health component %s is unavailable (%s)",
        component,
        type(exc).__name__,
    )


def local_model_enablement(usage: LocalRuntimeUsage) -> dict[str, bool]:
    """Render the two independent local-model enablement flags for health."""

    return {
        "embedding_enabled": usage.embedding,
        "reranker_enabled": usage.reranker,
    }


class HealthService:
    def __init__(
        self,
        paths: DataPaths,
        registry: SourceRegistry,
        canonical_repository: CanonicalRepository,
        mineru: MinerUClient,
        llm: LLMClient,
        local_inference: LocalInferenceRuntime,
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
            mineru = await self.mineru.provider.health_check()
            components["mineru"] = {
                "status": "healthy",
                "provider": mineru.get("provider"),
                "configured": mineru.get("configured"),
                "reachable": mineru.get("reachable"),
            }
        except Exception as exc:  # noqa: BLE001 - health reports component failures.
            _log_component_failure("mineru", exc)
            components["mineru"] = _component_failure("mineru_unavailable")
        try:
            model_status = await self.llm.health_check()
            components["llm"] = {
                "status": "healthy",
                "provider": model_status["provider"],
                "model": model_status["model"],
            }
        except Exception as exc:  # noqa: BLE001 - health reports component failures.
            _log_component_failure("llm", exc)
            components["llm"] = _component_failure("llm_unavailable")
        local_usage = local_runtime_usage(
            self.local_inference.settings,
            self.local_inference.cognee_config,
        )
        model_enablement = local_model_enablement(local_usage)
        if local_usage.required:
            try:
                local = await self.local_inference.client.health()
                embedding = local.get("embedding", {})
                reranker = local.get("reranker", {})
                components["local_models"] = {
                    "status": "healthy",
                    **model_enablement,
                    "embedding": {
                        "model": embedding.get("model"),
                        "dimensions": embedding.get("dimensions"),
                        "loaded": embedding.get("loaded"),
                    },
                    "reranker": {
                        "model": reranker.get("model"),
                        "loaded": reranker.get("loaded"),
                    },
                }
            except Exception as exc:  # noqa: BLE001 - health reports component failures.
                _log_component_failure("local_models", exc)
                components["local_models"] = {
                    **_component_failure("local_models_unavailable"),
                    **model_enablement,
                }
        else:
            components["local_models"] = {
                "status": "disabled",
                **model_enablement,
                "reason": "local embedding and reranker are disabled",
            }
        components["lexical"] = {
            "status": "healthy",
            **self.indexes.lexical.status(),
        }
        bundles = self.canonical_repository.list_active_bundles()
        dataset_name: str | None = None
        if bundles:
            manifest = self.cognee.read_manifest(bundles[-1].snapshot.id)
            dataset = manifest.get("dataset")
            if isinstance(dataset, dict) and dataset.get("name"):
                dataset_name = str(dataset["name"])
        if not bundles:
            # Constructing Cognee's vector engine can initialize its embedding
            # provider. Health must remain a read-only check, so an empty store
            # is validated from PaperOS-owned state without creating that engine.
            cognee_config = self.llm.runtime_config.read()
            components["vector"] = {
                "status": "healthy",
                "backend": "cognee",
                "collection_count": 0,
                "record_count": 0,
                "dimensions": cognee_config.embedding_dimensions,
            }
        else:
            try:
                vector = await self.cognee.vector_status(dataset_name=dataset_name)
                components["vector"] = {
                    "status": "healthy",
                    "backend": vector.get("backend"),
                    "collection_count": vector.get("collection_count"),
                    "record_count": vector.get("record_count"),
                    "dimensions": vector.get("dimensions"),
                }
            except Exception as exc:  # noqa: BLE001 - health reports component failures.
                _log_component_failure("vector", exc)
                components["vector"] = {
                    **_component_failure("vector_unavailable"),
                    "status": "degraded",
                }
        try:
            if bundles:
                await self.cognee.get_datapoint(
                    bundles[-1].document.id,
                    dataset_name=dataset_name,
                    snapshot_id=bundles[-1].snapshot.id,
                )
            components["cognee_graph"] = {
                "status": "healthy",
                "document_count": len(bundles),
            }
        except Exception as exc:  # noqa: BLE001 - health reports component failures.
            _log_component_failure("cognee_graph", exc)
            components["cognee_graph"] = {
                **_component_failure("cognee_graph_unavailable"),
                "status": "degraded",
            }
        registry = self.registry.status()
        components["job_database"] = {
            "status": "healthy",
            "ingestion_jobs": registry["ingestion_job_count"],
            "operational_jobs": len(self.queue.list_jobs()),
        }
        components["data_paths"] = {
            "status": "healthy",
            "all_within_root": True,
        }
        overall = (
            "healthy"
            if all(
                item["status"] in {"healthy", "disabled"}
                for item in components.values()
            )
            else "degraded"
        )
        return {"status": overall, "components": components}
