"""PaperOS dependency assembly and the single owned application lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from paperos_core.adapters.mineru.client import MinerUClient
from paperos_core.adapters.mineru.mapper import MinerUCanonicalMapper
from paperos_core.adapters.mineru.providers import MinerUCloudProvider
from paperos_core.config import RuntimeSettings, load_settings
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.parser_artifacts import ParserArtifactRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.ingestion.service import IngestionService
from paperos_core.paths import DataPaths, build_data_paths

if TYPE_CHECKING:
    from paperos_core.adapters.cognee.pipeline import CogneePipeline
    from paperos_core.adapters.llm import DeepSeekClient
    from paperos_core.runtime.local_inference.client import LocalInferenceClient
    from paperos_core.runtime.local_inference.runtime import LocalInferenceRuntime
    from paperos_core.documents import DocumentService
    from paperos_core.feedback.service import FeedbackService
    from paperos_core.health import HealthService
    from paperos_core.indexes.rebuild import DerivedDataRebuilder
    from paperos_core.jobs.queue import JobQueue
    from paperos_core.jobs.worker import BackgroundWorker
    from paperos_core.retrieval.service import RetrievalService


@dataclass(slots=True)
class ApplicationServices:
    ingestion: IngestionService
    retrieval: RetrievalService
    documents: DocumentService
    feedback: FeedbackService
    health: HealthService
    rebuilder: DerivedDataRebuilder


@dataclass(slots=True)
class ManagedRuntime:
    local_inference: LocalInferenceRuntime
    worker: BackgroundWorker


@dataclass(slots=True)
class Application:
    settings: RuntimeSettings
    services: ApplicationServices
    runtime: ManagedRuntime
    paths: DataPaths
    registry: SourceRegistry
    parser_artifacts: ParserArtifactRepository
    canonical_repository: CanonicalRepository
    canonical_mapper: MinerUCanonicalMapper
    mineru: MinerUClient
    local_inference_client: LocalInferenceClient
    deepseek: DeepSeekClient
    knowledge_pipeline: CogneePipeline
    queue: JobQueue
    _started: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)

    async def start(self) -> None:
        """Initialize owned resources in dependency order exactly once."""

        if self._closed:
            raise RuntimeError("A closed PaperOS Application cannot be restarted.")
        if self._started:
            return
        self.paths.initialize()
        try:
            await self.runtime.local_inference.start()
            await self.runtime.worker.start()
        except BaseException:
            await self.aclose()
            raise
        self._started = True

    async def aclose(self) -> None:
        """Close owned resources in reverse startup order."""

        if self._closed:
            return
        self._closed = True
        await self.runtime.worker.stop()
        await self.runtime.local_inference.stop()
        await self.local_inference_client.aclose()
        await self.deepseek.aclose()
        await self.mineru.aclose()
        self._started = False


def create_application(settings: RuntimeSettings) -> Application:
    """Assemble the object graph without starting a process or background task."""

    paths = build_data_paths(settings.data_dir)
    registry = SourceRegistry(paths)
    parser_artifacts = ParserArtifactRepository(paths)
    canonical_repository = CanonicalRepository(paths)
    canonical_mapper = MinerUCanonicalMapper(settings.ingestion)
    if settings.config_path is None:
        from paperos_core.errors import CogneeConfigurationError

        raise CogneeConfigurationError(
            "Project configuration is required to configure Cognee."
        )
    from paperos_core.adapters.cognee.config import (
        configure_cognee_environment,
        reassert_cognee_runtime,
    )

    cognee_config = configure_cognee_environment(
        paths, env_path=settings.config_path.parent.parent / ".env"
    )
    from paperos_core.adapters.cognee.pipeline import CogneePipeline
    from paperos_core.adapters.cognee.repository import CogneeRepository
    from paperos_core.adapters.llm import DeepSeekClient
    from paperos_core.documents import DocumentService
    from paperos_core.feedback.service import FeedbackService
    from paperos_core.health import HealthService
    from paperos_core.indexes.manager import IndexManager
    from paperos_core.indexes.rebuild import DerivedDataRebuilder
    from paperos_core.jobs.queue import JobQueue
    from paperos_core.jobs.worker import BackgroundWorker
    from paperos_core.runtime.local_inference.client import LocalInferenceClient
    from paperos_core.runtime.local_inference.runtime import LocalInferenceRuntime
    from paperos_core.retrieval.service import RetrievalService

    reassert_cognee_runtime(paths)
    local = settings.local_inference
    local_inference_client = LocalInferenceClient(
        f"http://{local.host}:{local.port}",
        local.request_timeout_seconds,
    )
    local_inference_runtime = LocalInferenceRuntime(
        settings, paths, local_inference_client
    )
    deepseek = DeepSeekClient(
        cognee_config,
        timeout_seconds=local.request_timeout_seconds,
    )
    cognee_repository = CogneeRepository(paths)
    index_manager = IndexManager(
        paths,
        embedding_model=cognee_config.embedding_model,
        embedding_dimensions=cognee_config.embedding_dimensions,
    )
    knowledge_pipeline = CogneePipeline(
        paths,
        canonical_repository,
        registry,
        cognee_repository,
        index_manager,
        deepseek,
    )
    rebuilder = DerivedDataRebuilder(paths, canonical_repository, knowledge_pipeline)
    feedback = FeedbackService(paths, canonical_repository)
    queue = JobQueue(paths)
    retrieval = RetrievalService(
        settings,
        paths,
        canonical_repository,
        registry,
        cognee_repository,
        index_manager,
        local_inference_client,
        deepseek,
        feedback,
    )
    if settings.mineru_ocr.provider != "mineru_cloud":
        from paperos_core.errors import MinerUConfigurationError

        raise MinerUConfigurationError(
            f"Unsupported configured MinerU provider: {settings.mineru_ocr.provider}",
            affected="mineru_ocr.provider",
        )
    provider = MinerUCloudProvider(settings.mineru_ocr)
    mineru = MinerUClient(provider, settings.mineru_ocr)
    ingestion = IngestionService(
        settings,
        registry,
        parser_artifacts,
        mineru,
        canonical_mapper,
        canonical_repository,
        knowledge_pipeline,
    )
    documents = DocumentService(
        paths,
        canonical_repository,
        ingestion,
        rebuilder,
        index_manager,
        cognee_repository,
    )
    health = HealthService(
        paths,
        registry,
        canonical_repository,
        mineru,
        deepseek,
        local_inference_client,
        cognee_repository,
        index_manager,
        queue,
    )
    services = ApplicationServices(
        ingestion=ingestion,
        retrieval=retrieval,
        documents=documents,
        feedback=feedback,
        health=health,
        rebuilder=rebuilder,
    )
    worker = BackgroundWorker(
        queue,
        ingestion,
        rebuilder,
        documents,
        feedback,
        poll_interval_seconds=settings.worker.poll_interval_seconds,
    )
    runtime = ManagedRuntime(
        local_inference=local_inference_runtime,
        worker=worker,
    )
    return Application(
        settings=settings,
        services=services,
        runtime=runtime,
        paths=paths,
        registry=registry,
        parser_artifacts=parser_artifacts,
        canonical_repository=canonical_repository,
        canonical_mapper=canonical_mapper,
        mineru=mineru,
        local_inference_client=local_inference_client,
        deepseek=deepseek,
        knowledge_pipeline=knowledge_pipeline,
        queue=queue,
    )


def application_from_config(
    *,
    config_path: Path | None = None,
    data_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Application:
    """Debug/test helper that still performs assembly only."""

    return create_application(
        load_settings(config_path, data_dir=data_dir, environ=environ)
    )
