"""PaperOS dependency assembly and the single owned application lifecycle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from paperos_core.adapters.mineru.client import MinerUClient
from paperos_core.adapters.mineru.mapper import MinerUCanonicalMapper
from paperos_core.adapters.mineru.providers import MinerUCloudProvider
from paperos_core.config import RuntimeSettings
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.parser_artifacts import ParserArtifactRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry
from paperos_core.ingestion.service import IngestionService
from paperos_core.paths import DataPaths, build_data_paths

if TYPE_CHECKING:
    from paperos_core.adapters.cognee.llm import LLMClient
    from paperos_core.adapters.cognee.pipeline import CogneePipelineAdapter
    from paperos_core.documents import DocumentService
    from paperos_core.feedback.service import FeedbackService
    from paperos_core.health import HealthService
    from paperos_core.indexes.rebuild import DerivedDataRebuilder
    from paperos_core.jobs.queue import JobQueue
    from paperos_core.jobs.worker import BackgroundWorker
    from paperos_core.retrieval.service import RetrievalService
    from paperos_core.runtime.local_inference.client import LocalInferenceClient
    from paperos_core.runtime.local_inference.runtime import LocalInferenceRuntime
    from paperos_core.storage.initializer import StorageInitializer


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
    scholarly_registry: ScholarlyRegistry
    parser_artifacts: ParserArtifactRepository
    canonical_repository: CanonicalRepository
    canonical_mapper: MinerUCanonicalMapper
    mineru: MinerUClient
    local_inference_client: LocalInferenceClient
    llm: LLMClient
    knowledge_pipeline: CogneePipelineAdapter
    queue: JobQueue
    storage: StorageInitializer
    _started: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)

    async def start(self) -> None:
        """Initialize owned resources in dependency order exactly once."""

        if self._closed:
            raise RuntimeError("A closed PaperOS Application cannot be restarted.")
        if self._started:
            return
        self.storage.initialize()
        self.runtime.local_inference.cleanup_stale_record()
        self.runtime.worker.cleanup_stale_record()
        status = self.storage.validate()
        if not status.valid:
            raise RuntimeError(
                "PaperOS local schema validation failed: " + ", ".join(status.missing_tables)
            )
        try:
            if self.runtime.local_inference.required:
                await self.runtime.local_inference.start()
            self.queue.recover_interrupted_jobs()
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
        failures: list[Exception] = []
        for close in (
            self.runtime.worker.stop,
            self.knowledge_pipeline.compat.aclose,
            self.runtime.local_inference.stop,
            self.local_inference_client.aclose,
            self.mineru.aclose,
        ):
            try:
                await close()
            except Exception as exc:  # noqa: BLE001 - all owners must still close.
                failures.append(exc)
        self._started = False
        if failures:
            raise RuntimeError(
                "PaperOS shutdown failed for one or more owned resources: "
                + "; ".join(f"{type(exc).__name__}: {exc}" for exc in failures)
            ) from failures[0]


def create_application(settings: RuntimeSettings) -> Application:
    """Assemble the object graph without starting a process or background task."""

    paths = build_data_paths(settings.data_dir)
    from paperos_core.adapters.cognee.configurator import CogneeConfigurator

    CogneeConfigurator().apply(settings, paths)
    from paperos_core.storage.initializer import StorageInitializer

    storage = StorageInitializer(paths)
    registry = SourceRegistry(paths)
    scholarly_registry = ScholarlyRegistry(paths)
    parser_artifacts = ParserArtifactRepository(paths)
    canonical_repository = CanonicalRepository(paths)
    canonical_mapper = MinerUCanonicalMapper()
    from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
    from paperos_core.adapters.cognee.runtime_config import CogneeRuntimeConfigReader

    compat = CogneeCompatibilityAdapter(paths)
    from paperos_core.adapters.cognee.llm import LLMClient
    from paperos_core.adapters.cognee.pipeline import CogneePipelineAdapter
    from paperos_core.adapters.cognee.search import CogneeSearchAdapter
    from paperos_core.documents import DocumentService
    from paperos_core.feedback.service import FeedbackService
    from paperos_core.health import HealthService
    from paperos_core.indexes.manager import IndexManager
    from paperos_core.indexes.rebuild import DerivedDataRebuilder
    from paperos_core.jobs.queue import JobQueue
    from paperos_core.jobs.worker import BackgroundWorker
    from paperos_core.prompt_repository import PromptRepository
    from paperos_core.retrieval.service import RetrievalService
    from paperos_core.runtime.local_inference.client import LocalInferenceClient
    from paperos_core.runtime.local_inference.runtime import LocalInferenceRuntime

    local = settings.local_inference
    local_inference_client = LocalInferenceClient(
        f"http://{local.host}:{local.port}",
        local.request_timeout,
    )
    cognee_config = CogneeRuntimeConfigReader()
    local_inference_runtime = LocalInferenceRuntime(
        settings, paths, local_inference_client, cognee_config
    )
    llm = LLMClient(PromptRepository(), cognee_config)
    search = CogneeSearchAdapter(paths, compat)
    index_manager = IndexManager(paths)
    knowledge_pipeline = CogneePipelineAdapter(
        paths,
        canonical_repository,
        registry,
        scholarly_registry,
        compat,
        index_manager,
        llm,
        settings.ingestion,
    )
    rebuilder = DerivedDataRebuilder(paths, canonical_repository, knowledge_pipeline, storage)
    feedback = FeedbackService(paths, canonical_repository)
    queue = JobQueue(paths)
    retrieval = RetrievalService(
        settings,
        paths,
        canonical_repository,
        registry,
        scholarly_registry,
        search,
        compat,
        index_manager,
        local_inference_client,
        llm,
    )
    if settings.mineru.provider not in {"cloud", "mineru_cloud"}:
        from paperos_core.errors import MinerUConfigurationError

        raise MinerUConfigurationError(
            f"Unsupported configured MinerU provider: {settings.mineru.provider}",
            affected="mineru.provider",
        )
    provider = MinerUCloudProvider(settings.mineru)
    mineru = MinerUClient(provider, settings.mineru)
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
        compat,
    )
    worker = BackgroundWorker(
        queue,
        ingestion,
        rebuilder,
        documents,
        feedback,
        poll_interval_seconds=1.0,
    )
    health = HealthService(
        paths,
        registry,
        canonical_repository,
        mineru,
        llm,
        local_inference_runtime,
        compat,
        index_manager,
        queue,
        worker,
    )
    services = ApplicationServices(
        ingestion=ingestion,
        retrieval=retrieval,
        documents=documents,
        feedback=feedback,
        health=health,
        rebuilder=rebuilder,
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
        scholarly_registry=scholarly_registry,
        parser_artifacts=parser_artifacts,
        canonical_repository=canonical_repository,
        canonical_mapper=canonical_mapper,
        mineru=mineru,
        local_inference_client=local_inference_client,
        llm=llm,
        knowledge_pipeline=knowledge_pipeline,
        queue=queue,
        storage=storage,
    )
