"""Application bootstrap and dependency assembly."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from paperos_core.adapters.mineru.client import MinerUClient
from paperos_core.adapters.mineru.mapper import MinerUCanonicalMapper
from paperos_core.adapters.mineru.providers import MinerUCloudProvider
from paperos_core.config import PaperOSConfig, load_config
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.parser_artifacts import ParserArtifactRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.ingestion.service import IngestionService
from paperos_core.paths import DataPaths, build_data_paths

if TYPE_CHECKING:
    from paperos_core.adapters.cognee.pipeline import CogneePipeline
    from paperos_core.adapters.llm import DeepSeekClient
    from paperos_core.adapters.models.client import (
        LocalModelGatewayClient,
        LocalModelGatewayProcess,
    )
    from paperos_core.documents import DocumentService
    from paperos_core.feedback.service import FeedbackService
    from paperos_core.health import HealthService
    from paperos_core.indexes.rebuild import DerivedDataRebuilder
    from paperos_core.jobs.queue import JobQueue
    from paperos_core.jobs.worker import Worker
    from paperos_core.retrieval.service import RetrievalService


@dataclass(slots=True)
class Application:
    config: PaperOSConfig
    paths: DataPaths
    registry: SourceRegistry
    parser_artifacts: ParserArtifactRepository
    canonical_repository: CanonicalRepository
    canonical_mapper: MinerUCanonicalMapper
    mineru: MinerUClient
    ingestion: IngestionService
    model_client: LocalModelGatewayClient
    model_process: LocalModelGatewayProcess
    deepseek: DeepSeekClient
    knowledge_pipeline: CogneePipeline
    rebuilder: DerivedDataRebuilder
    retrieval: RetrievalService
    feedback: FeedbackService
    documents: DocumentService
    health: HealthService
    queue: JobQueue
    worker: Worker | None

    async def aclose(self) -> None:
        if self.worker is not None:
            self.worker.stop()
        await self.model_process.stop()
        await self.model_client.aclose()
        await self.deepseek.aclose()
        await self.mineru.aclose()


def build_application(
    *,
    config_path: Path | None = None,
    data_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Application:
    """Load configuration and assemble cumulative ingestion services."""
    config = load_config(config_path, data_dir=data_dir, environ=environ)
    paths = build_data_paths(config.data_dir)
    registry = SourceRegistry(paths)
    parser_artifacts = ParserArtifactRepository(paths)
    canonical_repository = CanonicalRepository(paths)
    canonical_mapper = MinerUCanonicalMapper(config.ingestion)
    if config.config_path is None:
        from paperos_core.errors import CogneeConfigurationError

        raise CogneeConfigurationError(
            "Project configuration is required to locate the Cognee .env file."
        )
    from paperos_core.adapters.cognee.config import (
        configure_cognee_environment,
        reassert_cognee_runtime,
    )

    cognee_config = configure_cognee_environment(
        paths, env_path=config.config_path.parent.parent / ".env"
    )
    # These modules import Cognee. Import only after its runtime paths are isolated.
    from paperos_core.adapters.cognee.pipeline import CogneePipeline
    from paperos_core.adapters.cognee.repository import CogneeRepository
    from paperos_core.adapters.llm import DeepSeekClient
    from paperos_core.adapters.models.client import (
        LocalModelGatewayClient,
        LocalModelGatewayProcess,
    )
    from paperos_core.documents import DocumentService
    from paperos_core.feedback.service import FeedbackService
    from paperos_core.health import HealthService
    from paperos_core.indexes.manager import IndexManager
    from paperos_core.indexes.rebuild import DerivedDataRebuilder
    from paperos_core.jobs.queue import JobQueue
    from paperos_core.jobs.worker import Worker
    from paperos_core.retrieval.service import RetrievalService

    reassert_cognee_runtime(paths)
    model_client = LocalModelGatewayClient(
        config.models.gateway_endpoint,
        config.models.request_timeout_seconds,
    )
    model_process = LocalModelGatewayProcess(config, paths, model_client)
    deepseek = DeepSeekClient(
        cognee_config,
        timeout_seconds=config.models.request_timeout_seconds,
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
        cognee_repository,
        index_manager,
        deepseek,
        model_process,
    )
    rebuilder = DerivedDataRebuilder(paths, canonical_repository, knowledge_pipeline)
    feedback = FeedbackService(paths, canonical_repository)
    queue = JobQueue(paths)
    retrieval = RetrievalService(
        config,
        paths,
        canonical_repository,
        registry,
        cognee_repository,
        index_manager,
        model_client,
        model_process,
        deepseek,
        feedback,
    )
    if config.mineru_ocr.provider != "mineru_cloud":
        from paperos_core.errors import MinerUConfigurationError

        raise MinerUConfigurationError(
            f"Unsupported configured MinerU provider: {config.mineru_ocr.provider}",
            affected="mineru_ocr.provider",
        )
    provider = MinerUCloudProvider(config.mineru_ocr)
    mineru = MinerUClient(provider, config.mineru_ocr)
    ingestion = IngestionService(
        config,
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
        model_process,
        cognee_repository,
        index_manager,
        queue,
    )
    application = Application(
        config=config,
        paths=paths,
        registry=registry,
        parser_artifacts=parser_artifacts,
        canonical_repository=canonical_repository,
        canonical_mapper=canonical_mapper,
        mineru=mineru,
        ingestion=ingestion,
        model_client=model_client,
        model_process=model_process,
        deepseek=deepseek,
        knowledge_pipeline=knowledge_pipeline,
        rebuilder=rebuilder,
        retrieval=retrieval,
        feedback=feedback,
        documents=documents,
        health=health,
        queue=queue,
        worker=None,
    )
    application.worker = Worker(application, queue)
    return application
