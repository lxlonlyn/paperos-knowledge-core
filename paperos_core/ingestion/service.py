"""Public cumulative ingestion application service."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from paperos_core.adapters.mineru.client import MinerUClient
from paperos_core.adapters.mineru.mapper import MinerUCanonicalMapper
from paperos_core.config import RuntimeSettings
from paperos_core.domain.canonical import CanonicalIngestionResult
from paperos_core.domain.documents import IngestionJob, IngestionResult, SourceFile
from paperos_core.domain.enums import IngestionJobStatus, ParseRunStatus
from paperos_core.domain.parsing import ParsedIngestionResult
from paperos_core.errors import InvalidDatasetError, PaperOSError
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.parser_artifacts import ParserArtifactRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.ingestion.validation import validate_pdf

if TYPE_CHECKING:
    from paperos_core.adapters.cognee.pipeline import (
        CogneePipelineAdapter,
        KnowledgeIngestionResult,
    )


class IngestionService:
    """Orchestrate validation, source registration, preservation, and job creation."""

    def __init__(
        self,
        config: RuntimeSettings,
        registry: SourceRegistry,
        parser_artifacts: ParserArtifactRepository,
        mineru: MinerUClient,
        canonical_mapper: MinerUCanonicalMapper,
        canonical_repository: CanonicalRepository,
        knowledge_pipeline: CogneePipelineAdapter | None = None,
    ) -> None:
        self.config = config
        self.registry = registry
        self.parser_artifacts = parser_artifacts
        self.mineru = mineru
        self.canonical_mapper = canonical_mapper
        self.canonical_repository = canonical_repository
        self.knowledge_pipeline = knowledge_pipeline

    def ingest_pdf(
        self,
        path: Path,
        *,
        dataset: str | None = None,
        user_metadata: dict[str, Any] | None = None,
        requested_options: dict[str, Any] | None = None,
    ) -> IngestionResult:
        dataset_id = (dataset or self.config.dataset).strip()
        if not dataset_id:
            raise InvalidDatasetError("Dataset must not be empty.", affected="dataset")
        validated = validate_pdf(path, max_file_mb=self.config.ingestion.max_file_mb)
        source, duplicate = self.registry.register_source(
            validated, dataset_id=dataset_id, user_metadata=user_metadata
        )
        job = self.registry.create_job(
            source.id,
            dataset_id=dataset_id,
            requested_options=requested_options,
        )
        return IngestionResult(source_file=source, job=job, duplicate=duplicate)

    def get_job(self, job_id: str) -> IngestionJob:
        return self.registry.get_job(job_id)

    def get_source(self, source_id: str) -> SourceFile:
        return self.registry.get_source(source_id)

    def status(self, *, limit: int = 20) -> dict[str, Any]:
        return self.registry.status(limit=limit)

    async def ingest_pdf_to_parser(
        self,
        path: Path,
        *,
        dataset: str | None = None,
        user_metadata: dict[str, Any] | None = None,
        requested_options: dict[str, Any] | None = None,
    ) -> ParsedIngestionResult:
        """Run genuine PDF intake through the live parser artifact stage."""
        intake = self.ingest_pdf(
            path,
            dataset=dataset,
            user_metadata=user_metadata,
            requested_options=requested_options,
        )
        source = intake.source_file
        options = requested_options or {}
        backend = str(options.get("model_version") or self.config.mineru.preferred_backend)
        if backend == "auto":
            backend = "vlm"
        parse_run = self.parser_artifacts.create_parse_run(
            source,
            provider=self.mineru.provider.name,
            backend=backend,
            request_options=options,
        )
        self.registry.update_job(
            intake.job.id,
            status=IngestionJobStatus.PARSING,
            current_operation="submitting_to_mineru",
        )
        try:
            result = await self.mineru.parse_pdf(source, request_options=requested_options)
            parse_run = self.parser_artifacts.update_parse_run(
                parse_run.id,
                status=ParseRunStatus.RUNNING,
                provider_task_id=result.provider_task_id,
                provider_model=result.backend,
                raw_metadata=result.final_metadata,
            )
            artifacts = self.parser_artifacts.persist_result(parse_run, result)
            self.parser_artifacts.verify_artifact_checksums(parse_run.id)
            parse_run = self.parser_artifacts.update_parse_run(
                parse_run.id,
                status=ParseRunStatus.COMPLETED,
                provider_task_id=result.provider_task_id,
                provider_model=result.backend,
                raw_metadata=result.final_metadata,
            )
            self.registry.update_job(
                intake.job.id,
                status=IngestionJobStatus.NORMALIZING,
                current_operation="awaiting_canonical_transformation",
            )
            return ParsedIngestionResult(
                source_file_id=source.id,
                ingestion_job_id=intake.job.id,
                duplicate_source=intake.duplicate,
                parse_run=parse_run,
                artifacts=artifacts,
            )
        except PaperOSError as exc:
            self.parser_artifacts.update_parse_run(
                parse_run.id,
                status=ParseRunStatus.FAILED,
                error_code=exc.code,
                error_message=exc.message,
            )
            self.registry.update_job(
                intake.job.id,
                status=IngestionJobStatus.FAILED,
                current_operation="mineru_failed",
                error_code=exc.code,
                error_message=exc.message,
            )
            raise

    async def ingest_pdf_to_knowledge(
        self,
        path: Path,
        *,
        dataset: str | None = None,
        user_metadata: dict[str, Any] | None = None,
        requested_options: dict[str, Any] | None = None,
    ) -> KnowledgeIngestionResult:
        """Run genuine PDF intake through the complete knowledge pipeline."""
        if self.knowledge_pipeline is None:
            from paperos_core.errors import CogneeConfigurationError

            raise CogneeConfigurationError("Knowledge pipeline is not configured.")
        canonical = await self.ingest_pdf_to_canonical(
            path,
            dataset=dataset,
            user_metadata=user_metadata,
            requested_options=requested_options,
        )
        self.registry.update_job(
            canonical.parsed.ingestion_job_id,
            status=IngestionJobStatus.INDEXING,
            current_operation="writing_cognee_and_derived_indexes",
        )
        try:
            result = await self.knowledge_pipeline.ingest_canonical_snapshot(canonical)
            self.registry.update_job(
                canonical.parsed.ingestion_job_id,
                status=IngestionJobStatus.POSTPROCESSING,
                current_operation="validating_knowledge_consistency",
            )
            self.registry.update_job(
                canonical.parsed.ingestion_job_id,
                status=IngestionJobStatus.COMPLETED,
                current_operation="completed",
            )
            return result
        except PaperOSError as exc:
            self.registry.update_job(
                canonical.parsed.ingestion_job_id,
                status=IngestionJobStatus.FAILED,
                current_operation="knowledge_indexing_failed",
                error_code=exc.code,
                error_message=exc.message,
            )
            raise

    async def ingest_pdf_to_canonical(
        self,
        path: Path,
        *,
        dataset: str | None = None,
        user_metadata: dict[str, Any] | None = None,
        requested_options: dict[str, Any] | None = None,
    ) -> CanonicalIngestionResult:
        """Run genuine PDF intake through canonical transformation."""
        parsed = await self.ingest_pdf_to_parser(
            path,
            dataset=dataset,
            user_metadata=user_metadata,
            requested_options=requested_options,
        )
        try:
            source = self.registry.get_source(parsed.source_file_id)
            self.parser_artifacts.verify_artifact_checksums(parsed.parse_run.id)
            manifest_path = self.canonical_repository.snapshot_manifest_path(
                source.id, parsed.parse_run.id
            )
            bundle = self.canonical_mapper.build_canonical_snapshot(
                source=source,
                parse_run=parsed.parse_run,
                artifacts=parsed.artifacts,
                manifest_path=manifest_path,
                dataset_id=self.registry.get_job(parsed.ingestion_job_id).dataset_id,
            )
            self.registry.update_job(
                parsed.ingestion_job_id,
                status=IngestionJobStatus.WRITING,
                current_operation="persisting_canonical_snapshot",
            )
            persisted = self.canonical_repository.save_snapshot(bundle)
            self.registry.update_job(
                parsed.ingestion_job_id,
                status=IngestionJobStatus.WRITING,
                current_operation="awaiting_derived_indexing",
            )
            return CanonicalIngestionResult(parsed=parsed, canonical=persisted)
        except PaperOSError as exc:
            self.registry.update_job(
                parsed.ingestion_job_id,
                status=IngestionJobStatus.FAILED,
                current_operation="canonical_transformation_failed",
                error_code=exc.code,
                error_message=exc.message,
            )
            raise
