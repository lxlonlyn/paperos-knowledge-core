"""Durable MinerU task resume contracts for interrupted operational jobs."""

from __future__ import annotations

import asyncio
import hashlib
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from paperos_core.adapters.mineru.schemas import MinerUParseResult, MinerUTask
from paperos_core.config import RuntimeSettings
from paperos_core.domain.canonical import CanonicalBundle, CanonicalSnapshot, Document
from paperos_core.domain.documents import SourceFile
from paperos_core.domain.enums import ParseRunStatus
from paperos_core.domain.ids import canonical_snapshot_id, document_id
from paperos_core.domain.parsing import ParserArtifact, ParseRun
from paperos_core.errors import MinerUTaskUnavailableError
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.parser_artifacts import ParserArtifactRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.ingestion.service import IngestionService
from paperos_core.jobs.queue import JobQueue
from paperos_core.jobs.worker import BackgroundWorker
from paperos_core.paths import DataPaths, build_data_paths
from paperos_core.storage.initializer import StorageInitializer


class _InjectedProcessExit(BaseException):
    """Model an abrupt process exit that normal error handling cannot persist."""


class _ProviderIdentity:
    name = "mineru_resume_contract"


def _result_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr("paper.md", "# Durable resume contract\n")
        archive.writestr("content_list.json", "[]")
        archive.writestr("model_output.json", "{}")
    return buffer.getvalue()


class _MinerUResumeProbe:
    def __init__(
        self,
        *,
        crash_before_submit: bool = False,
        crash_before_poll: bool = False,
    ) -> None:
        self.provider = _ProviderIdentity()
        self.crash_before_submit = crash_before_submit
        self.crash_before_poll = crash_before_poll
        self.task_unavailable = False
        self.submit_count = 0
        self.poll_task_ids: list[str] = []
        self.fetch_task_ids: list[str] = []
        self.submitted_options: list[dict[str, Any]] = []

    async def submit_pdf(
        self,
        source: SourceFile,
        *,
        request_options: dict[str, Any] | None = None,
    ) -> MinerUTask:
        if self.crash_before_submit:
            self.crash_before_submit = False
            raise _InjectedProcessExit
        self.submit_count += 1
        self.submitted_options.append(dict(request_options or {}))
        return MinerUTask(
            provider=self.provider.name,
            task_id=f"remote_task_{self.submit_count}",
            state="submitted",
            backend="vlm",
            data_id=source.id,
            raw_metadata={"submission": self.submit_count},
        )

    async def poll_task(
        self,
        task: MinerUTask,
    ) -> tuple[MinerUTask, list[dict[str, Any]]]:
        self.poll_task_ids.append(task.task_id)
        if self.crash_before_poll:
            self.crash_before_poll = False
            raise _InjectedProcessExit
        if self.task_unavailable:
            raise MinerUTaskUnavailableError(
                "Retained contract task expired.",
                affected=task.task_id,
            )
        completed = task.model_copy(
            update={
                "state": "done",
                "result_archive_url": "memory://result.zip",
                "raw_metadata": {
                    **task.raw_metadata,
                    "state": "done",
                    "poll_attempt": len(self.poll_task_ids),
                },
            }
        )
        return completed, [
            {
                "state": "done",
                "task_id": task.task_id,
                "poll_attempt": len(self.poll_task_ids),
            }
        ]

    async def fetch_result(
        self,
        task: MinerUTask,
        *,
        poll_history: list[dict[str, Any]],
    ) -> MinerUParseResult:
        self.fetch_task_ids.append(task.task_id)
        return MinerUParseResult(
            provider=self.provider.name,
            provider_task_id=task.task_id,
            backend=task.backend,
            archive_bytes=_result_archive(),
            final_metadata=task.raw_metadata,
            poll_history=poll_history,
        )


class _CanonicalMapperProbe:
    def __init__(self, *, crash_once: bool = False) -> None:
        self.crash_once = crash_once
        self.calls = 0

    def build_canonical_snapshot(
        self,
        *,
        source: SourceFile,
        parse_run: ParseRun,
        artifacts: list[ParserArtifact],
        manifest_path: Path,
        dataset_id: str | None = None,
    ) -> CanonicalBundle:
        self.calls += 1
        if self.crash_once:
            self.crash_once = False
            raise _InjectedProcessExit
        assert artifacts
        snapshot_id = canonical_snapshot_id(parse_run.id)
        resolved_document_id = document_id(source.id)
        snapshot = CanonicalSnapshot(
            id=snapshot_id,
            source_file_id=source.id,
            parse_run_id=parse_run.id,
            document_id=resolved_document_id,
            manifest_path=manifest_path,
            dataset_id=dataset_id or "resume-contract",
        )
        return CanonicalBundle(
            snapshot=snapshot,
            document=Document(
                id=resolved_document_id,
                source_file_id=source.id,
                parse_run_id=parse_run.id,
                canonical_snapshot_id=snapshot_id,
                language="en",
                title="Durable resume contract",
            ),
            sections=[],
            elements=[],
            references=[],
        )


def _service(
    root: Path,
    mineru: _MinerUResumeProbe,
    *,
    mapper: _CanonicalMapperProbe | None = None,
) -> tuple[
    IngestionService,
    SourceRegistry,
    ParserArtifactRepository,
    CanonicalRepository,
    Path,
]:
    paths: DataPaths = build_data_paths(root / "data")
    StorageInitializer(paths).initialize()
    registry = SourceRegistry(paths)
    parser_artifacts = ParserArtifactRepository(paths)
    canonical_repository = CanonicalRepository(paths)
    source_path = root / "source.pdf"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"%PDF-1.4\n%%EOF\n")
    service = IngestionService(
        RuntimeSettings(),
        registry,
        parser_artifacts,
        mineru,  # type: ignore[arg-type]
        mapper or _CanonicalMapperProbe(),  # type: ignore[arg-type]
        canonical_repository,
    )
    return service, registry, parser_artifacts, canonical_repository, source_path


def _recover_attempts(
    registry: SourceRegistry,
    parser_artifacts: ParserArtifactRepository,
) -> None:
    registry.recover_interrupted_jobs()
    parser_artifacts.recover_interrupted_runs()


def _registered_source(registry: SourceRegistry, path: Path) -> SourceFile:
    source = registry.find_source_by_sha256(
        hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert source is not None
    return source


def test_submit_before_crash_replays_with_one_submission(tmp_path: Path) -> None:
    async def scenario() -> None:
        mineru = _MinerUResumeProbe(crash_before_submit=True)
        service, registry, parser_artifacts, _canonical, source = _service(
            tmp_path,
            mineru,
        )
        with pytest.raises(_InjectedProcessExit):
            await service.ingest_pdf_to_parser(
                source,
                operation_id="op_submit_before_crash",
            )
        _recover_attempts(registry, parser_artifacts)

        parsed = await service.ingest_pdf_to_parser(
            source,
            operation_id="op_submit_before_crash",
        )

        assert parsed.parse_run.status == ParseRunStatus.COMPLETED
        assert mineru.submit_count == 1
        assert mineru.poll_task_ids == ["remote_task_1"]

    asyncio.run(scenario())


def test_saved_task_id_resumes_without_duplicate_submission(tmp_path: Path) -> None:
    async def scenario() -> None:
        mineru = _MinerUResumeProbe(crash_before_poll=True)
        service, registry, parser_artifacts, _canonical, source = _service(
            tmp_path,
            mineru,
        )
        operation_id = "op_saved_task_id_before_crash"
        with pytest.raises(_InjectedProcessExit):
            await service.ingest_pdf_to_parser(source, operation_id=operation_id)

        checkpoint = parser_artifacts.find_replay_run(
            _registered_source(registry, source).id,
            operation_id=operation_id,
        )
        assert checkpoint is not None
        assert checkpoint.status == ParseRunStatus.SUBMITTED
        assert checkpoint.provider_task_id == "remote_task_1"
        assert checkpoint.provider == mineru.provider.name
        assert checkpoint.backend == "vlm"
        assert checkpoint.raw_metadata is not None
        checkpoint_metadata = checkpoint.raw_metadata["_paperos_checkpoint"]
        assert checkpoint_metadata["state"] == "submitted"
        assert checkpoint_metadata["updated_at"]
        _recover_attempts(registry, parser_artifacts)

        parsed = await service.ingest_pdf_to_parser(source, operation_id=operation_id)

        assert parsed.parse_run.id == checkpoint.id
        assert parsed.parse_run.provider_task_id == "remote_task_1"
        assert mineru.submit_count == 1
        assert mineru.poll_task_ids == ["remote_task_1", "remote_task_1"]
        assert all("_paperos_operational_job_id" not in options for options in mineru.submitted_options)

    asyncio.run(scenario())


def test_durable_result_resumes_canonical_without_submission(tmp_path: Path) -> None:
    async def scenario() -> None:
        mineru = _MinerUResumeProbe()
        mapper = _CanonicalMapperProbe(crash_once=True)
        service, registry, parser_artifacts, canonical, source = _service(
            tmp_path,
            mineru,
            mapper=mapper,
        )
        operation_id = "op_result_before_crash"
        with pytest.raises(_InjectedProcessExit):
            await service.ingest_pdf_to_canonical(source, operation_id=operation_id)

        source_record = _registered_source(registry, source)
        checkpoint = parser_artifacts.find_replay_run(
            source_record.id,
            operation_id=operation_id,
        )
        assert checkpoint is not None
        assert checkpoint.status == ParseRunStatus.COMPLETED
        assert parser_artifacts.durable_result_artifacts(checkpoint)
        _recover_attempts(registry, parser_artifacts)

        result = await service.ingest_pdf_to_canonical(
            source,
            operation_id=operation_id,
        )

        assert result.parsed.parse_run.id == checkpoint.id
        assert canonical.get_snapshot(result.canonical.snapshot.id).id == result.canonical.snapshot.id
        assert mapper.calls == 2
        assert mineru.submit_count == 1
        assert mineru.fetch_task_ids == ["remote_task_1"]

    asyncio.run(scenario())


def test_partial_result_is_cleaned_before_resuming_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        mineru = _MinerUResumeProbe()
        service, registry, parser_artifacts, canonical, source = _service(
            tmp_path,
            mineru,
        )
        operation_id = "op_partial_result_before_manifest"
        original_write = parser_artifacts._write_immutable

        def interrupt_manifest(path: Path, content: bytes) -> None:
            if path.name == "manifest.json":
                raise _InjectedProcessExit
            original_write(path, content)

        monkeypatch.setattr(parser_artifacts, "_write_immutable", interrupt_manifest)
        with pytest.raises(_InjectedProcessExit):
            await service.ingest_pdf_to_canonical(source, operation_id=operation_id)

        source_record = _registered_source(registry, source)
        checkpoint = parser_artifacts.find_replay_run(
            source_record.id,
            operation_id=operation_id,
        )
        assert checkpoint is not None
        assert checkpoint.provider_task_id == "remote_task_1"
        assert not checkpoint.artifact_manifest_path.exists()
        partial_artifacts = parser_artifacts.list_artifacts(checkpoint.id)
        assert partial_artifacts
        partial_ids = {artifact.id for artifact in partial_artifacts}
        partial_response = (
            checkpoint.artifact_manifest_path.parent / "provider_response.json"
        ).read_bytes()
        orphan = checkpoint.artifact_manifest_path.parent / "orphan-partial.bin"
        orphan.write_bytes(b"unregistered partial state")

        _recover_attempts(registry, parser_artifacts)
        monkeypatch.setattr(parser_artifacts, "_write_immutable", original_write)
        original_cleanup = parser_artifacts.cleanup_uncommitted_result
        cleanup_results: list[bool] = []

        def observe_cleanup(parse_run: ParseRun) -> bool:
            cleaned = original_cleanup(parse_run)
            if cleaned:
                assert parser_artifacts.list_artifacts(parse_run.id) == []
                assert tuple(parse_run.artifact_manifest_path.parent.iterdir()) == ()
                assert original_cleanup(parse_run) is False
            cleanup_results.append(cleaned)
            return cleaned

        monkeypatch.setattr(
            parser_artifacts,
            "cleanup_uncommitted_result",
            observe_cleanup,
        )
        result = await service.ingest_pdf_to_canonical(
            source,
            operation_id=operation_id,
        )

        final_run = result.parsed.parse_run
        final_artifacts = parser_artifacts.durable_result_artifacts(final_run)
        assert final_artifacts
        assert final_run.id == checkpoint.id
        assert final_run.provider_task_id == "remote_task_1"
        assert canonical.get_snapshot(result.canonical.snapshot.id).id == result.canonical.snapshot.id
        assert mineru.submit_count == 1
        assert mineru.poll_task_ids == ["remote_task_1", "remote_task_1"]
        assert mineru.fetch_task_ids == ["remote_task_1", "remote_task_1"]
        assert cleanup_results == [True]
        assert not orphan.exists()
        assert (
            checkpoint.artifact_manifest_path.parent / "provider_response.json"
        ).read_bytes() != partial_response
        final_ids = {artifact.id for artifact in final_artifacts}
        assert partial_ids - final_ids

    asyncio.run(scenario())


def test_unavailable_retained_task_fails_without_resubmitting(tmp_path: Path) -> None:
    async def scenario() -> None:
        mineru = _MinerUResumeProbe(crash_before_poll=True)
        service, registry, parser_artifacts, _canonical, source = _service(
            tmp_path,
            mineru,
        )
        operation_id = "op_expired_task"
        with pytest.raises(_InjectedProcessExit):
            await service.ingest_pdf_to_parser(source, operation_id=operation_id)
        _recover_attempts(registry, parser_artifacts)
        source_record = _registered_source(registry, source)
        checkpoint = parser_artifacts.find_replay_run(
            source_record.id,
            operation_id=operation_id,
        )
        assert checkpoint is not None
        mineru.task_unavailable = True

        with pytest.raises(MinerUTaskUnavailableError):
            await service.ingest_pdf_to_parser(source, operation_id=operation_id)

        failed_run = parser_artifacts.get_parse_run(checkpoint.id)
        assert failed_run.status == ParseRunStatus.FAILED
        assert failed_run.error_code == "mineru_task_unavailable"
        failed = parser_artifacts.find_replay_run(
            source_record.id,
            operation_id=operation_id,
        )
        assert failed is None
        assert mineru.submit_count == 1

    asyncio.run(scenario())


def test_worker_propagates_logical_operation_id(tmp_path: Path) -> None:
    class _IngestionProbe:
        operation_id: str | None = None

        async def ingest_pdf_to_knowledge(
            self,
            path: Path,
            *,
            dataset: str | None = None,
            user_metadata: dict[str, Any] | None = None,
            operation_id: str | None = None,
        ) -> Any:
            self.operation_id = operation_id
            return SimpleNamespace(public_dict=lambda: {"status": "completed"})

    class _DocumentsProbe:
        operation_id: str | None = None

        async def reprocess(
            self,
            document_id: str,
            *,
            operation_id: str | None = None,
        ) -> dict[str, object]:
            self.operation_id = operation_id
            return {"status": "completed"}

    async def scenario() -> None:
        paths = build_data_paths(tmp_path / "worker-data")
        StorageInitializer(paths).initialize()
        queue = JobQueue(paths)
        ingestion = _IngestionProbe()
        documents = _DocumentsProbe()
        worker = BackgroundWorker(
            queue,
            ingestion,  # type: ignore[arg-type]
            SimpleNamespace(),  # type: ignore[arg-type]
            documents,  # type: ignore[arg-type]
            SimpleNamespace(),  # type: ignore[arg-type]
            poll_interval_seconds=0.01,
        )

        staged = paths.tmp / "operation-contract" / "source.pdf"
        staged.parent.mkdir(parents=True)
        staged.write_bytes(b"%PDF-1.4\n%%EOF\n")
        ingest_job = queue.enqueue("ingest", {"path": staged})
        completed_ingest = await worker.run_once()
        assert completed_ingest is not None
        assert ingestion.operation_id == ingest_job.id

        reprocess_job = queue.enqueue(
            "reprocess",
            {"document_id": "doc_operation_contract"},
        )
        completed_reprocess = await worker.run_once()
        assert completed_reprocess is not None
        assert documents.operation_id == reprocess_job.id

    asyncio.run(scenario())
