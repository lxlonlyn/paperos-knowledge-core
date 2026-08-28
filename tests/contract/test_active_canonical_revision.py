"""Direct Task 2A contracts for active canonical revision isolation.

Run from the repository root without pytest:

    python tests/contract/test_active_canonical_revision.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import httpx
from pydantic import SecretStr

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.adapters.cognee.compat import (
    CogneeCompatibilityAdapter,
    CogneeDatasetBinding,
    PipelineItem,
    cognee_data_identity,
    cognee_snapshot_uuid,
    resolve_cognee_tokenizer,
)
from paperos_core.adapters.cognee.configurator import CogneeConfigurator
from paperos_core.adapters.cognee.datapoints import PaperOSChunkDataPoint
from paperos_core.adapters.cognee.models import DataPointGraph
from paperos_core.adapters.cognee.pipeline import CogneePipelineAdapter
from paperos_core.adapters.cognee.search import CogneeSearchAdapter
from paperos_core.api.visualize import visualize_dataset
from paperos_core.config import RuntimeSettings, load_settings
from paperos_core.documents import DocumentService
from paperos_core.domain.canonical import (
    CanonicalBundle,
    CanonicalSnapshot,
    Chunk,
    Document,
    Element,
    ReferenceEntry,
    SourceSpan,
)
from paperos_core.domain.documents import utc_now
from paperos_core.domain.enums import ElementType
from paperos_core.domain.ids import canonical_snapshot_id, document_id
from paperos_core.errors import CanonicalStorageError, CogneeStorageError
from paperos_core.health import HealthService
from paperos_core.indexes.manager import IndexManager
from paperos_core.indexes.manifest import IndexingReport
from paperos_core.indexes.rebuild import DerivedDataRebuilder
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.chunking import build_chunks
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.ingestion.retrieval_text import effective_index_text
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry
from paperos_core.paths import DataPaths, build_data_paths
from paperos_core.retrieval.candidates import QueryRequest
from paperos_core.retrieval.corpus import CorpusView
from paperos_core.retrieval.service import (
    NO_EVIDENCE_ANSWER,
    NO_EVIDENCE_MODEL,
    RetrievalService,
)
from paperos_core.storage.initializer import StorageInitializer
from paperos_core.storage.path_refs import DataPathCodec

_SOURCE_ID = "src_active_revision_contract"
_SOURCE_SHA256 = "a" * 64
_DOCUMENT_ID = document_id(_SOURCE_ID)
_DATASET = "active-revision-contract"
_SHARED_ELEMENT_ID = "element_shared_revision_contract"
_SHARED_CHUNK_ID = "chunk_shared_revision_contract"
_VECTOR_DATA = (
    REPOSITORY_ROOT / "data" / "validation" / "scholarly_work_reference" / "output"
)
_VECTOR_QUERY = "Explicit flows for implicit surfaces shape morphing deformation"
_LOCAL_INFERENCE_PORT = 18081
_REAL_REVISION_IDS = (
    "snapshot_889743f265a49cee32253bd3bedbd256",
    "snapshot_99c70e0395047da22051170050947a98",
)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class _ForbiddenDependency:
    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"No-active query touched forbidden dependency: {name}")


class _HealthProbe:
    def __init__(self) -> None:
        self.runtime_config = _LocalRuntimeReader()

    async def health_check(self) -> dict[str, object]:
        return {"provider": "contract", "model": "contract"}


class _MinerUProbe:
    async def health_check(self) -> dict[str, object]:
        return {"provider": "contract", "configured": True, "reachable": True}


class _CogneeHealthProbe:
    async def vector_status(self, *, dataset_name: str | None = None) -> dict[str, object]:
        return {
            "backend": "contract",
            "collection_count": 1,
            "record_count": 1,
            "dimensions": 768,
        }

    async def get_datapoint(
        self,
        canonical_id: str,
        *,
        dataset_name: str | None = None,
        snapshot_id: str | None = None,
    ) -> dict[str, object]:
        return {"canonical_id": canonical_id, "canonical_snapshot_id": snapshot_id}

    def read_manifest(self, snapshot_id: str) -> dict[str, object]:
        return {"dataset": {"name": _DATASET}}


class _DeletionCognee:
    async def delete_document_data(self, snapshot_id: str) -> int:
        _require(bool(snapshot_id), "Delete received an empty snapshot ID")
        return 1


class _LocalRuntimeConfig:
    embedding_dimensions = 768

    def embedding_targets(self, host: str, port: int) -> bool:
        return False


class _LocalRuntimeReader:
    def read(self) -> _LocalRuntimeConfig:
        return _LocalRuntimeConfig()


def _insert_source_and_parse_runs(paths: DataPaths, parse_ids: list[str]) -> None:
    codec = DataPathCodec(paths.root)
    created_at = utc_now().isoformat()
    source_path = paths.raw / _SOURCE_ID / "source.pdf"
    with sqlite3.connect(paths.registry_db) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO source_files (
                id, sha256, original_filename, stored_filename, media_type,
                size_bytes, storage_path, created_at, schema_version, id_version,
                source_url, user_metadata, dataset_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _SOURCE_ID,
                _SOURCE_SHA256,
                "active-revision-contract.pdf",
                "source.pdf",
                "application/pdf",
                1,
                codec.encode(source_path),
                created_at,
                "1.0",
                "1",
                None,
                None,
                _DATASET,
            ),
        )
        for parse_id in parse_ids:
            connection.execute(
                """
                INSERT INTO parse_runs (
                    id, source_file_id, provider, backend, status,
                    request_options, created_at, completed_at,
                    artifact_manifest_path, schema_version, pipeline_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    parse_id,
                    _SOURCE_ID,
                    "contract",
                    "contract",
                    "completed",
                    "{}",
                    created_at,
                    created_at,
                    codec.encode(paths.parsed / parse_id / "manifest.json"),
                    "1.0",
                    "contract",
                ),
            )


def _revision(
    repository: CanonicalRepository,
    parse_id: str,
    *,
    title: str,
    chunk_text: str,
) -> tuple[CanonicalBundle, Chunk]:
    snapshot_id = canonical_snapshot_id(parse_id)
    snapshot = CanonicalSnapshot(
        id=snapshot_id,
        source_file_id=_SOURCE_ID,
        parse_run_id=parse_id,
        document_id=_DOCUMENT_ID,
        dataset_id=_DATASET,
        manifest_path=repository.snapshot_manifest_path(_SOURCE_ID, parse_id),
    )
    document = Document(
        id=_DOCUMENT_ID,
        source_file_id=_SOURCE_ID,
        parse_run_id=parse_id,
        canonical_snapshot_id=snapshot_id,
        language="en",
        title=title,
    )
    element = Element(
        id=_SHARED_ELEMENT_ID,
        document_id=_DOCUMENT_ID,
        canonical_snapshot_id=snapshot_id,
        element_type=ElementType.PARAGRAPH,
        order=0,
        text=chunk_text,
        source_span=SourceSpan(artifact_id=f"artifact:{parse_id}", item_index=0),
    )
    bundle = CanonicalBundle(
        snapshot=snapshot,
        document=document,
        sections=[],
        elements=[element],
        references=[],
    )
    chunk = Chunk(
        id=_SHARED_CHUNK_ID,
        document_id=_DOCUMENT_ID,
        canonical_snapshot_id=snapshot_id,
        text=chunk_text,
        order=0,
        element_ids=[element.id],
        token_count=max(1, len(chunk_text.split())),
    )
    return bundle, chunk


async def _save_and_index(
    repository: CanonicalRepository,
    indexes: IndexManager,
    bundle: CanonicalBundle,
    chunk: Chunk,
) -> None:
    repository.save_snapshot(bundle)
    repository.save_chunks(bundle.snapshot.id, [chunk])
    await indexes.index_bundle(bundle, chunks=[chunk])


def _with_scholarly_identity(
    bundle: CanonicalBundle,
    chunk: Chunk,
    *,
    document_doi: str,
    reference_doi: str,
    reference_title: str,
) -> tuple[CanonicalBundle, Chunk]:
    reference_id = "reference_shared_scholarly_revision"
    reference = ReferenceEntry(
        id=reference_id,
        document_id=bundle.document.id,
        canonical_snapshot_id=bundle.snapshot.id,
        raw_text=f"Scholar, A. 2024. {reference_title}.",
        order=0,
        title=reference_title,
        authors=["A. Scholar"],
        year=2024,
        doi=reference_doi,
        source_element_id=bundle.elements[0].id,
    )
    return (
        bundle.model_copy(
            update={
                "document": bundle.document.model_copy(
                    update={"doi": document_doi}
                ),
                "references": [reference],
            }
        ),
        chunk.model_copy(
            update={"citation_reference_entry_ids": [reference_id]}
        ),
    )


async def scholarly_isolation_contract(root: Path) -> dict[str, object]:
    paths = build_data_paths(root / "scholarly-data")
    StorageInitializer(paths).initialize()
    repository = CanonicalRepository(paths)
    registry = ScholarlyRegistry(paths)
    parse_ids = ["parse_scholarly_revision_1", "parse_scholarly_revision_2"]
    _insert_source_and_parse_runs(paths, parse_ids)

    first, first_chunk = _revision(
        repository,
        parse_ids[0],
        title="Old active scholarly title",
        chunk_text="old scholarly evidence",
    )
    first, first_chunk = _with_scholarly_identity(
        first,
        first_chunk,
        document_doi="10.1000/old-document",
        reference_doi="10.1000/merge-target",
        reference_title="Old cited work",
    )
    repository.save_snapshot(first)
    repository.save_chunks(first.snapshot.id, [first_chunk])
    registry.resolve_candidate_bundle(first, [first_chunk])
    _require(registry.identity_snapshot()["works"] == [], "Candidate touched main registry")
    registry.publish_candidate(first.snapshot.id, repository)
    old_identity = registry.identity_snapshot()
    old_document_work = registry.work_for_document(_DOCUMENT_ID)
    old_reference_work = registry.work_for_reference(
        "reference_shared_scholarly_revision"
    )
    _require(old_document_work is not None, "Active document Work is missing")
    _require(old_reference_work is not None, "Active reference Work is missing")
    _require(
        old_document_work.id != old_reference_work.id,
        "Contract setup did not create distinct merge inputs",
    )

    second, second_chunk = _revision(
        repository,
        parse_ids[1],
        title="Candidate scholarly title",
        chunk_text="candidate scholarly evidence",
    )
    second, second_chunk = _with_scholarly_identity(
        second,
        second_chunk,
        document_doi="10.1000/merge-target",
        reference_doi="10.1000/merge-target",
        reference_title="Candidate cited work",
    )
    repository.save_snapshot(second)
    repository.save_chunks(second.snapshot.id, [second_chunk])
    with sqlite3.connect(paths.registry_db) as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_staged_scholarly_update
            BEFORE UPDATE ON scholarly_works
            BEGIN
                SELECT RAISE(ABORT, 'injected scholarly resolution failure');
            END;
            """
        )
    try:
        registry.resolve_candidate_bundle(second, [second_chunk])
    except sqlite3.IntegrityError:
        pass
    else:
        raise RuntimeError("Injected scholarly resolution failure did not occur")
    _require(
        registry.identity_snapshot() == old_identity,
        "Failed scholarly resolution polluted active-visible registry",
    )
    _require(
        not registry.candidate_database_path(second.snapshot.id).exists(),
        "Failed scholarly resolution retained staging state",
    )
    with sqlite3.connect(paths.registry_db) as connection:
        connection.execute("DROP TRIGGER fail_staged_scholarly_update")

    candidate_context = registry.resolve_candidate_bundle(second, [second_chunk])
    staged = ScholarlyRegistry(
        paths,
        database_path=registry.candidate_database_path(second.snapshot.id),
    )
    _require(staged.list_redirects(), "Candidate did not exercise Work redirect staging")
    _require(
        registry.identity_snapshot() == old_identity,
        "Candidate scholarly resolution polluted active-visible registry",
    )
    _require(
        candidate_context.document_work.title == "Candidate scholarly title",
        "Candidate staging did not contain new document Work state",
    )
    registry.discard_candidate(second.snapshot.id)
    _require(
        registry.identity_snapshot() == old_identity,
        "Failed candidate left active-visible scholarly state",
    )

    registry.resolve_candidate_bundle(second, [second_chunk])
    previous = registry.publish_candidate(second.snapshot.id, repository)
    _require(previous == first.snapshot.id, "Scholarly publish replaced wrong revision")
    _require(
        registry.work_for_document(_DOCUMENT_ID).title == "Candidate scholarly title",  # type: ignore[union-attr]
        "Successful activation did not publish candidate scholarly mapping",
    )
    redirects_after_publish = registry.list_redirects()
    repository.cleanup_snapshot(first.snapshot.id)
    repository.cleanup_snapshot(first.snapshot.id)
    _require(
        registry.list_redirects() == redirects_after_publish,
        "Old canonical cleanup damaged new active scholarly mapping",
    )
    return {
        "status": "passed",
        "candidate_merge_isolated": True,
        "failed_candidate_clean": True,
        "published_redirect_count": len(redirects_after_publish),
    }


class _LocalRebuildPipeline:
    def __init__(
        self,
        paths: DataPaths,
        repository: CanonicalRepository,
        registry: SourceRegistry,
        indexes: IndexManager,
        *,
        fail_build: bool,
    ) -> None:
        self.paths = paths
        self.repository = repository
        self.indexes = indexes
        self.fail_build = fail_build
        self.scholarly_registry = ScholarlyRegistry(paths)
        self.cleanup_adapter = CogneePipelineAdapter(
            paths,
            repository,
            registry,
            self.scholarly_registry,
            SimpleNamespace(),  # type: ignore[arg-type]
            indexes,
            SimpleNamespace(),  # type: ignore[arg-type]
            RuntimeSettings().ingestion,
        )

    def reproject_enrichment(
        self,
        source_snapshot_id: str,
        target_snapshot_id: str,
    ) -> Path:
        source = self.paths.cognee / "enrichment" / f"{source_snapshot_id}.json"
        target = self.paths.cognee / "enrichment" / f"{target_snapshot_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return target

    async def ingest_bundle(
        self,
        bundle: CanonicalBundle,
        *,
        rebuilt: bool,
        reuse_existing_enrichment: bool,
        generate_enrichment_if_missing: bool,
    ) -> tuple[IndexingReport, Path]:
        _require(rebuilt, "Rebuild pipeline did not mark candidate as rebuilt")
        _require(reuse_existing_enrichment, "Rebuild did not reuse current enrichment")
        _require(
            not generate_enrichment_if_missing,
            "Safe rebuild unexpectedly enabled enrichment generation",
        )
        if self.fail_build:
            raise RuntimeError("injected isolated rebuild build failure")
        active_snapshot_id = self.repository.active_snapshot_id(bundle.document.id)
        _require(active_snapshot_id is not None, "Rebuild lost old active pointer")
        source_projection = self.repository.get_chunk_projection(active_snapshot_id)
        chunks = [
            chunk.model_copy(
                update={"canonical_snapshot_id": bundle.snapshot.id}
            )
            for chunk in source_projection.chunks
        ]
        self.repository.save_chunks(bundle.snapshot.id, chunks)
        self.scholarly_registry.resolve_candidate_bundle(bundle, chunks)
        manifest, manifest_path = await self.indexes.index_bundle(
            bundle,
            chunks=chunks,
        )
        enrichment_path = (
            self.paths.cognee / "enrichment" / f"{bundle.snapshot.id}.json"
        )
        return (
            IndexingReport(
                canonical_snapshot_id=bundle.snapshot.id,
                document_id=bundle.document.id,
                dataset_name=bundle.snapshot.dataset_id,
                cognee_dataset_id="contract",
                cognee_data_id="contract",
                cognee_pipeline_run_id="contract",
                cognee_provenance_backend="contract",
                manifest_path=manifest_path,
                cognee_manifest_path=(
                    self.paths.cognee / "manifests" / f"{bundle.snapshot.id}.json"
                ),
                lexical_database=manifest.lexical_database,
                vector_database="contract",
                cognee_object_count=0,
                relation_count=0,
                lexical_object_count=len(manifest.lexical_object_ids),
                vector_object_count=0,
                embedding_dimensions=1,
                semantic_entity_count=0,
                semantic_claim_count=0,
                semantic_relation_count=0,
                consistency_valid=True,
                rebuilt=True,
            ),
            enrichment_path,
        )

    async def _cleanup_after_failure(self, snapshot_id: str, *, phase: str) -> None:
        _require(phase == "rebuild_candidate", "Unexpected rebuild cleanup phase")
        self.indexes.lexical.delete_snapshot(snapshot_id)
        for path in (
            self.repository.chunk_store_path(snapshot_id),
            self.repository.citation_mention_store_path(snapshot_id),
            self.paths.cognee / "enrichment" / f"{snapshot_id}.json",
            self.paths.indexes / "manifests" / f"{snapshot_id}.json",
        ):
            path.unlink(missing_ok=True)
        self.scholarly_registry.discard_candidate(snapshot_id)
        self.repository.cleanup_snapshot(snapshot_id)

    async def cleanup_snapshot_revision(self, snapshot_id: str) -> list[Path]:
        return await self.cleanup_adapter.cleanup_snapshot_revision(snapshot_id)

    def _record_cleanup_retry(
        self,
        snapshot_id: str,
        *,
        phase: str,
        exc: Exception,
    ) -> None:
        raise RuntimeError(
            f"Unexpected cleanup retry for {snapshot_id} in {phase}: {exc}"
        )


async def safe_rebuild_contract(root: Path) -> dict[str, object]:
    paths = build_data_paths(root / "rebuild-data")
    StorageInitializer(paths).initialize()
    repository = CanonicalRepository(paths)
    registry = SourceRegistry(paths)
    indexes = IndexManager(paths)
    parse_id = "parse_safe_rebuild"
    _insert_source_and_parse_runs(paths, [parse_id])
    active, active_chunk = _revision(
        repository,
        parse_id,
        title="Safe rebuild active",
        chunk_text="saferebuildtoken active evidence",
    )
    await _save_and_index(repository, indexes, active, active_chunk)
    scholarly = ScholarlyRegistry(paths)
    scholarly.resolve_candidate_bundle(active, [active_chunk])
    scholarly.publish_candidate(active.snapshot.id, repository)
    enrichment = paths.cognee / "enrichment" / f"{active.snapshot.id}.json"
    enrichment.parent.mkdir(parents=True, exist_ok=True)
    enrichment.write_text('{"contract": true}', encoding="utf-8")

    failing_pipeline = _LocalRebuildPipeline(
        paths,
        repository,
        registry,
        indexes,
        fail_build=True,
    )
    failing_rebuilder = DerivedDataRebuilder(
        paths,
        repository,
        failing_pipeline,  # type: ignore[arg-type]
        StorageInitializer(paths),
    )
    try:
        await failing_rebuilder.rebuild()
    except RuntimeError as exc:
        _require("injected isolated rebuild" in str(exc), "Wrong rebuild failure")
    else:
        raise RuntimeError("Injected rebuild failure unexpectedly succeeded")
    _require(
        repository.active_snapshot_id(_DOCUMENT_ID) == active.snapshot.id,
        "Failed rebuild changed active pointer",
    )
    old_hits = indexes.lexical.search(
        '"saferebuildtoken"',
        active_snapshot_ids={active.snapshot.id},
        limit=10,
    )
    _require(old_hits, "Failed rebuild deleted active FTS")
    _require(
        repository.list_all_snapshot_ids() == [active.snapshot.id],
        "Failed rebuild retained canonical candidate",
    )

    successful_pipeline = _LocalRebuildPipeline(
        paths,
        repository,
        registry,
        indexes,
        fail_build=False,
    )
    report = await DerivedDataRebuilder(
        paths,
        repository,
        successful_pipeline,  # type: ignore[arg-type]
        StorageInitializer(paths),
    ).rebuild()
    _require(len(report.rebuilt_snapshot_ids) == 1, "Rebuild did not publish one revision")
    rebuilt_snapshot_id = report.rebuilt_snapshot_ids[0]
    _require(
        repository.active_snapshot_id(_DOCUMENT_ID) == rebuilt_snapshot_id,
        "Successful rebuild did not switch active pointer",
    )
    _require(
        repository.list_all_snapshot_ids() == [rebuilt_snapshot_id],
        "Successful rebuild retained old canonical revision",
    )
    rebuilt_hits = indexes.lexical.search(
        '"saferebuildtoken"',
        active_snapshot_ids={rebuilt_snapshot_id},
        limit=10,
    )
    _require(rebuilt_hits, "Successful rebuild did not publish candidate FTS")
    return {
        "status": "passed",
        "failure_preserved_snapshot_id": active.snapshot.id,
        "published_snapshot_id": rebuilt_snapshot_id,
        "old_canonical_removed": True,
    }


async def _no_active_query(
    paths: DataPaths,
    repository: CanonicalRepository,
    registry: SourceRegistry,
    indexes: IndexManager,
) -> dict[str, object]:
    settings = RuntimeSettings.model_validate(
        {
            "data": {"directory": paths.root, "dataset": _DATASET},
            "local_inference": {"enabled": True},
            "retrieval": {"rerank_enabled": True},
        }
    )
    forbidden = _ForbiddenDependency()
    service = RetrievalService(
        settings,
        paths,
        repository,
        registry,
        ScholarlyRegistry(paths),
        forbidden,  # type: ignore[arg-type]
        forbidden,  # type: ignore[arg-type]
        indexes,
        forbidden,  # type: ignore[arg-type]
        forbidden,  # type: ignore[arg-type]
    )
    response = await service.query(QueryRequest(query="revisiontoken"))
    _require(response.answer == NO_EVIDENCE_ANSWER, "No-active query answer changed")
    _require(response.answer_model == NO_EVIDENCE_MODEL, "No-active model is misleading")
    _require(response.candidates == [], "No-active query returned candidates")
    _require(response.evidence == [], "No-active query returned evidence")
    _require("no_evidence" in response.stages, "No-active query omitted no_evidence")
    return {"status": "passed", "answer_model": response.answer_model}


async def _health_document_count(
    paths: DataPaths,
    repository: CanonicalRepository,
    registry: SourceRegistry,
    indexes: IndexManager,
) -> int:
    settings = RuntimeSettings.model_validate(
        {
            "data": {"directory": paths.root, "dataset": _DATASET},
            "cognee": {"embedding": {"endpoint": "https://embedding.invalid/v1"}},
            "local_inference": {"enabled": False},
            "retrieval": {"rerank_enabled": False},
        }
    )
    health = HealthService(
        paths,
        registry,
        repository,
        SimpleNamespace(provider=_MinerUProbe()),
        _HealthProbe(),  # type: ignore[arg-type]
        SimpleNamespace(
            settings=settings,
            cognee_config=_LocalRuntimeReader(),
        ),
        _CogneeHealthProbe(),  # type: ignore[arg-type]
        indexes,
        SimpleNamespace(list_jobs=list),
    )
    report = await health.report()
    return int(report["components"]["cognee_graph"]["document_count"])


async def local_revision_contract(root: Path) -> dict[str, object]:
    paths = build_data_paths(root / "active-data")
    storage = StorageInitializer(paths)
    storage.initialize()
    repository = CanonicalRepository(paths)
    registry = SourceRegistry(paths)
    indexes = IndexManager(paths)
    parse_ids = ["parse_active_revision_1", "parse_active_revision_2"]
    _insert_source_and_parse_runs(paths, parse_ids)

    with sqlite3.connect(paths.registry_db) as connection:
        columns = [
            str(row[1])
            for row in connection.execute(
                "PRAGMA table_info(active_canonical_snapshots)"
            )
        ]
    _require(
        columns == ["document_id", "snapshot_id", "activated_at"],
        f"Active pointer schema changed: {columns}",
    )

    first, first_chunk = _revision(
        repository,
        parse_ids[0],
        title="Active revision one",
        chunk_text="revisiontoken old active evidence",
    )
    await _save_and_index(repository, indexes, first, first_chunk)
    _require(repository.list_active_snapshot_ids() == [], "Candidate auto-activated")
    empty_corpus = CorpusView.load(paths, repository, registry)
    _require(empty_corpus.bundles == {}, "Candidate entered CorpusView before activation")
    no_active = await _no_active_query(paths, repository, registry, indexes)

    repository.activate_snapshot(first.snapshot.id)
    _require(
        repository.active_snapshot_id(_DOCUMENT_ID) == first.snapshot.id,
        "First snapshot did not activate",
    )
    first_corpus = CorpusView.load(paths, repository, registry)
    _require(
        first_corpus.chunks[_SHARED_CHUNK_ID].text == first_chunk.text,
        "First active Chunk is unavailable",
    )

    second, second_chunk = _revision(
        repository,
        parse_ids[1],
        title="Active revision two",
        chunk_text="revisiontoken new candidate evidence",
    )
    await _save_and_index(repository, indexes, second, second_chunk)
    _require(
        repository.active_snapshot_id(_DOCUMENT_ID) == first.snapshot.id,
        "Saving/indexing the second candidate changed active",
    )
    with sqlite3.connect(indexes.lexical.path) as connection:
        same_object_rows = int(
            connection.execute(
                "SELECT COUNT(*) FROM lexical_records WHERE object_id = ?",
                (_SHARED_CHUNK_ID,),
            ).fetchone()[0]
        )
    _require(same_object_rows == 2, "FTS could not retain the same object in two snapshots")
    active_rows = indexes.lexical.search(
        '"revisiontoken"',
        active_snapshot_ids={first.snapshot.id},
        limit=20,
    )
    active_chunk_rows = [row for row in active_rows if row["object_id"] == _SHARED_CHUNK_ID]
    _require(len(active_chunk_rows) == 1, "FTS active filter returned duplicate revisions")
    _require(active_chunk_rows[0]["text"] == first_chunk.text, "FTS returned candidate text")

    first_data_identity = cognee_data_identity(_SOURCE_SHA256, first.snapshot.id)
    second_data_identity = cognee_data_identity(_SOURCE_SHA256, second.snapshot.id)
    _require(first_data_identity != second_data_identity, "Cognee Data identity is source-only")
    _require(
        cognee_snapshot_uuid(first.snapshot.id, _SHARED_CHUNK_ID)
        != cognee_snapshot_uuid(second.snapshot.id, _SHARED_CHUNK_ID),
        "Canonical-backed DataPoint storage identity is not snapshot-scoped",
    )

    with sqlite3.connect(paths.registry_db) as connection:
        connection.execute(
            f"""
            CREATE TRIGGER block_second_activation
            BEFORE UPDATE ON active_canonical_snapshots
            WHEN NEW.snapshot_id = '{second.snapshot.id}'
            BEGIN
                SELECT RAISE(ABORT, 'injected activation failure');
            END;
            """
        )
    try:
        repository.activate_snapshot(second.snapshot.id)
    except CanonicalStorageError:
        pass
    else:
        raise RuntimeError("Injected activation failure unexpectedly succeeded")
    _require(
        repository.active_snapshot_id(_DOCUMENT_ID) == first.snapshot.id,
        "Activation transaction changed pointer on failure",
    )
    with sqlite3.connect(paths.registry_db) as connection:
        connection.execute("DROP TRIGGER block_second_activation")

    rebuilder = DerivedDataRebuilder(
        paths,
        repository,
        SimpleNamespace(),  # type: ignore[arg-type]
        storage,
    )
    _require(
        rebuilder.select_snapshot_ids() == [first.snapshot.id],
        "Default rebuild selected candidate/history",
    )
    try:
        rebuilder.select_snapshot_ids(second.snapshot.id)
    except CogneeStorageError as exc:
        _require(
            exc.details.get("reason") == "inactive_canonical_snapshot",
            "Inactive rebuild rejection reason changed",
        )
    else:
        raise RuntimeError("Public rebuild accepted an inactive candidate")

    document_service = DocumentService(
        paths,
        repository,
        SimpleNamespace(get_source=registry.get_source),  # type: ignore[arg-type]
        rebuilder,
        indexes,
        SimpleNamespace(),  # type: ignore[arg-type]
    )
    listed = document_service.list_documents()
    detail = document_service.inspect(_DOCUMENT_ID)
    _require(len(listed) == 1, "Document list exposed candidate/history")
    _require(
        listed[0].canonical_snapshot_id == first.snapshot.id,
        "Document list did not select active",
    )
    detail_payload = detail.model_dump(mode="json")
    _require("snapshot_ids" not in detail_payload, "Inspect exposed snapshot history")
    _require(
        "deleted" not in listed[0].model_dump(mode="json")
        and "deleted" not in detail_payload,
        "Document API exposed obsolete deleted state",
    )
    _require(
        detail.canonical_snapshot_id == first.snapshot.id,
        "Inspect did not select active",
    )
    _require(
        await _health_document_count(paths, repository, registry, indexes) == 1,
        "Health document count is not active-only",
    )

    graph_root = paths.cognee / "graphs"
    graph_root.mkdir(parents=True, exist_ok=True)
    for bundle, label in ((first, "old-active"), (second, "new-candidate")):
        (graph_root / f"{bundle.snapshot.id}.json").write_text(
            json.dumps(
                {
                    "nodes": [
                        {
                            "canonical_id": f"node:{label}",
                            "__type__": "ContractNode",
                            "title": label,
                        }
                    ],
                    "relations": [],
                }
            ),
            encoding="utf-8",
        )
    visual_before = await visualize_dataset(
        application=SimpleNamespace(
            settings=SimpleNamespace(dataset=_DATASET),
            canonical_repository=repository,
            paths=paths,
        ),
        dataset=None,
    )
    _require(
        visual_before["active_snapshot_ids"] == [first.snapshot.id],
        "Visualize exposed inactive candidate",
    )

    previous = repository.activate_snapshot(second.snapshot.id)
    _require(previous == first.snapshot.id, "Activation did not return retired revision")
    switched = CorpusView.load(paths, repository, registry)
    _require(
        switched.chunks[_SHARED_CHUNK_ID].text == second_chunk.text,
        "Successful switch did not expose only the new Chunk",
    )
    _require(
        repository.list_active_snapshot_ids() == [second.snapshot.id],
        "Document has more than one active pointer",
    )
    with sqlite3.connect(paths.registry_db) as connection:
        pointer_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM active_canonical_snapshots WHERE document_id = ?",
                (_DOCUMENT_ID,),
            ).fetchone()[0]
        )
    _require(pointer_count == 1, "Active pointer uniqueness failed")

    visual_after = await visualize_dataset(
        application=SimpleNamespace(
            settings=SimpleNamespace(dataset=_DATASET),
            canonical_repository=repository,
            paths=paths,
        ),
        dataset=None,
    )
    _require(
        visual_after["active_snapshot_ids"] == [second.snapshot.id],
        "Visualize did not switch atomically to new active",
    )

    cleanup_pipeline = CogneePipelineAdapter(
        paths,
        repository,
        registry,
        ScholarlyRegistry(paths),
        SimpleNamespace(),  # type: ignore[arg-type]
        indexes,
        SimpleNamespace(),  # type: ignore[arg-type]
        RuntimeSettings().ingestion,
    )
    first_cleanup = await cleanup_pipeline.cleanup_snapshot_revision(first.snapshot.id)
    second_cleanup = await cleanup_pipeline.cleanup_snapshot_revision(first.snapshot.id)
    _require(first_cleanup, "First old-revision cleanup removed nothing")
    _require(second_cleanup == [], "Cleanup is not idempotent")
    _require(
        first.snapshot.id not in repository.list_all_snapshot_ids(),
        "Old canonical revision survived successful cleanup",
    )
    _require(
        repository.active_snapshot_id(_DOCUMENT_ID) == second.snapshot.id,
        "Old cleanup changed the new active pointer",
    )
    _require(
        indexes.lexical.object_ids(second.snapshot.id),
        "Old cleanup deleted new active FTS rows",
    )
    _require(
        repository.chunk_store_path(second.snapshot.id).is_file(),
        "Old cleanup deleted new active ChunkProjection",
    )
    try:
        repository.cleanup_snapshot(second.snapshot.id)
    except CanonicalStorageError:
        pass
    else:
        raise RuntimeError("Canonical cleanup accepted the active snapshot")

    delete_service = DocumentService(
        paths,
        repository,
        SimpleNamespace(get_source=registry.get_source),  # type: ignore[arg-type]
        rebuilder,
        indexes,
        _DeletionCognee(),  # type: ignore[arg-type]
    )
    deletion = await delete_service.delete(_DOCUMENT_ID)
    _require(deletion.removed_vector_objects == 1, "Delete vector count changed")
    _require(
        repository.active_snapshot_id(_DOCUMENT_ID) is None,
        "Delete left a public active pointer",
    )
    with sqlite3.connect(paths.registry_db) as connection:
        raw_pointer_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM active_canonical_snapshots WHERE document_id = ?",
                (_DOCUMENT_ID,),
            ).fetchone()[0]
        )
    _require(raw_pointer_count == 0, "Delete retained the raw active pointer")
    _require(delete_service.list_documents() == [], "Delete remained in document list")
    _require(
        CorpusView.load(paths, repository, registry).bundles == {},
        "Delete remained query-visible",
    )
    _require(
        await _health_document_count(paths, repository, registry, indexes) == 0,
        "Deleted document remained in health active count",
    )
    visual_deleted = await visualize_dataset(
        application=SimpleNamespace(
            settings=SimpleNamespace(dataset=_DATASET),
            canonical_repository=repository,
            paths=paths,
        ),
        dataset=None,
    )
    _require(
        visual_deleted["active_snapshot_ids"] == [],
        "Deleted document remained visualization-visible",
    )
    deleted_query = await _no_active_query(paths, repository, registry, indexes)

    return {
        "status": "passed",
        "schema": columns,
        "no_active_query": no_active,
        "candidate_rows_with_shared_object_id": same_object_rows,
        "activation_failure_preserved": first.snapshot.id,
        "active_after_switch": second.snapshot.id,
        "cleanup_first_count": len(first_cleanup),
        "cleanup_second_count": len(second_cleanup),
        "document_history_hidden": "snapshot_ids" not in detail_payload,
        "document_deleted_state_hidden": "deleted" not in detail_payload,
        "old_canonical_removed": True,
        "delete_active_pointer_count": raw_pointer_count,
        "delete_no_active_query": deleted_query,
    }


async def _start_embedding_service() -> tuple[asyncio.subprocess.Process, str]:
    token = "task2a-contract-shutdown"
    environment = dict(os.environ)
    environment.update(
        {
            # "CUDA_VISIBLE_DEVICES": "",
            "PAPEROS_LOCAL_INFERENCE_HOST": "127.0.0.1",
            "PAPEROS_LOCAL_INFERENCE_PORT": str(_LOCAL_INFERENCE_PORT),
            "PAPEROS_EMBEDDING_ENABLED": "true",
            "PAPEROS_EMBEDDING_MODEL_PATH": str(
                REPOSITORY_ROOT
                / "data"
                / "models"
                / "embedding"
                / "embeddinggemma-300M-Q8_0.gguf"
            ),
            "PAPEROS_EMBEDDING_MODEL_NAME": "default",
            "PAPEROS_EMBEDDING_DIMENSIONS": "768",
            "PAPEROS_EMBEDDING_MAX_TOKENS": "2048",
            "PAPEROS_RERANKER_ENABLED": "false",
            "PAPEROS_SHUTDOWN_TOKEN": token,
        }
    )
    process = await asyncio.create_subprocess_exec(
        "node",
        "dist/server.js",
        cwd=REPOSITORY_ROOT / "services" / "local_models",
        env=environment,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    endpoint = f"http://127.0.0.1:{_LOCAL_INFERENCE_PORT}"
    async with httpx.AsyncClient(timeout=2, trust_env=False) as client:
        for _ in range(240):
            if process.returncode is not None:
                raise RuntimeError(
                    f"BLOCKED: local embedding service exited with {process.returncode}"
                )
            try:
                response = await client.get(f"{endpoint}/health")
                if response.status_code == 200:
                    return process, token
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
    process.terminate()
    await process.wait()
    raise RuntimeError("BLOCKED: local embedding service did not become healthy")


async def _stop_embedding_service(
    process: asyncio.subprocess.Process,
    token: str,
) -> None:
    endpoint = f"http://127.0.0.1:{_LOCAL_INFERENCE_PORT}"
    if process.returncode is None:
        async with httpx.AsyncClient(timeout=5, trust_env=False) as client:
            try:
                await client.post(
                    f"{endpoint}/internal/shutdown",
                    headers={"x-paperos-shutdown-token": token},
                )
            except httpx.HTTPError:
                process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=10)
        except TimeoutError:
            process.kill()
            await process.wait()


def _manifest_index(paths: DataPaths) -> tuple[dict[str, str], set[str], str]:
    mapping: dict[str, str] = {}
    snapshot_ids: set[str] = set()
    dataset_names: set[str] = set()
    for path in sorted((paths.cognee / "manifests").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        snapshot_id = str(payload["canonical_snapshot_id"])
        snapshot_ids.add(snapshot_id)
        dataset = payload.get("dataset")
        if isinstance(dataset, dict) and dataset.get("name"):
            dataset_names.add(str(dataset["name"]))
        ids = payload.get("canonical_to_cognee_id")
        if isinstance(ids, dict):
            mapping.update({str(value): str(key) for key, value in ids.items()})
    _require(len(dataset_names) == 1, "Current Cognee dataset is ambiguous")
    _require(len(snapshot_ids) >= 2, "Current boundary has fewer than two snapshots")
    return mapping, snapshot_ids, next(iter(dataset_names))


def _real_chunk_graph(bundle: CanonicalBundle, chunks: list[Chunk]) -> DataPointGraph:
    snapshot = bundle.snapshot
    nodes = [
        PaperOSChunkDataPoint(
            id=cognee_snapshot_uuid(snapshot.id, chunk.id),
            canonical_id=chunk.id,
            canonical_snapshot_id=snapshot.id,
            source_file_id=snapshot.source_file_id,
            parse_run_id=snapshot.parse_run_id,
            document_id=bundle.document.id,
            section_id=chunk.section_id,
            section_path=chunk.section_path,
            text=effective_index_text(chunk),
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            source_chunk_ids=[chunk.id],
            derived_from_ids=chunk.element_ids,
        )
        for chunk in chunks
    ]
    _require(nodes, f"Retained revision {snapshot.id} has no real chunks")
    return DataPointGraph(nodes=nodes, relations=[])


def _current_chunks(bundle: CanonicalBundle) -> list[Chunk]:
    ingestion = RuntimeSettings().ingestion
    chunks, _ = build_chunks(
        document=bundle.document,
        snapshot_id=bundle.snapshot.id,
        sections=bundle.sections,
        elements=bundle.elements,
        references=bundle.references,
        target_tokens=ingestion.chunk_target_tokens,
        hard_max_tokens=ingestion.chunk_hard_max_tokens,
        overlap_tokens=ingestion.chunk_overlap_tokens,
        tokenizer=resolve_cognee_tokenizer(),
    )
    _require(chunks, f"Real revision {bundle.snapshot.id} produced no chunks")
    return chunks[:8]


def _register_live_rebuild_source(
    paths: DataPaths,
    repository: CanonicalRepository,
    bundle: CanonicalBundle,
    chunks: list[Chunk],
    source: Any,
) -> None:
    codec = DataPathCodec(paths.root)
    created_at = utc_now().isoformat()
    with sqlite3.connect(paths.registry_db) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT OR IGNORE INTO source_files (
                id, sha256, original_filename, stored_filename, media_type,
                size_bytes, storage_path, created_at, schema_version, id_version,
                source_url, user_metadata, dataset_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source.id,
                source.sha256,
                source.original_filename,
                source.stored_filename,
                source.media_type,
                source.size_bytes,
                codec.encode(paths.raw / source.id / source.stored_filename),
                created_at,
                source.schema_version,
                source.id_version,
                source.source_url,
                json.dumps(source.user_metadata) if source.user_metadata else None,
                bundle.snapshot.dataset_id,
            ),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO parse_runs (
                id, source_file_id, provider, backend, status,
                request_options, created_at, completed_at,
                artifact_manifest_path, schema_version, pipeline_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bundle.snapshot.parse_run_id,
                source.id,
                "contract",
                "contract",
                "completed",
                "{}",
                created_at,
                created_at,
                codec.encode(
                    paths.parsed / bundle.snapshot.parse_run_id / "manifest.json"
                ),
                "1.0",
                "contract",
            ),
        )
    local_bundle = bundle.model_copy(
        update={
            "snapshot": bundle.snapshot.model_copy(
                update={
                    "manifest_path": repository.snapshot_manifest_path(
                        bundle.snapshot.source_file_id,
                        bundle.snapshot.parse_run_id,
                        snapshot_id=bundle.snapshot.id,
                    )
                }
            ),
            "elements": [
                element.model_copy(update={"asset_path": None})
                for element in bundle.elements
            ],
        }
    )
    repository.save_snapshot(local_bundle)
    repository.save_chunks(bundle.snapshot.id, chunks)


async def _store_real_vector_revision(
    pipeline: CogneePipelineAdapter,
    compat: CogneeCompatibilityAdapter,
    bundle: CanonicalBundle,
    chunks: list[Chunk],
    source: Any,
) -> tuple[DataPointGraph, UUID]:
    graph = _real_chunk_graph(bundle, chunks)
    dataset = await compat.ensure_dataset(bundle.snapshot.dataset_id)
    data_id = await compat.register_data_item(
        dataset=dataset,
        source=source,
        snapshot_id=bundle.snapshot.id,
        document_id=bundle.document.id,
        title=bundle.document.title,
    )
    pipeline._persist_candidate_manifest(
        bundle=bundle,
        source=source,
        dataset=dataset,
        data_id=data_id,
    )
    from cognee.modules.pipelines.models import PipelineContext
    from cognee.modules.users.methods import get_default_user

    run_id = uuid4()
    user = await get_default_user()
    context = PipelineContext(
        user=user,
        data_item=PipelineItem(
            id=data_id,
            data_id=data_id,
            bundle=bundle,
            source=source,
        ),
        dataset=dataset,
        pipeline_run_id=run_id,
        pipeline_name="paperos_knowledge_ingestion",
    )
    async with await compat._dataset_scope(str(dataset.name)):
        await compat.add_data_points(
            graph.nodes,
            custom_edges=[],
            embed_triplets=False,
            ctx=context,
        )
    provenance = await compat.provenance_counts(
        dataset_id=UUID(str(dataset.id)),
        data_id=data_id,
        pipeline_run_id=run_id,
    )
    _require(provenance.provenance_node_count > 0, "Real vector nodes lack provenance")
    binding = CogneeDatasetBinding(
        user_id=str(dataset.owner_id),
        dataset_id=str(dataset.id),
        dataset_name=str(dataset.name),
        data_id=str(data_id),
        data_name=source.original_filename,
        pipeline_id="",
        pipeline_run_id=str(run_id),
        pipeline_name="paperos_knowledge_ingestion",
        provenance_backend=provenance.provenance_backend,
        provenance_node_count=provenance.provenance_node_count,
        provenance_edge_count=provenance.provenance_edge_count,
    )
    pipeline._persist_manifest(
        snapshot_id=bundle.snapshot.id,
        document_id=bundle.document.id,
        source=source,
        binding=binding,
        graph=graph,
    )
    return graph, data_id


async def live_vector_contract(root: Path) -> dict[str, object]:
    _require(_VECTOR_DATA.is_dir(), f"BLOCKED: validation pool missing: {_VECTOR_DATA}")
    process, token = await _start_embedding_service()
    paths = build_data_paths(root / "vector-data")
    StorageInitializer(paths).initialize()
    compat: CogneeCompatibilityAdapter | None = None
    try:
        retained_paths = build_data_paths(_VECTOR_DATA)
        retained_repository = CanonicalRepository(retained_paths)
        retained_registry = SourceRegistry(retained_paths)
        bundles = [retained_repository.get_bundle(item) for item in _REAL_REVISION_IDS]
        _require(
            len({bundle.document.id for bundle in bundles}) == 1,
            "Retained revisions do not belong to the same real document",
        )
        base = load_settings(REPOSITORY_ROOT / "config" / "paperos.example.toml")
        dataset_name = bundles[0].snapshot.dataset_id
        settings = base.model_copy(
            update={
                "data": base.data.model_copy(
                    update={"directory": paths.root, "dataset": dataset_name}
                ),
                "cognee": base.cognee.model_copy(
                    update={
                        "embedding": base.cognee.embedding.model_copy(
                            update={
                                "endpoint": (
                                    f"http://127.0.0.1:{_LOCAL_INFERENCE_PORT}/v1"
                                ),
                                "model": "openai/default",
                                "api_key": SecretStr("contract-local"),
                            }
                        )
                    }
                ),
                "local_inference": base.local_inference.model_copy(
                    update={"host": "127.0.0.1", "port": _LOCAL_INFERENCE_PORT}
                ),
            }
        )
        CogneeConfigurator().apply(settings, paths)
        import litellm

        litellm.drop_params = True
        compat = CogneeCompatibilityAdapter(paths)
        search = CogneeSearchAdapter(paths, compat)
        temp_repository = CanonicalRepository(paths)
        temp_registry = SourceRegistry(paths)
        temp_indexes = IndexManager(paths)
        pipeline = CogneePipelineAdapter(
            paths,
            temp_repository,
            temp_registry,
            ScholarlyRegistry(paths),
            compat,
            temp_indexes,
            _ForbiddenDependency(),  # type: ignore[arg-type]
            RuntimeSettings().ingestion,
        )
        data_ids: list[UUID] = []
        for bundle in bundles:
            chunks = _current_chunks(bundle)
            source = retained_registry.get_source(bundle.snapshot.source_file_id)
            _register_live_rebuild_source(
                paths,
                temp_repository,
                bundle,
                chunks,
                source,
            )
            await temp_indexes.index_bundle(bundle, chunks=chunks)
            _, data_id = await _store_real_vector_revision(
                pipeline,
                compat,
                bundle,
                chunks,
                source,
            )
            data_ids.append(data_id)
        _require(data_ids[0] != data_ids[1], "Real Cognee Data IDs reused one PDF")
        mapping, snapshot_ids, dataset_name = _manifest_index(paths)
        from cognee.infrastructure.databases.vector import get_vector_engine_async

        async with await compat._dataset_scope(dataset_name):
            engine = await get_vector_engine_async()
            direct = await engine.search(
                "PaperOSChunkDataPoint_text",
                query_text=_VECTOR_QUERY,
                query_vector=None,
                limit=4,
                include_payload=True,
            )
        _require(direct, "Real PaperOS vector collection returned no direct hits")
        direct_payloads = [hit.payload or {} for hit in direct]
        _require(
            all(payload.get("canonical_id") for payload in direct_payloads),
            "PaperOS vector payload dropped canonical_id; fields=" + repr([sorted(payload) for payload in direct_payloads]),
        )
        _require(
            all(
                str(payload.get("canonical_snapshot_id")) in snapshot_ids
                for payload in direct_payloads
            ),
            "PaperOS vector payload dropped canonical_snapshot_id",
        )
        raw = await compat.search_datapoint_vectors(
            _VECTOR_QUERY,
            dataset_name=dataset_name,
            search_type="PAPEROS_CHUNKS",
            canonical_ids=mapping,
            active_snapshot_ids=snapshot_ids,
            top_k=256,
        )
        _require(raw, "BLOCKED: real Cognee vector boundary returned no hits")
        first_positions: dict[str, int] = {}
        for rank, hit in enumerate(raw):
            if hit.canonical_snapshot_id is not None:
                first_positions.setdefault(hit.canonical_snapshot_id, rank)
        eligible = {
            snapshot_id: rank
            for snapshot_id, rank in first_positions.items()
            if rank >= 1
        }
        _require(
            eligible,
            "BLOCKED: current ranking cannot prove candidates exceed the pool",
        )
        active_snapshot_id, candidate_higher_count = max(
            eligible.items(),
            key=lambda item: item[1],
        )
        active_manifest = json.loads(
            (paths.cognee / "manifests" / f"{active_snapshot_id}.json").read_text(
                encoding="utf-8"
            )
        )
        active_mapping = active_manifest.get("canonical_to_cognee_id")
        _require(isinstance(active_mapping, dict), "Active manifest has no ID mapping")
        active_canonical_ids = {str(item) for item in active_mapping}
        filtered = await search.graph_search(
            _VECTOR_QUERY,
            dataset=dataset_name,
            top_k=1,
            active_snapshot_ids={active_snapshot_id},
        )
        _require(filtered, "Bounded overfetch did not reach an active vector hit")
        _require(
            all(hit.canonical_id in active_canonical_ids for hit in filtered),
            "Candidate vector crossed the final active filter",
        )
        temp_repository.activate_snapshot(active_snapshot_id)
        active_enrichment = (
            paths.cognee / "enrichment" / f"{active_snapshot_id}.json"
        )
        active_enrichment.parent.mkdir(parents=True, exist_ok=True)
        active_enrichment.write_text('{"contract": true}', encoding="utf-8")
        canonical_before_failure = set(temp_repository.list_all_snapshot_ids())
        failing_pipeline = _LocalRebuildPipeline(
            paths,
            temp_repository,
            temp_registry,
            temp_indexes,
            fail_build=True,
        )
        try:
            await DerivedDataRebuilder(
                paths,
                temp_repository,
                failing_pipeline,  # type: ignore[arg-type]
                StorageInitializer(paths),
            ).rebuild(active_snapshot_id)
        except RuntimeError as exc:
            _require(
                "injected isolated rebuild" in str(exc),
                "Real-boundary rebuild failed for an unexpected reason",
            )
        else:
            raise RuntimeError("Real-boundary injected rebuild unexpectedly succeeded")
        _require(
            temp_repository.active_snapshot_id(bundles[0].document.id)
            == active_snapshot_id,
            "Real-boundary rebuild failure changed active pointer",
        )
        _require(
            set(temp_repository.list_all_snapshot_ids()) == canonical_before_failure,
            "Real-boundary rebuild failure retained a canonical candidate",
        )
        filtered_after_rebuild_failure = await search.graph_search(
            _VECTOR_QUERY,
            dataset=dataset_name,
            top_k=1,
            active_snapshot_ids={active_snapshot_id},
        )
        _require(
            filtered_after_rebuild_failure
            and all(
                hit.canonical_id in active_canonical_ids
                for hit in filtered_after_rebuild_failure
            ),
            "Failed rebuild deleted or replaced active real vector data",
        )
        retired_snapshot_id = next(iter(snapshot_ids - {active_snapshot_id}))
        await compat.delete_document_data(retired_snapshot_id)
        filtered_after_cleanup = await search.graph_search(
            _VECTOR_QUERY,
            dataset=dataset_name,
            top_k=1,
            active_snapshot_ids={active_snapshot_id},
        )
        _require(
            filtered_after_cleanup
            and all(
                hit.canonical_id in active_canonical_ids
                for hit in filtered_after_cleanup
            ),
            "Retired vector cleanup deleted the active revision",
        )
        return {
            "status": "passed",
            "dataset": dataset_name,
            "raw_hit_count": len(raw),
            "active_snapshot_id": active_snapshot_id,
            "candidate_hits_before_active": candidate_higher_count,
            "pool_size": 1,
            "filtered_chunk_ids": [hit.canonical_id for hit in filtered],
            "retired_cleanup_preserved_active": True,
            "rebuild_failure_preserved_active_vector": True,
            "data_ids_distinct": True,
        }
    except Exception as exc:
        if "BLOCKED:" in str(exc):
            raise
        raise RuntimeError(f"BLOCKED: real Cognee retrieval boundary failed: {exc}") from exc
    finally:
        if compat is not None:
            await compat.aclose()
        await _stop_embedding_service(process, token)


async def run_contract() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="paperos-task2a-") as directory:
        root = Path(directory)
        scholarly = await scholarly_isolation_contract(root)
        rebuild = await safe_rebuild_contract(root)
        local = await local_revision_contract(root)
        vector = await live_vector_contract(root)
    return {
        "status": "passed",
        "scholarly": scholarly,
        "rebuild": rebuild,
        "local": local,
        "vector": vector,
    }


def main() -> None:
    try:
        report = asyncio.run(run_contract())
    except Exception as exc:
        if "BLOCKED:" in str(exc):
            print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, indent=2))
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
