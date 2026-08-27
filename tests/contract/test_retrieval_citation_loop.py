"""Task 3 retained-canonical retrieval text and CitationMention contracts.

Run from the repository root without pytest or MinerU:

    conda run -n paperos python tests/contract/test_retrieval_citation_loop.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.adapters.cognee.datapoints import PaperOSChunkDataPoint
from paperos_core.adapters.cognee.models import canonical_to_datapoints
from paperos_core.adapters.cognee.pipeline_tasks import (
    academic_chunk_task,
    scholarly_identity_task,
)
from paperos_core.domain.canonical import (
    CanonicalBundle,
    CanonicalSnapshot,
    Document,
    Element,
    ReferenceEntry,
    Section,
)
from paperos_core.domain.knowledge import SemanticEnrichment
from paperos_core.indexes.manager import IndexManager
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.retrieval_text import (
    _strip_leading_marker,
    effective_index_text,
)
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry
from paperos_core.paths import DataPaths, build_data_paths
from paperos_core.storage.initializer import StorageInitializer
from paperos_core.storage.path_refs import DataPathCodec

_CANONICAL_ROOT = (
    REPOSITORY_ROOT
    / "data"
    / "validation"
    / "chunk"
    / "output"
    / "canonical"
    / "src_0a6d556646aec7d48b873ddd1d800a5b"
    / "snapshot_b74905b26696d818f57381e61331302f"
)
_SOURCE_PDF = (
    REPOSITORY_ROOT / "data" / "validation" / "corpus" / "papers" / "isogeometric.pdf"
)
_DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "data" / "validation" / "retrieval_citation" / "output"
)
_TARGET_TOKENS = 900
_HARD_MAX_TOKENS = 1200
_OVERLAP_TOKENS = 0


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json_lines(path: Path, model: type[Any]) -> list[Any]:
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_retained_bundle(repository: CanonicalRepository) -> CanonicalBundle:
    _require(_CANONICAL_ROOT.is_dir(), "Retained canonical validation corpus is missing")
    _require(_SOURCE_PDF.is_file(), "Validation PDF pool is missing isogeometric.pdf")
    snapshot = CanonicalSnapshot.model_validate_json(
        (_CANONICAL_ROOT / "snapshot.json").read_text(encoding="utf-8")
    )
    snapshot = snapshot.model_copy(
        update={
            "manifest_path": repository.snapshot_manifest_path(
                snapshot.source_file_id,
                snapshot.parse_run_id,
                snapshot_id=snapshot.id,
            )
        }
    )
    return CanonicalBundle(
        snapshot=snapshot,
        document=Document.model_validate_json(
            (_CANONICAL_ROOT / "document.json").read_text(encoding="utf-8")
        ),
        sections=_json_lines(_CANONICAL_ROOT / "sections.jsonl", Section),
        elements=[
            element.model_copy(
                update={
                    "asset_path": (
                        repository.paths.parsed
                        / snapshot.source_file_id
                        / snapshot.parse_run_id
                        / "artifacts"
                        / "images"
                        / element.asset_path.name
                    )
                }
            )
            if element.asset_path is not None
            else element
            for element in _json_lines(
                _CANONICAL_ROOT / "elements.jsonl", Element
            )
        ],
        references=_json_lines(_CANONICAL_ROOT / "references.jsonl", ReferenceEntry),
        warnings=json.loads(
            (_CANONICAL_ROOT / "warnings.json").read_text(encoding="utf-8")
        ),
    )


def _insert_source_and_parse(paths: DataPaths, bundle: CanonicalBundle) -> None:
    codec = DataPathCodec(paths.root)
    snapshot = bundle.snapshot
    created_at = snapshot.created_at.isoformat()
    source_path = paths.raw / snapshot.source_file_id / "source.pdf"
    parsed_manifest = (
        paths.parsed
        / snapshot.source_file_id
        / snapshot.parse_run_id
        / "manifest.json"
    )
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
                snapshot.source_file_id,
                "a" * 64,
                _SOURCE_PDF.name,
                "source.pdf",
                "application/pdf",
                _SOURCE_PDF.stat().st_size,
                codec.encode(source_path),
                created_at,
                snapshot.schema_version,
                snapshot.id_version,
                None,
                None,
                snapshot.dataset_id,
            ),
        )
        connection.execute(
            """
            INSERT INTO parse_runs (
                id, source_file_id, provider, backend, status,
                request_options, created_at, completed_at,
                artifact_manifest_path, schema_version, pipeline_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.parse_run_id,
                snapshot.source_file_id,
                "retained-canonical",
                "validation",
                "completed",
                "{}",
                created_at,
                created_at,
                codec.encode(parsed_manifest),
                snapshot.schema_version,
                snapshot.pipeline_version,
            ),
        )


async def _reproject(
    bundle: CanonicalBundle,
    *,
    repository: CanonicalRepository,
    registry: ScholarlyRegistry,
) -> tuple[Any, Any]:
    chunked = await academic_chunk_task(
        [SimpleNamespace(bundle=bundle)],
        repository=repository,
        chunk_target_tokens=_TARGET_TOKENS,
        chunk_hard_max_tokens=_HARD_MAX_TOKENS,
        chunk_overlap_tokens=_OVERLAP_TOKENS,
    )
    _require(len(chunked) == 1, "Academic projection returned an unexpected batch")
    source_text = {chunk.id: chunk.text for chunk in chunked[0].projection.chunks}
    identity_bound = await scholarly_identity_task(
        chunked,
        scholarly_registry=registry,
        repository=repository,
    )
    _require(len(identity_bound) == 1, "Scholarly projection returned an unexpected batch")
    projection = identity_bound[0].projection
    _require(
        source_text == {chunk.id: chunk.text for chunk in projection.chunks},
        "Scholarly binding modified authoritative Chunk.text",
    )
    return identity_bound[0], chunked[0].projection


def _empty_enrichment(chunk_ids: list[str]) -> SemanticEnrichment:
    return SemanticEnrichment(
        entities=[],
        claims=[],
        relations=[],
        model="paperos/retained-canonical-contract",
        provider="paperos",
        model_version="1",
        prompt_name="no-claim-contract",
        prompt_version="1",
        prompt_sha256="0" * 64,
        covered_chunk_ids=chunk_ids,
        uncovered_chunk_ids=[],
        coverage_ratio=1.0,
    )


def _projection_signature(projection: Any) -> dict[str, Any]:
    return {
        "chunks": [
            {
                "id": chunk.id,
                "text": chunk.text,
                "retrieval_text": chunk.retrieval_text,
                "document_region": chunk.document_region,
                "section_path": chunk.section_path,
                "citation_mention_ids": chunk.citation_mention_ids,
                "citation_reference_entry_ids": chunk.citation_reference_entry_ids,
                "citation_work_ids": chunk.citation_work_ids,
                "metadata": chunk.metadata,
            }
            for chunk in projection.chunks
        ],
        "mentions": [
            {
                "id": mention.id,
                "chunk_id": mention.chunk_id,
                "atomic_key": mention.atomic_key,
                "reference_entry_id": mention.reference_entry_id,
                "resolved_work_id": mention.resolved_work_id,
                "resolution_status": mention.resolution_status,
                "failure_reason": mention.failure_reason,
            }
            for mention in projection.citation_mentions
        ],
    }


def _require_same_projection(
    first: dict[str, Any],
    second: dict[str, Any],
) -> None:
    for category in ("chunks", "mentions"):
        left_items = first[category]
        right_items = second[category]
        _require(
            len(left_items) == len(right_items),
            f"Reprojection changed {category} count",
        )
        for left, right in zip(left_items, right_items, strict=True):
            if left == right:
                continue
            differing = sorted(
                key
                for key in set(left) | set(right)
                if left.get(key) != right.get(key)
            )
            values = {
                key: {"first": left.get(key), "second": right.get(key)}
                for key in differing
            }
            raise RuntimeError(
                f"Reprojection changed {category} {left.get('id')}: "
                f"{json.dumps(values, ensure_ascii=False, sort_keys=True)}"
            )


def _validate_headers(bundle: CanonicalBundle, projection: Any) -> None:
    for chunk in projection.chunks:
        retrieval = effective_index_text(chunk)
        _require(retrieval == (chunk.retrieval_text or "").strip(), "Index text bypass")
        _require(
            retrieval.count(f"Paper:\n{bundle.document.title}") == 1,
            f"Paper title metadata is missing or duplicated in {chunk.id}",
        )
        if bundle.document.year is not None:
            _require(
                retrieval.count(f"Year:\n{bundle.document.year}") == 1,
                f"Year metadata is missing or duplicated in {chunk.id}",
            )
        if chunk.document_region:
            _require(
                retrieval.count(f"Region:\n{chunk.document_region}") == 1,
                f"Document region is missing or duplicated in {chunk.id}",
            )
        breadcrumb = chunk.section_path or chunk.major_section_title
        if breadcrumb:
            _require(
                retrieval.count(f"Section:\n{breadcrumb}") == 1,
                f"Section breadcrumb is missing or duplicated in {chunk.id}",
            )


def _validate_citation_binding(
    projection: Any,
    registry: ScholarlyRegistry,
) -> dict[str, Any]:
    mentions_by_chunk: dict[str, list[Any]] = {}
    for mention in projection.citation_mentions:
        if mention.chunk_id:
            mentions_by_chunk.setdefault(mention.chunk_id, []).append(mention)

    all_work_ids = {
        mention.resolved_work_id
        for mention in projection.citation_mentions
        if mention.resolved_work_id is not None
    }
    for chunk in projection.chunks:
        mentions = mentions_by_chunk.get(chunk.id, [])
        expected = list(
            dict.fromkeys(
                mention.resolved_work_id
                for mention in mentions
                if mention.resolved_work_id is not None
            )
        )
        _require(
            chunk.citation_work_ids == expected,
            f"Chunk citation Work identities do not match its mentions: {chunk.id}",
        )
        retrieval = chunk.retrieval_text or ""
        for work_id in expected:
            _require(
                f"Work ID: {work_id}" in retrieval,
                f"Resolved Work is absent from citing Chunk retrieval text: {work_id}",
            )
        for work_id in all_work_ids - set(expected):
            _require(
                f"Work ID: {work_id}" not in retrieval,
                f"Resolved Work leaked into a non-citing Chunk: {work_id}",
            )

    range_mentions = [
        mention
        for mention in projection.citation_mentions
        if mention.surface_text == "[2–4]"
    ]
    _require(
        {mention.atomic_key for mention in range_mentions} == {"2", "3", "4"},
        "The real [2–4] citation did not expand to three atomic mentions",
    )
    range_targets: dict[str, str] = {}
    for mention in range_mentions:
        _require(mention.reference_entry_id is not None, "Range target lost ReferenceEntry")
        _require(mention.resolved_work_id is not None, "Range target lost final Work")
        active_work = registry.work_for_reference(mention.reference_entry_id)
        _require(active_work is not None, "Active registry lacks range target Work")
        _require(
            mention.resolved_work_id == active_work.id,
            "CitationMention does not store the active registry's final Work ID",
        )
        _require(
            registry.canonicalize_work_id(mention.resolved_work_id)
            == mention.resolved_work_id,
            "CitationMention stored a redirected loser Work ID",
        )
        range_targets[mention.atomic_key] = mention.resolved_work_id

    unresolved = [
        mention
        for mention in projection.citation_mentions
        if mention.reference_entry_id is not None
        and mention.resolved_work_id is None
    ]
    _require(unresolved, "Real corpus did not retain any conservative unresolved Work")
    for mention in unresolved:
        active_work = registry.work_for_reference(mention.reference_entry_id)
        _require(
            active_work is None,
            "Unresolved CitationMention was guessed into an active Work",
        )
    return {
        "range_surface": "[2–4]",
        "range_targets": dict(sorted(range_targets.items())),
        "resolved_mention_count": len(projection.citation_mentions) - len(unresolved),
        "unresolved_work_mention_count": len(unresolved),
    }


def _validate_index_inputs(
    bundle: CanonicalBundle,
    projection: Any,
    scholarly: Any,
    manager: IndexManager,
) -> dict[str, int]:
    expected = {
        chunk.id: effective_index_text(chunk) for chunk in projection.chunks
    }
    with sqlite3.connect(manager.lexical.path) as connection:
        lexical = {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT object_id, text FROM lexical_records "
                "WHERE canonical_snapshot_id = ? AND object_type = 'chunk'",
                (bundle.snapshot.id,),
            ).fetchall()
        }
    _require(lexical == expected, "FTS did not index effective_index_text() exactly")

    graph = canonical_to_datapoints(
        bundle,
        projection.chunks,
        _empty_enrichment(list(expected)),
        scholarly,
    )
    vector_input = {
        node.canonical_id: node.text
        for node in graph.nodes
        if isinstance(node, PaperOSChunkDataPoint)
    }
    _require(
        vector_input == expected,
        "Cognee vector DataPoint mapping did not use effective_index_text() exactly",
    )
    return {
        "lexical_chunk_count": len(lexical),
        "vector_chunk_count": len(vector_input),
    }


def _validate_reference_marker(bundle: CanonicalBundle) -> None:
    reference = next(
        item for item in bundle.references if item.citation_label == "2"
    )
    with_marker = _strip_leading_marker(reference.raw_text, atomic_key="2")
    without_marker = _strip_leading_marker(with_marker, atomic_key="2")
    _require(with_marker.startswith("J.W. Barrett"), "Reference marker removed first author")
    _require(
        without_marker.startswith("J.W. Barrett"),
        "Unkeyed reference cleaning removed the first author token",
    )


async def run_contract(output_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="paperos-task3-contract-") as temporary:
        paths = build_data_paths(Path(temporary))
        StorageInitializer(paths).initialize()
        repository = CanonicalRepository(paths)
        registry = ScholarlyRegistry(paths)
        bundle = _load_retained_bundle(repository)
        _insert_source_and_parse(paths, bundle)
        repository.save_snapshot(bundle)
        _validate_reference_marker(bundle)

        first, initial_first = await _reproject(
            bundle,
            repository=repository,
            registry=registry,
        )
        _validate_headers(bundle, first.projection)
        manager = IndexManager(paths)
        await manager.index_bundle(bundle, chunks=first.projection.chunks)
        index_report = _validate_index_inputs(
            bundle,
            first.projection,
            first.scholarly,
            manager,
        )
        previous = registry.publish_candidate(
            bundle.snapshot.id,
            repository,
        )
        _require(previous is None, "First activation unexpectedly replaced a revision")
        persisted = repository.get_chunk_projection(bundle.snapshot.id)
        _require(
            _projection_signature(persisted) == _projection_signature(first.projection),
            "Final ChunkProjection was not persisted before activation",
        )
        citation_report = _validate_citation_binding(persisted, registry)

        active_pointer = repository.active_snapshot_id(bundle.document.id)
        active_registry = registry.identity_snapshot()
        active_signature = _projection_signature(persisted)
        candidate = repository.create_rebuild_candidate(bundle.snapshot.id)
        candidate_first, initial_candidate_first = await _reproject(
            candidate,
            repository=repository,
            registry=registry,
        )
        candidate_second, initial_candidate_second = await _reproject(
            candidate,
            repository=repository,
            registry=registry,
        )
        _require_same_projection(
            _projection_signature(candidate_first.projection),
            _projection_signature(candidate_second.projection),
        )
        _require(
            {chunk.id: chunk.text for chunk in initial_candidate_first.chunks}
            == {chunk.id: chunk.text for chunk in initial_candidate_second.chunks}
            == {chunk.id: chunk.text for chunk in initial_first.chunks},
            "Repeated retained-canonical reprojection changed Chunk.text",
        )
        _require(
            [chunk.retrieval_text for chunk in candidate_second.projection.chunks]
            == [chunk.retrieval_text for chunk in persisted.chunks],
            "Retained-canonical candidate changed deterministic retrieval text",
        )
        _require(
            [chunk.metadata for chunk in candidate_second.projection.chunks]
            == [chunk.metadata for chunk in persisted.chunks],
            "Retained-canonical candidate duplicated or changed metadata",
        )
        _require(
            repository.active_snapshot_id(bundle.document.id) == active_pointer,
            "Candidate reprojection changed the active pointer before publication",
        )
        _require(
            registry.identity_snapshot() == active_registry,
            "Candidate reprojection polluted the active scholarly registry",
        )

        injected_failure = False
        try:
            raise RuntimeError("injected candidate reprojection failure")
        except RuntimeError:
            injected_failure = True
            registry.discard_candidate(candidate.snapshot.id)
            repository.chunk_store_path(candidate.snapshot.id).unlink(missing_ok=True)
            repository.citation_mention_store_path(candidate.snapshot.id).unlink(
                missing_ok=True
            )
            repository.cleanup_snapshot(candidate.snapshot.id)
        _require(injected_failure, "Candidate failure injection did not execute")
        _require(
            repository.active_snapshot_id(bundle.document.id) == active_pointer,
            "Failed candidate changed the active pointer",
        )
        _require(
            registry.identity_snapshot() == active_registry,
            "Failed candidate changed active scholarly mappings",
        )
        _require(
            _projection_signature(repository.get_chunk_projection(bundle.snapshot.id))
            == active_signature,
            "Failed candidate changed the active ChunkProjection",
        )

        report: dict[str, Any] = {
            "status": "PASS",
            "source": {
                "pdf": str(_SOURCE_PDF.relative_to(REPOSITORY_ROOT)),
                "canonical_snapshot": bundle.snapshot.id,
                "document_id": bundle.document.id,
                "title": bundle.document.title,
                "year": bundle.document.year,
            },
            "retained_canonical": {
                "mineru_call_count": 0,
                "reprojection_count": 2,
                "deterministic": True,
                "chunk_text_unchanged": True,
                "metadata_not_duplicated": True,
                "chunk_count": len(persisted.chunks),
            },
            "retrieval_text": {
                "title": True,
                "year": bundle.document.year is not None,
                "region": all(chunk.document_region for chunk in persisted.chunks),
                "breadcrumb_chunk_count": sum(
                    bool(chunk.section_path or chunk.major_section_title)
                    for chunk in persisted.chunks
                ),
                **index_report,
            },
            "citations": citation_report,
            "candidate_failure": {
                "injected": True,
                "active_pointer_unchanged": True,
                "active_registry_unchanged": True,
                "active_projection_unchanged": True,
            },
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "acceptance.json"
    markdown_path = output_dir / "acceptance.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    citation = report["citations"]
    retained = report["retained_canonical"]
    retrieval = report["retrieval_text"]
    markdown_path.write_text(
        "\n".join(
            [
                "# Task 3 Retained-Canonical Acceptance",
                "",
                f"- Status: **{report['status']}**",
                f"- Paper: {report['source']['title']} ({report['source']['year']})",
                f"- Canonical snapshot: `{report['source']['canonical_snapshot']}`",
                f"- Chunks: {retained['chunk_count']}",
                f"- Retained reprojections: {retained['reprojection_count']}",
                f"- MinerU calls: {retained['mineru_call_count']}",
                (
                    f"- Lexical/vector inputs: {retrieval['lexical_chunk_count']}/"
                    f"{retrieval['vector_chunk_count']}"
                ),
                f"- Resolved citation mentions: {citation['resolved_mention_count']}",
                (
                    f"- Conservatively unresolved Work mentions: "
                    f"{citation['unresolved_work_mention_count']}"
                ),
                "",
                "## [2–4] final Work targets",
                "",
                *[
                    f"- [{key}] → `{value}`"
                    for key, value in citation["range_targets"].items()
                ],
                "",
                (
                    "Candidate failure left the active pointer, scholarly registry, and "
                    "ChunkProjection unchanged."
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )
    report["artifacts"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = asyncio.run(run_contract(args.output_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
