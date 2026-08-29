"""Direct contracts for persisted rerank projections and reranker IDs.

Run from the repository root without pytest:

    python tests/contract/test_rerank_projection_integrity.py
"""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.domain.canonical import (
    CanonicalBundle,
    CanonicalSnapshot,
    Chunk,
    Document,
    Element,
    RerankProjection,
    RerankSpan,
    SourceSpan,
)
from paperos_core.domain.documents import utc_now
from paperos_core.domain.enums import ElementType
from paperos_core.domain.ids import canonical_snapshot_id
from paperos_core.errors import CanonicalValidationError, LocalInferenceResponseError
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.tokenization import AUTHORITATIVE_CHUNK_TOKENIZER
from paperos_core.paths import build_data_paths
from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.rerank import rerank_candidates
from paperos_core.runtime.local_inference.schemas import RerankResult
from paperos_core.storage.initializer import StorageInitializer
from paperos_core.storage.path_refs import DataPathCodec


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _repository_fixture(root: Path, text: str, case_name: str) -> tuple[
    CanonicalRepository, Chunk
]:
    paths = build_data_paths(root)
    StorageInitializer(paths).initialize()
    repository = CanonicalRepository(paths)
    source_id = f"source_{case_name}"
    parse_id = f"parse_{case_name}"
    created_at = utc_now().isoformat()
    codec = DataPathCodec(paths.root)
    with sqlite3.connect(paths.registry_db) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO source_files (
                id, sha256, original_filename, stored_filename, media_type,
                size_bytes, storage_path, created_at, schema_version, id_version,
                dataset_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_id,
                "a" * 64,
                "contract.pdf",
                "source.pdf",
                "application/pdf",
                1,
                codec.encode(paths.raw / source_id / "source.pdf"),
                created_at,
                "1.0",
                "1",
                "rerank-integrity-contract",
            ),
        )
        connection.execute(
            """
            INSERT INTO parse_runs (
                id, source_file_id, provider, backend, status, request_options,
                created_at, completed_at, artifact_manifest_path, schema_version,
                pipeline_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                parse_id,
                source_id,
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
    snapshot_id = canonical_snapshot_id(parse_id)
    document_id = f"document_{case_name}"
    element_id = f"element_{case_name}"
    snapshot = CanonicalSnapshot(
        id=snapshot_id,
        source_file_id=source_id,
        parse_run_id=parse_id,
        document_id=document_id,
        dataset_id="rerank-integrity-contract",
        manifest_path=repository.snapshot_manifest_path(
            source_id,
            parse_id,
            snapshot_id=snapshot_id,
        ),
    )
    document = Document(
        id=document_id,
        source_file_id=source_id,
        parse_run_id=parse_id,
        canonical_snapshot_id=snapshot_id,
        language="en",
        title="Rerank integrity contract",
    )
    element = Element(
        id=element_id,
        document_id=document_id,
        canonical_snapshot_id=snapshot_id,
        element_type=ElementType.PARAGRAPH,
        order=0,
        text=text,
        source_span=SourceSpan(artifact_id=f"artifact_{case_name}", item_index=0),
    )
    bundle = CanonicalBundle(
        snapshot=snapshot,
        document=document,
        sections=[],
        elements=[element],
        references=[],
    )
    chunk = Chunk(
        id=f"chunk_{case_name}",
        document_id=document_id,
        canonical_snapshot_id=snapshot_id,
        text=text,
        order=0,
        element_ids=[element_id],
        token_count=AUTHORITATIVE_CHUNK_TOKENIZER.count_tokens(text),
    )
    repository.save_snapshot(bundle)
    repository.save_chunks(snapshot_id, [chunk])
    return repository, chunk


def _projection(
    chunk: Chunk,
    ranges: list[tuple[int, int]],
    *,
    stored_token_counts: list[int] | None = None,
) -> RerankProjection:
    spans: list[RerankSpan] = []
    for ordinal, (start, end) in enumerate(ranges):
        actual = AUTHORITATIVE_CHUNK_TOKENIZER.count_tokens(chunk.text[start:end])
        token_count = (
            stored_token_counts[ordinal]
            if stored_token_counts is not None
            else actual
        )
        spans.append(
            RerankSpan(
                id=f"rerank_span_{chunk.id}_{ordinal}",
                parent_chunk_id=chunk.id,
                canonical_snapshot_id=chunk.canonical_snapshot_id,
                ordinal=ordinal,
                character_start_in_chunk=start,
                character_end_in_chunk=end,
                unit_start=ordinal,
                unit_end=ordinal + 1,
                token_count=token_count,
            )
        )
    return RerankProjection(
        snapshot_id=chunk.canonical_snapshot_id,
        spans=spans,
    )


def _assert_persisted_projection(
    case_name: str,
    text: str,
    ranges: list[tuple[int, int]],
    *,
    stored_token_counts: list[int] | None = None,
    expected_error: str | None = None,
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"paperos-{case_name}-") as temporary:
        repository, chunk = _repository_fixture(Path(temporary), text, case_name)
        projection = _projection(
            chunk,
            ranges,
            stored_token_counts=stored_token_counts,
        )
        if expected_error is None:
            repository.save_rerank_projection(projection)
            loaded = repository.get_chunk_projection(chunk.canonical_snapshot_id)
            _require(
                loaded.rerank_projection == projection,
                f"{case_name}: valid persisted projection changed during loading",
            )
            return

        # Corruption contracts intentionally bypass the validated writer, then use
        # the real persisted loading boundary that production activation consumes.
        store = repository.rerank_projection_store_path(chunk.canonical_snapshot_id)
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(projection.model_dump_json(indent=2), encoding="utf-8")
        try:
            repository.get_chunk_projection(chunk.canonical_snapshot_id)
        except CanonicalValidationError as exc:
            _require(
                expected_error in exc.message,
                f"{case_name}: unexpected validation error: {exc.message}",
            )
        else:
            raise RuntimeError(f"{case_name}: corrupt persisted projection was accepted")


def _projection_contracts() -> None:
    text = "x" * 200
    _assert_persisted_projection("valid", text, [(0, 100), (100, 200)])
    _assert_persisted_projection(
        "gap", text, [(0, 100), (120, 200)], expected_error="contain a gap"
    )
    _assert_persisted_projection(
        "missing_prefix", text, [(10, 200)], expected_error="missing a parent Chunk prefix"
    )
    _assert_persisted_projection(
        "missing_suffix", text, [(0, 190)], expected_error="missing a parent Chunk suffix"
    )
    _assert_persisted_projection(
        "overlap", text, [(0, 120), (100, 200)], expected_error="ranges overlap"
    )
    _assert_persisted_projection(
        "token_mismatch",
        text,
        [(0, 200)],
        stored_token_counts=[199],
        expected_error="persisted token count does not match",
    )
    _assert_persisted_projection(
        "actual_hard_max",
        "x" * 385,
        [(0, 385)],
        stored_token_counts=[384],
        expected_error="exceeds its hard maximum",
    )


class _RerankClient:
    def __init__(self, returned_ids: list[str]) -> None:
        self.returned_ids = returned_ids

    async def rerank(
        self,
        query: str,
        candidate_ids: list[str],
        texts: list[str],
        *,
        limit: int,
    ) -> list[RerankResult]:
        _require(query == "contract query", "reranker query changed")
        _require(limit == len(candidate_ids), "reranker span limit changed")
        _require(len(texts) == len(candidate_ids), "reranker scoring input changed")
        return [_rerank_result(candidate_id, index) for index, candidate_id in enumerate(self.returned_ids)]


def _rerank_result(candidate_id: str, index: int) -> RerankResult:
    return RerankResult(
        candidate_id=candidate_id,
        original_index=index,
        relevance_score=max(0.1, 0.9 - index * 0.1),
        final_rank=index + 1,
        document_token_count=10,
        input_token_count=15,
        effective_input_token_count=15,
        model_max_input_tokens=512,
        query_token_count=2,
        special_prompt_token_count=3,
        truncated=False,
        window_count=1,
        winning_window_document_token_count=10,
        winning_window_index=0,
        winning_window_text="contract scoring text",
    )


def _rerank_fixture() -> tuple[Candidate, Any, list[str]]:
    text = "abcdefghij" * 3
    snapshot_id = "snapshot_rerank_response_contract"
    chunk = Chunk(
        id="chunk_rerank_response_contract",
        document_id="document_rerank_response_contract",
        canonical_snapshot_id=snapshot_id,
        text=text,
        order=0,
        element_ids=["element_rerank_response_contract"],
        token_count=len(text),
    )
    projection = _projection(chunk, [(0, 10), (10, 20), (20, 30)])
    spans = projection.spans
    corpus = SimpleNamespace(
        chunks={chunk.id: chunk},
        rerank_spans_by_chunk={chunk.id: spans},
    )
    candidate = Candidate(
        id="candidate_rerank_response_contract",
        object_id=chunk.id,
        object_type="chunk",
        document_id=chunk.document_id,
        source_file_id="source_rerank_response_contract",
        source_filename="contract.pdf",
        canonical_snapshot_id=snapshot_id,
        chunk_id=chunk.id,
        text=chunk.text,
        channels=["contract"],
    )
    return candidate, corpus, [span.id for span in spans]


async def _expect_rerank_error(
    case_name: str,
    returned_ids: list[str],
    *,
    expected_reason: str | None = None,
) -> None:
    candidate, corpus, _ = _rerank_fixture()
    try:
        await rerank_candidates(
            _RerankClient(returned_ids),  # type: ignore[arg-type]
            "contract query",
            [candidate],
            corpus=corpus,  # type: ignore[arg-type]
            limit=1,
        )
    except LocalInferenceResponseError as exc:
        if expected_reason is not None:
            _require(
                exc.details.get("reason") == expected_reason,
                f"{case_name}: wrong response-contract reason",
            )
    else:
        raise RuntimeError(f"{case_name}: invalid reranker response IDs were accepted")


async def _reranker_response_contracts() -> None:
    candidate, corpus, expected_ids = _rerank_fixture()
    passed = await rerank_candidates(
        _RerankClient(list(reversed(expected_ids))),  # type: ignore[arg-type]
        "contract query",
        [candidate],
        corpus=corpus,  # type: ignore[arg-type]
        limit=1,
    )
    _require(passed.span_count == 3, "valid reranker ID set did not pass")

    await _expect_rerank_error(
        "duplicate",
        [expected_ids[0], expected_ids[0], expected_ids[1]],
        expected_reason="rerank_candidate_id_mismatch",
    )
    await _expect_rerank_error(
        "unknown",
        [expected_ids[0], expected_ids[1], "unknown_span_id"],
        expected_reason="rerank_candidate_id_mismatch",
    )
    await _expect_rerank_error("missing", expected_ids[:2])
    await _expect_rerank_error(
        "same_count_missing_and_duplicate",
        [expected_ids[1], expected_ids[1], expected_ids[2]],
        expected_reason="rerank_candidate_id_mismatch",
    )


def main() -> None:
    _projection_contracts()
    asyncio.run(_reranker_response_contracts())
    print("PASS: rerank projection integrity and response ID contracts")


if __name__ == "__main__":
    main()
