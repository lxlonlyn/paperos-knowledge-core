"""Direct Task 2B contracts for pre-truncation query filters.

Run from the repository root without pytest:

    conda run -n paperos python tests/contract/test_query_filter_contracts.py
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import SecretStr

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from test_active_canonical_revision import (
    _current_chunks,
    _ForbiddenDependency,
    _manifest_index,
    _register_live_rebuild_source,
    _start_embedding_service,
    _stop_embedding_service,
    _store_real_vector_revision,
)

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.adapters.cognee.configurator import CogneeConfigurator
from paperos_core.adapters.cognee.pipeline import CogneePipelineAdapter
from paperos_core.adapters.cognee.search import CogneeSearchAdapter
from paperos_core.config import RuntimeSettings, load_settings
from paperos_core.domain.canonical import CanonicalBundle, Chunk
from paperos_core.indexes.manager import IndexManager
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry
from paperos_core.paths import DataPaths, build_data_paths
from paperos_core.retrieval.candidates import QueryRequest, VectorSearchDiagnostics
from paperos_core.retrieval.corpus import CorpusView
from paperos_core.retrieval.expansion import local_neighbor_expand
from paperos_core.retrieval.lexical import lexical_retrieve
from paperos_core.retrieval.semantic import semantic_retrieve
from paperos_core.retrieval.service import NO_EVIDENCE_MODEL, RetrievalService
from paperos_core.storage.initializer import StorageInitializer

_VALIDATION_DATA = REPOSITORY_ROOT / "data" / "validation" / "retrieval" / "output"
_VECTOR_QUERY = "Explicit flows for implicit surfaces shape morphing deformation"
_LOCAL_INFERENCE_PORT = 18081
_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{3,}")


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class _ContractLLM:
    model = "paperos/filter-contract"

    def __init__(self) -> None:
        self.call_count = 0

    async def synthesize_answer(
        self,
        *,
        prompt: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        _require(prompt and evidence, "Synthesis did not receive real evidence")
        self.call_count += 1
        return f"contract answer [{evidence[0]['evidence_id']}]"


class _ForbiddenBoundary:
    def __getattr__(self, name: str) -> Any:
        raise RuntimeError(f"Empty filter touched forbidden boundary: {name}")


async def _configure_vector_runtime(
    paths: DataPaths,
    *,
    dataset_name: str,
) -> RuntimeSettings:
    base = load_settings(REPOSITORY_ROOT / "config" / "paperos.example.toml")
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
            "retrieval": base.retrieval.model_copy(
                update={
                    "candidate_pool_size": 1,
                    "top_k": 1,
                    "rerank_enabled": False,
                    "synthesis_max_input_tokens": 48_000,
                }
            ),
        }
    )
    CogneeConfigurator().apply(settings, paths)
    import litellm

    litellm.drop_params = True
    return settings


def _candidate_chunks(chunks: list[Chunk], snapshot_id: str) -> list[Chunk]:
    return [
        chunk.model_copy(update={"canonical_snapshot_id": snapshot_id})
        for chunk in chunks
    ]


async def _store_revision(
    *,
    paths: DataPaths,
    repository: CanonicalRepository,
    indexes: IndexManager,
    pipeline: CogneePipelineAdapter,
    compat: CogneeCompatibilityAdapter,
    bundle: CanonicalBundle,
    chunks: list[Chunk],
    source: Any,
) -> None:
    repository.save_chunks(bundle.snapshot.id, chunks)
    await indexes.index_bundle(bundle, chunks=chunks)
    await _store_real_vector_revision(pipeline, compat, bundle, chunks, source)


def _lexical_case(
    corpus: CorpusView,
    indexes: IndexManager,
) -> tuple[str, str, int]:
    active_snapshot_ids = corpus.active_snapshot_ids
    token_documents: dict[str, set[str]] = {}
    for chunk in corpus.chunks.values():
        for token in set(_TOKEN_PATTERN.findall(chunk.text.casefold())):
            token_documents.setdefault(token, set()).add(chunk.document_id)
    for token, documents in sorted(token_documents.items()):
        if len(documents) < 2:
            continue
        global_rows = indexes.lexical.search(
            f'"{token}"',
            active_snapshot_ids=active_snapshot_ids,
            limit=256,
        )
        first_rank: dict[str, int] = {}
        for rank, row in enumerate(global_rows):
            first_rank.setdefault(str(row["document_id"]), rank)
        for document_id, rank in sorted(first_rank.items(), key=lambda item: -item[1]):
            if rank < 1:
                continue
            allowed_snapshots = corpus.snapshot_ids_for_documents({document_id})
            filtered = indexes.lexical.search(
                f'"{token}"',
                active_snapshot_ids=allowed_snapshots,
                allowed_document_ids={document_id},
                limit=1,
            )
            if filtered:
                return token, document_id, rank
    raise RuntimeError(
        "BLOCKED: validation FTS ranking cannot place an allowed document after pool=1"
    )


def _lexical_top_k_query(corpus: CorpusView, document_id: str, top_k: int) -> str:
    chunks = [
        chunk for chunk in corpus.chunks.values() if chunk.document_id == document_id
    ]
    frequencies: Counter[str] = Counter()
    for chunk in chunks:
        frequencies.update(set(_TOKEN_PATTERN.findall(chunk.text.casefold())))
    common = [token for token, count in frequencies.most_common() if count >= top_k]
    if common:
        return common[0]
    representatives: list[str] = []
    for chunk in chunks:
        tokens = _TOKEN_PATTERN.findall(chunk.text.casefold())
        if tokens:
            representatives.append(tokens[0])
        if len(representatives) >= top_k:
            break
    _require(
        len(representatives) >= top_k,
        "BLOCKED: allowed real document has too few searchable chunks",
    )
    return " ".join(representatives)


def _assert_no_evidence(response: Any, *, requested_kind: str) -> None:
    _require(response.answer_model == NO_EVIDENCE_MODEL, "Wrong no-evidence model")
    _require(response.evidence == [] and response.candidates == [], "Empty filter leaked hits")
    _require(
        response.stages == ["explicit_filters", "no_evidence"],
        "Empty filter executed retrieval stages",
    )
    _require(response.trace.lexical_request_limits == [], "Empty filter called FTS")
    _require(response.trace.vector_request_limits == [], "Empty filter called vector")
    _require(response.trace.first_reranked_chunk_ids == [], "Empty filter called rerank")
    _require(response.replay.replay_text == "", "Empty filter called synthesis")
    _require(requested_kind, "Missing no-evidence case label")


async def run_contract() -> dict[str, object]:
    _require(
        _VALIDATION_DATA.is_dir(),
        f"BLOCKED: validation pool missing: {_VALIDATION_DATA}",
    )
    process, token = await _start_embedding_service()
    compat: CogneeCompatibilityAdapter | None = None
    with tempfile.TemporaryDirectory(prefix="paperos-task2b-") as directory:
        paths = build_data_paths(Path(directory) / "data")
        StorageInitializer(paths).initialize()
        try:
            retained_paths = build_data_paths(_VALIDATION_DATA)
            retained_repository = CanonicalRepository(retained_paths)
            retained_registry = SourceRegistry(retained_paths)
            retained_bundles = [
                retained_repository.get_bundle(snapshot_id)
                for snapshot_id in retained_repository.list_all_snapshot_ids()
            ]
            bundles_by_document: dict[str, CanonicalBundle] = {}
            for bundle in retained_bundles:
                bundles_by_document.setdefault(bundle.document.id, bundle)
            bundles = list(bundles_by_document.values())
            _require(
                len(bundles) >= 2,
                "BLOCKED: validation pool has fewer than two real documents",
            )
            dataset_name = bundles[0].snapshot.dataset_id
            bundles = [
                bundle for bundle in bundles if bundle.snapshot.dataset_id == dataset_name
            ]
            _require(
                len(bundles) >= 2,
                "BLOCKED: validation pool has fewer than two documents in one dataset",
            )

            settings = await _configure_vector_runtime(
                paths,
                dataset_name=dataset_name,
            )
            compat = CogneeCompatibilityAdapter(paths)
            search = CogneeSearchAdapter(paths, compat)
            repository = CanonicalRepository(paths)
            registry = SourceRegistry(paths)
            scholarly = ScholarlyRegistry(paths)
            indexes = IndexManager(paths)
            pipeline = CogneePipelineAdapter(
                paths,
                repository,
                registry,
                scholarly,
                compat,
                indexes,
                _ForbiddenDependency(),  # type: ignore[arg-type]
                settings.ingestion,
            )

            chunks_by_document: dict[str, list[Chunk]] = {}
            sources_by_document: dict[str, Any] = {}
            original_snapshot_by_document: dict[str, str] = {}
            for bundle in bundles:
                chunks = _current_chunks(bundle)
                source = retained_registry.get_source(bundle.snapshot.source_file_id)
                _register_live_rebuild_source(
                    paths,
                    repository,
                    bundle,
                    chunks,
                    source,
                )
                await indexes.index_bundle(bundle, chunks=chunks)
                await _store_real_vector_revision(
                    pipeline,
                    compat,
                    bundle,
                    chunks,
                    source,
                )
                scholarly.resolve_candidate_bundle(bundle, chunks)
                scholarly.publish_candidate(bundle.snapshot.id, repository)
                chunks_by_document[bundle.document.id] = chunks
                sources_by_document[bundle.document.id] = source
                original_snapshot_by_document[bundle.document.id] = bundle.snapshot.id

            revision_document_id = bundles[0].document.id
            original_snapshot_id = original_snapshot_by_document[revision_document_id]
            original_chunks = chunks_by_document[revision_document_id]
            source = sources_by_document[revision_document_id]

            candidate = repository.create_rebuild_candidate(original_snapshot_id)
            candidate_chunks = _candidate_chunks(original_chunks, candidate.snapshot.id)
            await _store_revision(
                paths=paths,
                repository=repository,
                indexes=indexes,
                pipeline=pipeline,
                compat=compat,
                bundle=candidate,
                chunks=candidate_chunks,
                source=source,
            )

            replacement = repository.create_rebuild_candidate(original_snapshot_id)
            replacement_chunks = _candidate_chunks(
                original_chunks,
                replacement.snapshot.id,
            )
            await _store_revision(
                paths=paths,
                repository=repository,
                indexes=indexes,
                pipeline=pipeline,
                compat=compat,
                bundle=replacement,
                chunks=replacement_chunks,
                source=source,
            )
            scholarly.resolve_candidate_bundle(replacement, replacement_chunks)
            previous = scholarly.publish_candidate(replacement.snapshot.id, repository)
            _require(previous == original_snapshot_id, "Replacement switched wrong revision")

            corpus = CorpusView.load(paths, repository, registry, scholarly)
            _require(
                corpus.active_snapshot_ids == set(repository.list_active_snapshot_ids()),
                "Corpus active snapshot set diverged from repository",
            )
            _require(
                {original_snapshot_id, candidate.snapshot.id, replacement.snapshot.id}
                <= set(repository.list_all_snapshot_ids()),
                "Old/candidate/active revision coexistence setup failed",
            )

            lexical_token, lexical_document_id, lexical_global_rank = _lexical_case(
                corpus,
                indexes,
            )
            lexical_snapshots = corpus.snapshot_ids_for_documents(
                {lexical_document_id}
            )
            lexical_diagnostics: dict[str, list[int]] = {}
            lexical_hits = lexical_retrieve(
                indexes.lexical,
                corpus,
                [lexical_token],
                limit=1,
                document_ids={lexical_document_id},
                active_snapshot_ids=lexical_snapshots,
                diagnostics=lexical_diagnostics,
            )
            _require(lexical_hits, "FTS allowlist did not reach the post-pool hit")
            _require(
                {hit.document_id for hit in lexical_hits} == {lexical_document_id},
                "FTS returned a disallowed document",
            )
            _require(
                lexical_diagnostics["request_limits"]
                and set(lexical_diagnostics["request_limits"]) == {1},
                "FTS did not receive candidate pool=1",
            )

            manifest_mapping, manifest_snapshots, _ = _manifest_index(paths)
            raw_vector = await compat.search_datapoint_vectors(
                _VECTOR_QUERY,
                dataset_name=dataset_name,
                search_type="PAPEROS_CHUNKS",
                canonical_ids=manifest_mapping,
                active_snapshot_ids=manifest_snapshots,
                top_k=10_000,
            )
            _require(raw_vector, "BLOCKED: real vector boundary returned no hits")
            first_vector_rank: dict[str, int] = {}
            for rank, hit in enumerate(raw_vector):
                if hit.canonical_snapshot_id is not None:
                    first_vector_rank.setdefault(hit.canonical_snapshot_id, rank)
            eligible_vector_snapshots = {
                snapshot_id: rank
                for snapshot_id, rank in first_vector_rank.items()
                if snapshot_id in corpus.active_snapshot_ids and rank >= 1
            }
            _require(
                eligible_vector_snapshots,
                "BLOCKED: vector ranking cannot place an allowed active hit after pool=1",
            )
            vector_snapshot_id, vector_global_rank = max(
                eligible_vector_snapshots.items(),
                key=lambda item: item[1],
            )
            vector_document_id = repository.get_snapshot(vector_snapshot_id).document_id
            vector_diagnostics = VectorSearchDiagnostics()
            vector_hits = await semantic_retrieve(
                search,
                corpus,
                _VECTOR_QUERY,
                dataset_name=dataset_name,
                limit=1,
                document_ids={vector_document_id},
                active_snapshot_ids={vector_snapshot_id},
                diagnostics=vector_diagnostics,
            )
            _require(vector_hits, "Bounded vector overfetch did not reach allowed hit")
            _require(
                {hit.document_id for hit in vector_hits} == {vector_document_id},
                "Vector returned a disallowed document",
            )
            _require(
                vector_diagnostics.request_limits
                and vector_diagnostics.request_limits[0] > 1,
                "Vector did not overfetch beyond candidate pool",
            )

            vector_top_k = min(4, len([
                chunk
                for chunk in corpus.chunks.values()
                if chunk.document_id == vector_document_id
            ]))
            _require(vector_top_k >= 2, "BLOCKED: vector document has too few chunks")
            expanded_vector_diagnostics = VectorSearchDiagnostics()
            expanded_vector_hits = await semantic_retrieve(
                search,
                corpus,
                _VECTOR_QUERY,
                dataset_name=dataset_name,
                limit=vector_top_k,
                document_ids={vector_document_id},
                active_snapshot_ids={vector_snapshot_id},
                diagnostics=expanded_vector_diagnostics,
            )
            _require(
                len(expanded_vector_hits) == vector_top_k,
                "Eligible vector results were truncated below top_k",
            )

            lexical_top_k = min(4, len([
                chunk
                for chunk in corpus.chunks.values()
                if chunk.document_id == lexical_document_id
            ]))
            lexical_top_k_query = _lexical_top_k_query(
                corpus,
                lexical_document_id,
                lexical_top_k,
            )
            expanded_lexical_diagnostics: dict[str, list[int]] = {}
            expanded_lexical_hits = lexical_retrieve(
                indexes.lexical,
                corpus,
                [lexical_top_k_query],
                limit=lexical_top_k,
                document_ids={lexical_document_id},
                active_snapshot_ids=lexical_snapshots,
                diagnostics=expanded_lexical_diagnostics,
            )
            _require(
                len(expanded_lexical_hits) == lexical_top_k,
                "Eligible lexical results were truncated below top_k",
            )
            _require(
                set(expanded_lexical_diagnostics["request_limits"])
                == {lexical_top_k},
                "Expanded top_k did not reach FTS",
            )

            work = scholarly.work_for_document(vector_document_id)
            _require(work is not None, "Active vector document has no published Work")
            llm = _ContractLLM()
            service = RetrievalService(
                settings,
                paths,
                repository,
                registry,
                scholarly,
                search,
                compat,
                indexes,
                _ForbiddenDependency(),  # type: ignore[arg-type]
                llm,  # type: ignore[arg-type]
            )
            combined = await service.query(
                QueryRequest(
                    query=_VECTOR_QUERY,
                    document_ids=[vector_document_id],
                    work_ids=[work.id],
                    top_k=vector_top_k,
                )
            )
            _require(
                len(combined.candidates) == vector_top_k,
                "Service did not return top_k eligible candidates",
            )
            _require(
                {candidate.document_id for candidate in combined.candidates}
                == {vector_document_id},
                "RRF/dedup admitted a disallowed candidate",
            )
            _require(
                combined.trace.requested_document_ids == [vector_document_id]
                and combined.trace.requested_work_ids == [work.id]
                and combined.trace.resolved_work_document_ids == [vector_document_id]
                and combined.trace.applied_document_ids == [vector_document_id]
                and combined.trace.applied_snapshot_ids == [vector_snapshot_id],
                "Trace allowlist does not match actual filters",
            )
            _require(
                combined.trace.candidate_pool_sizes == [vector_top_k],
                "Service pool did not expand to top_k",
            )
            _require(
                combined.trace.lexical_request_limits
                and min(combined.trace.lexical_request_limits) == vector_top_k,
                "Expanded pool did not reach lexical retrieval",
            )
            _require(
                combined.trace.vector_request_limits
                and combined.trace.vector_request_limits[0] >= vector_top_k,
                "Expanded pool did not reach vector retrieval",
            )
            _require(llm.call_count == 1, "Evidence synthesis call count changed")

            work_only = await service.query(
                QueryRequest(query=_VECTOR_QUERY, work_ids=[work.id], top_k=1)
            )
            _require(
                work_only.trace.applied_document_ids == [vector_document_id],
                "Work filter did not resolve to its active document",
            )

            forbidden_service = RetrievalService(
                settings,
                paths,
                repository,
                registry,
                scholarly,
                _ForbiddenBoundary(),  # type: ignore[arg-type]
                _ForbiddenBoundary(),  # type: ignore[arg-type]
                _ForbiddenBoundary(),  # type: ignore[arg-type]
                _ForbiddenBoundary(),  # type: ignore[arg-type]
                _ForbiddenBoundary(),  # type: ignore[arg-type]
            )
            unknown_document = await forbidden_service.query(
                QueryRequest(query="unknown", document_ids=["document_unknown"])
            )
            _assert_no_evidence(unknown_document, requested_kind="document")
            unknown_work = await forbidden_service.query(
                QueryRequest(query="unknown", work_ids=["work_unknown"])
            )
            _assert_no_evidence(unknown_work, requested_kind="work")
            other_document_id = next(
                document_id
                for document_id in corpus.bundles
                if document_id != vector_document_id
            )
            empty_intersection = await forbidden_service.query(
                QueryRequest(
                    query="empty intersection",
                    document_ids=[other_document_id],
                    work_ids=[work.id],
                )
            )
            _assert_no_evidence(empty_intersection, requested_kind="intersection")
            _require(
                empty_intersection.trace.resolved_work_document_ids
                == [vector_document_id]
                and empty_intersection.trace.applied_document_ids == [],
                "Document/work filters were unioned instead of intersected",
            )

            expansion_case = next(
                (
                    (seed, neighbors)
                    for chunk in corpus.chunks.values()
                    if chunk.document_id == vector_document_id
                    for seed in [
                        corpus.candidate_for_chunk(
                            chunk.id,
                            channel="contract_seed",
                            score=1.0,
                        )
                    ]
                    for neighbors in [
                        local_neighbor_expand(
                            corpus,
                            [seed],
                            document_ids={vector_document_id},
                        )
                    ]
                    if neighbors
                ),
                None,
            )
            _require(expansion_case is not None, "BLOCKED: no real local expansion seed")
            _expansion_seed, expanded = expansion_case
            _require(
                all(
                    candidate.document_id == vector_document_id
                    and candidate.canonical_snapshot_id == vector_snapshot_id
                    for candidate in expanded
                ),
                "Post-hit expansion crossed document/active boundary",
            )

            replacement_manifest = json.loads(
                (
                    paths.cognee
                    / "manifests"
                    / f"{replacement.snapshot.id}.json"
                ).read_text(encoding="utf-8")
            )
            replacement_node_ids = {
                str(node_id)
                for node_id in replacement_manifest["canonical_to_cognee_id"].values()
            }
            revision_vector_diagnostics = VectorSearchDiagnostics()
            revision_vector_hits = await search.graph_search(
                _VECTOR_QUERY,
                dataset=dataset_name,
                top_k=1,
                active_snapshot_ids={replacement.snapshot.id},
                diagnostics=revision_vector_diagnostics,
            )
            _require(
                revision_vector_hits
                and all(
                    hit.node_id in replacement_node_ids
                    for hit in revision_vector_hits
                ),
                "Old/candidate vector revision crossed active filter",
            )
            revision_fts_query = _lexical_top_k_query(
                corpus,
                revision_document_id,
                1,
            )
            active_fts_rows = indexes.lexical.search(
                revision_fts_query,
                active_snapshot_ids=corpus.active_snapshot_ids,
                allowed_document_ids={revision_document_id},
                limit=256,
            )
            _require(
                active_fts_rows
                and all(
                    row["canonical_snapshot_id"] == replacement.snapshot.id
                    for row in active_fts_rows
                ),
                "Old/candidate FTS revision crossed active filter",
            )

            return {
                "status": "passed",
                "documents": len(corpus.bundles),
                "lexical": {
                    "allowed_document_id": lexical_document_id,
                    "global_rank": lexical_global_rank,
                    "pool": 1,
                    "top_k": lexical_top_k,
                    "request_limits": expanded_lexical_diagnostics["request_limits"],
                },
                "vector": {
                    "allowed_document_id": vector_document_id,
                    "global_rank": vector_global_rank,
                    "pool": 1,
                    "top_k": vector_top_k,
                    "request_limits": expanded_vector_diagnostics.request_limits,
                    "raw_hit_counts": expanded_vector_diagnostics.raw_hit_counts,
                    "filtered_counts": expanded_vector_diagnostics.filtered_hit_counts,
                    "backend_exhausted": expanded_vector_diagnostics.backend_exhausted,
                },
                "filters": {
                    "work_id": work.id,
                    "intersection": "passed",
                    "unknown_document": "no_evidence",
                    "unknown_work": "no_evidence",
                    "trace_applied_snapshot_ids": combined.trace.applied_snapshot_ids,
                },
                "revisions": {
                    "old": original_snapshot_id,
                    "candidate": candidate.snapshot.id,
                    "active": replacement.snapshot.id,
                    "active_only": True,
                },
                "post_hit_expansion": "active_allowed_only",
            }
        except Exception as exc:
            if "BLOCKED:" in str(exc):
                raise
            raise RuntimeError(f"BLOCKED: real 2B retrieval boundary failed: {exc}") from exc
        finally:
            if compat is not None:
                await compat.aclose()
            await _stop_embedding_service(process, token)


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
