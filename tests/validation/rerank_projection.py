"""Task 6A live acceptance for structured rerank projection.

This is deliberately a pipeline smoke, not a semantic quality benchmark.
It uses real PDFs, real providers, the production retrieval service, and writes
review artifacts under data/validation/rerank_projection_acceptance/output.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.application import create_application
from paperos_core.config import load_settings
from paperos_core.errors import PaperOSError
from paperos_core.ingestion.rerank_projection import RERANK_HARD_MAX_TOKENS
from paperos_core.ingestion.tokenization import AUTHORITATIVE_CHUNK_TOKENIZER
from paperos_core.retrieval.candidates import QueryRequest, QueryResponse
from paperos_core.retrieval.corpus import CorpusView

_DEFAULT_VALIDATION_ROOT = Path("data/validation/rerank_projection_acceptance")
_DEFAULT_CORPUS_ROOT = Path("data/validation/corpus/papers")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _trace_value(response: QueryResponse, name: str, default: Any = None) -> Any:
    trace = getattr(response, "trace", None)
    if trace is None:
        return default
    return getattr(trace, name, default)


def _trace_list(response: QueryResponse, name: str) -> list[str]:
    value = _trace_value(response, name, [])
    return list(value) if isinstance(value, (list, tuple)) else []


def _contains_any(text: str, snippets: list[str]) -> bool:
    folded = text.casefold()
    return any(snippet.casefold() in folded for snippet in snippets)


def _canonical_grounding(response: QueryResponse, corpus: CorpusView) -> bool:
    if not response.evidence or not response.candidates:
        return False
    evidence_ok = all(
        item.chunk_id in corpus.chunks and item.text == corpus.chunks[item.chunk_id].text
        for item in response.evidence
    )
    span_ids = {
        span.id
        for spans in corpus.rerank_spans_by_chunk.values()
        for span in spans
    }
    candidate_ok = all(
        item.chunk_id in corpus.chunks and item.id not in span_ids
        for item in response.candidates
    )
    return bool(response.provenance_complete and evidence_ok and candidate_ok)


def _work_filter_ok(
    response: QueryResponse,
    corpus: CorpusView,
    allowed_work_ids: set[str],
) -> bool:
    if not allowed_work_ids:
        return True
    evidence_work_ids = {
        corpus.work_id_by_document.get(item.document_id)
        for item in response.evidence
    }
    evidence_work_ids.discard(None)
    return bool(evidence_work_ids) and evidence_work_ids.issubset(allowed_work_ids)


def _structured_trace(response: QueryResponse) -> dict[str, Any]:
    return {
        "rerank_projection_version": _trace_value(response, "rerank_projection_version"),
        "first_rerank_span_count": _trace_value(response, "first_rerank_span_count", 0),
        "second_rerank_span_count": _trace_value(response, "second_rerank_span_count", 0),
        "first_reranked_chunk_ids": _trace_list(response, "first_reranked_chunk_ids"),
        "local_new_chunk_ids": _trace_list(response, "local_new_chunk_ids"),
        "semantic_new_chunk_ids": _trace_list(response, "semantic_new_chunk_ids"),
        "second_rerank_candidate_ids": _trace_list(response, "second_rerank_candidate_ids"),
    }


def _skipped_semantic_case(case: dict[str, Any]) -> dict[str, Any]:
    """Record the retained semantic diagnostic without executing it when disabled."""

    return {
        "id": case["id"],
        "mode": case["mode"],
        "hard_6a": False,
        "query": case["query"],
        "requested_work_symbols": list(case.get("work_ids", [])),
        "elapsed_seconds": 0.0,
        "status": "SKIPPED",
        "skip_reason": "semantic_enrichment_disabled",
        "pipeline_ok": False,
        "structured_first_rerank": False,
        "canonical_parent_chunk_grounding": False,
        "explicit_filter_ok": True,
        "expansion_path_ok": True,
        "stages": [],
        "structured_rerank_trace": {
            "rerank_projection_version": None,
            "first_rerank_span_count": 0,
            "second_rerank_span_count": 0,
            "first_reranked_chunk_ids": [],
            "local_new_chunk_ids": [],
            "semantic_new_chunk_ids": [],
            "second_rerank_candidate_ids": [],
        },
        "final_evidence_chunk_ids": [],
        "final_source_work_symbols": [],
        "quality_probe_status": "SKIPPED",
        "quality_matched_anchors": [],
        "quality_source_expectation_met": None,
        "answer": "",
    }


def _audit_projection(corpus: CorpusView) -> dict[str, Any]:
    """Verify every active parent Chunk has a rebuildable hard-bounded span set."""

    missing_chunk_ids = sorted(set(corpus.chunks) - set(corpus.rerank_spans_by_chunk))
    invalid_parent_ids: list[str] = []
    invalid_range_ids: list[str] = []
    hard_max_violation_ids: list[str] = []
    span_ids: set[str] = set()
    maximum_tokens = 0
    for chunk_id, spans in corpus.rerank_spans_by_chunk.items():
        chunk = corpus.chunks.get(chunk_id)
        if chunk is None:
            invalid_parent_ids.extend(span.id for span in spans)
            continue
        for span in spans:
            if span.id in span_ids:
                invalid_parent_ids.append(span.id)
            span_ids.add(span.id)
            try:
                scoring_text = span.scoring_text(chunk)
            except ValueError:
                invalid_range_ids.append(span.id)
                continue
            actual_tokens = AUTHORITATIVE_CHUNK_TOKENIZER.count_tokens(scoring_text)
            maximum_tokens = max(maximum_tokens, actual_tokens)
            if actual_tokens != span.token_count:
                invalid_range_ids.append(span.id)
            if actual_tokens > RERANK_HARD_MAX_TOKENS:
                hard_max_violation_ids.append(span.id)
    passed = not (
        missing_chunk_ids
        or invalid_parent_ids
        or invalid_range_ids
        or hard_max_violation_ids
    ) and bool(span_ids)
    return {
        "passed": passed,
        "active_chunk_count": len(corpus.chunks),
        "rerank_span_count": len(span_ids),
        "projection_versions": sorted(corpus.rerank_projection_versions),
        "maximum_span_tokens": maximum_tokens,
        "hard_max_tokens": RERANK_HARD_MAX_TOKENS,
        "missing_chunk_ids": missing_chunk_ids,
        "invalid_parent_span_ids": invalid_parent_ids,
        "invalid_range_span_ids": invalid_range_ids,
        "hard_max_violation_span_ids": hard_max_violation_ids,
    }


def _review_case(
    *,
    case: dict[str, Any],
    response: QueryResponse,
    corpus: CorpusView,
    work_by_symbol: dict[str, str],
    quality_probe: dict[str, Any] | None,
    elapsed_seconds: float,
) -> dict[str, Any]:
    requested_symbols = list(case.get("work_ids", []))
    requested_work_ids = {work_by_symbol[item] for item in requested_symbols}
    stages = set(response.stages)
    trace = _structured_trace(response)

    structured_first_rerank = (
        "first_rerank" in stages
        and bool(trace["rerank_projection_version"])
        and int(trace["first_rerank_span_count"] or 0) > 0
    )
    canonical_grounding = _canonical_grounding(response, corpus)
    filter_ok = _work_filter_ok(response, corpus, requested_work_ids)

    local_requested = bool(case["expansion"]["local"])
    semantic_requested = bool(case["expansion"]["semantic"])
    second_candidates = set(trace["second_rerank_candidate_ids"])

    if local_requested:
        local_new = trace["local_new_chunk_ids"]
        expansion_path_ok = (
            "local_post_hit_expansion" in stages
            and "second_rerank" in stages
            and int(trace["second_rerank_span_count"] or 0) > 0
            and bool(local_new)
            and set(local_new).issubset(second_candidates)
        )
    elif semantic_requested:
        semantic_new = trace["semantic_new_chunk_ids"]
        if semantic_new:
            expansion_path_ok = (
                "semantic_relation_expansion" in stages
                and "second_rerank" in stages
                and int(trace["second_rerank_span_count"] or 0) > 0
                and set(semantic_new).issubset(second_candidates)
            )
        else:
            expansion_path_ok = True  # soft diagnostic: graph may expose no new case
    else:
        expansion_path_ok = True

    pipeline_ok = bool(response.answer.strip()) and structured_first_rerank and canonical_grounding
    hard_runtime_ok = pipeline_ok and filter_ok and expansion_path_ok

    evidence_text = "\n".join(item.text for item in response.evidence)
    source_symbols = []
    reverse_work = {value: key for key, value in work_by_symbol.items()}
    for item in response.evidence:
        work_id = getattr(item, "source_work_id", None) or corpus.work_id_by_document.get(
            item.document_id
        )
        symbol = reverse_work.get(work_id, work_id)
        if symbol and symbol not in source_symbols:
            source_symbols.append(symbol)

    quality_probe_status = "NOT_CONFIGURED"
    matched_anchors: list[str] = []
    source_expectation_met: bool | None = None
    if quality_probe is not None:
        anchors = list(quality_probe.get("evidence_any_of", []))
        matched_anchors = [item for item in anchors if _contains_any(evidence_text, [item])]
        expected_sources = set(quality_probe.get("expected_source_works_any_of", []))
        source_expectation_met = bool(set(source_symbols).intersection(expected_sources))
        quality_probe_status = (
            "MATCH" if matched_anchors and source_expectation_met else "MISS"
        )

    status = "PASS" if hard_runtime_ok else ("FAIL" if case["hard_6a"] else "WARN")
    return {
        "id": case["id"],
        "mode": case["mode"],
        "hard_6a": bool(case["hard_6a"]),
        "query": case["query"],
        "requested_work_symbols": requested_symbols,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "status": status,
        "pipeline_ok": pipeline_ok,
        "structured_first_rerank": structured_first_rerank,
        "canonical_parent_chunk_grounding": canonical_grounding,
        "explicit_filter_ok": filter_ok,
        "expansion_path_ok": expansion_path_ok,
        "stages": response.stages,
        "structured_rerank_trace": trace,
        "final_evidence_chunk_ids": [item.chunk_id for item in response.evidence],
        "final_source_work_symbols": source_symbols,
        "quality_probe_status": quality_probe_status,
        "quality_matched_anchors": matched_anchors,
        "quality_source_expectation_met": source_expectation_met,
        "answer": response.answer,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Task 6A Structured RerankProjection Acceptance",
        "",
        f"Overall: **{report['overall_status']}**",
        "",
        f"Hard pipeline cases passed: **{report['hard_cases_passed']}**",
        "",
        "> Quality probe misses are diagnostic in Task 6A and do not block PASS.",
        "",
        "## Cases",
        "",
    ]
    for item in report["queries"]:
        trace = item["structured_rerank_trace"]
        lines.extend(
            [
                f"### {item['id']}",
                "",
                f"- 6A status: **{item['status']}**",
                f"- Quality probe: **{item['quality_probe_status']}**",
                f"- Pipeline: {item['pipeline_ok']}",
                f"- Canonical parent grounding: {item['canonical_parent_chunk_grounding']}",
                f"- Rerank projection version: {trace['rerank_projection_version']}",
                f"- First/second rerank span count: {trace['first_rerank_span_count']}/{trace['second_rerank_span_count']}",
                f"- Expansion path: {item['expansion_path_ok']}",
                f"- Evidence: {', '.join(item['final_evidence_chunk_ids'])}",
                "",
            ]
        )
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    validation_root = args.validation_root.resolve()
    config_root = validation_root / "config"
    output_root = args.output.resolve()
    runtime_root = output_root / "runtime"

    if args.reset and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    papers_config = _read_json(config_root / "papers.json")
    queries_config = _read_json(config_root / "queries.json")
    ground_truth = _read_json(config_root / "ground_truth.json")
    probe_by_case = {
        item["case_id"]: item for item in ground_truth.get("quality_probes", [])
    }

    corpus_root = args.corpus.resolve()
    for paper in papers_config["papers"]:
        pdf = corpus_root / paper["filename"]
        if not pdf.is_file():
            raise RuntimeError(f"Validation PDF missing: {pdf}")
        actual = _sha256(pdf)
        if actual != paper["sha256"]:
            raise RuntimeError(
                f"Validation PDF checksum mismatch: {paper['filename']} {actual}"
            )

    base = load_settings(args.config)
    settings = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={"directory": runtime_root, "dataset": args.dataset}
            ),
            "ingestion": base.ingestion.model_copy(
                update={
                    "semantic_enrichment_enabled": False,
                    "claim_enrichment_enabled": False,
                }
            ),
        }
    )

    if not settings.retrieval.rerank_enabled:
        raise RuntimeError("Task 6A acceptance requires retrieval.rerank_enabled=true")
    if settings.local_inference.cuda_devices != [6, 7]:
        raise RuntimeError("Task 6A acceptance requires local CUDA devices [6, 7]")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "6,7":
        raise RuntimeError("Task 6A acceptance requires CUDA_VISIBLE_DEVICES=6,7")

    application = create_application(settings)
    await application.start()
    try:
        local_health = await application.local_inference_client.health()
        if local_health.get("cuda_visible_devices") != "6,7":
            raise RuntimeError("Local inference runtime is not restricted to CUDA 6,7")
        work_by_symbol: dict[str, str] = {}
        document_by_symbol: dict[str, str] = {}
        ingest_seconds: dict[str, float] = {}
        disabled_ingestion_reports: list[dict[str, Any]] = []
        enrichment_root = application.paths.cognee / "enrichment"
        enrichment_artifacts_before = {
            path.name for path in enrichment_root.glob("*.json")
        }

        retained_by_filename = {
            application.registry.get_source(bundle.document.source_file_id).original_filename: bundle
            for bundle in application.canonical_repository.list_active_bundles()
            if bundle.snapshot.dataset_id == args.dataset
        }

        for paper in papers_config["papers"]:
            pdf = corpus_root / paper["filename"]
            retained = retained_by_filename.get(pdf.name)
            if retained is None:
                started = time.perf_counter()
                result = await application.services.ingestion.ingest_pdf_to_knowledge(
                    pdf, dataset=args.dataset
                )
                ingest_seconds[paper["id"]] = round(time.perf_counter() - started, 3)
                document_id = result.canonical_result.canonical.document.id
                if result.enrichment_path is not None:
                    raise RuntimeError("Disabled ingestion returned an enrichment artifact")
                semantic_counts = (
                    result.indexing.semantic_entity_count,
                    result.indexing.semantic_claim_count,
                    result.indexing.semantic_relation_count,
                )
                if result.indexing.semantic_enrichment_enabled or any(semantic_counts):
                    raise RuntimeError(
                        f"Disabled ingestion reported semantic output: {semantic_counts}"
                    )
                disabled_ingestion_reports.append(
                    {
                        "paper_id": paper["id"],
                        "semantic_counts": list(semantic_counts),
                        "enrichment_path": None,
                    }
                )
            else:
                ingest_seconds[paper["id"]] = 0.0
                document_id = retained.document.id

            work = application.scholarly_registry.work_for_document(document_id)
            if work is None:
                raise RuntimeError(f"No ScholarlyWork for {paper['id']}: {document_id}")
            work_by_symbol[paper["id"]] = work.id
            document_by_symbol[paper["id"]] = document_id

        enrichment_artifacts_after = {
            path.name for path in enrichment_root.glob("*.json")
        }
        new_enrichment_artifacts = sorted(
            enrichment_artifacts_after - enrichment_artifacts_before
        )
        removed_enrichment_artifacts = sorted(
            enrichment_artifacts_before - enrichment_artifacts_after
        )
        if new_enrichment_artifacts:
            raise RuntimeError(
                f"Disabled ingestion created enrichment artifacts: "
                f"{new_enrichment_artifacts}"
            )

        disabled_active_projections: list[dict[str, Any]] = []
        for active_bundle in application.canonical_repository.list_active_bundles():
            if active_bundle.snapshot.dataset_id != args.dataset:
                continue
            active_snapshot_id = active_bundle.snapshot.id
            graph_manifest_path = (
                application.paths.cognee / "manifests" / f"{active_snapshot_id}.json"
            )
            graph_manifest = _read_json(graph_manifest_path)
            active_enrichment_path = enrichment_root / f"{active_snapshot_id}.json"
            if graph_manifest.get("semantic_enrichment_enabled") is not False:
                raise RuntimeError(
                    f"Active projection does not report disabled enrichment: "
                    f"{active_snapshot_id}"
                )
            if active_enrichment_path.exists():
                raise RuntimeError(
                    f"Active disabled projection has an enrichment artifact: "
                    f"{active_snapshot_id}"
                )
            disabled_active_projections.append(
                {
                    "snapshot_id": active_snapshot_id,
                    "semantic_enrichment_enabled": False,
                    "enrichment_artifact": None,
                }
            )
        if len(disabled_active_projections) != len(papers_config["papers"]):
            raise RuntimeError("Not every 6A paper has an active disabled projection")

        corpus = CorpusView.load(
            application.paths,
            application.canonical_repository,
            application.registry,
            application.scholarly_registry,
        )
        projection_audit = _audit_projection(corpus)
        if not projection_audit["passed"]:
            raise RuntimeError("Active structured rerank projection audit failed")

        reviews: list[dict[str, Any]] = []
        for case in queries_config["cases"]:
            if (
                bool(case["expansion"]["semantic"])
                and not settings.ingestion.semantic_enrichment_enabled
            ):
                reviews.append(_skipped_semantic_case(case))
                continue
            work_ids = [work_by_symbol[item] for item in case.get("work_ids", [])] or None
            request = QueryRequest(
                query=case["query"],
                dataset=args.dataset,
                top_k=case.get("top_k"),
                work_ids=work_ids,
                expand_context=bool(case["expansion"]["local"]),
                expand_graph=bool(case["expansion"]["semantic"]),
            )
            started = time.perf_counter()
            response = await application.services.retrieval.query(request)
            elapsed = time.perf_counter() - started
            reviews.append(
                _review_case(
                    case=case,
                    response=response,
                    corpus=corpus,
                    work_by_symbol=work_by_symbol,
                    quality_probe=probe_by_case.get(case["id"]),
                    elapsed_seconds=elapsed,
                )
            )

        hard_reviews = [item for item in reviews if item["hard_6a"]]
        hard_cases_passed = bool(hard_reviews) and all(
            item["status"] == "PASS" for item in hard_reviews
        )
        report = {
            "overall_status": "PASS" if hard_cases_passed else "FAIL",
            "acceptance_scope": "Task 6A pipeline smoke; semantic quality probes are non-gating",
            "dataset": args.dataset,
            "hard_cases_passed": hard_cases_passed,
            "ingest_seconds": ingest_seconds,
            "semantic_enrichment_enabled": settings.ingestion.semantic_enrichment_enabled,
            "disabled_ingestion_reports": disabled_ingestion_reports,
            "disabled_active_projections": disabled_active_projections,
            "new_enrichment_artifacts": new_enrichment_artifacts,
            "removed_retired_enrichment_artifacts": removed_enrichment_artifacts,
            "runtime_work_ids": work_by_symbol,
            "runtime_document_ids": document_by_symbol,
            "projection_audit": projection_audit,
            "queries": reviews,
            "counts": {
                "papers": len(papers_config["papers"]),
                "chunks": len(corpus.chunks),
                "hard_cases": len(hard_reviews),
                "quality_probe_matches": sum(
                    item["quality_probe_status"] == "MATCH" for item in reviews
                ),
            },
        }
        _write_json(output_root / "acceptance.json", report)
        _write_json(output_root / "review" / "queries.json", reviews)
        (output_root / "acceptance.md").write_text(_markdown(report), encoding="utf-8")
        return report
    finally:
        await application.aclose()


def _external_boundary_failure(exc: BaseException) -> bool:
    external_codes = {
        "mineru_quota_error",
        "mineru_timeout",
        "mineru_provider_error",
        "local_inference_unavailable",
        "semantic_enrichment_error",
    }
    external_exception_names = {
        "APIConnectionError",
        "APITimeoutError",
        "InstructorRetryException",
    }
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if (
            isinstance(current, PaperOSError)
            and current.code in external_codes
        ) or type(current).__name__ in external_exception_names:
            return True
        current = current.__cause__ or current.__context__
    return False


def _failure_report(output_root: Path, exc: Exception) -> dict[str, Any]:
    blocked = _external_boundary_failure(exc)
    status = "BLOCKED" if blocked else "FAIL"
    report = {
        "overall_status": status,
        "hard_cases_passed": False,
        "failure_stage": "live_validation",
        "external_boundary_blocked": blocked,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "acceptance.json", report)
    (output_root / "acceptance.md").write_text(
        "# Task 6A Structured RerankProjection Acceptance\n\n"
        f"Overall: **{status}**\n\n"
        f"Error: {report['error_type']}: {report['error']}\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/paperos.toml"))
    parser.add_argument(
        "--validation-root", type=Path, default=_DEFAULT_VALIDATION_ROOT
    )
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_VALIDATION_ROOT / "output",
    )
    parser.add_argument("--dataset", default="rerank_projection_acceptance")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    try:
        report = asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001 - acceptance must persist failure
        report = _failure_report(args.output.resolve(), exc)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["overall_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
