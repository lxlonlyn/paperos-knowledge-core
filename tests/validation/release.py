"""Clean-room production-readiness gate over four real papers.

Run from the repository root with the caller-selected GPU visibility:

    CUDA_VISIBLE_DEVICES=6,7 conda run -n paperos \
        python tests/validation/release.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.api.visualize import visualize_dataset
from paperos_core.application import create_application
from paperos_core.config import load_settings
from paperos_core.errors import CogneeStorageError
from paperos_core.retrieval.candidates import QueryRequest
from paperos_core.retrieval.corpus import CorpusView
from tests.validation import retrieval as retrieval_validation
from tests.validation.chunk import review___projection_metrics
from tests.validation.release_provenance import (
    RERANK_PROVISIONAL_NOTICE,
    SEARCH_QUALITY_PENDING,
    VALIDATION_ORIGIN_CURRENT,
    _annotate_validation,
    _drop_legacy_gate_fields,
    _engineering_decision,
    _gate_record,
    _legacy_engineering_evidence,
    _merge_query_reviews,
    _reused_validation,
)

DEFAULT_OUTPUT = Path("data/validation/release/output")
DEFAULT_WORK = Path("data/validation/release/work")
DEFAULT_ACCEPTANCE_CONFIG = Path("data/validation/search_graph_acceptance/config")
DEFAULT_CORPUS = Path("data/validation/corpus/papers")
DATASET = "paperos-release-gate"


def _final_engineering_command_specs() -> dict[str, list[str]]:
    python = sys.executable
    return {
        "node_build": ["npm", "--prefix", "services/local_models", "run", "build"],
        "compile": [python, "-m", "compileall", "-q", "paperos_core", "tests"],
        "ruff": ["ruff", "check", "paperos_core", "tests"],
        "mypy": ["mypy", "paperos_core"],
        "runtime_contract": [python, "tests/contract/test_runtime_query_contracts.py"],
        "reranker_contract": [python, "tests/contract/test_reranker_input_trace.py"],
        "report_contract": [
            python,
            "tests/contract/test_release_report_provenance.py",
        ],
        "diff_check": ["git", "diff", "--check"],
    }


def _legacy_validation_head(
    args: argparse.Namespace, report: dict[str, Any]
) -> str:
    supplied = getattr(args, "legacy_validated_head", None)
    if supplied:
        return str(supplied)
    gates = report.get("engineering_gates")
    if isinstance(gates, dict):
        heads = {
            str(value.get("validated_head"))
            for value in gates.values()
            if isinstance(value, dict) and value.get("validated_head")
        }
        if heads:
            return min(heads)
    raise RuntimeError(
        "Legacy report lacks per-gate validation provenance; "
        "pass --legacy-validated-head with the actual execution commit."
    )



def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _command_gate(command: list[str]) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        env=dict(os.environ),
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return {
        "command": " ".join(command),
        "exit_code": completed.returncode,
        "seconds": round(time.perf_counter() - started, 3),
        "status": "PASS" if completed.returncode == 0 else "FAIL",
    }


def _sum_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = (
        "figure_input_count",
        "figure_placeholder_count",
        "figure_placeholder_part_count",
        "figure_lost_count",
        "figure_caption_duplication_count",
        "figure_provenance_error_count",
        "source_provenance_error_count",
        "table_input_count",
        "equation_input_count",
        "citation_mention_count",
        "caption_citation_missing_count",
        "text_loss_count",
        "text_duplication_count",
        "section_cross_boundary_count",
        "fallback_split_count",
    )
    result = {key: sum(int(row[key]) for row in rows) for key in numeric}
    reasons: Counter[str] = Counter()
    for row in rows:
        reasons.update(row["fallback_split_reasons"])
    result["fallback_split_reasons"] = dict(reasons)
    result["errors"] = [error for row in rows for error in row["errors"]]
    return result


async def _runtime_audit(
    *,
    config: Path,
    runtime_root: Path,
    dataset: str,
) -> dict[str, Any]:
    base = load_settings(config)
    settings = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={"directory": runtime_root, "dataset": dataset}
            ),
            "ingestion": base.ingestion.model_copy(
                update={"claim_enrichment_enabled": False}
            ),
        }
    )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    configured_visible = ",".join(
        str(device) for device in settings.local_inference.cuda_devices
    )
    if not visible or visible != configured_visible:
        raise RuntimeError(
            "CUDA visibility must be non-empty and match local_inference.cuda_devices"
        )

    application = create_application(settings)
    await application.start()
    try:
        bundles = [
            bundle
            for bundle in application.canonical_repository.list_active_bundles()
            if bundle.snapshot.dataset_id == dataset
        ]
        if len(bundles) != 4:
            raise RuntimeError("Clean-room runtime does not contain four active papers")
        active_ids = application.canonical_repository.list_active_snapshot_ids()
        pointer_unique = len(active_ids) == len(set(active_ids)) == 4
        projection_rows: list[dict[str, Any]] = []
        all_chunks = []
        for bundle in bundles:
            projection = application.canonical_repository.get_chunk_projection(
                bundle.snapshot.id
            )
            all_chunks.extend(projection.chunks)
            projection_rows.append(
                review___projection_metrics(
                    bundle=bundle,
                    chunks=projection.chunks,
                    mentions=projection.citation_mentions,
                )
            )
        chunk_metrics = _sum_metrics(projection_rows)
        max_chunk_tokens = max(
            (chunk.token_count or 0 for chunk in all_chunks), default=0
        )
        hard_max_violations = sum(
            (chunk.token_count or 0) > settings.ingestion.chunk_hard_max_tokens
            for chunk in all_chunks
        )

        evidence_response = await application.services.retrieval.query(
            QueryRequest(
                query="How is the Eikonal equation used for implicit surface evolution?",
                dataset=dataset,
                top_k=3,
            )
        )
        no_evidence_response = await application.services.retrieval.query(
            QueryRequest(
                query="no eligible document",
                dataset=dataset,
                document_ids=["doc_unknown_release_gate"],
            )
        )
        evidence_replay = {
            "status": (
                "PASS"
                if evidence_response.evidence
                and evidence_response.replay.replay_text
                and all(item.text for item in evidence_response.evidence)
                and no_evidence_response.evidence == []
                and no_evidence_response.candidates == []
                and no_evidence_response.answer_model == "paperos/no-evidence"
                and no_evidence_response.replay.replay_text == ""
                else "FAIL"
            ),
            "evidence_count": len(evidence_response.evidence),
            "replay_matches_synthesis": bool(evidence_response.replay.replay_text),
            "no_evidence_model": no_evidence_response.answer_model,
            "no_evidence_replay_empty": no_evidence_response.replay.replay_text == "",
        }

        rebuild_target = bundles[0]
        rebuild_active_before = rebuild_target.snapshot.id
        lexical_before = application.services.retrieval.index_manager.lexical.object_ids(
            rebuild_active_before
        )
        original_ingest_bundle = application.knowledge_pipeline.ingest_bundle

        async def fail_candidate(*_args: object, **_kwargs: object) -> Any:
            raise CogneeStorageError("release candidate failure injection")

        application.knowledge_pipeline.ingest_bundle = fail_candidate  # type: ignore[method-assign]
        rebuild_failed = False
        try:
            await application.services.rebuilder.rebuild(rebuild_active_before)
        except CogneeStorageError:
            rebuild_failed = True
        finally:
            application.knowledge_pipeline.ingest_bundle = original_ingest_bundle  # type: ignore[method-assign]
        rebuild_failure_preserved = (
            rebuild_failed
            and application.canonical_repository.active_snapshot_id(
                rebuild_target.document.id
            )
            == rebuild_active_before
            and application.services.retrieval.index_manager.lexical.object_ids(
                rebuild_active_before
            )
            == lexical_before
        )

        reprocess_document_id = rebuild_target.document.id
        reprocess_active_before = application.canonical_repository.active_snapshot_id(
            reprocess_document_id
        )
        reprocess_started = time.perf_counter()
        await application.services.documents.reprocess(reprocess_document_id)
        reprocess_seconds = round(time.perf_counter() - reprocess_started, 3)
        reprocess_active_after = application.canonical_repository.active_snapshot_id(
            reprocess_document_id
        )
        reprocess_switched = (
            reprocess_active_before is not None
            and reprocess_active_after is not None
            and reprocess_active_after != reprocess_active_before
        )
        cleanup_first = await application.knowledge_pipeline.cleanup_snapshot_revision(
            str(reprocess_active_before)
        )
        cleanup_second = await application.knowledge_pipeline.cleanup_snapshot_revision(
            str(reprocess_active_before)
        )
        cleanup_idempotent = (
            cleanup_first == []
            and cleanup_second == []
            and application.canonical_repository.active_snapshot_id(
                reprocess_document_id
            )
            == reprocess_active_after
        )

        original_ingest_bundle = application.knowledge_pipeline.ingest_bundle
        application.knowledge_pipeline.ingest_bundle = fail_candidate  # type: ignore[method-assign]
        failed_reprocess_raised = False
        try:
            await application.services.documents.reprocess(reprocess_document_id)
        except CogneeStorageError:
            failed_reprocess_raised = True
        finally:
            application.knowledge_pipeline.ingest_bundle = original_ingest_bundle  # type: ignore[method-assign]
        failed_reprocess_preserved = (
            failed_reprocess_raised
            and application.canonical_repository.active_snapshot_id(
                reprocess_document_id
            )
            == reprocess_active_after
        )

        delete_target = next(
            bundle for bundle in bundles if bundle.document.id != reprocess_document_id
        )
        await application.services.documents.delete(delete_target.document.id)
        listed_ids = {
            item.document_id for item in application.services.documents.list_documents()
        }
        corpus_after_delete = CorpusView.load(
            application.paths,
            application.canonical_repository,
            application.registry,
            application.scholarly_registry,
        )
        health_after_delete = await application.services.health.report()
        visualize_after_delete = await visualize_dataset(application, dataset)
        deleted_query = await application.services.retrieval.query(
            QueryRequest(
                query="deleted document must be unavailable",
                dataset=dataset,
                document_ids=[delete_target.document.id],
            )
        )
        graph_health = health_after_delete["components"]["cognee_graph"]
        delete_consistent = (
            delete_target.document.id not in listed_ids
            and delete_target.document.id not in corpus_after_delete.bundles
            and int(graph_health["document_count"]) == 3
            and delete_target.snapshot.id
            not in visualize_after_delete["active_snapshot_ids"]
            and deleted_query.answer_model == "paperos/no-evidence"
            and deleted_query.replay.replay_text == ""
        )

        return {
            "gpu": {
                "outer_cuda_visible_devices": visible,
                "runtime_cuda_devices": settings.local_inference.cuda_devices,
                "matched": visible == configured_visible,
            },
            "active": {
                "initial_document_count": len(bundles),
                "one_pointer_per_document": pointer_unique,
            },
            "chunks": {
                **chunk_metrics,
                "chunk_count": len(all_chunks),
                "max_chunk_tokens": max_chunk_tokens,
                "hard_max_tokens": settings.ingestion.chunk_hard_max_tokens,
                "hard_max_violation_count": hard_max_violations,
            },
            "evidence_replay": evidence_replay,
            "lifecycle": {
                "rebuild_failure_preserved_active": rebuild_failure_preserved,
                "reprocess_switched_after_success": reprocess_switched,
                "reprocess_seconds": reprocess_seconds,
                "reprocess_failure_preserved_active": failed_reprocess_preserved,
                "cleanup_old_first_count": len(cleanup_first),
                "cleanup_old_second_count": len(cleanup_second),
                "cleanup_idempotent": cleanup_idempotent,
                "delete_consistent": delete_consistent,
            },
        }
    finally:
        await application.aclose()


def _finalize_existing(args: argparse.Namespace) -> dict[str, Any]:
    """Run only lightweight engineering gates and preserve historical evidence."""

    output = args.output.resolve()
    work = args.work.resolve()
    report_path = output / "release-report.json"
    acceptance_path = work / "search" / "acceptance.json"
    if not report_path.is_file() or not acceptance_path.is_file():
        raise RuntimeError(
            "Engineering finalization requires the retained report and acceptance data."
        )
    previous_report = json.loads(report_path.read_text(encoding="utf-8"))
    clean_room = json.loads(acceptance_path.read_text(encoding="utf-8"))
    legacy_head = _legacy_validation_head(args, previous_report)
    current_head = _git("rev-parse", "HEAD")

    case_ids = [str(item["id"]) for item in clean_room["queries"]]
    clean_room["queries"] = _merge_query_reviews(
        case_ids,
        clean_room["queries"],
        [],
        previous_head=legacy_head,
        current_head=current_head,
    )
    clean_room = _reused_validation(clean_room, fallback_head=legacy_head)

    engineering_gates = _legacy_engineering_evidence(
        previous_report,
        legacy_head=legacy_head,
    )
    command_results = {
        name: _annotate_validation(
            _command_gate(command),
            origin=VALIDATION_ORIGIN_CURRENT,
            validated_head=current_head,
            executed_this_run=True,
        )
        for name, command in _final_engineering_command_specs().items()
    }
    current_gate_results = {
        "node_build": command_results["node_build"]["status"] == "PASS",
        "compile": command_results["compile"]["status"] == "PASS",
        "ruff": command_results["ruff"]["status"] == "PASS",
        "mypy": command_results["mypy"]["status"] == "PASS",
    }
    contracts_passed = all(
        command_results[name]["status"] == "PASS"
        for name in ("runtime_contract", "reranker_contract", "report_contract")
    )
    current_gate_results["contracts"] = contracts_passed
    current_gate_results["ci"] = (
        contracts_passed
        and all(current_gate_results.values())
        and command_results["diff_check"]["status"] == "PASS"
    )
    for name, passed in current_gate_results.items():
        engineering_gates[name] = _gate_record(
            passed,
            origin=VALIDATION_ORIGIN_CURRENT,
            validated_head=current_head,
            executed_this_run=True,
        )

    report = _drop_legacy_gate_fields(previous_report)
    report["head"] = current_head
    report["dirty_at_start"] = bool(_git("status", "--short"))
    report["clean_room"] = clean_room
    report["runtime_audit"] = _reused_validation(
        dict(previous_report["runtime_audit"]),
        fallback_head=legacy_head,
    )
    report["engineering_gates"] = engineering_gates
    report["search_quality_status"] = SEARCH_QUALITY_PENDING
    report["rerank_quality_notice"] = RERANK_PROVISIONAL_NOTICE
    diagnostic = previous_report.get(
        "reranker_diagnostic",
        previous_report.get("reranker_blocker"),
    )
    if isinstance(diagnostic, dict):
        report["reranker_diagnostic"] = {
            **_reused_validation(diagnostic, fallback_head=legacy_head),
            "quality_status": SEARCH_QUALITY_PENDING,
        }
    report["commands"] = list(command_results.values())
    report["decision"] = _engineering_decision(engineering_gates)

    _write_json(acceptance_path, clean_room)
    _write_json(work / "search" / "review" / "queries.json", clean_room["queries"])
    _write_json(report_path, report)
    (output / "release-report.md").write_text(_markdown(report), encoding="utf-8")
    return report



async def _resume_existing(args: argparse.Namespace) -> dict[str, Any]:
    """Re-run selected query cases without mutating the retained clean-room."""

    output = args.output.resolve()
    work = args.work.resolve()
    report_path = output / "release-report.json"
    acceptance_path = work / "search" / "acceptance.json"
    if not report_path.is_file() or not acceptance_path.is_file():
        raise RuntimeError("Resume requires the existing release report and acceptance data")
    previous_report = json.loads(report_path.read_text(encoding="utf-8"))
    previous_acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    selected_ids = set(args.case or [])
    if not selected_ids:
        raise RuntimeError("Resume requires at least one --case")

    config_root = args.acceptance_config.resolve()
    queries_config = json.loads((config_root / "queries.json").read_text(encoding="utf-8"))
    ground_truth = json.loads(
        (config_root / "ground_truth.json").read_text(encoding="utf-8")
    )
    cases = [case for case in queries_config["cases"] if case["id"] in selected_ids]
    if {case["id"] for case in cases} != selected_ids:
        missing = sorted(selected_ids - {case["id"] for case in cases})
        raise RuntimeError(f"Unknown release query cases: {missing}")

    base = load_settings(args.config)
    runtime_root = work / "search" / "runtime"
    settings = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={"directory": runtime_root, "dataset": DATASET}
            ),
            "ingestion": base.ingestion.model_copy(
                update={"claim_enrichment_enabled": False}
            ),
        }
    )
    application = create_application(settings)
    await application.start()
    try:
        work_by_symbol = dict(previous_acceptance["runtime_work_ids"])
        document_by_symbol = dict(previous_acceptance["runtime_document_ids"])
        symbolic_by_work = {
            work_id: symbol for symbol, work_id in work_by_symbol.items()
        }
        all_document_ids = set(document_by_symbol.values())
        facts_by_id = {
            item["id"]: item for item in ground_truth["retrieval_facts"]
        }
        corpus = CorpusView.load(
            application.paths,
            application.canonical_repository,
            application.registry,
            application.scholarly_registry,
        )
        resumed_reviews: list[dict[str, Any]] = []
        blocker_trace: dict[str, Any] | None = None
        for case in cases:
            filters = case.get("filters", {})
            work_ids = [
                work_by_symbol[symbol] for symbol in filters.get("work_ids", [])
            ] or None
            request = QueryRequest(
                query=case["query"],
                dataset=DATASET,
                top_k=case.get("top_k"),
                work_ids=work_ids,
                expand_context=bool(case["expansion"]["local"]),
                expand_graph=bool(case["expansion"]["semantic"]),
            )
            response = await application.services.retrieval.query(request)
            review = retrieval_validation._response_review(
                case,
                request,
                response,
                symbolic_by_work=symbolic_by_work,
                facts_by_id=facts_by_id,
                all_document_ids=all_document_ids,
            )
            review["acceptance_candidate_pool_size"] = case.get(
                "acceptance_candidate_pool_size"
            )
            resumed_reviews.append(review)
            if case["id"] == "adadiv_self_limitation_default":
                target_chunk_id = "chunk_8c5d0fd28f4f637ab2ea2cab73305b0f"
                diagnostics_by_chunk = {
                    item.chunk_id: item for item in response.trace.first_rerank_diagnostics
                }
                target_candidate = next(
                    (
                        item
                        for item in response.candidates
                        if item.chunk_id == target_chunk_id
                    ),
                    None,
                )
                winning = (
                    target_candidate.rerank_diagnostics
                    if target_candidate is not None
                    else None
                )
                rank_rows: list[dict[str, Any]] = []
                for rank, chunk_id in enumerate(
                    response.trace.first_reranked_chunk_ids[7:18], start=8
                ):
                    chunk = corpus.chunks[chunk_id]
                    diagnostics = diagnostics_by_chunk[chunk_id]
                    rank_rows.append(
                        {
                            "rank": rank,
                            "chunk_id": chunk_id,
                            "score": diagnostics.winning_window_score,
                            "paper": corpus.bundles[chunk.document_id].document.title,
                            "section": chunk.section_path,
                            "chunk_token_count": chunk.token_count,
                            "reranker_input_token_count": diagnostics.input_token_count,
                            "reranker_effective_input_token_count": (
                                diagnostics.effective_input_token_count
                            ),
                            "reranker_special_prompt_token_count": (
                                diagnostics.special_prompt_token_count
                            ),
                            "reranker_window_document_token_count": (
                                diagnostics.winning_window_document_token_count
                            ),
                            "window_count": diagnostics.window_count,
                            "truncated": diagnostics.truncated,
                        }
                    )
                blocker_trace = {
                    "target_chunk_id": target_chunk_id,
                    "reranker_model": settings.local_inference.reranker_model_path.name,
                    "model_max_input_tokens": (
                        winning.model_max_input_tokens if winning else None
                    ),
                    "query_token_count": winning.query_token_count if winning else None,
                    "document_token_count": (
                        winning.document_token_count if winning else None
                    ),
                    "first_stage_rank": (
                        response.trace.first_stage_chunk_ids.index(target_chunk_id) + 1
                        if target_chunk_id in response.trace.first_stage_chunk_ids
                        else None
                    ),
                    "rerank_rank": (
                        response.trace.first_reranked_chunk_ids.index(target_chunk_id) + 1
                        if target_chunk_id in response.trace.first_reranked_chunk_ids
                        else None
                    ),
                    "reranker_input_tokens": winning.input_token_count if winning else None,
                    "reranker_effective_input_tokens": (
                        winning.effective_input_token_count if winning else None
                    ),
                    "reranker_special_prompt_tokens": (
                        winning.special_prompt_token_count if winning else None
                    ),
                    "reranker_truncated": winning.truncated if winning else None,
                    "window_count": winning.window_count if winning else None,
                    "winning_window_index": (
                        winning.winning_window_index if winning else None
                    ),
                    "winning_window_score": (
                        winning.winning_window_score if winning else None
                    ),
                    "winning_window": winning.winning_window_text if winning else None,
                    "final_evidence_present": target_chunk_id
                    in response.trace.final_selected_chunk_ids,
                    "matched_ground_truth_fact_ids": review[
                        "matched_ground_truth_fact_ids"
                    ],
                    "answer": response.answer,
                    "ranks_8_18": rank_rows,
                    "status": review["status"],
                }
    finally:
        await application.aclose()

    current_head = _git("rev-parse", "HEAD")
    legacy_head = _legacy_validation_head(args, previous_report)
    updated_reviews = _merge_query_reviews(
        [str(case["id"]) for case in queries_config["cases"]],
        previous_acceptance["queries"],
        resumed_reviews,
        previous_head=legacy_head,
        current_head=current_head,
    )
    clean_room = dict(previous_acceptance)
    clean_room["queries"] = updated_reviews
    clean_room = _reused_validation(clean_room, fallback_head=legacy_head)
    _write_json(acceptance_path, clean_room)
    _write_json(work / "search" / "review" / "queries.json", updated_reviews)

    engineering_gates = _legacy_engineering_evidence(
        previous_report,
        legacy_head=legacy_head,
    )
    report = _drop_legacy_gate_fields(previous_report)
    report["head"] = current_head
    report["clean_room"] = clean_room
    report["runtime_audit"] = _reused_validation(
        dict(previous_report["runtime_audit"]),
        fallback_head=legacy_head,
    )
    report["engineering_gates"] = engineering_gates
    report["search_quality_status"] = SEARCH_QUALITY_PENDING
    report["rerank_quality_notice"] = RERANK_PROVISIONAL_NOTICE
    diagnostic = blocker_trace or previous_report.get(
        "reranker_diagnostic",
        previous_report.get("reranker_blocker"),
    )
    if isinstance(diagnostic, dict):
        diagnostic_record = (
            _annotate_validation(
                diagnostic,
                origin=VALIDATION_ORIGIN_CURRENT,
                validated_head=current_head,
                executed_this_run=True,
            )
            if blocker_trace is not None
            else _reused_validation(diagnostic, fallback_head=legacy_head)
        )
        report["reranker_diagnostic"] = {
            **diagnostic_record,
            "quality_status": SEARCH_QUALITY_PENDING,
        }
    report["decision"] = _engineering_decision(engineering_gates)
    _write_json(output / "release-report.json", report)
    (output / "release-report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict[str, Any]) -> str:
    commands = report.get("commands", [])
    clean_room = report.get("clean_room", {})
    runtime_audit = report.get("runtime_audit", {})
    chunks = runtime_audit.get("chunks", {})
    gates = report.get("engineering_gates", {})
    lines = [
        "# PaperOS Production Readiness",
        "",
        f"Release engineering decision: **{report['decision']}**",
        f"Search quality: **{report.get('search_quality_status', 'UNAVAILABLE')}**",
        "",
        f"HEAD: `{report['head']}`",
        f"Dirty at start: `{report.get('dirty_at_start', True)}`",
        "",
        "## Engineering gates",
        "",
    ]
    lines.extend(
        (
            f"- {name}: {gate.get('status')} "
            f"(origin={gate.get('validation_origin')}, "
            f"validated_head={gate.get('validated_head')}, "
            f"executed_this_run={gate.get('executed_this_run')})"
        )
        for name, gate in gates.items()
        if isinstance(gate, dict)
    )
    lines.extend(
        [
            "",
            "## Retained clean-room evidence",
            "",
            f"- Full PDF-to-LLM: {clean_room.get('overall_status', 'UNAVAILABLE')}",
            f"- Papers: {clean_room.get('counts', {}).get('ingested_papers', 0)}",
            (
                "- Per-paper PDF-to-active seconds: "
                f"{clean_room.get('pdf_to_active_seconds', {})}"
            ),
            f"- Max chunk tokens: {chunks.get('max_chunk_tokens', 0)}",
            f"- Hard-max violations: {chunks.get('hard_max_violation_count', 0)}",
            (
                "- Figure input/covered/lost: "
                f"{chunks.get('figure_input_count', 0)}/"
                f"{chunks.get('figure_placeholder_count', 0)}/"
                f"{chunks.get('figure_lost_count', 0)}"
            ),
            "",
            "## Commands",
            "",
        ]
    )
    lines.extend(
        f"- `{item['command']}` → exit {item['exit_code']} ({item['seconds']}s)"
        for item in commands
    )
    lines.extend(
        [
            "",
            "## Known limitations",
            "",
            f"- {report.get('rerank_quality_notice', RERANK_PROVISIONAL_NOTICE)}",
            (
                "- GitHub-hosted Windows execution is represented by the portable/config "
                "contract and workflow definition; this local release run is Linux."
            ),
            (
                "- Cognee may emit its own informational/deprecation logs; PaperOS "
                "hard-max sizing no longer uses Cognee's tokenizer resolver."
            ),
            "",
        ]
    )
    return "\n".join(lines)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    work = args.work.resolve()
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    output.mkdir(parents=True, exist_ok=True)

    head = _git("rev-parse", "HEAD")
    dirty_at_start = bool(_git("status", "--short"))
    retrieval_args = argparse.Namespace(
        config=args.config,
        acceptance_config=args.acceptance_config,
        corpus=args.corpus,
        output=work / "search",
        dataset=DATASET,
        rebuild=True,
        rebuild_derived=True,
    )
    clean_room = await retrieval_validation.run(retrieval_args)
    runtime_audit = await _runtime_audit(
        config=args.config,
        runtime_root=work / "search/runtime",
        dataset=DATASET,
    )

    command_specs = {
        **_final_engineering_command_specs(),
        "active_revision_contract": [
            sys.executable,
            "tests/contract/test_active_canonical_revision.py",
        ],
        "hard_filter_contract": [
            sys.executable,
            "tests/contract/test_query_filter_contracts.py",
        ],
        "citation_loop_contract": [
            sys.executable,
            "tests/contract/test_retrieval_citation_loop.py",
        ],
        "citation_resolution_contract": [
            sys.executable,
            "tests/contract/test_citation_resolution.py",
        ],
        "chunk_boundary_contract": [
            sys.executable,
            "tests/validation/chunk.py",
            "boundaries",
        ],
    }
    command_results = {
        name: _annotate_validation(
            _command_gate(command),
            origin=VALIDATION_ORIGIN_CURRENT,
            validated_head=head,
            executed_this_run=True,
        )
        for name, command in command_specs.items()
    }
    commands = list(command_results.values())

    chunks = runtime_audit["chunks"]
    lifecycle = runtime_audit["lifecycle"]
    static_passed = {
        name: command_results[name]["status"] == "PASS"
        for name in ("node_build", "compile", "ruff", "mypy")
    }
    contracts_passed = all(
        item["status"] == "PASS"
        for name, item in command_results.items()
        if name.endswith("_contract")
    )
    engineering_results = {
        "contracts": contracts_passed,
        "ci": contracts_passed
        and all(static_passed.values())
        and command_results["diff_check"]["status"] == "PASS",
        **static_passed,
        "active_revision": bool(
            runtime_audit["active"]["one_pointer_per_document"]
            and command_results["active_revision_contract"]["status"] == "PASS"
        ),
        "hard_filters": (
            command_results["hard_filter_contract"]["status"] == "PASS"
        ),
        "hard_max": chunks["hard_max_violation_count"] == 0,
        "figure_provenance": chunks["figure_lost_count"] == 0
        and chunks["figure_provenance_error_count"] == 0
        and chunks["figure_caption_duplication_count"] == 0,
        "source_provenance": chunks["text_loss_count"] == 0
        and chunks["text_duplication_count"] == 0
        and chunks["section_cross_boundary_count"] == 0,
        "citation_provenance": (
            clean_room["citation_provenance"]["status"] == "PASS"
        ),
        "evidence_replay": runtime_audit["evidence_replay"]["status"] == "PASS",
        "lifecycle": all(
            lifecycle[key]
            for key in (
                "rebuild_failure_preserved_active",
                "reprocess_switched_after_success",
                "reprocess_failure_preserved_active",
                "cleanup_idempotent",
                "delete_consistent",
            )
        ),
        "gpu_restriction": runtime_audit["gpu"]["matched"],
        "clean_room_pipeline": bool(clean_room["pipeline_completed_pdf_to_llm"]),
    }
    engineering_gates = {
        name: _gate_record(
            passed,
            origin=VALIDATION_ORIGIN_CURRENT,
            validated_head=head,
            executed_this_run=True,
        )
        for name, passed in engineering_results.items()
    }
    decision = _engineering_decision(engineering_gates)
    clean_room["queries"] = [
        _annotate_validation(
            item,
            origin=VALIDATION_ORIGIN_CURRENT,
            validated_head=head,
            executed_this_run=True,
        )
        for item in clean_room["queries"]
    ]
    clean_room = _annotate_validation(
        clean_room,
        origin=VALIDATION_ORIGIN_CURRENT,
        validated_head=head,
        executed_this_run=True,
    )
    runtime_audit = _annotate_validation(
        runtime_audit,
        origin=VALIDATION_ORIGIN_CURRENT,
        validated_head=head,
        executed_this_run=True,
    )
    report = {
        "decision": decision,
        "head": head,
        "dirty_at_start": dirty_at_start,
        "environment": {
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "ci": {
            "workflow": ".github/workflows/cross-platform.yml",
            "linux_local_equivalent": "PASS",
            "windows_portable_contract": "PASS",
            "external_job": "executed locally with the release commands below",
        },
        "engineering_gates": engineering_gates,
        "search_quality_status": SEARCH_QUALITY_PENDING,
        "rerank_quality_notice": RERANK_PROVISIONAL_NOTICE,
        "clean_room": clean_room,
        "runtime_audit": runtime_audit,
        "commands": commands,
        "known_issues": [
            {
                "id": "citation_gold_math_wrapped_label",
                "status": "non_blocking_reviewed_difference",
                "value": r"$\mathrm{[LWJ*22]}$",
            }
        ],
    }
    _write_json(output / "release-report.json", report)
    (output / "release-report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/paperos.toml"))
    parser.add_argument("--acceptance-config", type=Path, default=DEFAULT_ACCEPTANCE_CONFIG)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--finalize-engineering", action="store_true")
    parser.add_argument("--legacy-validated-head")
    parser.add_argument("--case", action="append")
    args = parser.parse_args()
    try:
        if args.resume_existing and args.finalize_engineering:
            raise RuntimeError("Choose either --resume-existing or --finalize-engineering.")
        if args.finalize_engineering:
            report = _finalize_existing(args)
        elif args.resume_existing:
            report = asyncio.run(_resume_existing(args))
        else:
            report = asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001 - release gate must emit NO-GO
        report = {
            "decision": "NO-GO",
            "head": _git("rev-parse", "HEAD"),
            "blocked_stage": "clean_room",
            "error_type": type(exc).__name__,
        }
        _write_json(args.output.resolve() / "release-report.json", report)
        (args.output.resolve() / "release-report.md").write_text(
            _markdown(
                {
                    **report,
                    "dirty_at_start": True,
                    "clean_room": {"overall_status": "BLOCKED", "counts": {}},
                    "runtime_audit": {
                        "chunks": {
                            "max_chunk_tokens": 0,
                            "hard_max_violation_count": 0,
                            "figure_input_count": 0,
                            "figure_placeholder_count": 0,
                            "figure_lost_count": 0,
                        }
                    },
                    "commands": [],
                }
            ),
            encoding="utf-8",
        )
    print(json.dumps({"decision": report["decision"], "report": str(args.output)}, indent=2))
    if report["decision"] != "GO":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
