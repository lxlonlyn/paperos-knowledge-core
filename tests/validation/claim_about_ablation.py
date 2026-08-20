"""Claim/ABOUT necessity ablation over the retained scholarly-work corpus.

    python tests/validation/claim_about_ablation.py \\
      --run-root data/validation/runs/scholarly-work-reference \\
      --dataset paperos-scholarly-work-reference

Optional:
      --resume
      --with-synthesis
      --pools 40
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.application import Application, create_application
from paperos_core.config import load_settings
from paperos_core.domain.provenance import RelationType
from paperos_core.errors import LocalInferenceUnavailableError
from paperos_core.retrieval.ablation import (
    AblationTrace,
    ablation_policy_context,
    policy_from_spec,
)
from paperos_core.retrieval.candidates import (
    QueryRequest,
    QueryScopeInput,
    RetrievalProfile,
)
from paperos_core.retrieval.corpus import CorpusView
from paperos_core.retrieval.scope import resolve_query_scope_async

FIXTURE_ROOT = (
    REPOSITORY_ROOT / "tests" / "fixtures" / "scholarly_work_reference"
)
ABLATION_ROOT = FIXTURE_ROOT / "claim_about_ablation"
MANIFEST = FIXTURE_ROOT / "reference_corpus_manifest.json"
GROUND_TRUTH = FIXTURE_ROOT / "reference_ground_truth.json"
CORE_PAPERS = ("nise_2023", "adadiv_2025", "efis_2026", "lipmlp_2022")
REPORT_NAME = "claim-about-ablation.json"
REVIEW_NAME = "claim-about-manual-review.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _paper_map() -> dict[str, dict[str, Any]]:
    return {str(item["id"]): dict(item) for item in _load_json(MANIFEST)["papers"]}


def _latest_bundles_by_file(application: Application) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for bundle in application.canonical_repository.list_bundles():
        filename = application.registry.get_source(
            bundle.document.source_file_id
        ).original_filename
        current = latest.get(filename)
        if current is None or (
            bundle.snapshot.created_at,
            bundle.snapshot.id,
        ) > (
            current.snapshot.created_at,
            current.snapshot.id,
        ):
            latest[filename] = bundle
    return latest


def _paper_work_ids(application: Application) -> dict[str, str]:
    papers = _paper_map()
    latest = _latest_bundles_by_file(application)
    mapping: dict[str, str] = {}
    for paper_id, paper in papers.items():
        bundle = latest.get(str(paper["file"]))
        if bundle is None:
            continue
        work = application.scholarly_registry.work_for_document(bundle.document.id)
        if work is not None:
            mapping[paper_id] = work.id
    return mapping


async def _ensure_runtime(application: Application) -> dict[str, Any]:
    application.storage.initialize()
    application.runtime.local_inference.cleanup_stale_record()
    application.runtime.worker.cleanup_stale_record()
    status = application.storage.validate()
    if not status.valid:
        raise RuntimeError(
            "PaperOS local schema validation failed: "
            + ", ".join(status.missing_tables)
        )
    llm = await application.llm.health_check()
    local = await application.local_inference_client.health()
    if local.get("status") != "healthy":
        try:
            await application.start()
            return {
                "llm": llm,
                "local_inference": {"status": "started_by_application"},
                "reused_existing_embedding": False,
            }
        except LocalInferenceUnavailableError as exc:
            raise RuntimeError(
                "Local embedding endpoint is unhealthy and owned start failed: "
                f"{exc}; health={local}"
            ) from exc
    try:
        await application.runtime.worker.start()
        worker_status = "started"
    except Exception as worker_exc:  # noqa: BLE001
        worker_status = f"skipped:{type(worker_exc).__name__}:{worker_exc}"
    application._started = True
    return {
        "llm": llm,
        "local_inference": {"status": "reused_existing", "health": local},
        "worker": worker_status,
        "reused_existing_embedding": True,
    }


def _remap_scope(
    scope: dict[str, Any] | None, work_ids: dict[str, str]
) -> QueryScopeInput | None:
    if not scope:
        return None

    def remap(keys: list[str] | None) -> list[str] | None:
        if keys is None:
            return None
        return [work_ids[key] for key in keys]

    return QueryScopeInput(
        source_work_ids=remap(scope.get("source_work_ids")),
        exclude_source_work_ids=remap(scope.get("exclude_source_work_ids")),
        subject_work_ids=remap(scope.get("subject_work_ids")),
        work_set_work_ids=remap(scope.get("work_set_work_ids")),
        topic_queries=list(scope.get("topic_queries") or []),
    )


def _about_targets(fact: dict[str, Any]) -> list[str]:
    value = fact.get("about_work")
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def resolve_gold_chunks(
    fact: dict[str, Any],
    *,
    work_ids: dict[str, str],
    corpus: CorpusView,
) -> dict[str, Any]:
    source_key = str(fact["source_work"])
    source_work_id = work_ids[source_key]
    document_ids = corpus.document_ids_by_work.get(source_work_id, set())
    phrases = [_normalize_text(item) for item in fact.get("evidence_any_of") or []]
    page = fact.get("pdf_page")
    matched: list[tuple[str, int, bool]] = []
    for chunk_id, chunk in corpus.chunks.items():
        if chunk.document_id not in document_ids:
            continue
        text = _normalize_text(chunk.text)
        if not any(phrase and phrase in text for phrase in phrases):
            continue
        page_hit = False
        if isinstance(page, int):
            starts = chunk.page_start
            ends = chunk.page_end if chunk.page_end is not None else starts
            if starts is not None and ends is not None and starts <= page <= ends:
                page_hit = True
        distance = 0 if page_hit else 1
        matched.append((chunk_id, distance, page_hit))
    matched.sort(key=lambda item: (item[1], item[0]))
    chunk_ids = [chunk_id for chunk_id, _, _ in matched]
    return {
        "fact_id": fact["id"],
        "source_work_key": source_key,
        "source_work_id": source_work_id,
        "pdf_page": page,
        "gold_chunk_ids": chunk_ids,
        "page_assisted_chunk_ids": [
            chunk_id for chunk_id, _, page_hit in matched if page_hit
        ],
        "resolved": bool(chunk_ids),
    }


async def claim_coverage_report(
    application: Application,
    *,
    dataset: str,
    facts: list[dict[str, Any]],
    work_ids: dict[str, str],
    gold_by_fact: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    covered: list[str] = []
    missing: list[str] = []
    wrong_about: list[str] = []
    wrong_source: list[str] = []
    for fact in facts:
        fact_id = str(fact["id"])
        gold = gold_by_fact[fact_id]
        gold_chunks = set(gold["gold_chunk_ids"])
        source_work_id = work_ids[str(fact["source_work"])]
        target_ids = [work_ids[key] for key in _about_targets(fact)]
        relations = await application.services.retrieval.compat.incoming_typed_relations(
            target_ids,
            dataset_name=dataset,
            relation_type=RelationType.ABOUT.value,
            depth=1,
            limit=500,
        )
        matched_about = False
        matched_provenance = False
        wrong_target_hit = False
        for relation in relations:
            if relation.source_work_id != source_work_id:
                continue
            if relation.target_canonical_id not in target_ids:
                wrong_target_hit = True
                continue
            matched_about = True
            if gold_chunks.intersection(relation.source_chunk_ids):
                matched_provenance = True
                break
        if matched_provenance:
            covered.append(fact_id)
        elif not matched_about:
            missing.append(fact_id)
            if wrong_target_hit:
                wrong_about.append(fact_id)
        else:
            wrong_source.append(fact_id)
    total = len(facts)
    return {
        "covered_fact_ids": covered,
        "missing_claim_fact_ids": missing,
        "wrong_about_target_fact_ids": wrong_about,
        "wrong_source_provenance_fact_ids": wrong_source,
        "claim_coverage": (len(covered) / total) if total else 0.0,
    }


async def export_manual_review(
    application: Application,
    *,
    dataset: str,
    work_ids: dict[str, str],
) -> list[dict[str, Any]]:
    core_ids = [work_ids[key] for key in CORE_PAPERS]
    records: dict[str, dict[str, Any]] = {}
    for target_id in core_ids:
        relations = await application.services.retrieval.compat.incoming_typed_relations(
            [target_id],
            dataset_name=dataset,
            relation_type=RelationType.ABOUT.value,
            depth=1,
            limit=500,
        )
        for relation in relations:
            if relation.source_work_id not in core_ids:
                continue
            claim_id = relation.source_canonical_id
            current = records.get(claim_id)
            subjects = list(current["subject_work_ids"] if current else [])
            if target_id not in subjects:
                subjects.append(target_id)
            records[claim_id] = {
                "claim_id": claim_id,
                "claim_text": relation.text,
                "source_work_id": relation.source_work_id,
                "subject_work_ids": subjects,
                "source_chunk_ids": list(relation.source_chunk_ids),
                "roles": list(relation.roles),
            }
    return sorted(records.values(), key=lambda item: item["claim_id"])


def _fact_hit(selected_chunk_ids: list[str], gold_chunk_ids: list[str]) -> bool:
    return bool(set(selected_chunk_ids).intersection(gold_chunk_ids))


def _case_expected_facts(case: dict[str, Any]) -> list[str]:
    if "expected_fact_ids" in case:
        return list(case.get("expected_fact_ids") or [])
    return list(case.get("expected_fact_ids_min") or [])


def _case_is_min_coverage(case: dict[str, Any]) -> bool:
    return "expected_fact_ids_min" in case and "expected_fact_ids" not in case


def _score_case(
    case: dict[str, Any],
    *,
    selected_chunk_ids: list[str],
    pool_chunk_ids: list[str],
    gold_by_fact: dict[str, dict[str, Any]],
    source_work_ids: list[str],
    work_ids: dict[str, str],
) -> dict[str, Any]:
    expected = _case_expected_facts(case)
    forbidden = list(case.get("forbidden_fact_ids") or [])
    matched = [
        fact_id
        for fact_id in expected
        if _fact_hit(selected_chunk_ids, gold_by_fact[fact_id]["gold_chunk_ids"])
    ]
    missed = [fact_id for fact_id in expected if fact_id not in matched]
    pool_matched = [
        fact_id
        for fact_id in expected
        if _fact_hit(pool_chunk_ids, gold_by_fact[fact_id]["gold_chunk_ids"])
    ]
    forbidden_hits = [
        fact_id
        for fact_id in forbidden
        if _fact_hit(selected_chunk_ids, gold_by_fact[fact_id]["gold_chunk_ids"])
    ]
    if _case_is_min_coverage(case):
        required = len(expected)
        fact_recall = (len(matched) / required) if required else 1.0
        success = len(matched) >= required and not forbidden_hits
    else:
        required = len(expected)
        fact_recall = (len(matched) / required) if required else 1.0
        success = len(missed) == 0 and not forbidden_hits

    expected_sources = case.get("expected_source_works") or case.get(
        "expected_source_works_min"
    )
    source_ok = True
    if expected_sources:
        mapped = {work_ids[key] for key in expected_sources}
        observed = {item for item in source_work_ids if item}
        if case.get("expected_source_works_min"):
            source_ok = mapped.issubset(observed)
        else:
            source_ok = observed == mapped or mapped == observed
            # Allow extra evidence from the same expected sources only.
            source_ok = bool(observed) and observed.issubset(mapped) and mapped.issubset(
                observed
            )

    return {
        "matched_fact_ids": matched,
        "missed_fact_ids": missed,
        "pool_matched_fact_ids": pool_matched,
        "forbidden_fact_hits": forbidden_hits,
        "fact_recall": fact_recall,
        "gold_evidence_recall_at_candidate_pool": (
            (len(pool_matched) / required) if required else 1.0
        ),
        "gold_evidence_recall_at_final_k": fact_recall,
        "success": success and not forbidden_hits,
        "source_works_ok": source_ok,
        "hard_error": bool(forbidden_hits),
    }


def _recommendation(
    *,
    unique_rescue_rate: float,
    claim_only_fraction: float,
    recall_gap_pp: float,
    evidence_reduction: float | None,
    claim_coverage: float,
    guidance: dict[str, Any],
) -> str:
    keep = guidance.get("keep_signal") or {}
    remove = guidance.get("remove_from_retrieval_signal") or {}
    short = guidance.get("claim_only_short_circuit_signal") or {}
    if unique_rescue_rate * 100 >= float(
        keep.get("unique_claim_rescue_percentage_points_gte", 10)
    ):
        return "KEEP"
    if evidence_reduction is not None and evidence_reduction * 100 >= float(
        keep.get("or_evidence_token_reduction_percent_gte", 30)
    ):
        return "KEEP"
    if claim_only_fraction >= float(
        short.get("claim_only_fact_recall_fraction_of_full_gte", 0.97)
    ):
        return "REDUCE"
    if abs(recall_gap_pp) <= float(
        remove.get("fact_recall_gap_vs_CITE_ANCHOR_RAG_percentage_points_lte", 2)
    ) and (evidence_reduction is None or evidence_reduction < 0.1):
        return "REMOVE_FROM_RETRIEVAL"
    if claim_coverage < float(keep.get("claim_coverage_gte", 0.85)):
        return "INSUFFICIENT_EVIDENCE"
    if unique_rescue_rate > 0:
        return "REDUCE"
    return "INSUFFICIENT_EVIDENCE"


async def _run_one(
    application: Application,
    *,
    dataset: str,
    case: dict[str, Any],
    config: dict[str, Any],
    pool: int,
    top_k: int,
    work_ids: dict[str, str],
    gold_by_fact: dict[str, dict[str, Any]],
    with_synthesis: bool,
) -> dict[str, Any]:
    policy = policy_from_spec(
        config,
        candidate_pool_size=pool,
        final_top_k=top_k,
        skip_synthesis=not with_synthesis,
        bypass_query_cache=True,
    )
    request = QueryRequest(
        query=str(case["query"]),
        profile=RetrievalProfile.COMPREHENSIVE,
        dataset=dataset,
        top_k=top_k,
        scope=_remap_scope(case.get("scope"), work_ids),
    )
    with ablation_policy_context(policy) as trace:
        response = await application.services.retrieval.query(request)
    selected_chunk_ids = [item.chunk_id for item in response.evidence]
    pool_chunk_ids = list(trace.fused_before_rerank_chunk_ids)
    source_work_ids = [
        item.source_work_id for item in response.evidence if item.source_work_id
    ]
    score = _score_case(
        case,
        selected_chunk_ids=selected_chunk_ids,
        pool_chunk_ids=pool_chunk_ids,
        gold_by_fact=gold_by_fact,
        source_work_ids=source_work_ids,
        work_ids=work_ids,
    )
    leak_errors: list[str] = []
    if config["id"] == "NO_CLAIM":
        if "subject_claim" in response.channels_used:
            leak_errors.append("NO_CLAIM leaked subject_claim channel")
        if "subject_about_retrieval" in response.stages:
            leak_errors.append("NO_CLAIM ran subject_about_retrieval")
        for candidate in response.candidates:
            if candidate.object_type == "claim" or "claim_about" in candidate.object_type:
                leak_errors.append(
                    f"NO_CLAIM leaked claim object_type={candidate.object_type}"
                )
                break
    evidence_chars = sum(len(item.text or "") for item in response.evidence)
    return {
        "configuration": config["id"],
        "candidate_pool": pool,
        "query_id": case["id"],
        "family": case.get("family"),
        "query": case["query"],
        "hard": bool(case.get("hard")),
        "resolved_explicit_scope": response.resolved_scope.model_dump(),
        "channels_used": response.channels_used,
        "stages": response.stages,
        "candidate_chunk_ids_before_rerank": pool_chunk_ids,
        "candidate_channels": trace.fused_candidate_channels,
        "candidate_ranks": trace.fused_candidate_ranks,
        "channel_candidate_chunk_ids": trace.channel_candidate_chunk_ids,
        "reranked_chunk_ids": list(trace.reranked_chunk_ids),
        "final_selected_chunk_ids": selected_chunk_ids,
        "source_work_ids": source_work_ids,
        "chunk_count": len(response.evidence),
        "total_text_chars": evidence_chars,
        "retrieval_latency_ms": trace.retrieval_latency_ms,
        "rerank_latency_ms": trace.rerank_latency_ms,
        "provenance_complete": response.provenance_complete,
        "citation_source_work_ids": list(trace.citation_source_work_ids),
        "no_claim_leak_errors": leak_errors,
        **score,
    }


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    fact_meta: dict[str, dict[str, Any]],
    gold_by_fact: dict[str, dict[str, Any]],
    guidance: dict[str, Any],
    claim_coverage: float,
) -> dict[str, Any]:
    primary_pool = 40
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row["candidate_pool"] == primary_pool
            and row.get("family") != "negative_source_attribution"
            and row.get("family") != "planner"
        ):
            by_config[row["configuration"]].append(row)

    def mean_recall(items: list[dict[str, Any]]) -> float:
        if not items:
            return 0.0
        return sum(float(item["fact_recall"]) for item in items) / len(items)

    config_metrics = {
        config_id: {
            "fact_recall_at_final_k": mean_recall(items),
            "gold_evidence_recall_at_candidate_pool": (
                sum(float(item["gold_evidence_recall_at_candidate_pool"]) for item in items)
                / len(items)
                if items
                else 0.0
            ),
            "mean_chunk_count": (
                sum(item["chunk_count"] for item in items) / len(items) if items else 0.0
            ),
            "mean_text_chars": (
                sum(item["total_text_chars"] for item in items) / len(items)
                if items
                else 0.0
            ),
            "case_count": len(items),
        }
        for config_id, items in by_config.items()
    }

    full_rows = {row["query_id"]: row for row in by_config.get("FULL_CURRENT", [])}
    cite_rows = {row["query_id"]: row for row in by_config.get("CITE_ANCHOR_RAG", [])}
    claim_only_rows = {
        row["query_id"]: row for row in by_config.get("CLAIM_ONLY", [])
    }

    rescue_ids: list[str] = []
    claim_supported: set[str] = set()
    also_cite: set[str] = set()
    for query_id, full in full_rows.items():
        cite = cite_rows.get(query_id)
        if cite is None:
            continue
        full_matched = set(full["matched_fact_ids"])
        cite_matched = set(cite["matched_fact_ids"])
        for fact_id in full_matched - cite_matched:
            rescue_ids.append(fact_id)
        claim_channels = full.get("channel_candidate_chunk_ids", {}).get(
            "subject_claim", []
        )
        selected = set(full["final_selected_chunk_ids"])
        # Facts matched by FULL and present via subject_claim channel pool.
        for fact_id in full_matched:
            claim_supported.add(fact_id)
            if fact_id in cite_matched:
                also_cite.add(fact_id)
        _ = claim_channels
        _ = selected

    # Redefine claim_supported using subject_claim channel contribution when available.
    claim_supported = set()
    also_cite = set()
    for query_id, full in full_rows.items():
        cite = cite_rows.get(query_id)
        if cite is None:
            continue
        subject_chunks = set(
            full.get("channel_candidate_chunk_ids", {}).get("subject_claim", [])
        )
        for fact_id in full["matched_fact_ids"]:
            gold_chunks = set(gold_by_fact.get(fact_id, {}).get("gold_chunk_ids") or [])
            if subject_chunks and gold_chunks.intersection(subject_chunks):
                claim_supported.add(fact_id)
                if fact_id in cite["matched_fact_ids"]:
                    also_cite.add(fact_id)

    unique_ids = sorted(set(rescue_ids))
    expected_fact_universe = sorted(
        {
            fact_id
            for row in full_rows.values()
            for fact_id in row["matched_fact_ids"] + row["missed_fact_ids"]
        }
    )
    unique_rate = (
        len(unique_ids) / len(expected_fact_universe) if expected_fact_universe else 0.0
    )
    redundancy = (
        len(also_cite) / len(claim_supported) if claim_supported else 0.0
    )

    by_family: dict[str, Any] = {}
    for family in sorted({row["family"] for row in by_config.get("FULL_CURRENT", [])}):
        full_f = [row for row in by_config.get("FULL_CURRENT", []) if row["family"] == family]
        cite_f = [
            row for row in by_config.get("CITE_ANCHOR_RAG", []) if row["family"] == family
        ]
        by_family[family] = {
            "FULL_CURRENT_fact_recall": mean_recall(full_f),
            "CITE_ANCHOR_RAG_fact_recall": mean_recall(cite_f),
            "case_count": len(full_f),
        }

    by_mention: dict[str, Any] = defaultdict(lambda: {"rescued": 0, "facts": 0})
    by_priority: dict[str, Any] = defaultdict(lambda: {"rescued": 0, "facts": 0})
    for fact_id in expected_fact_universe:
        meta = fact_meta.get(fact_id) or {}
        mention = str(meta.get("target_mention_mode") or "unknown")
        priority = str(meta.get("necessity_priority") or "unknown")
        by_mention[mention]["facts"] += 1
        by_priority[priority]["facts"] += 1
        if fact_id in unique_ids:
            by_mention[mention]["rescued"] += 1
            by_priority[priority]["rescued"] += 1

    full_recall = config_metrics.get("FULL_CURRENT", {}).get("fact_recall_at_final_k", 0.0)
    cite_recall = config_metrics.get("CITE_ANCHOR_RAG", {}).get(
        "fact_recall_at_final_k", 0.0
    )
    claim_only_recall = config_metrics.get("CLAIM_ONLY", {}).get(
        "fact_recall_at_final_k", 0.0
    )
    full_chars = config_metrics.get("FULL_CURRENT", {}).get("mean_text_chars", 0.0)
    cite_chars = config_metrics.get("CITE_ANCHOR_RAG", {}).get("mean_text_chars", 0.0)
    claim_chars = config_metrics.get("CLAIM_ONLY", {}).get("mean_text_chars", 0.0)
    evidence_reduction = None
    if full_chars and cite_chars and abs(full_recall - cite_recall) < 0.05:
        # Compare CLAIM_ONLY vs FULL when recalls are comparable; else FULL vs CITE.
        if abs(claim_only_recall - full_recall) < 0.05 and claim_chars:
            evidence_reduction = max(0.0, (full_chars - claim_chars) / full_chars)
        else:
            evidence_reduction = max(0.0, (cite_chars - full_chars) / cite_chars)

    recommendation = _recommendation(
        unique_rescue_rate=unique_rate,
        claim_only_fraction=(claim_only_recall / full_recall) if full_recall else 0.0,
        recall_gap_pp=(full_recall - cite_recall) * 100,
        evidence_reduction=evidence_reduction,
        claim_coverage=claim_coverage,
        guidance=guidance,
    )
    return {
        "by_configuration": config_metrics,
        "by_family": by_family,
        "by_target_mention_mode": dict(by_mention),
        "by_necessity_priority": dict(by_priority),
        "unique_claim_rescue": {
            "unique_claim_rescue_count": len(unique_ids),
            "unique_claim_rescue_rate": unique_rate,
            "unique_claim_rescue_fact_ids": unique_ids,
        },
        "claim_redundancy": {
            "claim_supported_gold_facts": sorted(claim_supported),
            "also_found_by_cite_anchor_rag": sorted(also_cite),
            "claim_redundancy_rate": redundancy,
        },
        "evidence_input": {
            "FULL_CURRENT_mean_chars": full_chars,
            "CITE_ANCHOR_RAG_mean_chars": cite_chars,
            "CLAIM_ONLY_mean_chars": claim_chars,
            "estimated_evidence_reduction_vs_baseline": evidence_reduction,
        },
        "recommendation": recommendation,
        "recommendation_inputs": {
            "full_fact_recall": full_recall,
            "cite_anchor_fact_recall": cite_recall,
            "claim_only_fact_recall": claim_only_recall,
            "unique_claim_rescue_rate": unique_rate,
            "claim_redundancy_rate": redundancy,
            "claim_coverage": claim_coverage,
            "evidence_reduction": evidence_reduction,
        },
    }


async def run(
    run_root: Path,
    dataset: str,
    *,
    resume: bool,
    with_synthesis: bool,
    pools: list[int] | None,
) -> dict[str, Any]:
    spec = _load_json(ABLATION_ROOT / "ablation_experiment_spec.json")
    queries = _load_json(ABLATION_ROOT / "ablation_queries.json")
    fact_meta_doc = _load_json(ABLATION_ROOT / "ablation_fact_metadata.json")
    ground_truth = _load_json(GROUND_TRUTH)
    facts = list(ground_truth["cross_paper_facts"])
    fact_by_id = {str(item["id"]): item for item in facts}
    fact_meta = {str(item["fact_id"]): item for item in fact_meta_doc["facts"]}
    for fact_id in fact_meta:
        if fact_id not in fact_by_id:
            raise RuntimeError(f"ablation fact_id missing from ground truth: {fact_id}")

    configured = load_settings()
    settings = configured.model_copy(
        update={
            "data": configured.data.model_copy(
                update={"directory": run_root.resolve(), "dataset": dataset}
            )
        }
    )
    application = create_application(settings)
    runtime = await _ensure_runtime(application)
    work_ids = _paper_work_ids(application)
    missing_papers = [key for key in CORE_PAPERS if key not in work_ids]
    if missing_papers:
        raise RuntimeError(f"Missing core papers in retained run: {missing_papers}")
    corpus = CorpusView.load(
        application.paths,
        application.canonical_repository,
        application.registry,
        application.scholarly_registry,
    )
    if not corpus.chunks:
        raise RuntimeError("Retained run has no canonical chunks.")
    about_probe = await application.services.retrieval.compat.incoming_typed_relations(
        [work_ids["nise_2023"]],
        dataset_name=dataset,
        relation_type=RelationType.ABOUT.value,
        depth=1,
        limit=20,
    )
    if not about_probe:
        raise RuntimeError("Retained run has no Claim/ABOUT edges for NISE.")

    gold_by_fact = {
        fact_id: resolve_gold_chunks(fact, work_ids=work_ids, corpus=corpus)
        for fact_id, fact in fact_by_id.items()
        if fact_id in fact_meta
    }
    unresolved = [
        fact_id
        for fact_id, payload in gold_by_fact.items()
        if fact_meta[fact_id].get("necessity_priority") != "ignore"
        and not payload["resolved"]
        and fact_id
        in {
            fact_id
            for case in queries["cases"]
            for fact_id in _case_expected_facts(case) + list(case.get("forbidden_fact_ids") or [])
        }
    ]
    hard_failures: list[str] = []
    if unresolved:
        hard_failures.append(
            f"fixture_resolution_failure: unresolved gold chunks for {unresolved}"
        )

    coverage = await claim_coverage_report(
        application,
        dataset=dataset,
        facts=[fact_by_id[fact_id] for fact_id in fact_meta],
        work_ids=work_ids,
        gold_by_fact=gold_by_fact,
    )
    manual_review = await export_manual_review(
        application, dataset=dataset, work_ids=work_ids
    )

    report_path = run_root / "logs" / "contracts" / REPORT_NAME
    review_path = run_root / "logs" / "contracts" / REVIEW_NAME
    existing_rows: list[dict[str, Any]] = []
    if resume and report_path.is_file():
        previous = _load_json(report_path)
        existing_rows = list(previous.get("per_case_results") or [])
    done = {
        (row["configuration"], row["candidate_pool"], row["query_id"])
        for row in existing_rows
    }

    pool_sizes = pools or list(spec["candidate_pool_sizes"])
    top_k = int(spec["final_top_k"])
    rows = list(existing_rows)
    warnings: list[str] = []

    try:
        for pool in pool_sizes:
            for config in spec["configurations"]:
                for case in queries["cases"]:
                    key = (config["id"], pool, case["id"])
                    if key in done:
                        continue
                    if hard_failures and any(
                        fact_id in unresolved for fact_id in _case_expected_facts(case)
                    ):
                        continue
                    row = await _run_one(
                        application,
                        dataset=dataset,
                        case=case,
                        config=config,
                        pool=int(pool),
                        top_k=top_k,
                        work_ids=work_ids,
                        gold_by_fact=gold_by_fact,
                        with_synthesis=with_synthesis,
                    )
                    if row["no_claim_leak_errors"]:
                        hard_failures.extend(row["no_claim_leak_errors"])
                    if row["hard_error"]:
                        hard_failures.append(
                            f"negative_control_violation:{case['id']}:{row['forbidden_fact_hits']}"
                        )
                    if not row["provenance_complete"]:
                        hard_failures.append(f"provenance_incomplete:{case['id']}")
                    rows.append(row)
                    done.add(key)
                    _atomic_json(
                        report_path,
                        {
                            "partial": True,
                            "per_case_results": rows,
                            "hard_failures": hard_failures,
                        },
                    )

        planner_results: list[dict[str, Any]] = []
        for item in queries.get("planner_diagnostics") or []:
            request = QueryRequest(
                query=str(item["query"]),
                profile=RetrievalProfile.COMPREHENSIVE,
                dataset=dataset,
            )
            scope, trace = await resolve_query_scope_async(
                request,
                corpus,
                application.scholarly_registry,
                llm=application.services.retrieval.llm,
            )
            expected = [work_ids[key] for key in item["expected_subject_work_ids"]]
            planner_results.append(
                {
                    "id": item["id"],
                    "query": item["query"],
                    "expected_subject_work_ids": expected,
                    "resolved_subject_work_ids": list(scope.subject_work_ids),
                    "resolution": trace.resolution,
                    "warnings": list(trace.warnings),
                }
            )

        negative = [
            row
            for row in rows
            if row["family"] == "negative_source_attribution"
            and row["candidate_pool"] == int(spec["primary_candidate_pool_size"])
            and row["configuration"] == "FULL_CURRENT"
        ]
        aggregates = _aggregate(
            rows,
            fact_meta=fact_meta,
            gold_by_fact=gold_by_fact,
            guidance=spec.get("decision_guidance") or {},
            claim_coverage=float(coverage["claim_coverage"]),
        )
        benchmark_status = "FAIL" if hard_failures else "PASS"
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "benchmark_status": benchmark_status,
            "run_metadata": {
                "run_root": str(run_root.resolve()),
                "dataset": dataset,
                "fixture_schema_version": spec.get("schema_version"),
                "corpus_work_count": len(corpus.work_titles),
                "corpus_chunk_count": len(corpus.chunks),
                "core_work_count": len(CORE_PAPERS),
                "candidate_pool_sizes": pool_sizes,
                "primary_candidate_pool_size": spec["primary_candidate_pool_size"],
                "final_top_k": top_k,
                "reranker_enabled": bool(settings.retrieval.rerank_enabled),
                "runtime": runtime,
                "paper_work_ids": work_ids,
            },
            "fixture_resolution": {
                "gold_by_fact": gold_by_fact,
                "unresolved_fact_ids": unresolved,
            },
            "claim_coverage": coverage,
            "configurations": spec["configurations"],
            "per_case_results": rows,
            "per_family_results": aggregates["by_family"],
            "aggregate_metrics": aggregates["by_configuration"],
            "unique_claim_rescue": aggregates["unique_claim_rescue"],
            "claim_redundancy": aggregates["claim_redundancy"],
            "evidence_input": aggregates["evidence_input"],
            "negative_controls": negative,
            "planner_diagnostics": planner_results,
            "recommendation_inputs": aggregates["recommendation_inputs"],
            "recommendation": aggregates["recommendation"],
            "hard_failures": hard_failures,
            "warnings": warnings,
            "breakdowns": {
                "target_mention_mode": aggregates["by_target_mention_mode"],
                "necessity_priority": aggregates["by_necessity_priority"],
            },
        }
        _atomic_json(report_path, report)
        _atomic_json(review_path, {"claims": manual_review})
        return report
    finally:
        await application.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("data/validation/runs/scholarly-work-reference"),
    )
    parser.add_argument("--dataset", default="paperos-scholarly-work-reference")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--with-synthesis", action="store_true")
    parser.add_argument(
        "--pools",
        type=int,
        nargs="+",
        default=None,
        help="Override candidate pool sizes (default: 40 80 160).",
    )
    args = parser.parse_args()
    report = asyncio.run(
        run(
            args.run_root,
            args.dataset,
            resume=args.resume,
            with_synthesis=args.with_synthesis,
            pools=args.pools,
        )
    )
    summary = {
        "benchmark_status": report["benchmark_status"],
        "recommendation": report["recommendation"],
        "aggregate_metrics": report["aggregate_metrics"],
        "unique_claim_rescue": report["unique_claim_rescue"],
        "claim_redundancy": report["claim_redundancy"],
        "claim_coverage": report["claim_coverage"]["claim_coverage"],
        "evidence_input": report["evidence_input"],
        "hard_failures": report["hard_failures"],
        "report_path": str(
            args.run_root.resolve() / "logs" / "contracts" / REPORT_NAME
        ),
        "review_path": str(
            args.run_root.resolve() / "logs" / "contracts" / REVIEW_NAME
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if report["benchmark_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
