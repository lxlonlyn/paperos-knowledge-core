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
    is_claim_object_type,
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
CASES_NAME = "claim-about-ablation-cases.jsonl"
CLAIM_BLIND_CONFIGS = frozenset({"NO_CLAIM", "CITE_SCOPE_RAG", "CITE_ANCHOR_RAG"})
PRIMARY_CLAIM_CONFIG = "FULL_CLAIM_NO_PRIVILEGE"
EXCLUDED_PRIMARY_FAMILIES = frozenset({"negative_source_attribution"})
HARD_FAMILIES = frozenset({"implicit_subject", "citation_only_multi_target"})


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

def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
                stream.write("\n")
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


def _benign_leak_stage(stage: str) -> bool:
    return "raw_hits_filtered" in stage or stage.endswith("_filtered")


def _leak_failure_records(
    *,
    query_id: str,
    configuration: str,
    channels_used: list[str],
    candidates: list[Any],
    selected: list[dict[str, Any]],
    trace: AblationTrace,
) -> list[dict[str, Any]]:
    """Hard-fail leakage for claim-blind configs; filtered raw hits are OK."""
    failures: list[dict[str, Any]] = []

    def add(
        *,
        stage: str,
        object_id: str | None = None,
        object_type: str | None = None,
        relation_type: str | None = None,
    ) -> None:
        failures.append(
            {
                "query_id": query_id,
                "configuration": configuration,
                "stage": stage,
                "object_id": object_id,
                "object_type": object_type,
                "relation_type": relation_type,
            }
        )

    for event in trace.claim_leakage:
        stage = str(event.get("stage") or "")
        if _benign_leak_stage(stage):
            continue
        add(
            stage=stage,
            object_id=event.get("object_id"),
            object_type=event.get("object_type"),
            relation_type=event.get("relation_type"),
        )

    if "subject_claim" in channels_used:
        add(stage="channels_used", object_type="subject_claim")

    for candidate in candidates:
        object_type = getattr(candidate, "object_type", None)
        channels = list(getattr(candidate, "channels", []) or [])
        if is_claim_object_type(object_type):
            add(
                stage="selected_candidates",
                object_id=getattr(candidate, "object_id", None),
                object_type=object_type,
            )
        if "subject_claim" in channels:
            add(
                stage="selected_candidates",
                object_id=getattr(candidate, "object_id", None),
                object_type=object_type,
                relation_type="subject_claim_channel",
            )

    for item in selected:
        object_type = item.get("object_type")
        channels = list(item.get("channels") or [])
        if is_claim_object_type(object_type if isinstance(object_type, str) else None):
            add(
                stage="final_candidates",
                object_id=item.get("object_id"),
                object_type=object_type if isinstance(object_type, str) else None,
            )
        if "subject_claim" in channels:
            add(
                stage="final_candidates",
                object_id=item.get("object_id"),
                object_type=object_type if isinstance(object_type, str) else None,
                relation_type="subject_claim_channel",
            )

    for relation in trace.graph_traversal:
        if relation.get("relation_type") == RelationType.ABOUT.value:
            add(
                stage="graph_traversal",
                object_id=relation.get("source_canonical_id"),
                relation_type=RelationType.ABOUT.value,
            )

    for seed in trace.graph_seeds:
        object_type = seed.get("object_type")
        if is_claim_object_type(object_type if isinstance(object_type, str) else None):
            add(
                stage="graph_seed",
                object_id=seed.get("object_id"),
                object_type=object_type if isinstance(object_type, str) else None,
            )

    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in failures:
        key = tuple(sorted(item.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _summarize_raw_hits(raw_hits: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for channel, hits in raw_hits.items():
        summary[channel] = {
            "count": len(hits),
            "object_types": sorted(
                {
                    str(item.get("object_type"))
                    for item in hits
                    if item.get("object_type")
                }
            ),
            "sample_object_ids": [
                item.get("object_id") for item in hits[:8] if item.get("object_id")
            ],
        }
    return summary


def _evidence_metrics(
    *,
    candidates: list[Any],
    selected_chunk_ids: list[str],
    corpus: CorpusView,
    channel_candidate_ids: dict[str, list[str]],
    channels_used: list[str],
) -> dict[str, Any]:
    # Digest cost is measured on final selected evidence that carries Claim text,
    # not the entire candidate pool (pool digests inflate no-privilege runs).
    digest_items: list[Any] = []
    for item in candidates:
        object_type = getattr(item, "object_type", "") or ""
        channels = list(getattr(item, "channels", []) or [])
        if object_type in {"claim", "claim_about"} or "subject_claim" in channels:
            digest_items.append(item)
    claim_digest_count = len(digest_items)
    claim_digest_char_count = sum(
        len(getattr(item, "text", "") or "") for item in digest_items
    )

    unique_chunk_ids = list(dict.fromkeys(selected_chunk_ids))
    canonical_chars = 0
    for chunk_id in unique_chunk_ids:
        chunk = corpus.chunks.get(chunk_id)
        if chunk is not None:
            canonical_chars += len(chunk.text or "")

    digest_payload: dict[str, Any] = {
        "digest_candidate_count": 0,
        "canonical_chunks_materialized": len(unique_chunk_ids),
        "canonical_chars_materialized": canonical_chars,
        "digest_prefilter_ratio": None,
    }
    if "subject_claim" in channels_used or channel_candidate_ids.get("subject_claim"):
        claim_ids = list(dict.fromkeys(channel_candidate_ids.get("subject_claim") or []))
        digest_payload["digest_candidate_count"] = len(claim_ids)
        if claim_ids:
            digest_payload["digest_prefilter_ratio"] = len(unique_chunk_ids) / len(
                claim_ids
            )

    return {
        "claim_digest_count": claim_digest_count,
        "claim_digest_char_count": claim_digest_char_count,
        "canonical_evidence_chunk_count": len(unique_chunk_ids),
        "canonical_evidence_char_count": canonical_chars,
        "digest_prefilter": digest_payload,
    }


async def claim_coverage_report(
    application: Application,
    *,
    dataset: str,
    facts: list[dict[str, Any]],
    work_ids: dict[str, str],
    gold_by_fact: dict[str, dict[str, Any]],
    corpus: CorpusView,
) -> dict[str, Any]:
    covered: list[str] = []
    failures: dict[str, list[str]] = {
        "NO_CLAIM_EXTRACTED": [],
        "WRONG_ABOUT_TARGET": [],
        "RIGHT_ABOUT_WRONG_PROVENANCE": [],
        "CLAIM_EXISTS_BUT_SOURCE_CHUNK_MISSING": [],
    }
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
        all_core_targets = [work_ids[key] for key in CORE_PAPERS]
        broad = await application.services.retrieval.compat.incoming_typed_relations(
            all_core_targets,
            dataset_name=dataset,
            relation_type=RelationType.ABOUT.value,
            depth=1,
            limit=500,
        )
        matched_about = False
        matched_provenance = False
        wrong_target_hit = False
        right_about_missing_chunks = False
        right_about_wrong_chunks = False
        for relation in relations:
            if relation.source_work_id != source_work_id:
                continue
            if relation.target_canonical_id not in target_ids:
                continue
            matched_about = True
            chunk_ids = list(relation.source_chunk_ids or [])
            present = [chunk_id for chunk_id in chunk_ids if chunk_id in corpus.chunks]
            missing = [
                chunk_id for chunk_id in chunk_ids if chunk_id not in corpus.chunks
            ]
            if gold_chunks.intersection(present):
                matched_provenance = True
                break
            if missing and not present:
                right_about_missing_chunks = True
            else:
                right_about_wrong_chunks = True
        if not matched_about:
            for relation in broad:
                if relation.source_work_id != source_work_id:
                    continue
                if relation.target_canonical_id not in target_ids:
                    wrong_target_hit = True
                    break
        if matched_provenance:
            covered.append(fact_id)
        elif not matched_about:
            reason = "WRONG_ABOUT_TARGET" if wrong_target_hit else "NO_CLAIM_EXTRACTED"
            failures[reason].append(fact_id)
        elif right_about_missing_chunks and not right_about_wrong_chunks:
            failures["CLAIM_EXISTS_BUT_SOURCE_CHUNK_MISSING"].append(fact_id)
        else:
            failures["RIGHT_ABOUT_WRONG_PROVENANCE"].append(fact_id)
    total = len(facts)
    return {
        "covered_fact_ids": covered,
        "failure_reasons": failures,
        "missing_claim_fact_ids": [
            *failures["NO_CLAIM_EXTRACTED"],
            *failures["WRONG_ABOUT_TARGET"],
            *failures["RIGHT_ABOUT_WRONG_PROVENANCE"],
            *failures["CLAIM_EXISTS_BUT_SOURCE_CHUNK_MISSING"],
        ],
        "wrong_about_target_fact_ids": failures["WRONG_ABOUT_TARGET"],
        "wrong_source_provenance_fact_ids": failures["RIGHT_ABOUT_WRONG_PROVENANCE"],
        "claim_coverage": (len(covered) / total) if total else 0.0,
    }


async def export_manual_review(
    application: Application,
    *,
    dataset: str,
    work_ids: dict[str, str],
    gold_by_fact: dict[str, dict[str, Any]],
    facts: list[dict[str, Any]],
) -> dict[str, Any]:
    core_ids = [work_ids[key] for key in CORE_PAPERS]
    fixture_chunk_ids = {
        chunk_id
        for gold in gold_by_fact.values()
        for chunk_id in gold.get("gold_chunk_ids") or []
    }
    fixture_pairs: set[tuple[str, str]] = set()
    for fact in facts:
        source = work_ids.get(str(fact["source_work"]))
        if source is None:
            continue
        for target_key in _about_targets(fact):
            target = work_ids.get(target_key)
            if target is not None:
                fixture_pairs.add((source, target))

    edges: list[dict[str, Any]] = []
    claim_ids: set[str] = set()
    source_chunks: set[str] = set()
    claims_by_chunk: dict[str, set[str]] = defaultdict(set)
    edges_by_claim: dict[str, int] = defaultdict(int)

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
            claim_ids.add(claim_id)
            edges_by_claim[claim_id] += 1
            chunk_ids = list(relation.source_chunk_ids or [])
            for chunk_id in chunk_ids:
                source_chunks.add(chunk_id)
                claims_by_chunk[chunk_id].add(claim_id)
            in_fixture = (
                (relation.source_work_id, relation.target_canonical_id) in fixture_pairs
                and bool(set(chunk_ids).intersection(fixture_chunk_ids))
            )
            edges.append(
                {
                    "claim_id": claim_id,
                    "claim_text": relation.text,
                    "source_work_id": relation.source_work_id,
                    "subject_work_id": relation.target_canonical_id,
                    "source_chunk_ids": chunk_ids,
                    "roles": list(relation.roles),
                    "review_status": "FIXTURE_COVERED" if in_fixture else "UNREVIEWED",
                }
            )

    claims_per_source_chunk = (
        (sum(len(items) for items in claims_by_chunk.values()) / len(claims_by_chunk))
        if claims_by_chunk
        else 0.0
    )
    about_edges_per_claim = (
        (sum(edges_by_claim.values()) / len(edges_by_claim)) if edges_by_claim else 0.0
    )
    return {
        "unique_claim_count": len(claim_ids),
        "unique_about_edge_count": len(edges),
        "unique_source_chunk_count": len(source_chunks),
        "claims_per_source_chunk": claims_per_source_chunk,
        "about_edges_per_claim": about_edges_per_claim,
        "edges": sorted(
            edges,
            key=lambda item: (
                item["review_status"] != "UNREVIEWED",
                item["claim_id"],
                item["subject_work_id"],
            ),
        ),
    }


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
    required = len(expected)
    fact_recall = (len(matched) / required) if required else 1.0
    if _case_is_min_coverage(case):
        success = len(matched) >= required and not forbidden_hits
    else:
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
            source_ok = (
                bool(observed)
                and observed.issubset(mapped)
                and mapped.issubset(observed)
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


def _primary_rows(
    rows: list[dict[str, Any]], *, pool: int | None = None
) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        if row.get("family") in EXCLUDED_PRIMARY_FAMILIES:
            continue
        if pool is not None and row.get("candidate_pool") != pool:
            continue
        selected.append(row)
    return selected


def _rows_by_config(
    rows: list[dict[str, Any]], *, pool: int
) -> dict[str, list[dict[str, Any]]]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _primary_rows(rows, pool=pool):
        by_config[row["configuration"]].append(row)
    return by_config


def _mean(values: list[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _unique_rescue(
    left_rows: dict[str, dict[str, Any]],
    right_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rescued: list[str] = []
    universe: set[str] = set()
    for query_id, left in left_rows.items():
        right = right_rows.get(query_id)
        if right is None:
            continue
        left_matched = set(left["matched_fact_ids"])
        right_matched = set(right["matched_fact_ids"])
        universe.update(left["matched_fact_ids"])
        universe.update(left["missed_fact_ids"])
        rescued.extend(sorted(left_matched - right_matched))
    unique_ids = sorted(set(rescued))
    rate = (len(unique_ids) / len(universe)) if universe else 0.0
    return {
        "unique_claim_rescue_count": len(unique_ids),
        "unique_claim_rescue_rate": rate,
        "unique_claim_rescue_fact_ids": unique_ids,
    }


def _claim_redundancy(
    claim_rows: dict[str, dict[str, Any]],
    baseline_rows: dict[str, dict[str, Any]],
    *,
    gold_by_fact: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    claim_supported: set[str] = set()
    also_baseline: set[str] = set()
    for query_id, claim in claim_rows.items():
        baseline = baseline_rows.get(query_id)
        if baseline is None:
            continue
        subject_chunks = set(
            claim.get("channel_candidate_chunk_ids", {}).get("subject_claim", [])
        )
        for fact_id in claim["matched_fact_ids"]:
            gold_chunks = set(gold_by_fact.get(fact_id, {}).get("gold_chunk_ids") or [])
            via_claim = bool(subject_chunks.intersection(gold_chunks))
            if not via_claim and "subject_claim" not in (
                claim.get("channels_used") or []
            ):
                # Still count matched facts under claim-enabled config when the
                # subject_claim channel pool was empty/unavailable.
                via_claim = not subject_chunks
            if not via_claim:
                continue
            claim_supported.add(fact_id)
            if fact_id in baseline["matched_fact_ids"]:
                also_baseline.add(fact_id)
    rate = (len(also_baseline) / len(claim_supported)) if claim_supported else 0.0
    return {
        "claim_supported_gold_facts": sorted(claim_supported),
        "also_found_by_baseline": sorted(also_baseline),
        "claim_redundancy_rate": rate,
    }


def _privilege_delta(
    full_rows: dict[str, dict[str, Any]],
    no_priv_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    improved: list[dict[str, Any]] = []
    regressed: list[dict[str, Any]] = []
    unchanged: list[str] = []
    for query_id, full in full_rows.items():
        no_priv = no_priv_rows.get(query_id)
        if no_priv is None:
            continue
        full_recall = float(full["fact_recall"])
        no_priv_recall = float(no_priv["fact_recall"])
        payload = {
            "query_id": query_id,
            "FULL_CURRENT_fact_recall": full_recall,
            "FULL_CLAIM_NO_PRIVILEGE_fact_recall": no_priv_recall,
            "FULL_CURRENT_matched_fact_ids": full["matched_fact_ids"],
            "FULL_CLAIM_NO_PRIVILEGE_matched_fact_ids": no_priv["matched_fact_ids"],
            "FULL_CURRENT_chunk_ids": full.get("final_selected_chunk_ids"),
            "FULL_CLAIM_NO_PRIVILEGE_chunk_ids": no_priv.get(
                "final_selected_chunk_ids"
            ),
            "claim_ids_no_privilege": sorted(
                {
                    claim_id
                    for item in no_priv.get("rank_trace", {}).get("selected") or []
                    for claim_id in (item.get("claim_ids") or [])
                }
            ),
        }
        if no_priv_recall > full_recall + 1e-9:
            improved.append(payload)
        elif no_priv_recall < full_recall - 1e-9:
            regressed.append(payload)
        else:
            unchanged.append(query_id)
    return {
        "cases_improved_without_privilege": improved,
        "cases_regressed_without_privilege": regressed,
        "unchanged": unchanged,
    }


def _cite_equality(
    scope_rows: dict[str, dict[str, Any]],
    anchor_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    candidate_equal = 0
    final_equal = 0
    recall_equal = 0
    compared = 0
    unique_rescue = _unique_rescue(anchor_rows, scope_rows)
    for query_id, scope in scope_rows.items():
        anchor = anchor_rows.get(query_id)
        if anchor is None:
            continue
        compared += 1
        if set(scope.get("candidate_chunk_ids_before_rerank") or []) == set(
            anchor.get("candidate_chunk_ids_before_rerank") or []
        ):
            candidate_equal += 1
        if set(scope.get("final_selected_chunk_ids") or []) == set(
            anchor.get("final_selected_chunk_ids") or []
        ):
            final_equal += 1
        if abs(float(scope["fact_recall"]) - float(anchor["fact_recall"])) < 1e-12:
            recall_equal += 1
    return {
        "citation_anchor_unique_rescue": unique_rescue["unique_claim_rescue_count"],
        "citation_anchor_unique_rescue_fact_ids": unique_rescue[
            "unique_claim_rescue_fact_ids"
        ],
        "candidate_set_equality_rate": (
            candidate_equal / compared if compared else 1.0
        ),
        "final_evidence_equality_rate": (final_equal / compared if compared else 1.0),
        "fact_recall_equality_rate": (recall_equal / compared if compared else 1.0),
        "cases_compared": compared,
    }


def _recommendation(
    *,
    unique_rescue_vs_scope: float,
    unique_rescue_count: int,
    redundancy_vs_scope: float,
    no_priv_recall: float,
    cite_scope_recall: float,
    full_current_recall: float,
    claim_only_recall: float,
    no_claim_recall: float,
    cite_anchor_equal: bool,
    claim_coverage: float,
) -> dict[str, Any]:
    notes: list[str] = []
    recall_gap = no_priv_recall - cite_scope_recall
    if claim_only_recall + 1e-9 < min(no_priv_recall, cite_scope_recall) * 0.85:
        notes.append(
            "CLAIM_ONLY much lower than Claim/CITE baselines; Claim coverage "
            "insufficient to short-circuit Chunk RAG."
        )
    if no_priv_recall > full_current_recall + 1e-9:
        notes.append(
            "FULL_CLAIM_NO_PRIVILEGE > FULL_CURRENT; privileged prepend may harm ranking."
        )
    if cite_anchor_equal:
        notes.append(
            "CITE_SCOPE_RAG == CITE_ANCHOR_RAG; citation_anchor_unique_rescue = 0."
        )
    if no_claim_recall + 0.05 < cite_scope_recall:
        notes.append(
            "NO_CLAIM << CITE_SCOPE_RAG; subject-aware source narrowing helps, "
            "but is not itself proof of ABOUT necessity."
        )

    if unique_rescue_count > 0 and (
        recall_gap >= 0.05 or unique_rescue_vs_scope >= 0.08
    ):
        decision = "KEEP"
        notes.append(
            "FULL_CLAIM_NO_PRIVILEGE clearly beats CITE_SCOPE_RAG with unique rescue."
        )
    elif unique_rescue_count == 0 and redundancy_vs_scope >= 0.8 and (
        abs(recall_gap) <= 0.03 or no_priv_recall <= cite_scope_recall + 0.03
    ):
        decision = "REMOVE_FROM_RETRIEVAL"
        notes.append(
            "No unique Claim rescue vs CITE_SCOPE, high redundancy, and "
            "FULL_CLAIM_NO_PRIVILEGE does not beat CITE_SCOPE; Claim as recall "
            "layer is not necessary (consider optional semantic cache later)."
        )
    elif (
        no_priv_recall > no_claim_recall + 0.03
        and unique_rescue_count == 0
        and abs(recall_gap) <= 0.05
    ):
        decision = "REDUCE"
        notes.append("Claim helps vs NO_CLAIM but not vs CITE_SCOPE; prefer REDUCE.")
    elif no_priv_recall > full_current_recall + 1e-9 and unique_rescue_count == 0:
        decision = "REDUCE"
        notes.append(
            "Privilege appears harmful and Claim lacks unique rescue vs CITE_SCOPE."
        )
    elif claim_coverage < 0.7:
        decision = "INSUFFICIENT_EVIDENCE"
        notes.append("Claim coverage too low for a firm KEEP/REMOVE call.")
    else:
        decision = "INSUFFICIENT_EVIDENCE"
        notes.append("Results are ambiguous under corrected fairness metrics.")

    return {"recommendation": decision, "notes": notes}


def _pool_saturation(rows: list[dict[str, Any]], pools: list[int]) -> bool:
    if len(pools) < 2:
        return False
    by_config_pool: dict[str, dict[int, float]] = defaultdict(dict)
    for row in _primary_rows(rows):
        by_config_pool[row["configuration"]][int(row["candidate_pool"])] = float(
            row["fact_recall"]
        )
    configs = sorted(by_config_pool)
    if not configs:
        return False
    for config_id in configs:
        recalls = [by_config_pool[config_id].get(pool) for pool in pools]
        if any(value is None for value in recalls):
            return False
        if len({round(value, 12) for value in recalls if value is not None}) != 1:
            return False
    return True


def _aggregate(
    rows: list[dict[str, Any]],
    *,
    pools: list[int],
    primary_pool: int,
    fact_meta: dict[str, dict[str, Any]],
    gold_by_fact: dict[str, dict[str, Any]],
    claim_coverage: float,
) -> dict[str, Any]:
    by_pool: dict[int, dict[str, Any]] = {}
    for pool in pools:
        by_config = _rows_by_config(rows, pool=pool)
        config_metrics: dict[str, Any] = {}
        for config_id, items in by_config.items():
            config_metrics[config_id] = {
                "fact_recall_at_final_k": _mean(
                    [float(item["fact_recall"]) for item in items]
                ),
                "gold_evidence_recall_at_candidate_pool": _mean(
                    [
                        float(item["gold_evidence_recall_at_candidate_pool"])
                        for item in items
                    ]
                ),
                "mean_canonical_evidence_chunk_count": _mean(
                    [float(item["canonical_evidence_chunk_count"]) for item in items]
                ),
                "mean_canonical_evidence_char_count": _mean(
                    [float(item["canonical_evidence_char_count"]) for item in items]
                ),
                "mean_claim_digest_count": _mean(
                    [float(item["claim_digest_count"]) for item in items]
                ),
                "mean_claim_digest_char_count": _mean(
                    [float(item["claim_digest_char_count"]) for item in items]
                ),
                "case_count": len(items),
            }
        by_pool[pool] = config_metrics

    primary = _rows_by_config(rows, pool=primary_pool)
    index = {
        config_id: {row["query_id"]: row for row in items}
        for config_id, items in primary.items()
    }
    claim_rows = index.get(PRIMARY_CLAIM_CONFIG, {})
    cite_scope = index.get("CITE_SCOPE_RAG", {})
    cite_anchor = index.get("CITE_ANCHOR_RAG", {})
    full_current = index.get("FULL_CURRENT", {})
    claim_only = index.get("CLAIM_ONLY", {})
    no_claim = index.get("NO_CLAIM", {})

    rescue_scope = _unique_rescue(claim_rows, cite_scope)
    rescue_anchor = _unique_rescue(claim_rows, cite_anchor)
    redundancy_scope = _claim_redundancy(
        claim_rows, cite_scope, gold_by_fact=gold_by_fact
    )
    redundancy_anchor = _claim_redundancy(
        claim_rows, cite_anchor, gold_by_fact=gold_by_fact
    )
    privilege = _privilege_delta(full_current, claim_rows)
    cite_eq = _cite_equality(cite_scope, cite_anchor)

    by_family: dict[str, Any] = {}
    families = sorted(
        {
            row["family"]
            for items in primary.values()
            for row in items
            if row.get("family")
        }
    )
    for family in families:
        family_metrics: dict[str, Any] = {"case_count": 0}
        for config_id, items in primary.items():
            subset = [row for row in items if row["family"] == family]
            if config_id == PRIMARY_CLAIM_CONFIG:
                family_metrics["case_count"] = len(subset)
            family_metrics[config_id] = {
                "fact_recall": _mean([float(row["fact_recall"]) for row in subset]),
                "gold_evidence_recall_at_candidate_pool": _mean(
                    [
                        float(row["gold_evidence_recall_at_candidate_pool"])
                        for row in subset
                    ]
                ),
                "mean_canonical_evidence_char_count": _mean(
                    [float(row["canonical_evidence_char_count"]) for row in subset]
                ),
            }
        by_family[family] = family_metrics

    hard_family_focus = {
        family: by_family[family] for family in HARD_FAMILIES if family in by_family
    }

    expected_universe = sorted(
        {
            fact_id
            for row in claim_rows.values()
            for fact_id in row["matched_fact_ids"] + row["missed_fact_ids"]
        }
    )
    by_mention: dict[str, Any] = defaultdict(lambda: {"rescued": 0, "facts": 0})
    by_priority: dict[str, Any] = defaultdict(lambda: {"rescued": 0, "facts": 0})
    rescued_ids = set(rescue_scope["unique_claim_rescue_fact_ids"])
    for fact_id in expected_universe:
        meta = fact_meta.get(fact_id) or {}
        mention = str(meta.get("target_mention_mode") or "unknown")
        priority = str(meta.get("necessity_priority") or "unknown")
        by_mention[mention]["facts"] += 1
        by_priority[priority]["facts"] += 1
        if fact_id in rescued_ids:
            by_mention[mention]["rescued"] += 1
            by_priority[priority]["rescued"] += 1

    primary_metrics = by_pool.get(primary_pool, {})
    no_priv_recall = primary_metrics.get(PRIMARY_CLAIM_CONFIG, {}).get(
        "fact_recall_at_final_k", 0.0
    )
    cite_scope_recall = primary_metrics.get("CITE_SCOPE_RAG", {}).get(
        "fact_recall_at_final_k", 0.0
    )
    full_recall = primary_metrics.get("FULL_CURRENT", {}).get(
        "fact_recall_at_final_k", 0.0
    )
    claim_only_recall = primary_metrics.get("CLAIM_ONLY", {}).get(
        "fact_recall_at_final_k", 0.0
    )
    no_claim_recall = primary_metrics.get("NO_CLAIM", {}).get(
        "fact_recall_at_final_k", 0.0
    )
    recommendation = _recommendation(
        unique_rescue_vs_scope=float(rescue_scope["unique_claim_rescue_rate"]),
        unique_rescue_count=int(rescue_scope["unique_claim_rescue_count"]),
        redundancy_vs_scope=float(redundancy_scope["claim_redundancy_rate"]),
        no_priv_recall=float(no_priv_recall),
        cite_scope_recall=float(cite_scope_recall),
        full_current_recall=float(full_recall),
        claim_only_recall=float(claim_only_recall),
        no_claim_recall=float(no_claim_recall),
        cite_anchor_equal=(
            cite_eq["citation_anchor_unique_rescue"] == 0
            and cite_eq["fact_recall_equality_rate"] >= 0.999
        ),
        claim_coverage=claim_coverage,
    )

    return {
        "by_configuration_by_pool": {
            str(pool): metrics for pool, metrics in by_pool.items()
        },
        "by_configuration": primary_metrics,
        "by_family": by_family,
        "hard_family_focus": hard_family_focus,
        "by_target_mention_mode": dict(by_mention),
        "by_necessity_priority": dict(by_priority),
        "unique_claim_rescue_vs_cite_scope": rescue_scope,
        "unique_claim_rescue_vs_cite_anchor": rescue_anchor,
        "claim_redundancy_vs_cite_scope": redundancy_scope,
        "claim_redundancy_vs_cite_anchor": redundancy_anchor,
        "privilege_comparison": privilege,
        "cite_scope_vs_cite_anchor": cite_eq,
        "candidate_pool_saturation_observed": _pool_saturation(rows, pools),
        "evidence_input": {
            config_id: {
                "mean_claim_digest_char_count": metrics.get(
                    "mean_claim_digest_char_count", 0.0
                ),
                "mean_canonical_evidence_chunk_count": metrics.get(
                    "mean_canonical_evidence_chunk_count", 0.0
                ),
                "mean_canonical_evidence_char_count": metrics.get(
                    "mean_canonical_evidence_char_count", 0.0
                ),
            }
            for config_id, metrics in primary_metrics.items()
        },
        "recommendation": recommendation["recommendation"],
        "recommendation_notes": recommendation["notes"],
        "recommendation_inputs": {
            "FULL_CLAIM_NO_PRIVILEGE_fact_recall": no_priv_recall,
            "CITE_SCOPE_RAG_fact_recall": cite_scope_recall,
            "FULL_CURRENT_fact_recall": full_recall,
            "CLAIM_ONLY_fact_recall": claim_only_recall,
            "NO_CLAIM_fact_recall": no_claim_recall,
            "unique_claim_rescue_rate_vs_cite_scope": rescue_scope[
                "unique_claim_rescue_rate"
            ],
            "claim_redundancy_rate_vs_cite_scope": redundancy_scope[
                "claim_redundancy_rate"
            ],
            "claim_coverage": claim_coverage,
        },
    }


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
    corpus: CorpusView,
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

    selected_chunk_ids = list(trace.selected_chunk_ids) or [
        item.chunk_id for item in response.evidence
    ]
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
    evidence_metrics = _evidence_metrics(
        candidates=list(response.candidates),
        selected_chunk_ids=selected_chunk_ids,
        corpus=corpus,
        channel_candidate_ids=dict(trace.channel_candidate_ids),
        channels_used=list(response.channels_used),
    )
    claim_leakage: list[dict[str, Any]] = []
    if config["id"] in CLAIM_BLIND_CONFIGS:
        claim_leakage = _leak_failure_records(
            query_id=str(case["id"]),
            configuration=str(config["id"]),
            channels_used=list(response.channels_used),
            candidates=list(response.candidates),
            selected=list(trace.selected_candidates),
            trace=trace,
        )

    rank_trace = {
        "raw_hits": _summarize_raw_hits(dict(trace.raw_hits)),
        "raw_hits_full": dict(trace.raw_hits),
        "channel_candidate_ids": dict(trace.channel_candidate_ids),
        "channel_candidate_chunk_ids": dict(trace.channel_candidate_chunk_ids),
        "fused_candidates": list(trace.fused_candidates),
        "post_dedup_chunk_ids": list(trace.post_dedup_chunk_ids),
        "duplicate_chunk_candidates_before_dedup": (
            trace.duplicate_chunk_candidates_before_dedup
        ),
        "unique_chunks_after_dedup": trace.unique_chunks_after_dedup,
        "reranked_chunk_ids": list(trace.reranked_chunk_ids),
        "selected": list(trace.selected_candidates),
        "graph_seeds": list(trace.graph_seeds),
        "graph_traversal": list(trace.graph_traversal),
        "claim_leakage_events": list(trace.claim_leakage),
    }
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
        "channel_candidate_ids": trace.channel_candidate_ids,
        "reranked_chunk_ids": list(trace.reranked_chunk_ids),
        "final_selected_chunk_ids": selected_chunk_ids,
        "source_work_ids": source_work_ids,
        "retrieval_latency_ms": trace.retrieval_latency_ms,
        "rerank_latency_ms": trace.rerank_latency_ms,
        "provenance_complete": response.provenance_complete,
        "citation_source_work_ids": list(trace.citation_source_work_ids),
        "claim_leakage": claim_leakage,
        "rank_trace": rank_trace,
        **evidence_metrics,
        **score,
    }


def _reranker_state(settings: Any, runtime: dict[str, Any]) -> dict[str, Any]:
    enabled = bool(settings.retrieval.rerank_enabled)
    local_health = (
        (runtime.get("local_inference") or {}).get("health")
        if isinstance(runtime.get("local_inference"), dict)
        else None
    )
    payload: dict[str, Any] = {
        "reranker_enabled": enabled,
        "production_reranker_configured": enabled,
        "reranker_provider": "local_inference" if enabled else None,
        "reranker_model": (
            str(settings.local_inference.reranker_model_path) if enabled else None
        ),
        "local_inference_health": local_health,
        "reranker_loaded": None,
        "blocked_reranker": False,
    }
    if not enabled:
        return payload
    if isinstance(local_health, dict):
        loaded = local_health.get("reranker_loaded")
        if loaded is None:
            loaded = local_health.get("reranker")
        if isinstance(loaded, dict):
            loaded = loaded.get("loaded")
        payload["reranker_loaded"] = loaded
        if loaded is False:
            payload["blocked_reranker"] = True
    return payload


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
    reranker = _reranker_state(settings, runtime)
    if reranker.get("blocked_reranker"):
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "benchmark_status": "BLOCKED_RERANKER",
            "run_metadata": {
                "run_root": str(run_root.resolve()),
                "dataset": dataset,
                "runtime": runtime,
                **reranker,
            },
            "hard_failures": ["production reranker enabled but not loaded"],
        }
        report_path = run_root / "logs" / "contracts" / REPORT_NAME
        _atomic_json(report_path, report)
        await application.aclose()
        return report

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
            for fact_id in _case_expected_facts(case)
            + list(case.get("forbidden_fact_ids") or [])
        }
    ]
    hard_failures: list[str] = []
    claim_leakage_failures: list[dict[str, Any]] = []
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
        corpus=corpus,
    )
    manual_review = await export_manual_review(
        application,
        dataset=dataset,
        work_ids=work_ids,
        gold_by_fact=gold_by_fact,
        facts=[fact_by_id[fact_id] for fact_id in fact_meta],
    )

    report_path = run_root / "logs" / "contracts" / REPORT_NAME
    review_path = run_root / "logs" / "contracts" / REVIEW_NAME
    cases_path = run_root / "logs" / "contracts" / CASES_NAME
    existing_rows: list[dict[str, Any]] = []
    if resume and report_path.is_file():
        previous = _load_json(report_path)
        existing_rows = list(previous.get("per_case_results") or [])
    done = {
        (row["configuration"], row["candidate_pool"], row["query_id"])
        for row in existing_rows
    }

    pool_sizes = [int(item) for item in (pools or list(spec["candidate_pool_sizes"]))]
    # Aggregation always covers every pool present in retained rows so resume
    # with a subset (e.g. --pools 80 160) does not drop primary-pool metrics.
    top_k = int(spec["final_top_k"])
    primary_pool = int(spec["primary_candidate_pool_size"])
    rows = list(existing_rows)
    warnings: list[str] = []
    if not reranker["production_reranker_configured"]:
        warnings.append(
            "production_reranker_configured=false; results are no-reranker condition."
        )

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
                        corpus=corpus,
                        with_synthesis=with_synthesis,
                    )
                    if row["claim_leakage"]:
                        claim_leakage_failures.extend(row["claim_leakage"])
                        hard_failures.append(
                            "claim_leakage:"
                            f"{row['configuration']}:{row['query_id']}:"
                            f"{len(row['claim_leakage'])}"
                        )
                    if row["hard_error"]:
                        hard_failures.append(
                            "negative_control_violation:"
                            f"{case['id']}:{row['forbidden_fact_hits']}"
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
                            "claim_leakage": claim_leakage_failures,
                        },
                    )
                    _atomic_jsonl(cases_path, rows)

        planner_results: list[dict[str, Any]] = []
        for item in queries.get("planner_diagnostics") or []:
            request = QueryRequest(
                query=str(item["query"]),
                profile=RetrievalProfile.COMPREHENSIVE,
                dataset=dataset,
            )
            scope, scope_trace = await resolve_query_scope_async(
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
                    "resolution": scope_trace.resolution,
                    "warnings": list(scope_trace.warnings),
                }
            )

        negative = [
            row
            for row in rows
            if row["family"] == "negative_source_attribution"
            and row["candidate_pool"] == primary_pool
        ]
        aggregates = _aggregate(
            rows,
            pools=sorted(
                {
                    *pool_sizes,
                    *[int(row["candidate_pool"]) for row in rows],
                    primary_pool,
                }
            ),
            primary_pool=primary_pool,
            fact_meta=fact_meta,
            gold_by_fact=gold_by_fact,
            claim_coverage=float(coverage["claim_coverage"]),
        )
        if unresolved and not rows:
            benchmark_status = "INSUFFICIENT_DATA"
        elif hard_failures or claim_leakage_failures:
            benchmark_status = "FAIL"
        else:
            benchmark_status = "PASS"
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
                "primary_candidate_pool_size": primary_pool,
                "final_top_k": top_k,
                "runtime": runtime,
                "paper_work_ids": work_ids,
                **reranker,
            },
            "fixture_resolution": {
                "gold_by_fact": gold_by_fact,
                "unresolved_fact_ids": unresolved,
            },
            "claim_coverage": coverage,
            "configurations": spec["configurations"],
            "per_case_results": rows,
            "per_family_results": aggregates["by_family"],
            "hard_family_focus": aggregates["hard_family_focus"],
            "aggregate_metrics": aggregates["by_configuration"],
            "aggregate_metrics_by_pool": aggregates["by_configuration_by_pool"],
            "unique_claim_rescue_vs_cite_scope": aggregates[
                "unique_claim_rescue_vs_cite_scope"
            ],
            "unique_claim_rescue_vs_cite_anchor": aggregates[
                "unique_claim_rescue_vs_cite_anchor"
            ],
            "claim_redundancy_vs_cite_scope": aggregates[
                "claim_redundancy_vs_cite_scope"
            ],
            "claim_redundancy_vs_cite_anchor": aggregates[
                "claim_redundancy_vs_cite_anchor"
            ],
            "privilege_comparison": aggregates["privilege_comparison"],
            "cite_scope_vs_cite_anchor": aggregates["cite_scope_vs_cite_anchor"],
            "candidate_pool_saturation_observed": aggregates[
                "candidate_pool_saturation_observed"
            ],
            "evidence_input": aggregates["evidence_input"],
            "negative_controls": negative,
            "planner_diagnostics": planner_results,
            "claim_leakage": claim_leakage_failures,
            "recommendation_inputs": aggregates["recommendation_inputs"],
            "recommendation_notes": aggregates["recommendation_notes"],
            "recommendation": aggregates["recommendation"],
            "hard_failures": hard_failures,
            "warnings": warnings,
            "breakdowns": {
                "target_mention_mode": aggregates["by_target_mention_mode"],
                "necessity_priority": aggregates["by_necessity_priority"],
            },
        }
        _atomic_json(report_path, report)
        _atomic_json(review_path, manual_review)
        _atomic_jsonl(cases_path, rows)
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
    privilege = report.get("privilege_comparison") or {}
    summary = {
        "benchmark_status": report["benchmark_status"],
        "recommendation": report.get("recommendation"),
        "recommendation_notes": report.get("recommendation_notes"),
        "aggregate_metrics": report.get("aggregate_metrics"),
        "aggregate_metrics_by_pool": report.get("aggregate_metrics_by_pool"),
        "unique_claim_rescue_vs_cite_scope": report.get(
            "unique_claim_rescue_vs_cite_scope"
        ),
        "unique_claim_rescue_vs_cite_anchor": report.get(
            "unique_claim_rescue_vs_cite_anchor"
        ),
        "claim_redundancy_vs_cite_scope": report.get("claim_redundancy_vs_cite_scope"),
        "claim_redundancy_vs_cite_anchor": report.get(
            "claim_redundancy_vs_cite_anchor"
        ),
        "privilege_comparison": {
            "cases_improved_without_privilege": len(
                privilege.get("cases_improved_without_privilege") or []
            ),
            "cases_regressed_without_privilege": len(
                privilege.get("cases_regressed_without_privilege") or []
            ),
            "unchanged": len(privilege.get("unchanged") or []),
        },
        "cite_scope_vs_cite_anchor": report.get("cite_scope_vs_cite_anchor"),
        "candidate_pool_saturation_observed": report.get(
            "candidate_pool_saturation_observed"
        ),
        "claim_coverage": (report.get("claim_coverage") or {}).get("claim_coverage"),
        "claim_coverage_failure_reasons": (report.get("claim_coverage") or {}).get(
            "failure_reasons"
        ),
        "evidence_input": report.get("evidence_input"),
        "production_reranker_configured": (report.get("run_metadata") or {}).get(
            "production_reranker_configured"
        ),
        "hard_failures": report.get("hard_failures"),
        "claim_leakage_count": len(report.get("claim_leakage") or []),
        "report_path": str(
            args.run_root.resolve() / "logs" / "contracts" / REPORT_NAME
        ),
        "review_path": str(
            args.run_root.resolve() / "logs" / "contracts" / REVIEW_NAME
        ),
        "cases_path": str(args.run_root.resolve() / "logs" / "contracts" / CASES_NAME),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if report["benchmark_status"] not in {"PASS"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
