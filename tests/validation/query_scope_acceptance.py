"""Task 03 scope-aware retrieval real acceptance.

Consumes the retained Task 02 scholarly-work-reference corpus. Does not call
MinerU, semantic enrichment, or rebuild.

    python tests/validation/query_scope_acceptance.py \\
      --live-data-dir data/validation/runs/scholarly-work-reference \\
      --dataset paperos-scholarly-work-reference
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.adapters.cognee.compat import cognee_uuid
from paperos_core.application import Application, create_application
from paperos_core.config import load_settings
from paperos_core.domain.provenance import RelationType
from paperos_core.errors import LocalInferenceUnavailableError
from paperos_core.retrieval.candidates import QueryRequest, QueryResponse
from paperos_core.retrieval.corpus import CorpusView
from paperos_core.retrieval.scope import (
    build_mention_index,
    proven_subject_work_ids,
)

MANIFEST = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "scholarly_work_reference"
    / "reference_corpus_manifest.json"
)
REPORT_NAME = "query-scope-acceptance.json"

QUERIES = {
    "A": "只根据 NISE 原文，说明它的方法和限制。",
    "B": "NISE 有哪些后来论文指出的问题？",
    "C": "只根据 Volume Preserving Neural Shape Morphing，NISE 有哪些问题？",
    "D": "Volume Preserving Neural Shape Morphing 自己报告了哪些限制？",
    "E": "比较 NISE、Volume Preserving Neural Shape Morphing 和 EFIS 在 volume preservation / intermediate shape 方面的差异。",
}

VOLUME_ANY = ("control", "disappear", "reappear", "preserve", "intermediate", "property")
SELF_ANY = ("blob", "detach", "reattach", "lse")
PREFERRED_EXTERNAL_TITLE = "geometry processing with neural fields"


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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


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


def _table_ids(db_path: Path, table: str) -> set[str]:
    with sqlite3.connect(db_path) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        if not exists:
            return set()
        return {
            str(row[0])
            for row in connection.execute(f"SELECT id FROM {table}").fetchall()
        }


def _text_blob(response: QueryResponse) -> str:
    parts = [item.text.casefold() for item in response.evidence]
    parts.extend(item.text.casefold() for item in response.candidates)
    return "\n".join(parts)


def _concept_hit(text: str, *, required: tuple[str, ...], any_of: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    if any(token not in lowered for token in required):
        return False
    return any(token in lowered for token in any_of)


def _structured_about(response: QueryResponse) -> list[Any]:
    return [
        item
        for item in response.evidence
        if "subject_claim" in item.channels
    ]


def _enrichment_fingerprint(application: Application) -> dict[str, str]:
    root = application.paths.cognee / "enrichment"
    if not root.is_dir():
        return {}
    digest: dict[str, str] = {}
    for path in sorted(root.glob("*.json")):
        digest[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest


def _mention_index(application: Application):
    return build_mention_index(
        {work.id: work for work in application.scholarly_registry.list_works()}
    )


def _evidence_subject_ids(
    item: Any,
    subject_work_ids: list[str],
    mention_index: dict[str, tuple[str, ...]],
) -> list[str]:
    return proven_subject_work_ids(
        text=item.text,
        structured_subject_ids=list(item.subject_work_ids or []),
        derived_from_ids=list(item.derived_from_ids or []),
        subject_work_ids=subject_work_ids,
        mention_index=mention_index,
    )


def _count_scope_violations(
    response: QueryResponse,
    mention_index: dict[str, tuple[str, ...]],
) -> int:
    scope = response.resolved_scope
    violations = 0
    source_allowed = set(scope.source_work_ids)
    excluded = set(scope.exclude_source_work_ids)
    work_set = set(scope.work_set_work_ids)
    for item in response.evidence:
        if source_allowed and item.source_work_id not in source_allowed:
            violations += 1
        if item.source_work_id in excluded:
            violations += 1
        if work_set and item.source_work_id not in work_set:
            violations += 1
        if scope.subject_work_ids and not _evidence_subject_ids(
            item, scope.subject_work_ids, mention_index
        ):
            violations += 1
    return violations


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


def _scope_ok(response: QueryResponse, **expected: list[str]) -> list[str]:
    failures = []
    scope = response.resolved_scope
    for field, values in expected.items():
        actual = getattr(scope, field)
        if actual != values:
            failures.append(f"{field} expected {values}, got {actual}")
    return failures


def _judge_a(response: QueryResponse, works: dict[str, str]) -> dict[str, Any]:
    nise = works["nise_2023"]
    failures = _scope_ok(response, source_work_ids=[nise])
    sources = [item.source_work_id for item in response.evidence]
    if not response.evidence:
        failures.append("no evidence")
    if any(item != nise for item in sources):
        failures.append(f"non-NISE evidence sources: {sources}")
    if response.scope_trace.recall_context_disabled is not True:
        failures.append("recall_context should be disabled under hard source scope")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "evidence_source_works": sources,
        "resolved_scope": response.resolved_scope.model_dump(),
    }


def _judge_b(
    response: QueryResponse,
    works: dict[str, str],
    mention_index: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    nise = works["nise_2023"]
    adadiv = works["adadiv_2025"]
    efis = works["efis_2026"]
    failures = _scope_ok(
        response,
        subject_work_ids=[nise],
        exclude_source_work_ids=[nise],
    )
    about = _structured_about(response)
    if not about:
        failures.append("no subject_claim ABOUT evidence")
    sources = {item.source_work_id for item in about}
    all_sources = {item.source_work_id for item in response.evidence}
    if nise in all_sources:
        failures.append("NISE appeared as evidence source under exclusion")
    if adadiv not in sources:
        failures.append(f"missing ADADIV ABOUT source, have {sources}")
    if efis not in sources:
        failures.append(f"missing EFIS ABOUT source, have {sources}")
    unrelated = [
        item.evidence_id
        for item in response.evidence
        if nise not in _evidence_subject_ids(item, [nise], mention_index)
    ]
    if unrelated:
        failures.append(f"final evidence lacking NISE subject relevance: {unrelated}")
    blob = "\n".join(item.text for item in about)
    volume_hit = _concept_hit(blob, required=("volume",), any_of=VOLUME_ANY)
    if not volume_hit:
        failures.append("ADADIV/NISE volume concept group not found in ABOUT text")
    missing_chunks = [
        item.evidence_id
        for item in about
        if not item.chunk_id or not item.derived_from_ids
    ]
    if missing_chunks:
        failures.append(f"ABOUT provenance incomplete: {missing_chunks}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "external_source_works": sorted(item for item in all_sources if item),
        "about_count": len(about),
        "all_evidence_subject_relevant": not unrelated,
        "volume_concept": volume_hit,
        "resolved_scope": response.resolved_scope.model_dump(),
    }


def _judge_c(
    response: QueryResponse,
    works: dict[str, str],
    mention_index: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    nise = works["nise_2023"]
    adadiv = works["adadiv_2025"]
    efis = works["efis_2026"]
    failures = _scope_ok(
        response,
        source_work_ids=[adadiv],
        subject_work_ids=[nise],
    )
    sources = [item.source_work_id for item in response.evidence]
    if any(item != adadiv for item in sources):
        failures.append(f"evidence sources must be ADADIV only: {sources}")
    if efis in sources:
        failures.append("EFIS leaked into source+subject evidence")
    unrelated = [
        item.evidence_id
        for item in response.evidence
        if nise not in _evidence_subject_ids(item, [nise], mention_index)
    ]
    if unrelated:
        failures.append(f"final evidence lacking NISE subject relevance: {unrelated}")
    about = _structured_about(response)
    if not about:
        failures.append("missing structured ABOUT evidence")
    blob = "\n".join(item.text for item in about) or _text_blob(response)
    if not _concept_hit(blob, required=("volume",), any_of=VOLUME_ANY):
        failures.append("volume concept group missing")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "evidence_source_works": sources,
        "about_count": len(about),
        "all_evidence_subject_relevant": not unrelated,
        "resolved_scope": response.resolved_scope.model_dump(),
    }


def _judge_d(
    response: QueryResponse,
    works: dict[str, str],
    mention_index: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    adadiv = works["adadiv_2025"]
    failures = _scope_ok(
        response,
        source_work_ids=[adadiv],
        subject_work_ids=[adadiv],
    )
    about = _structured_about(response)
    if not about:
        failures.append("missing self ABOUT evidence")
    if any(item.source_work_id != adadiv for item in response.evidence):
        failures.append("self query evidence source must be ADADIV")
    unrelated = [
        item.evidence_id
        for item in response.evidence
        if adadiv not in _evidence_subject_ids(item, [adadiv], mention_index)
    ]
    if unrelated:
        failures.append(f"final evidence lacking ADADIV subject relevance: {unrelated}")
    if any(adadiv not in item.subject_work_ids for item in about):
        failures.append("self ABOUT target must be ADADIV")
    roles = " ".join(" ".join(item.derived_from_ids) for item in about)
    if "about_role:self" not in roles:
        failures.append("ABOUT role does not include self")
    blob = "\n".join(item.text for item in about)
    if not any(token in blob.casefold() for token in SELF_ANY):
        failures.append("self limitation concept group missing")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "about_count": len(about),
        "all_evidence_subject_relevant": not unrelated,
        "resolved_scope": response.resolved_scope.model_dump(),
    }


def _judge_e(
    response: QueryResponse,
    works: dict[str, str],
    mention_index: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    allowed = {
        works["nise_2023"],
        works["adadiv_2025"],
        works["efis_2026"],
    }
    lipmlp = works["lipmlp_2022"]
    failures = _scope_ok(
        response,
        work_set_work_ids=sorted(allowed),
    )
    sources = [item.source_work_id for item in response.evidence]
    if any(item not in allowed for item in sources if item):
        failures.append(f"evidence escaped work-set: {sources}")
    if lipmlp in sources:
        failures.append("LipMLP leaked into work-set evidence")
    if len({item for item in sources if item}) < 2:
        failures.append(f"work-set evidence too narrow: {sources}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "evidence_source_works": sources,
        "resolved_scope": response.resolved_scope.model_dump(),
    }


def _judge_f(
    response: QueryResponse,
    target: dict[str, Any],
    ingested_work_ids: set[str],
    mention_index: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    work_id = str(target["work_id"])
    failures: list[str] = []
    if work_id not in response.resolved_scope.subject_work_ids:
        failures.append(
            "resolved subject missing external Work "
            f"{work_id}: {response.resolved_scope.subject_work_ids}"
        )
    if response.resolved_scope.source_work_ids:
        failures.append("external subject query must not require a source Work")
    if target["has_document"]:
        failures.append("selected subject Work unexpectedly has a Document")
    about = _structured_about(response)
    if not about:
        failures.append("ABOUT retrieval returned no structured evidence")
    unrelated = [
        item.evidence_id
        for item in response.evidence
        if work_id not in _evidence_subject_ids(
            item, response.resolved_scope.subject_work_ids, mention_index
        )
    ]
    if unrelated:
        failures.append(f"final evidence lacking subject relevance: {unrelated}")
    sources = sorted({item.source_work_id for item in response.evidence if item.source_work_id})
    if any(item not in ingested_work_ids for item in sources):
        failures.append(f"evidence source is not an ingested Work: {sources}")
    if any(not item.chunk_id or not item.derived_from_ids for item in response.evidence):
        failures.append("provenance incomplete")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "work_id": work_id,
        "identity_status": target["identity_status"],
        "title": target["title"],
        "document_exists": target["has_document"],
        "about_count": len(about),
        "source_works": sources,
        "all_evidence_subject_relevant": not unrelated,
        "resolved_scope": response.resolved_scope.model_dump(),
    }


async def _select_external_subject(
    application: Application, dataset: str
) -> dict[str, Any]:
    corpus = CorpusView.load(
        application.paths,
        application.canonical_repository,
        application.registry,
        application.scholarly_registry,
    )
    preferred: dict[str, Any] | None = None
    for work in application.scholarly_registry.list_works():
        has_document = work.id in corpus.document_ids_by_work
        if PREFERRED_EXTERNAL_TITLE not in work.title.casefold():
            continue
        relations = await application.services.retrieval.compat.incoming_typed_relations(
            [work.id],
            dataset_name=dataset,
            relation_type=RelationType.ABOUT.value,
            depth=1,
            limit=200,
        )
        candidate = {
            "work_id": work.id,
            "title": work.title,
            "identity_status": work.identity_status.value,
            "has_document": has_document,
            "about_count": len(relations),
            "query": f"现有论文对 {work.title} 有哪些评价或讨论？",
        }
        if not has_document and relations:
            return candidate
        preferred = candidate
    if preferred is not None:
        raise RuntimeError(
            "Preferred external Work has no unused-document ABOUT evidence: "
            f"{preferred}"
        )
    raise RuntimeError(
        "No un-ingested ScholarlyWork with ABOUT evidence was found for case F."
    )


def _serve(live_data_dir: Path, dataset: str) -> None:
    import uvicorn
    from paperos_core.api.app import create_app

    configured = load_settings()
    settings = configured.model_copy(
        update={
            "data": configured.data.model_copy(
                update={"directory": live_data_dir.resolve(), "dataset": dataset}
            )
        }
    )
    uvicorn.run(
        create_app(settings),
        host=settings.api.host,
        port=settings.api.port,
        log_level="warning",
    )


def _http_ready(payload: dict[str, Any]) -> str | None:
    status = payload.get("status")
    if status == "healthy":
        return None
    components = payload.get("components") if isinstance(payload, dict) else None
    if status != "degraded" or not isinstance(components, dict):
        return f"unexpected health status {status}"
    for name in ("vector", "cognee_graph", "local_models"):
        component = components.get(name) or {}
        if component.get("status") not in {"healthy", "disabled"}:
            return f"{name} not ready: {component}"
    return None


async def _wait_healthy(base_url: str, *, attempts: int = 90) -> dict[str, Any]:
    last_error = "not attempted"
    for _ in range(attempts):
        try:
            async with httpx.AsyncClient(base_url=base_url, timeout=15) as client:
                response = await client.get("/api/v1/health")
                if response.status_code == 200:
                    payload = response.json()
                    reason = _http_ready(payload)
                    if reason is None:
                        return payload
                    last_error = reason
                else:
                    last_error = f"HTTP {response.status_code}"
        except (httpx.HTTPError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        await asyncio.sleep(1)
    raise RuntimeError(f"PaperOS HTTP health check failed: {last_error}")


async def _http_query(
    base_url: str, query: str, dataset: str
) -> QueryResponse:
    async with httpx.AsyncClient(base_url=base_url, timeout=300) as client:
        response = await client.post(
            "/api/v1/query",
            json={
                "query": query,
                "profile": "comprehensive",
                "dataset": dataset,
            },
        )
        response.raise_for_status()
        return QueryResponse.model_validate(response.json())


async def _real_http_e2e(
    settings,
    dataset: str,
    live_data_dir: Path,
    works: dict[str, str],
    mention_index: dict[str, tuple[str, ...]],
) -> dict[str, Any]:
    base_url = f"http://{settings.api.host}:{settings.api.port}"
    process: subprocess.Popen[bytes] | None = None
    log_handle = None
    mode = "reused_running_server"
    try:
        health = await _wait_healthy(base_url, attempts=2)
    except RuntimeError:
        health = None
    if health is None:
        mode = "spawned_create_app_lifespan"
        log_path = live_data_dir / "logs" / "http-e2e-server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("ab", buffering=0)
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--serve",
                "--live-data-dir",
                str(live_data_dir),
                "--dataset",
                dataset,
            ],
            cwd=str(REPOSITORY_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        try:
            health = await _wait_healthy(base_url, attempts=90)
        except Exception as exc:
            output = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
            raise RuntimeError(
                "spawned PaperOS server did not become healthy "
                f"(exit={process.poll()}): {output}"
            ) from exc
    try:
        source_response = await _http_query(base_url, QUERIES["A"], dataset)
        subject_response = await _http_query(base_url, QUERIES["B"], dataset)
        source_judgment = _judge_a(source_response, works)
        subject_judgment = _judge_b(subject_response, works, mention_index)
        status = (
            "PASS"
            if source_judgment["status"] == "PASS"
            and subject_judgment["status"] == "PASS"
            else "FAIL"
        )
        return {
            "status": status,
            "mode": mode,
            "server_command": (
                "reused existing PaperOS on localhost socket"
                if process is None
                else "python tests/validation/query_scope_acceptance.py --serve "
                "(create_app + uvicorn lifespan, official server equivalent)"
            ),
            "health_status": health.get("status"),
            "source_query": {
                "query": QUERIES["A"],
                "judgment": source_judgment,
                "transport": "http://127.0.0.1 socket POST /api/v1/query",
            },
            "subject_query": {
                "query": QUERIES["B"],
                "judgment": subject_judgment,
                "transport": "http://127.0.0.1 socket POST /api/v1/query",
            },
        }
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log_handle is not None and not log_handle.closed:
            log_handle.close()


async def run(live_data_dir: Path, dataset: str) -> dict[str, Any]:
    configured = load_settings()
    settings = configured.model_copy(
        update={
            "data": configured.data.model_copy(
                update={"directory": live_data_dir.resolve(), "dataset": dataset}
            )
        }
    )
    application = create_application(settings)
    before_works = application.scholarly_registry.identity_snapshot()
    before_work_ids = sorted(work["id"] for work in before_works["works"])
    before_parse = _table_ids(application.paths.registry_db, "parse_runs")
    before_snapshots = {
        bundle.snapshot.id for bundle in application.canonical_repository.list_bundles()
    }
    before_enrichment = _enrichment_fingerprint(application)
    runtime = await _ensure_runtime(application)
    works = _paper_work_ids(application)
    required = {"nise_2023", "adadiv_2025", "efis_2026", "lipmlp_2022"}
    missing = required - set(works)
    if missing:
        raise RuntimeError(f"Missing Work IDs for {sorted(missing)}")
    mention_index = _mention_index(application)
    external_target = await _select_external_subject(application, dataset)
    queries = {**QUERIES, "F": str(external_target["query"])}

    about_probe = await application.services.retrieval.compat.incoming_typed_relations(
        [works["nise_2023"]],
        dataset_name=dataset,
        relation_type=RelationType.ABOUT.value,
        depth=1,
        limit=200,
    )
    about_sources = sorted(
        {item.source_work_id for item in about_probe if item.source_work_id}
    )

    case_reports: dict[str, Any] = {}
    responses: dict[str, QueryResponse] = {}
    for case_id, query in queries.items():
        response = await application.services.retrieval.query(
            QueryRequest(query=query, dataset=dataset, profile="comprehensive")
        )
        responses[case_id] = response
        if case_id == "A":
            judgment = _judge_a(response, works)
        elif case_id == "B":
            judgment = _judge_b(response, works, mention_index)
        elif case_id == "C":
            judgment = _judge_c(response, works, mention_index)
        elif case_id == "D":
            judgment = _judge_d(response, works, mention_index)
        elif case_id == "E":
            judgment = _judge_e(response, works, mention_index)
        else:
            judgment = _judge_f(
                response, external_target, set(works.values()), mention_index
            )
        case_reports[case_id] = {
            "query": query,
            "stages": response.stages,
            "channels_used": response.channels_used,
            "answer_preview": response.answer[:500],
            "judgment": judgment,
        }

    after_works = application.scholarly_registry.identity_snapshot()
    after_work_ids = sorted(work["id"] for work in after_works["works"])
    after_parse = _table_ids(application.paths.registry_db, "parse_runs")
    after_snapshots = {
        bundle.snapshot.id for bundle in application.canonical_repository.list_bundles()
    }
    after_enrichment = _enrichment_fingerprint(application)
    await application.aclose()

    http_report: dict[str, Any]
    try:
        http_report = await _real_http_e2e(
            settings, dataset, live_data_dir, works, mention_index
        )
    except Exception as exc:  # noqa: BLE001
        http_report = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }

    scope_violations = 0
    about_structured = 0
    provenance_complete = True
    recall_contamination = 0
    for case_id, response in responses.items():
        about_structured += len(_structured_about(response))
        provenance_complete = provenance_complete and response.provenance_complete
        if response.scope_trace.recall_context_disabled and any(
            stage == "cognee_recall" for stage in response.stages
        ):
            recall_contamination += 1
        scope_violations += _count_scope_violations(response, mention_index)

    pipeline_pass = all(
        "fusion" in responses[case_id].stages
        and "synthesis" in responses[case_id].stages
        and "subject_about_retrieval" in responses[case_id].stages
        for case_id in ("B", "C", "D", "F")
    ) and all("fusion" in responses[case_id].stages for case_id in queries)
    enrichment_unchanged = before_enrichment == after_enrichment
    reingest = before_parse != after_parse or before_snapshots != after_snapshots
    overall = (
        all(case_reports[case_id]["judgment"]["status"] == "PASS" for case_id in queries)
        and http_report.get("status") == "PASS"
        and before_work_ids == after_work_ids
        and not reingest
        and enrichment_unchanged
        and scope_violations == 0
        and recall_contamination == 0
        and pipeline_pass
        and http_report.get("mode") != "asgi_production_router"
    )
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "runtime": runtime,
        "work_ids_before": before_work_ids,
        "work_ids_after": after_work_ids,
        "parse_run_count_before": len(before_parse),
        "parse_run_count_after": len(after_parse),
        "snapshot_count_before": len(before_snapshots),
        "snapshot_count_after": len(after_snapshots),
        "paper_work_ids": works,
        "external_subject": {
            "work_id": external_target["work_id"],
            "identity_status": external_target["identity_status"],
            "document_exists": external_target["has_document"],
            "about_probe_count": external_target["about_count"],
            "title": external_target["title"],
        },
        "about_probe_count": len(about_probe),
        "about_probe_sources": about_sources,
        "cases": case_reports,
        "scope_violations_count": scope_violations,
        "unscoped_recall_contamination_count": recall_contamination,
        "about_structured_evidence_count": about_structured,
        "provenance_complete": provenance_complete,
        "re_ingest_observed": reingest,
        "re_enrichment_observed": not enrichment_unchanged,
        "enrichment_artifacts_unchanged": (
            "PASS" if enrichment_unchanged else "FAIL"
        ),
        "retrieval_service_pipeline": "PASS" if pipeline_pass else "FAIL",
        "http_end_to_end": http_report,
        "overall": "PASS" if overall else "FAIL",
        "cognee_uuid_nise": str(cognee_uuid(works["nise_2023"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-data-dir",
        type=Path,
        default=Path("data/validation/runs/scholarly-work-reference"),
    )
    parser.add_argument("--dataset", default="paperos-scholarly-work-reference")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start the official create_app/uvicorn PaperOS server for HTTP e2e.",
    )
    args = parser.parse_args()
    if args.serve:
        _serve(args.live_data_dir, args.dataset)
        return
    report = asyncio.run(run(args.live_data_dir, args.dataset))
    output = (
        args.live_data_dir.resolve()
        / "logs"
        / "contracts"
        / REPORT_NAME
    )
    _atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["overall"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
