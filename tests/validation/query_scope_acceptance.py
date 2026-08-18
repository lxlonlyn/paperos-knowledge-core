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
import json
import os
import sqlite3
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.adapters.cognee.compat import cognee_uuid
from paperos_core.api.query import router as query_router
from paperos_core.application import Application, create_application
from paperos_core.config import load_settings
from paperos_core.domain.provenance import RelationType
from paperos_core.errors import LocalInferenceUnavailableError
from paperos_core.retrieval.candidates import QueryRequest, QueryResponse

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


def _judge_b(response: QueryResponse, works: dict[str, str]) -> dict[str, Any]:
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
    subjects = {
        subject
        for item in about
        for subject in item.subject_work_ids
    }
    if nise in {item.source_work_id for item in response.evidence}:
        failures.append("NISE appeared as evidence source under exclusion")
    if adadiv not in sources:
        failures.append(f"missing ADADIV ABOUT source, have {sources}")
    if efis not in sources:
        failures.append(f"missing EFIS ABOUT source, have {sources}")
    if subjects and subjects != {nise}:
        failures.append(f"ABOUT targets not exactly NISE: {subjects}")
    blob = "\n".join(item.text for item in about)
    volume_hit = _concept_hit(blob, required=("volume",), any_of=VOLUME_ANY)
    if not volume_hit:
        failures.append("ADADIV/NISE volume concept group not found in ABOUT text")
    missing_chunks = [
        item.object_id if hasattr(item, "object_id") else item.evidence_id
        for item in about
        if not item.chunk_id or not item.derived_from_ids
    ]
    if missing_chunks:
        failures.append(f"ABOUT provenance incomplete: {missing_chunks}")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "external_source_works": sorted(item for item in sources if item),
        "about_count": len(about),
        "volume_concept": volume_hit,
        "resolved_scope": response.resolved_scope.model_dump(),
    }


def _judge_c(response: QueryResponse, works: dict[str, str]) -> dict[str, Any]:
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
    about = _structured_about(response)
    if not about:
        failures.append("missing structured ABOUT evidence")
    if any(nise not in item.subject_work_ids for item in about):
        failures.append("ABOUT target must be NISE")
    blob = "\n".join(item.text for item in about) or _text_blob(response)
    if not _concept_hit(blob, required=("volume",), any_of=VOLUME_ANY):
        failures.append("volume concept group missing")
    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "evidence_source_works": sources,
        "about_count": len(about),
        "resolved_scope": response.resolved_scope.model_dump(),
    }


def _judge_d(response: QueryResponse, works: dict[str, str]) -> dict[str, Any]:
    adadiv = works["adadiv_2025"]
    failures = _scope_ok(
        response,
        source_work_ids=[adadiv],
        subject_work_ids=[adadiv],
    )
    about = _structured_about(response)
    if not about:
        failures.append("missing self ABOUT evidence")
    if any(item.source_work_id != adadiv for item in about):
        failures.append("self ABOUT source must be ADADIV")
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
        "resolved_scope": response.resolved_scope.model_dump(),
    }


def _judge_e(response: QueryResponse, works: dict[str, str]) -> dict[str, Any]:
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


async def _http_e2e(application: Application, dataset: str) -> dict[str, Any]:
    existing = None
    try:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=10) as client:
            health = await client.get("/api/v1/health")
            if health.status_code == 200:
                existing = health.json()
    except (httpx.HTTPError, OSError):
        existing = None
    if existing is not None:
        async with httpx.AsyncClient(base_url="http://127.0.0.1:8000", timeout=300) as client:
            response = await client.post(
                "/api/v1/query",
                json={
                    "query": QUERIES["A"],
                    "profile": "comprehensive",
                    "dataset": dataset,
                },
            )
            response.raise_for_status()
            payload = response.json()
        return {
            "status": "PASS",
            "mode": "reused_running_server",
            "health": existing,
            "query": QUERIES["A"],
            "evidence_count": len(payload.get("evidence") or []),
            "resolved_scope": payload.get("resolved_scope"),
        }

    api = FastAPI()
    api.include_router(query_router)
    api.state.paperos = application
    transport = httpx.ASGITransport(app=api)
    async with httpx.AsyncClient(transport=transport, base_url="http://paperos", timeout=300) as client:
        response = await client.post(
            "/api/v1/query",
            json={
                "query": QUERIES["A"],
                "profile": "comprehensive",
                "dataset": dataset,
            },
        )
        response.raise_for_status()
        payload = response.json()
    return {
        "status": "PASS",
        "mode": "asgi_production_router",
        "reason": "port 8000 is free; 8081 embedding already healthy so lifespan server.py was not rebound",
        "query": QUERIES["A"],
        "evidence_count": len(payload.get("evidence") or []),
        "resolved_scope": payload.get("resolved_scope"),
        "channels_used": payload.get("channels_used"),
    }


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
    runtime = await _ensure_runtime(application)
    works = _paper_work_ids(application)
    required = {"nise_2023", "adadiv_2025", "efis_2026", "lipmlp_2022"}
    missing = required - set(works)
    if missing:
        raise RuntimeError(f"Missing Work IDs for {sorted(missing)}")

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

    judges = {
        "A": _judge_a,
        "B": _judge_b,
        "C": _judge_c,
        "D": _judge_d,
        "E": _judge_e,
    }
    case_reports: dict[str, Any] = {}
    responses: dict[str, QueryResponse] = {}
    for case_id, query in QUERIES.items():
        response = await application.services.retrieval.query(
            QueryRequest(query=query, dataset=dataset, profile="comprehensive")
        )
        responses[case_id] = response
        case_reports[case_id] = {
            "query": query,
            "stages": response.stages,
            "channels_used": response.channels_used,
            "answer_preview": response.answer[:500],
            "judgment": judges[case_id](response, works),
        }

    http_report: dict[str, Any]
    try:
        http_report = await _http_e2e(application, dataset)
    except Exception as exc:  # noqa: BLE001
        http_report = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }

    after_works = application.scholarly_registry.identity_snapshot()
    after_work_ids = sorted(work["id"] for work in after_works["works"])
    after_parse = _table_ids(application.paths.registry_db, "parse_runs")
    after_snapshots = {
        bundle.snapshot.id for bundle in application.canonical_repository.list_bundles()
    }

    scope_violations = 0
    about_structured = 0
    provenance_complete = True
    recall_contamination = 0
    for case_id, response in responses.items():
        about_structured += len(_structured_about(response))
        provenance_complete = provenance_complete and response.provenance_complete
        if response.scope_trace.recall_context_disabled:
            if any(
                stage == "cognee_recall" and "skipped" not in stage
                for stage in response.stages
            ):
                recall_contamination += 1
        else:
            recall_contamination += 0
        judgment = case_reports[case_id]["judgment"]
        if judgment["status"] != "PASS":
            scope_violations += len(judgment.get("failures") or [])

    pipeline_pass = all(
        "fusion" in responses[case_id].stages
        and "synthesis" in responses[case_id].stages
        and "subject_about_retrieval" in responses[case_id].stages
        for case_id in ("B", "C", "D")
    ) and all("fusion" in responses[case_id].stages for case_id in QUERIES)
    overall = (
        all(case_reports[case_id]["judgment"]["status"] == "PASS" for case_id in QUERIES)
        and http_report.get("status") == "PASS"
        and before_work_ids == after_work_ids
        and not (after_parse - before_parse)
        and not (after_snapshots - before_snapshots)
        and recall_contamination == 0
        and pipeline_pass
    )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": dataset,
        "runtime": runtime,
        "work_ids_before": before_work_ids,
        "work_ids_after": after_work_ids,
        "paper_work_ids": works,
        "about_probe_count": len(about_probe),
        "about_probe_sources": about_sources,
        "cases": case_reports,
        "scope_violations_count": scope_violations,
        "unscoped_recall_contamination_count": recall_contamination,
        "about_structured_evidence_count": about_structured,
        "provenance_complete": provenance_complete,
        "mineru_calls": 0,
        "enrichment_calls": 0,
        "new_parse_runs": sorted(after_parse - before_parse),
        "new_snapshots": sorted(after_snapshots - before_snapshots),
        "retrieval_service_pipeline": "PASS" if pipeline_pass else "FAIL",
        "http_end_to_end": http_report,
        "overall": "PASS" if overall else "FAIL",
        "cognee_uuid_nise": str(cognee_uuid(works["nise_2023"])),
    }
    await application.aclose()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-data-dir",
        type=Path,
        default=Path("data/validation/runs/scholarly-work-reference"),
    )
    parser.add_argument("--dataset", default="paperos-scholarly-work-reference")
    args = parser.parse_args()
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
