"""Task 02R Claim→ABOUT→ScholarlyWork real acceptance.

Refresh path (validation/maintenance, not rebuild lifecycle):

    production semantic_enrichment_task(reuse_existing=False)
    → persist enrichment for current Snapshots only
    → rebuild(refresh_enrichment=False) with 0 enrichment LLM calls
    → stored graph + live Cognee ABOUT readback

Resume is supported via a state file under the live data dir.

    python tests/validation/claim_about_acceptance.py \\
      --live-data-dir data/validation/runs/scholarly-work-reference \\
      --resume

Smoke one current Snapshot first:

    python tests/validation/claim_about_acceptance.py \\
      --live-data-dir data/validation/runs/scholarly-work-reference \\
      --smoke-one --resume
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.adapters.cognee.compat import cognee_uuid
from paperos_core.adapters.cognee.datapoints import TripletDataPoint
from paperos_core.adapters.cognee.models import canonical_to_datapoints
from paperos_core.adapters.cognee.pipeline_tasks import (
    IdentityBoundBundle,
    semantic_enrichment_task,
)
from paperos_core.application import Application, create_application
from paperos_core.config import load_settings
from paperos_core.domain.canonical import ChunkProjection
from paperos_core.domain.knowledge import SemanticEnrichment
from paperos_core.domain.provenance import RelationType
from paperos_core.errors import LocalInferenceUnavailableError

MANIFEST = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "scholarly_work_reference"
    / "reference_corpus_manifest.json"
)
STATE_NAME = "claim-about-acceptance.state.json"
REPORT_NAME = "claim-about-acceptance.json"

_GROUND_TRUTH = [
    {
        "id": "efis_about_nise_handcrafted",
        "source_key": "efis_2026",
        "about_key": "nise_2023",
        "evidence_any_of": ["handcrafted vector field", "main drawback"],
        "role_self": False,
    },
    {
        "id": "adadiv_about_nise_volume",
        "source_key": "adadiv_2025",
        "about_key": "nise_2023",
        "evidence_any_of": [
            "without any control",
            "volume disappear",
            "reappear",
            "volume of the intermediate",
        ],
        "role_self": False,
    },
    {
        "id": "adadiv_about_lipmlp_volume",
        "source_key": "adadiv_2025",
        "about_key": "lipmlp_2022",
        "evidence_any_of": [
            "does not offer any control over the volume",
            "control over volume",
        ],
        "role_self": False,
    },
    {
        "id": "efis_about_adadiv",
        "source_key": "efis_2026",
        "about_key": "adadiv_2025",
        "evidence_any_of": ["ADADIV", "adaptive divergence", "volume preservation"],
        "role_self": False,
    },
    {
        "id": "adadiv_self_limitation",
        "source_key": "adadiv_2025",
        "about_key": "adadiv_2025",
        "evidence_any_of": ["small blobs", "detach", "reattach", "nonzero LSE"],
        "role_self": True,
    },
    {
        "id": "efis_self_topology_constraint",
        "source_key": "efis_2026",
        "about_key": "efis_2026",
        "evidence_any_of": ["share the same topology", "same topology", "higher genus"],
        "role_self": True,
    },
]


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
    payload = _load_json(MANIFEST)
    return {str(item["id"]): dict(item) for item in payload["papers"]}


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


def _current_snapshot_plan(application: Application) -> list[dict[str, str]]:
    papers = _paper_map()
    latest = _latest_bundles_by_file(application)
    plan: list[dict[str, str]] = []
    for paper_id, paper in papers.items():
        bundle = latest.get(str(paper["file"]))
        if bundle is None:
            raise RuntimeError(f"Missing current Document for {paper_id}")
        plan.append(
            {
                "paper_id": paper_id,
                "snapshot_id": bundle.snapshot.id,
                "document_id": bundle.document.id,
                "file": str(paper["file"]),
            }
        )
    return sorted(plan, key=lambda item: item["paper_id"])


def _enrichment_is_task02(path: Path) -> bool:
    if not path.is_file():
        return False
    enrichment = SemanticEnrichment.model_validate_json(path.read_text(encoding="utf-8"))
    if enrichment.prompt_version == "2":
        return True
    return any(claim.source_work_id for claim in enrichment.claims) or any(
        claim.about for claim in enrichment.claims
    )


def _load_enrichment(application: Application, snapshot_id: str) -> SemanticEnrichment:
    path = application.paths.cognee / "enrichment" / f"{snapshot_id}.json"
    return SemanticEnrichment.model_validate_json(path.read_text(encoding="utf-8"))


async def _ensure_runtime(application: Application) -> dict[str, Any]:
    """Initialize storage and reuse a healthy local embedding service when present.

    Task 02R must not fail solely because application.start() cannot re-bind an
    already-healthy 8081 embedding process. LLM remains the configured remote
    provider (DeepSeek); embedding uses the existing local endpoint.
    """
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
        # Fall back to owned start only when nothing is listening.
        try:
            await application.start()
            return {
                "llm": llm,
                "local_inference": {"status": "started_by_application"},
                "started": True,
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
        "local_inference": {
            "status": "reused_existing",
            "health": local,
        },
        "worker": worker_status,
        "started": True,
        "reused_existing_embedding": True,
    }


async def refresh_snapshot_enrichment(
    application: Application,
    *,
    snapshot_id: str,
) -> dict[str, Any]:
    """Call production semantic_enrichment_task with reuse_existing=False."""
    bundle = application.canonical_repository.get_bundle(snapshot_id)
    projection = application.canonical_repository.get_chunk_projection(snapshot_id)
    scholarly = application.scholarly_registry.resolve_bundle(bundle, projection.chunks)
    identity = IdentityBoundBundle(
        bundle=bundle,
        projection=ChunkProjection(snapshot_id=snapshot_id, chunks=projection.chunks),
        scholarly=scholarly,
    )
    enrichment_root = application.paths.cognee / "enrichment"
    before = enrichment_root / f"{snapshot_id}.json"
    if before.is_file():
        before.unlink()
    started = datetime.now(UTC)
    results = await semantic_enrichment_task(
        [identity],
        llm=application.llm,
        enrichment_root=enrichment_root,
        reuse_existing=False,
        generate_if_missing=True,
    )
    if len(results) != 1:
        raise RuntimeError(f"Expected one enrichment result for {snapshot_id}")
    enrichment = results[0].enrichment
    about_count = sum(len(claim.about) for claim in enrichment.claims)
    return {
        "snapshot_id": snapshot_id,
        "status": "completed",
        "started_at": started.isoformat(),
        "completed_at": datetime.now(UTC).isoformat(),
        "prompt_version": enrichment.prompt_version,
        "claim_count": len(enrichment.claims),
        "about_target_count": about_count,
        "claims_with_source_work_id": sum(
            1 for claim in enrichment.claims if claim.source_work_id
        ),
    }


def validate_stored(application: Application) -> dict[str, Any]:
    work_ids = _paper_work_ids(application)
    about_edges = 0
    external_about = 0
    self_about = 0
    about_triplets = 0
    invalid_provenance = 0
    invalid_work_grounding = 0
    claim_about_links = 0
    graphs_checked = 0
    enrichments: dict[str, SemanticEnrichment] = {}

    for snapshot_id in application.canonical_repository.list_latest_snapshot_ids():
        enrichment = _load_enrichment(application, snapshot_id)
        enrichments[snapshot_id] = enrichment
        bundle = application.canonical_repository.get_bundle(snapshot_id)
        projection = application.canonical_repository.get_chunk_projection(snapshot_id)
        scholarly = application.scholarly_registry.resolve_bundle(
            bundle, projection.chunks
        )
        work_set = {work.id for work in scholarly.works}
        chunk_ids = {chunk.id for chunk in projection.chunks}
        graph = canonical_to_datapoints(
            bundle, projection.chunks, enrichment, scholarly
        )
        graphs_checked += 1
        about_triplets += sum(
            1
            for node in graph.nodes
            if isinstance(node, TripletDataPoint) and node.relation_type == "ABOUT"
        )
        for claim in enrichment.claims:
            for about in claim.about:
                claim_about_links += 1
                if about.work_id not in work_set:
                    invalid_work_grounding += 1
                if not about.source_chunk_ids or not set(
                    about.source_chunk_ids
                ).issubset(chunk_ids):
                    invalid_provenance += 1
        for relation in graph.relations:
            if relation.relation_type is not RelationType.ABOUT:
                continue
            about_edges += 1
            if not relation.source_chunk_ids or not set(
                relation.source_chunk_ids
            ).issubset(chunk_ids):
                invalid_provenance += 1
            claim = next(
                (item for item in enrichment.claims if item.id == relation.source_id),
                None,
            )
            if claim is None:
                invalid_provenance += 1
                continue
            source_work = claim.source_work_id or scholarly.document_work.id
            if source_work == relation.target_id and "self" in relation.roles:
                self_about += 1
            elif source_work != relation.target_id:
                external_about += 1

    ground = []
    for probe in _GROUND_TRUTH:
        source_work = work_ids.get(probe["source_key"])
        about_work = work_ids.get(probe["about_key"])
        status = "MISSING"
        if source_work and about_work:
            matched = []
            for enrichment in enrichments.values():
                for claim in enrichment.claims:
                    if (claim.source_work_id or "") != source_work:
                        continue
                    for about in claim.about:
                        if about.work_id != about_work:
                            continue
                        if probe["role_self"] and "self" not in about.roles:
                            continue
                        matched.append(claim)
            evidence_hits = [
                claim
                for claim in matched
                if any(
                    needle.casefold() in claim.text.casefold()
                    for needle in probe["evidence_any_of"]
                )
            ]
            if evidence_hits:
                status = "MATCH"
            elif matched:
                status = "PARTIAL"
        ground.append({"id": probe["id"], "status": status})

    stored_pass = (
        about_triplets == 0
        and invalid_provenance == 0
        and invalid_work_grounding == 0
        and graphs_checked == 4
        and about_edges > 0
        and external_about > 0
        and self_about > 0
    )
    return {
        "graphs_checked": graphs_checked,
        "about_claim_link_count": claim_about_links,
        "about_edge_count": about_edges,
        "external_about_count": external_about,
        "self_about_count": self_about,
        "about_triplet_count": about_triplets,
        "invalid_provenance_count": invalid_provenance,
        "invalid_work_grounding_count": invalid_work_grounding,
        "work_ids": work_ids,
        "ground_truth": ground,
        "stored_graph": "PASS" if stored_pass else "FAIL",
    }


async def validate_live_cognee(
    application: Application, stored: dict[str, Any]
) -> dict[str, Any]:
    claim_ids: list[str] = []
    work_ids = set(stored["work_ids"].values())
    for snapshot_id in application.canonical_repository.list_latest_snapshot_ids():
        enrichment = _load_enrichment(application, snapshot_id)
        for claim in enrichment.claims:
            if claim.about:
                claim_ids.append(claim.id)
    seeds = sorted(
        {
            str(cognee_uuid(item))
            for item in [*claim_ids, *work_ids]
        }
    )
    readback = await application.knowledge_pipeline.compat.read_graph_records(
        seeds,
        dataset_name=application.settings.dataset,
        depth=1,
    )
    canonical_by_cognee = {
        str(node.get("id")): str(node.get("canonical_id") or "")
        for node in readback["nodes"]
        if node.get("id")
    }
    about_edges = []
    for edge in readback["edges"]:
        if str(edge.get("relation_type") or "") != "ABOUT":
            continue
        source = canonical_by_cognee.get(str(edge.get("source_id")), "")
        target = canonical_by_cognee.get(str(edge.get("target_id")), "")
        # Some engines already store canonical ids in edge metadata.
        source = str(edge.get("canonical_source_id") or source)
        target = str(edge.get("canonical_target_id") or target)
        about_edges.append(
            {
                "source_id": source,
                "target_id": target,
                "roles": list(edge.get("roles") or []),
                "source_chunk_ids": list(edge.get("source_chunk_ids") or []),
                "derived_from_ids": list(edge.get("derived_from_ids") or []),
            }
        )
    unique = {
        json.dumps(item, ensure_ascii=False, sort_keys=True): item
        for item in about_edges
    }
    about_edges = [unique[key] for key in sorted(unique)]
    live_pass = (
        len(about_edges) > 0
        and all(edge["source_chunk_ids"] for edge in about_edges)
        and all(edge["source_id"].startswith("claim_") for edge in about_edges)
        and all(edge["target_id"].startswith("work_") for edge in about_edges)
    )
    return {
        "about_edge_count": len(about_edges),
        "about_edges_sample": about_edges[:12],
        "raw_node_count": len(readback["nodes"]),
        "raw_edge_count": len(readback["edges"]),
        "live_cognee": "PASS" if live_pass else "FAIL",
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-data-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        default="paperos-scholarly-work-reference",
        help="Cognee dataset name used by the scholarly-work-reference corpus.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke-one",
        action="store_true",
        help="Refresh only one current Snapshot, then stop before rebuild.",
    )
    parser.add_argument(
        "--paper-id",
        default=None,
        help="Optional paper id (e.g. efis_2026) to prefer for smoke/refresh order.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Skip enrichment refresh; validate retained artifacts and optionally rebuild.",
    )
    parser.add_argument(
        "--skip-rebuild",
        action="store_true",
        help="Skip normal rebuild after enrichment refresh.",
    )
    args = parser.parse_args()

    live_dir = args.live_data_dir.resolve()
    state_path = live_dir / "logs" / "contracts" / STATE_NAME
    report_path = live_dir / "logs" / "contracts" / REPORT_NAME

    configured = load_settings(args.config)
    settings = configured.model_copy(
        update={
            "data": configured.data.model_copy(
                update={
                    "directory": live_dir,
                    "dataset": args.dataset,
                }
            )
        }
    )
    application = create_application(settings)
    runtime_info = await _ensure_runtime(application)
    report: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "live_data_dir": str(live_dir),
        "runtime": runtime_info,
        "mineru_calls": 0,
        "new_parse_runs": 0,
        "new_canonical_snapshots": 0,
    }
    try:
        all_ids = application.canonical_repository.list_snapshot_ids()
        current_ids = application.canonical_repository.list_latest_snapshot_ids()
        plan = _current_snapshot_plan(application)
        if args.paper_id:
            preferred = [item for item in plan if item["paper_id"] == args.paper_id]
            if not preferred:
                raise RuntimeError(f"Unknown --paper-id: {args.paper_id}")
            plan = preferred + [item for item in plan if item["paper_id"] != args.paper_id]
        report["all_snapshot_count"] = len(all_ids)
        report["current_snapshot_count"] = len(current_ids)
        report["historical_snapshot_count"] = len(all_ids) - len(current_ids)
        report["plan"] = plan
        if len(current_ids) != 4 or len(plan) != 4:
            raise RuntimeError(
                f"Expected 4 current Snapshots/papers; got current={len(current_ids)} plan={len(plan)}"
            )

        if state_path.is_file() and args.resume:
            state = _load_json(state_path)
        else:
            state = {
                "schema_version": 1,
                "work_ids_before": _paper_work_ids(application),
                "enrichment": {},
                "rebuild": {},
            }
            _atomic_json(state_path, state)
        report["work_ids_before"] = dict(state.get("work_ids_before") or {})

        enrichment_root = application.paths.cognee / "enrichment"
        refreshed: list[str] = []
        if not args.validate_only:
            for item in plan:
                snapshot_id = item["snapshot_id"]
                paper_id = item["paper_id"]
                prior = dict(state.get("enrichment", {}).get(snapshot_id) or {})
                path = enrichment_root / f"{snapshot_id}.json"
                if (
                    args.resume
                    and prior.get("status") == "completed"
                    and _enrichment_is_task02(path)
                ):
                    print(f"enrichment reuse {paper_id} {snapshot_id}", flush=True)
                    refreshed.append(snapshot_id)
                    continue
                print(f"enrichment refresh {paper_id} {snapshot_id}", flush=True)
                result = await refresh_snapshot_enrichment(
                    application, snapshot_id=snapshot_id
                )
                state.setdefault("enrichment", {})[snapshot_id] = {
                    **result,
                    "paper_id": paper_id,
                }
                _atomic_json(state_path, state)
                refreshed.append(snapshot_id)
                if args.smoke_one:
                    report["smoke"] = state["enrichment"][snapshot_id]
                    report["status"] = "smoke_completed"
                    report["refreshed_snapshot_count"] = 1
                    _atomic_json(report_path, report)
                    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
                    return
        else:
            refreshed = [
                item["snapshot_id"]
                for item in plan
                if _enrichment_is_task02(
                    enrichment_root / f"{item['snapshot_id']}.json"
                )
            ]

        report["refreshed_snapshot_count"] = len(refreshed)
        if len(refreshed) != 4 and not args.smoke_one:
            raise RuntimeError(
                f"Expected 4 Task-02 enrichments; have {len(refreshed)}: {refreshed}"
            )

        if not args.skip_rebuild:
            print("normal rebuild refresh_enrichment=False", flush=True)
            rebuilt = await application.services.rebuilder.rebuild(
                refresh_enrichment=False
            )
            state["rebuild"] = {
                "status": "completed",
                **rebuilt.public_dict(),
                "completed_at": datetime.now(UTC).isoformat(),
            }
            _atomic_json(state_path, state)
            report["normal_rebuild"] = rebuilt.public_dict()
            report["rebuild_enrichment_llm_calls"] = rebuilt.llm_enrichment_call_count
            if rebuilt.llm_enrichment_call_count != 0:
                raise RuntimeError(
                    "Normal rebuild unexpectedly invoked semantic enrichment LLM calls."
                )
        else:
            report["rebuild_enrichment_llm_calls"] = 0

        stored = validate_stored(application)
        report["validation"] = stored
        report["work_ids_after"] = _paper_work_ids(application)
        report["work_ids_stable"] = (
            report["work_ids_before"] == report["work_ids_after"]
        )
        try:
            live = await validate_live_cognee(application, stored)
        except Exception as live_exc:  # noqa: BLE001
            live = {
                "live_cognee": "FAIL",
                "error": f"{type(live_exc).__name__}: {live_exc}",
                "about_edge_count": 0,
            }
            report["live_error_traceback"] = traceback.format_exc()
        report["live"] = live

        gt = stored["ground_truth"]
        matchish = sum(1 for item in gt if item["status"] in {"MATCH", "PARTIAL"})
        overall = (
            stored["stored_graph"] == "PASS"
            and live["live_cognee"] == "PASS"
            and report.get("rebuild_enrichment_llm_calls", 0) == 0
            and report["work_ids_stable"]
            and matchish >= 4
            and report["refreshed_snapshot_count"] == 4
        )
        report["overall"] = "PASS" if overall else "FAIL"
        if not overall:
            report["remaining_blocker"] = {
                "stored_graph": stored["stored_graph"],
                "live_cognee": live.get("live_cognee"),
                "live_error": live.get("error"),
                "ground_truth": gt,
                "rebuild_llm_calls": report.get("rebuild_enrichment_llm_calls"),
                "work_ids_stable": report["work_ids_stable"],
            }
        report["completed_at"] = datetime.now(UTC).isoformat()
        _atomic_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        if report["overall"] != "PASS":
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        report["completed_at"] = datetime.now(UTC).isoformat()
        _atomic_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        raise
    finally:
        try:
            await application.aclose()
        except Exception as shutdown_exc:  # noqa: BLE001
            print(f"shutdown warning: {shutdown_exc}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
