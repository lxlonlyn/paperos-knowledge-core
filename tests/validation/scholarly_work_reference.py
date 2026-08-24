"""Real-PDF acceptance for ScholarlyWork identity and the citation backbone.

This project intentionally does not use pytest. Run:

    python tests/validation/scholarly_work_reference.py \
      --corpus-dir data/validation/corpus \
      --run-dir data/validation/scholarly_work_reference/output \
      --dataset paperos-scholarly-work-reference --resume
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.adapters.cognee.compat import cognee_uuid
from paperos_core.application import Application, create_application
from paperos_core.config import load_settings
from paperos_core.domain.canonical import CanonicalBundle
from paperos_core.domain.scholarly import ScholarlyWork
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry

FIXTURE_ROOT = (
    REPOSITORY_ROOT / "data" / "validation" / "scholarly_work_reference" / "config"
)
REPORT_NAME = "scholarly-work-citation-backbone.json"
STATE_NAME = "scholarly-work-citation-backbone.state.json"
BACKBONE_RELATIONS = {"REPRESENTS_WORK", "RESOLVES_TO", "CITES"}
FAKE_PROVENANCE_FIELDS = {
    "source_file_id",
    "parse_run_id",
    "canonical_snapshot_id",
    "source_chunk_ids",
}


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pair(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["source"]), str(record["target"])


def _paper_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    papers = manifest.get("papers")
    if not isinstance(papers, list):
        raise TypeError("Fixture manifest has no papers list.")
    return {str(item["id"]): dict(item) for item in papers}


def _latest_bundles(application: Application) -> dict[str, CanonicalBundle]:
    latest: dict[str, CanonicalBundle] = {}
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


def _matching_works(
    registry: ScholarlyRegistry,
    paper: dict[str, Any],
) -> list[ScholarlyWork]:
    expected_doi = registry.normalize_doi(paper.get("doi"))
    matches = []
    for work in registry.list_works():
        if expected_doi and registry.normalize_doi(work.doi) == expected_doi:
            matches.append(work)
            continue
        if registry.identity_attributes_match(
            work,
            title=str(paper["title"]),
            year=int(paper["year"]),
            first_author=str(paper["first_author"]),
            doi=paper.get("doi"),
        ):
            matches.append(work)
    return sorted(
        {work.id: work for work in matches}.values(), key=lambda item: item.id
    )


def _identity_snapshot_path(contract_root: Path, position: int, paper_key: str) -> Path:
    return contract_root / f"identity-after-{position:02d}-{paper_key}.json"


def _reference_statuses(registry: ScholarlyRegistry) -> dict[str, str]:
    return {
        str(item["reference_id"]): str(item["resolution_status"])
        for item in registry.identity_snapshot()["reference_links"]
    }


def _corpus_signature(
    application: Application,
    papers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    latest = _latest_bundles(application)
    work_ids: dict[str, str] = {}
    document_links: dict[str, str] = {}
    bundles: dict[str, CanonicalBundle] = {}
    for key, paper in papers.items():
        filename = str(paper["file"])
        bundle = latest.get(filename)
        if bundle is None:
            continue
        work = application.scholarly_registry.work_for_document(bundle.document.id)
        if work is None:
            continue
        bundles[key] = bundle
        work_ids[key] = work.id
        document_links[bundle.document.id] = work.id

    key_by_work = {work_id: key for key, work_id in work_ids.items()}
    statuses = _reference_statuses(application.scholarly_registry)
    reference_links = []
    citation_records = []
    for source_key, bundle in sorted(bundles.items()):
        source_work_id = work_ids[source_key]
        for reference in bundle.references:
            target = application.scholarly_registry.work_for_reference(reference.id)
            if target is None or target.id not in key_by_work:
                continue
            target_key = key_by_work[target.id]
            record = {
                "source": source_key,
                "target": target_key,
                "source_work_id": source_work_id,
                "target_work_id": target.id,
                "reference_id": reference.id,
                "resolution_status": statuses.get(reference.id, "unknown"),
            }
            reference_links.append(record)
            citation_records.append(
                {
                    "source": source_key,
                    "target": target_key,
                    "reference_id": reference.id,
                    "derived_from_ids": [reference.id],
                }
            )
    return {
        "work_ids": dict(sorted(work_ids.items())),
        "document_links": dict(sorted(document_links.items())),
        "reference_links": sorted(
            reference_links,
            key=lambda item: (
                item["source"],
                item["target"],
                item["reference_id"],
            ),
        ),
        "citation_edges": sorted(
            citation_records,
            key=lambda item: (
                item["source"],
                item["target"],
                item["reference_id"],
            ),
        ),
    }


def _stored_graph_contract(
    application: Application,
    signature: dict[str, Any],
) -> dict[str, Any]:
    latest = _latest_bundles(application)
    corpus_work_ids = set(signature["work_ids"].values())
    observed: set[tuple[str, str, str]] = set()
    triplet_backbone_count = 0
    fake_external = []
    work_nodes: set[str] = set()
    for bundle in latest.values():
        graph_path = application.paths.cognee / "graphs" / f"{bundle.snapshot.id}.json"
        if not graph_path.is_file():
            continue
        graph = _load_json(graph_path)
        for node in graph.get("nodes", []):
            if not isinstance(node, dict):
                continue
            if (
                node.get("__type__") == "TripletDataPoint"
                and node.get("relation_type") in BACKBONE_RELATIONS
            ):
                triplet_backbone_count += 1
            if node.get("__type__") != "ScholarlyWorkDataPoint":
                continue
            canonical_id = str(node.get("canonical_id") or "")
            work_nodes.add(canonical_id)
            if canonical_id in corpus_work_ids:
                continue
            invalid = sorted(field for field in FAKE_PROVENANCE_FIELDS if field in node)
            if invalid:
                fake_external.append(
                    {
                        "work_id": canonical_id,
                        "fields": invalid,
                        "source": str(graph_path),
                    }
                )
        for relation in graph.get("relations", []):
            if not isinstance(relation, dict):
                continue
            kind = str(relation.get("relation_type") or "")
            if kind in BACKBONE_RELATIONS:
                observed.add(
                    (
                        str(relation.get("source_id") or ""),
                        kind,
                        str(relation.get("target_id") or ""),
                    )
                )
    backbone_records = []
    for bundle in latest.values():
        graph_path = application.paths.cognee / "graphs" / f"{bundle.snapshot.id}.json"
        if not graph_path.is_file():
            continue
        for relation in _load_json(graph_path).get("relations", []):
            if not isinstance(relation, dict):
                continue
            kind = str(relation.get("relation_type") or "")
            if kind not in BACKBONE_RELATIONS:
                continue
            backbone_records.append(
                {
                    "source_id": str(relation.get("source_id") or ""),
                    "target_id": str(relation.get("target_id") or ""),
                    "relation_type": kind,
                    "derived_from_ids": list(relation.get("derived_from_ids") or []),
                    "source_chunk_ids": list(relation.get("source_chunk_ids") or []),
                }
            )
    unique_records = {
        json.dumps(item, ensure_ascii=False, sort_keys=True): item
        for item in backbone_records
    }
    return {
        "work_node_ids": sorted(work_nodes),
        "backbone_edges": [list(item) for item in sorted(observed)],
        "backbone_records": [unique_records[key] for key in sorted(unique_records)],
        "triplet_backbone_count": triplet_backbone_count,
        "fake_external_provenance": fake_external,
    }


def _edge_canonical_ids(
    raw: dict[str, Any],
    canonical_by_cognee: dict[str, str],
) -> tuple[str, str]:
    source = str(
        raw.get("canonical_source_id")
        or canonical_by_cognee.get(str(raw.get("source_id")), raw.get("source_id"))
        or ""
    )
    target = str(
        raw.get("canonical_target_id")
        or canonical_by_cognee.get(str(raw.get("target_id")), raw.get("target_id"))
        or ""
    )
    return source, target


async def _live_cognee_contract(
    application: Application,
    signature: dict[str, Any],
) -> dict[str, Any]:
    canonical_ids = set(signature["work_ids"].values())
    canonical_ids.update(signature["document_links"].keys())
    canonical_ids.update(item["reference_id"] for item in signature["reference_links"])
    readback = await application.knowledge_pipeline.compat.read_graph_records(
        [str(cognee_uuid(item)) for item in sorted(canonical_ids)],
        dataset_name=application.settings.dataset,
        depth=1,
    )
    canonical_by_cognee = {
        str(node.get("id")): str(node.get("canonical_id"))
        for node in readback["nodes"]
        if node.get("id") and node.get("canonical_id")
    }
    work_ids = set(signature["work_ids"].values())
    work_nodes = []
    fake_external = []
    for node in readback["nodes"]:
        canonical_id = str(node.get("canonical_id") or "")
        object_type = str(node.get("type") or node.get("object_type") or "")
        if canonical_id not in work_ids and "ScholarlyWorkDataPoint" not in object_type:
            continue
        if "ScholarlyWorkDataPoint" in object_type or canonical_id in work_ids:
            work_nodes.append(canonical_id)
        if canonical_id in work_ids:
            continue
        invalid = sorted(
            field
            for field in FAKE_PROVENANCE_FIELDS
            if node.get(field) not in (None, [], "")
        )
        if invalid:
            fake_external.append({"work_id": canonical_id, "fields": invalid})

    edges: dict[str, list[dict[str, Any]]] = {
        "REPRESENTS_WORK": [],
        "RESOLVES_TO": [],
        "CITES": [],
    }
    for edge in readback["edges"]:
        kind = str(edge.get("relation_type") or "")
        if kind not in edges:
            continue
        source, target = _edge_canonical_ids(edge, canonical_by_cognee)
        edges[kind].append(
            {
                "source_id": source,
                "target_id": target,
                "derived_from_ids": list(edge.get("derived_from_ids") or []),
                "source_chunk_ids": list(edge.get("source_chunk_ids") or []),
            }
        )
    for kind, values in edges.items():
        unique = {
            json.dumps(item, ensure_ascii=False, sort_keys=True): item
            for item in values
        }
        edges[kind] = [unique[key] for key in sorted(unique)]
    return {
        "work_nodes": sorted(set(work_nodes)),
        "represents_work_edges": edges["REPRESENTS_WORK"],
        "resolves_to_edges": edges["RESOLVES_TO"],
        "cites_edges": edges["CITES"],
        "fake_external_provenance": fake_external,
        "fake_external_provenance_count": len(fake_external),
        "raw_node_count": len(readback["nodes"]),
        "raw_edge_count": len(readback["edges"]),
    }


def _semantic_signature(
    registry_signature: dict[str, Any],
    stored_graph: dict[str, Any],
    live_graph: dict[str, Any],
) -> dict[str, Any]:
    return {
        "work_ids": registry_signature["work_ids"],
        "document_links": registry_signature["document_links"],
        "reference_links": registry_signature["reference_links"],
        "citation_edges": registry_signature["citation_edges"],
        "stored_backbone_edges": stored_graph["backbone_edges"],
        "stored_backbone_records": stored_graph["backbone_records"],
        "live_represents_work_edges": live_graph["represents_work_edges"],
        "live_resolves_to_edges": live_graph["resolves_to_edges"],
        "live_cites_edges": live_graph["cites_edges"],
    }


def _resolution_diagnostics(
    application: Application,
    papers: dict[str, dict[str, Any]],
    expected_edges: set[tuple[str, str]],
    signature: dict[str, Any],
) -> list[dict[str, Any]]:
    observed = {
        (item["source"], item["target"]) for item in signature["citation_edges"]
    }
    latest = _latest_bundles(application)
    diagnostics = []
    for source_key, target_key in sorted(expected_edges - observed):
        source = papers[source_key]
        target = papers[target_key]
        bundle = latest.get(str(source["file"]))
        if bundle is None:
            diagnostics.append(
                {
                    "source": source_key,
                    "target": target_key,
                    "layer": "MinerU/Canonical document missing",
                    "candidate_references": [],
                }
            )
            continue
        expected_doi = application.scholarly_registry.normalize_doi(target.get("doi"))
        expected_title = application.scholarly_registry.normalize_text(
            str(target["title"])
        )
        candidates = []
        for reference in bundle.references:
            reference_doi = application.scholarly_registry.normalize_doi(reference.doi)
            reference_title = application.scholarly_registry.normalize_text(
                reference.title or reference.raw_text
            )
            if not (
                (expected_doi and reference_doi == expected_doi)
                or expected_title == reference_title
                or expected_title in reference_title
            ):
                continue
            linked = application.scholarly_registry.work_for_reference(reference.id)
            candidates.append(
                {
                    "reference_id": reference.id,
                    "title": reference.title,
                    "authors": reference.authors,
                    "year": reference.year,
                    "doi": reference.doi,
                    "linked_work_id": linked.id if linked else None,
                }
            )
        if not candidates:
            layer = "MinerU/Canonical did not extract a matching ReferenceEntry"
        elif any(
            not item["title"] or not item["authors"] or item["year"] is None
            for item in candidates
        ):
            layer = "ReferenceEntry bibliographic metadata incomplete"
        elif all(item["linked_work_id"] is None for item in candidates):
            layer = "ScholarlyRegistry matching unresolved"
        else:
            layer = "bibliographic normalization or Work reconciliation mismatch"
        diagnostics.append(
            {
                "source": source_key,
                "target": target_key,
                "layer": layer,
                "candidate_references": candidates,
            }
        )
    return diagnostics


def _record_failure(report: dict[str, Any], message: str) -> None:
    failures = report.setdefault("hard_failures", [])
    if message not in failures:
        failures.append(message)


def _validate_fixture_copy(corpus_dir: Path, report: dict[str, Any]) -> None:
    corpus_expectations = corpus_dir / "expectations"
    for name in (
        "acceptance_tasks.md",
        "reference_corpus_manifest.json",
        "reference_ground_truth.json",
        "reference_queries.json",
    ):
        fixture = FIXTURE_ROOT / name
        source = corpus_expectations / name
        if not fixture.is_file():
            raise RuntimeError(f"Missing committed fixture: {fixture}")
        if source.is_file() and fixture.read_bytes() != source.read_bytes():
            _record_failure(report, f"Corpus expectation differs from fixture: {name}")


def _validate_final(
    report: dict[str, Any],
    papers: dict[str, dict[str, Any]],
    expected_edges: set[tuple[str, str]],
    forbidden_edges: set[tuple[str, str]],
    signature: dict[str, Any],
    stored: dict[str, Any],
    cognee: dict[str, Any],
) -> None:
    work_ids = signature["work_ids"]
    if len(work_ids) != len(papers) or len(set(work_ids.values())) != len(papers):
        _record_failure(
            report, "Four supplied PDFs did not resolve to four distinct Works."
        )
    ingested = {
        work.id
        for work in report.pop("_active_works")
        if work.identity_status.value == "ingested"
    }
    if set(work_ids.values()) - ingested:
        _record_failure(report, "One or more corpus Works are not ingested.")

    observed = {
        (item["source"], item["target"]) for item in signature["citation_edges"]
    }
    missing = sorted(expected_edges - observed)
    reverse = sorted(forbidden_edges & observed)
    report["citation_edges"] = {
        "expected": [list(item) for item in sorted(expected_edges)],
        "observed": [list(item) for item in sorted(observed)],
        "missing": [list(item) for item in missing],
        "unexpected_reverse": [list(item) for item in reverse],
    }
    if missing:
        _record_failure(report, f"Missing expected citation edges: {missing}")
    if reverse:
        _record_failure(report, f"Forbidden reverse citation edges exist: {reverse}")

    resolutions = []
    for edge in signature["citation_edges"]:
        source_work_id = work_ids[edge["source"]]
        target_work_id = work_ids[edge["target"]]
        stored_provenance = any(
            item["relation_type"] == "CITES"
            and item["source_id"] == source_work_id
            and item["target_id"] == target_work_id
            and edge["reference_id"] in item["derived_from_ids"]
            for item in stored["backbone_records"]
        )
        live_provenance = any(
            item["source_id"] == source_work_id
            and item["target_id"] == target_work_id
            and edge["reference_id"] in item["derived_from_ids"]
            for item in cognee["cites_edges"]
        )
        status = (
            "resolved"
            if stored_provenance and live_provenance
            else "missing_provenance"
        )
        resolutions.append(
            {
                "reference_id": edge["reference_id"],
                "source_work_id": source_work_id,
                "target_work_id": target_work_id,
                "status": status,
            }
        )
        if not stored_provenance or not live_provenance:
            _record_failure(
                report,
                f"CITES edge lacks real ReferenceEntry provenance: {edge['reference_id']}",
            )
    report["reference_resolution"] = resolutions

    if stored["triplet_backbone_count"]:
        _record_failure(report, "Backbone edges were duplicated as TripletDataPoints.")
    if stored["fake_external_provenance"]:
        _record_failure(
            report, "Stored external Work contains fake canonical provenance."
        )
    if cognee["fake_external_provenance_count"]:
        _record_failure(report, "Live Cognee external Work contains fake provenance.")

    document_links = {
        (document_id, work_id)
        for document_id, work_id in signature["document_links"].items()
    }
    represents = {
        (item["source_id"], item["target_id"])
        for item in cognee["represents_work_edges"]
    }
    missing_represents = sorted(document_links - represents)
    if missing_represents:
        _record_failure(
            report,
            f"Cognee is missing Document REPRESENTS_WORK edges: {missing_represents}",
        )

    expected_resolves = {
        (item["reference_id"], item["target_work_id"])
        for item in signature["reference_links"]
    }
    resolves = {
        (item["source_id"], item["target_id"]) for item in cognee["resolves_to_edges"]
    }
    missing_resolves = sorted(expected_resolves - resolves)
    if missing_resolves:
        _record_failure(
            report,
            f"Cognee is missing Reference RESOLVES_TO edges: {missing_resolves}",
        )

    expected_cites = {
        (work_ids[source], work_ids[target]) for source, target in expected_edges
    }
    cites = {(item["source_id"], item["target_id"]) for item in cognee["cites_edges"]}
    missing_cites = sorted(expected_cites - cites)
    if missing_cites:
        _record_failure(report, f"Cognee is missing Work CITES edges: {missing_cites}")


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    started = monotonic()
    corpus_dir = args.corpus_dir.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    contract_root = run_dir / "logs" / "contracts"
    report_path = contract_root / REPORT_NAME
    state_path = contract_root / STATE_NAME
    report: dict[str, Any] = {
        "status": "running",
        "corpus": {"supplied_pdf_count": 0, "ingested_work_count": 0},
        "ingestion_steps": [],
        "citation_edges": {},
        "reference_resolution": [],
        "identity_reconciliation": [],
        "reprocess": [],
        "rebuild": {},
        "cognee": {},
        "hard_failures": [],
        "warnings": [],
        "started_at": datetime.now(UTC).isoformat(),
        "run_dir": str(run_dir),
        "dataset": args.dataset,
    }
    application: Application | None = None
    shutdown_error: Exception | None = None
    try:
        global FIXTURE_ROOT
        FIXTURE_ROOT = args.config_dir.expanduser().resolve()
        manifest = _hydrate_manifest(
            _load_json(FIXTURE_ROOT / "corpus_spec.json"), corpus_dir, run_dir
        )
        ground_truth = _load_json(FIXTURE_ROOT / "reference_ground_truth.json")
        papers = _paper_map(manifest)
        ingest_order = [str(item) for item in manifest["recommended_ingest_order"]]
        if set(ingest_order) != set(papers) or len(ingest_order) != 4:
            raise RuntimeError(
                "Fixture ingest order must describe exactly four papers."
            )
        expected_edges = {
            _pair(item) for item in ground_truth["citation_edges_within_corpus"]
        }
        forbidden_edges = {
            _pair(item)
            for item in ground_truth["forbidden_reverse_edges_within_corpus"]
        }

        pdfs = {
            key: corpus_dir / "papers" / str(paper["pool_file"])
            for key, paper in papers.items()
        }
        missing_pdfs = [str(path) for path in pdfs.values() if not path.is_file()]
        if missing_pdfs:
            raise RuntimeError(f"Missing supplied PDFs: {missing_pdfs}")
        report["corpus"]["supplied_pdf_count"] = len(pdfs)
        report["corpus"]["pdf_sha256"] = {
            key: _sha256(path) for key, path in sorted(pdfs.items())
        }

        if state_path.is_file():
            if not args.resume:
                raise RuntimeError(
                    f"Run state already exists; pass --resume or choose a new --run-dir: {state_path}"
                )
            state = _load_json(state_path)
            if state.get("dataset") != args.dataset:
                raise RuntimeError("Resume dataset does not match retained state.")
        else:
            state = {
                "schema_version": 1,
                "dataset": args.dataset,
                "ingest_order": ingest_order,
                "steps": {},
                "reprocess": {},
                "rebuild": {},
            }
            _atomic_json(state_path, state)

        configured = load_settings(args.settings)
        if not configured.mineru.api_key_value():
            raise RuntimeError("MinerU API key is not configured.")
        settings = configured.model_copy(
            update={
                "data": configured.data.model_copy(
                    update={"directory": run_dir, "dataset": args.dataset}
                )
            }
        )
        application = create_application(settings)
        await application.start()

        for position, paper_key in enumerate(ingest_order, 1):
            paper = papers[paper_key]
            step = dict(state["steps"].get(paper_key) or {})
            before_matches = _matching_works(application.scholarly_registry, paper)
            if step.get("status") != "completed":
                if len(before_matches) > 1:
                    _record_failure(
                        report,
                        f"Ambiguous pre-ingest Work identity for {paper_key}: "
                        f"{[item.id for item in before_matches]}",
                    )
                initial = before_matches[0] if len(before_matches) == 1 else None
                existing_before_ingest = step.get(
                    "existing_before_ingest", initial is not None
                )
                initial_work_id = step.get(
                    "initial_work_id", initial.id if initial else None
                )
                initial_status = step.get(
                    "initial_status",
                    initial.identity_status.value if initial else None,
                )
                step = {
                    "status": "pending",
                    "paper": paper_key,
                    "existing_before_ingest": existing_before_ingest,
                    "initial_work_id": initial_work_id,
                    "initial_status": initial_status,
                    "started_at": step.get("started_at")
                    or datetime.now(UTC).isoformat(),
                }
                state["steps"][paper_key] = step
                _atomic_json(state_path, state)

            if step.get("status") == "completed" and args.resume:
                bundle = application.canonical_repository.get_bundle(
                    str(step["snapshot_id"])
                )
                print(f"ingest {position}/4 reused {paper_key}", flush=True)
            else:
                latest = _latest_bundles(application)
                bundle = latest.get(str(paper["file"]))
                if bundle is not None and args.resume:
                    print(
                        f"ingest {position}/4 resume-knowledge {paper_key}", flush=True
                    )
                    await application.knowledge_pipeline.ingest_bundle(bundle)
                    bundle = application.canonical_repository.get_bundle(
                        bundle.snapshot.id
                    )
                else:
                    print(f"ingest {position}/4 live {paper_key}", flush=True)
                    result = (
                        await application.services.ingestion.ingest_pdf_to_knowledge(
                            pdfs[paper_key],
                            dataset=args.dataset,
                        )
                    )
                    bundle = result.canonical_result.canonical

            work = application.scholarly_registry.work_for_document(bundle.document.id)
            if work is None:
                raise RuntimeError(
                    f"No Document-to-Work link after ingest: {paper_key}"
                )
            step.update(
                {
                    "status": "completed",
                    "document_id": bundle.document.id,
                    "snapshot_id": bundle.snapshot.id,
                    "work_id": work.id,
                    "identity_status": work.identity_status.value,
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            )
            state["steps"][paper_key] = step
            _atomic_json(state_path, state)
            _atomic_json(
                _identity_snapshot_path(contract_root, position, paper_key),
                application.scholarly_registry.identity_snapshot(),
            )

        report["ingestion_steps"] = [
            state["steps"][paper_key] for paper_key in ingest_order
        ]
        for paper_key in ingest_order[1:]:
            step = state["steps"][paper_key]
            stable = bool(
                step.get("existing_before_ingest")
                and step.get("initial_work_id") == step.get("work_id")
            )
            report["identity_reconciliation"].append(
                {
                    "paper": paper_key,
                    "initial_work_id": step.get("initial_work_id"),
                    "final_work_id": step.get("work_id"),
                    "initial_status": step.get("initial_status"),
                    "final_status": step.get("identity_status"),
                    "stable": stable,
                }
            )
            if not stable:
                _record_failure(
                    report,
                    f"Later PDF ingest did not reuse a pre-existing Work: {paper_key}",
                )

        for position, paper_key in enumerate(ingest_order, 1):
            previous = dict(state["reprocess"].get(paper_key) or {})
            if previous.get("status") == "completed" and args.resume:
                report["reprocess"].append(previous)
                continue
            paper = papers[paper_key]
            bundle = _latest_bundles(application).get(str(paper["file"]))
            if bundle is None:
                raise RuntimeError(f"Cannot reprocess missing document: {paper_key}")
            before_work = application.scholarly_registry.work_for_document(
                bundle.document.id
            )
            if before_work is None:
                raise RuntimeError(f"Cannot reprocess unlinked document: {paper_key}")
            pending = {
                "status": "pending",
                "paper": paper_key,
                "document_id": bundle.document.id,
                "before_work_id": before_work.id,
                "before_snapshot_id": bundle.snapshot.id,
                "started_at": datetime.now(UTC).isoformat(),
            }
            state["reprocess"][paper_key] = pending
            _atomic_json(state_path, state)
            print(f"reprocess {position}/4 live {paper_key}", flush=True)
            await application.services.documents.reprocess(bundle.document.id)
            after = _latest_bundles(application).get(str(paper["file"]))
            if after is None:
                raise RuntimeError(f"Reprocess lost document: {paper_key}")
            after_work = application.scholarly_registry.work_for_document(
                after.document.id
            )
            if after_work is None:
                raise RuntimeError(f"Reprocess lost Work link: {paper_key}")
            completed = {
                **pending,
                "status": "completed",
                "after_work_id": after_work.id,
                "after_snapshot_id": after.snapshot.id,
                "stable": before_work.id == after_work.id,
                "completed_at": datetime.now(UTC).isoformat(),
            }
            state["reprocess"][paper_key] = completed
            _atomic_json(state_path, state)
            report["reprocess"].append(completed)
            if not completed["stable"]:
                _record_failure(report, f"Real reprocess changed Work ID: {paper_key}")
            if completed["before_snapshot_id"] == completed["after_snapshot_id"]:
                _record_failure(
                    report, f"Real reprocess did not create a snapshot: {paper_key}"
                )

        rebuild_state = dict(state.get("rebuild") or {})
        retained_before = rebuild_state.get("before_signature")
        if isinstance(retained_before, dict):
            before_signature = retained_before
        else:
            before_registry = _corpus_signature(application, papers)
            before_stored = _stored_graph_contract(application, before_registry)
            before_live = await _live_cognee_contract(application, before_registry)
            before_signature = _semantic_signature(
                before_registry,
                before_stored,
                before_live,
            )

        if (
            args.rerun_rebuild
            or rebuild_state.get("status") != "completed"
            or not args.resume
        ):
            state["rebuild"] = {
                "status": "pending",
                "before_signature": before_signature,
                "started_at": rebuild_state.get("started_at")
                or datetime.now(UTC).isoformat(),
            }
            _atomic_json(state_path, state)
            print(
                "rebuild first current snapshots with missing enrichment refresh",
                flush=True,
            )
            rebuilt = await application.services.rebuilder.rebuild(
                refresh_enrichment=True,
            )
            state["rebuild"] = {
                "status": "first_completed_second_pending",
                "before_signature": before_signature,
                **rebuilt.public_dict(),
                "first_rebuild_status": "passed",
                "first_rebuild": rebuilt.public_dict(),
                "rebuilt_snapshot_ids": rebuilt.rebuilt_snapshot_ids,
                "first_completed_at": datetime.now(UTC).isoformat(),
            }
            _atomic_json(state_path, state)
            print("rebuild second current snapshots without LLM enrichment", flush=True)
            second_rebuild = await application.services.rebuilder.rebuild(
                refresh_enrichment=False,
            )
            if second_rebuild.llm_enrichment_call_count != 0:
                raise RuntimeError(
                    "Second rebuild unexpectedly invoked semantic enrichment."
                )
            state["rebuild"].update(
                {
                    "status": "completed",
                    "second_rebuild_status": "passed",
                    "second_rebuild": second_rebuild.public_dict(),
                    "completed_at": datetime.now(UTC).isoformat(),
                }
            )
            _atomic_json(state_path, state)
        else:
            print("rebuild reused from retained state", flush=True)

        after_registry = _corpus_signature(application, papers)
        after_stored = _stored_graph_contract(application, after_registry)
        after_live = await _live_cognee_contract(application, after_registry)
        after_signature = _semantic_signature(after_registry, after_stored, after_live)
        rebuild_stable = all(
            before_signature.get(field) == after_signature.get(field)
            for field in (
                "work_ids",
                "document_links",
                "reference_links",
                "citation_edges",
            )
        )
        report["rebuild"] = {
            "before_signature": before_signature,
            "after_signature": after_signature,
            "stable": rebuild_stable,
            **state["rebuild"],
        }
        report["live_cognee"] = after_live
        if not rebuild_stable:
            _record_failure(report, "Real rebuild changed Work/citation signature.")

        active_works = application.scholarly_registry.list_works()
        report["_active_works"] = active_works
        report["corpus"]["ingested_work_count"] = len(
            set(after_registry["work_ids"].values())
        )
        report["cognee"] = {
            **after_live,
            "stored_triplet_backbone_count": after_stored["triplet_backbone_count"],
        }
        report["resolution_diagnostics"] = _resolution_diagnostics(
            application,
            papers,
            expected_edges,
            after_registry,
        )
        _validate_final(
            report,
            papers,
            expected_edges,
            forbidden_edges,
            after_registry,
            after_stored,
            after_live,
        )

        all_references = [
            reference
            for bundle in _latest_bundles(application).values()
            for reference in bundle.references
        ]
        metadata_complete = sum(
            bool(reference.title and reference.authors and reference.year)
            for reference in all_references
        )
        resolved_links = application.scholarly_registry.identity_snapshot()[
            "reference_links"
        ]
        references_by_id = {reference.id: reference for reference in all_references}
        title_only_count = sum(
            item["resolution_status"] == "resolved"
            and (reference := references_by_id.get(str(item["reference_id"])))
            is not None
            and bool(reference.title)
            and not reference.doi
            and not reference.arxiv_id
            for item in resolved_links
        )
        report["soft_observations"] = {
            "reference_count": len(all_references),
            "reference_metadata_complete_count": metadata_complete,
            "reference_metadata_completeness": (
                metadata_complete / len(all_references) if all_references else 0.0
            ),
            "external_provisional_work_count": sum(
                work.identity_status.value == "provisional"
                for work in active_works
                if work.id not in set(after_registry["work_ids"].values())
            ),
            "title_only_resolution_count": title_only_count,
            "citation_context_source_chunk_coverage": (
                sum(
                    bool(item["source_chunk_ids"]) for item in after_live["cites_edges"]
                )
                / len(after_live["cites_edges"])
                if after_live["cites_edges"]
                else 0.0
            ),
        }
        report["status"] = "passed" if not report["hard_failures"] else "failed"
    except Exception as exc:  # noqa: BLE001 - persist every live-run failure.
        _record_failure(report, f"{type(exc).__name__}: {exc}")
        report["failure_traceback"] = traceback.format_exc()
        report["status"] = "failed"
    finally:
        if application is not None:
            try:
                await application.aclose()
            except Exception as exc:  # noqa: BLE001 - report lifecycle failures.
                shutdown_error = exc
                _record_failure(report, f"Application shutdown failed: {exc}")
                report["status"] = "failed"
        report.pop("_active_works", None)
        report["completed_at"] = datetime.now(UTC).isoformat()
        report["runtime_seconds"] = round(monotonic() - started, 3)
        if shutdown_error is None:
            report["subprocess_cleanup"] = "completed"
        _atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        "--corpus-dir",
        dest="corpus_dir",
        type=Path,
        default=Path("data/validation/corpus"),
    )
    parser.add_argument("--rerun-rebuild", action="store_true")
    parser.add_argument(
        "--output",
        "--run-dir",
        dest="run_dir",
        type=Path,
        default=Path("data/validation/scholarly_work_reference/output"),
    )
    parser.add_argument("--dataset", default="paperos-scholarly-work-reference")
    parser.add_argument("--config", dest="config_dir", type=Path, default=FIXTURE_ROOT)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = asyncio.run(_run(args))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


def _hydrate_manifest(
    manifest: dict[str, Any], corpus_dir: Path, run_dir: Path
) -> dict[str, Any]:
    pool = _load_json(corpus_dir / "manifest.json")
    retained_by_sha: dict[str, str] = {}
    registry = run_dir / "jobs" / "registry.sqlite3"
    if registry.is_file():
        with sqlite3.connect(registry) as connection:
            sources = connection.execute(
                "SELECT id, original_filename FROM source_files"
            ).fetchall()
        for source_id, original_filename in sources:
            source_pdf = run_dir / "raw" / str(source_id) / "source.pdf"
            if source_pdf.is_file():
                retained_by_sha[_sha256(source_pdf)] = str(original_filename)
    for paper in manifest["papers"]:
        entry = pool[str(paper["paper_id"])]
        paper["pool_file"] = Path(str(entry["file"])).name
        paper["file"] = retained_by_sha.get(entry["sha256"], paper["pool_file"])
        paper["sha256"] = entry["sha256"]
    return manifest


if __name__ == "__main__":
    main()
