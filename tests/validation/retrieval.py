#!/usr/bin/env python3
from __future__ import annotations

"Shared live Cognee retrieval contract for direct scripts and acceptance."
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from paperos_core.adapters.cognee.compat import (
    CogneeCompatibilityAdapter,
    CogneeVectorHit,
)
from paperos_core.adapters.cognee.search import CogneeSearchAdapter, CogneeSearchHit
from paperos_core.application import Application
from paperos_core.domain.provenance import RelationType


@dataclass(frozen=True, slots=True)
class contract__CapabilitySpec:
    name: str
    datapoint_type: str
    public_search_type: str
    compat_search_type: str


contract___CAPABILITIES = (
    contract__CapabilitySpec("chunk", "ChunkDataPoint", "CHUNKS", "PAPEROS_CHUNKS"),
    contract__CapabilitySpec(
        "entity", "EntityDataPoint", "GRAPH_COMPLETION", "PAPEROS_ENTITIES"
    ),
    contract__CapabilitySpec(
        "claim", "ClaimDataPoint", "GRAPH_COMPLETION", "PAPEROS_CLAIMS"
    ),
    contract__CapabilitySpec(
        "summary", "SummaryDataPoint", "GRAPH_COMPLETION", "PAPEROS_SUMMARIES"
    ),
)


def contract___require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def contract___string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def contract___load_graph_nodes(graph_root: Path) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for path in sorted(graph_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_nodes = payload.get("nodes", [])
        if isinstance(raw_nodes, list):
            nodes.extend((item for item in raw_nodes if isinstance(item, dict)))
    return nodes


def contract___representative(
    nodes: list[dict[str, Any]], datapoint_type: str
) -> dict[str, Any]:
    matching = [
        node
        for node in nodes
        if str(node.get("__type__") or node.get("type") or "") == datapoint_type
        and node.get("canonical_id")
    ]
    contract___require(
        matching, f"""No real {datapoint_type } exists in retained graph snapshots."""
    )
    return min(matching, key=lambda node: str(node["canonical_id"]))


def contract___query_from_node(node: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    for field in ("name", "title", "text", "description"):
        value = node.get(field)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())[:500]
    return str(node["canonical_id"])


def contract___manifest_context(
    data_root: Path, dataset: str
) -> tuple[set[str], str | None]:
    canonical_ids: set[str] = set()
    dataset_ids: set[str] = set()
    for path in sorted((data_root / "cognee" / "manifests").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset_payload = payload.get("dataset")
        if not isinstance(dataset_payload, dict):
            continue
        if dataset_payload.get("name") != dataset:
            continue
        if dataset_payload.get("id"):
            dataset_ids.add(str(dataset_payload["id"]))
        mapping = payload.get("canonical_to_cognee_id", {})
        if isinstance(mapping, dict):
            canonical_ids.update((str(item) for item in mapping))
    contract___require(
        canonical_ids, "No canonical IDs are registered for the selected dataset."
    )
    contract___require(
        len(dataset_ids) <= 1, f"""Dataset name maps to multiple IDs: {dataset_ids }"""
    )
    return (canonical_ids, next(iter(dataset_ids), None))


def contract___surface_metrics(
    hits: list[CogneeSearchHit], *, datapoint_type: str, allowed_canonical_ids: set[str]
) -> dict[str, Any]:
    typed = [hit for hit in hits if hit.result_type == datapoint_type]
    canonical = bool(typed) and all(
        (hit.canonical_id in allowed_canonical_ids for hit in typed)
    )
    source_chunks = bool(typed) and all(
        (
            hit.source_chunk_ids
            or (
                datapoint_type == "ChunkDataPoint"
                and hit.canonical_id.startswith("chunk_")
            )
            for hit in typed
        )
    )
    scores = [hit.score for hit in hits]
    return {
        "result_count": len(hits),
        "matching_type_count": len(typed),
        "custom_node_type_preserved": bool(typed),
        "canonical_id_preserved": canonical,
        "source_chunk_ids_preserved": source_chunks,
        "dataset_scope_supported": canonical,
        "ranking_usable": bool(scores)
        and all((score > 0 for score in scores))
        and (scores == sorted(scores, reverse=True)),
        "graph_provenance_recoverable": bool(typed)
        and all((bool(hit.references) for hit in typed)),
        "results": [
            {
                "rank": rank,
                "node_id": hit.node_id,
                "canonical_id": hit.canonical_id,
                "result_type": hit.result_type,
                "score": hit.score,
                "source_chunk_ids": list(hit.source_chunk_ids),
                "references": list(hit.references),
            }
            for rank, hit in enumerate(hits, 1)
        ],
    }


async def contract___safe_surface(
    adapter: CogneeSearchAdapter,
    *,
    query: str,
    dataset: str,
    top_k: int,
    search_type: str,
    recall: bool,
) -> tuple[list[CogneeSearchHit], str | None]:
    try:
        if recall:
            return (
                await adapter.recall_context(
                    query, dataset=dataset, top_k=top_k, search_type=search_type
                ),
                None,
            )
        return (
            await adapter.graph_search(
                query, dataset=dataset, top_k=top_k, search_type=search_type
            ),
            None,
        )
    except Exception as exc:
        return ([], f"""{type (exc ).__name__ }: {exc }""")


async def contract___observe_public_recall(
    *, query: str, dataset: str, top_k: int, search_type: str
) -> tuple[dict[str, Any], str | None]:
    """Observe context separately from structured provenance preservation."""
    import cognee

    try:
        entries = await cognee.recall(
            query_text=query,
            query_type=cognee.SearchType(search_type),
            datasets=[dataset],
            only_context=True,
            top_k=top_k,
            auto_route=False,
        )
    except Exception as exc:
        return ({}, f"""{type (exc ).__name__ }: {exc }""")
    items = entries if isinstance(entries, list) else []
    context_count = 0
    structured_count = 0
    provenance_count = 0
    for entry in items:
        payload = (
            entry.model_dump(mode="json") if hasattr(entry, "model_dump") else entry
        )
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("text") or payload.get("content"), str):
            context_count += 1
        raw = payload.get("raw")
        if isinstance(raw, dict) and any(
            (
                key in raw
                for key in ("id", "node_id", "canonical_id", "source_chunk_ids")
            )
        ):
            structured_count += 1
            if raw.get("canonical_id") and raw.get("source_chunk_ids"):
                provenance_count += 1
    return (
        {
            "raw_entry_count": len(items),
            "context_returned": context_count > 0,
            "context_entry_count": context_count,
            "structured_objects_returned": structured_count,
            "provenance_preserved": bool(structured_count)
            and provenance_count == structured_count,
        },
        None,
    )


async def contract___datapoint_case(
    adapter: CogneeSearchAdapter,
    *,
    spec: contract__CapabilitySpec,
    node: dict[str, Any],
    query: str,
    dataset: str,
    dataset_id: str | None,
    allowed_canonical_ids: set[str],
    top_k: int,
    cognee_version: str,
) -> dict[str, Any]:
    searched, search_error = await contract___safe_surface(
        adapter,
        query=query,
        dataset=dataset,
        top_k=top_k,
        search_type=spec.public_search_type,
        recall=False,
    )
    recalled, recall_error = await contract___safe_surface(
        adapter,
        query=query,
        dataset=dataset,
        top_k=top_k,
        search_type=spec.public_search_type,
        recall=True,
    )
    recall_observation, recall_observation_error = (
        await contract___observe_public_recall(
            query=query,
            dataset=dataset,
            top_k=top_k,
            search_type=spec.public_search_type,
        )
    )
    compatible, compat_error = await contract___safe_surface(
        adapter,
        query=query,
        dataset=dataset,
        top_k=top_k,
        search_type=spec.compat_search_type,
        recall=False,
    )
    public_search = contract___surface_metrics(
        searched,
        datapoint_type=spec.datapoint_type,
        allowed_canonical_ids=allowed_canonical_ids,
    )
    public_recall = contract___surface_metrics(
        recalled,
        datapoint_type=spec.datapoint_type,
        allowed_canonical_ids=allowed_canonical_ids,
    )
    compat = contract___surface_metrics(
        compatible,
        datapoint_type=spec.datapoint_type,
        allowed_canonical_ids=allowed_canonical_ids,
    )
    public_typed_hits = [
        hit for hit in [*searched, *recalled] if hit.result_type == spec.datapoint_type
    ]
    resolved = await adapter.compat.resolve_graph_nodes(
        [hit.node_id for hit in public_typed_hits]
    )
    readback_hits = [
        CogneeSearchHit(
            node_id=hit.node_id,
            canonical_id=hit.canonical_id,
            source_chunk_ids=hit.source_chunk_ids
            or tuple(
                contract___string_list(
                    resolved.get(hit.node_id, {}).get("source_chunk_ids")
                )
            ),
            references=hit.references
            or tuple(
                contract___string_list(
                    resolved.get(hit.node_id, {}).get("derived_from_ids")
                )
            ),
            result_type=hit.result_type,
            text=hit.text,
            score=hit.score,
        )
        for hit in public_typed_hits
    ]
    readback = contract___surface_metrics(
        readback_hits,
        datapoint_type=spec.datapoint_type,
        allowed_canonical_ids=allowed_canonical_ids,
    )
    required_fields = (
        "custom_node_type_preserved",
        "canonical_id_preserved",
        "source_chunk_ids_preserved",
        "dataset_scope_supported",
        "ranking_usable",
    )
    public_search_supported = all(
        (bool(public_search[field]) for field in required_fields)
    )
    public_recall_supported = all(
        (bool(public_recall[field]) for field in required_fields)
    )
    compat_vector_supported = all((bool(compat[field]) for field in required_fields))
    compat_readback_supported = all(
        (bool(readback[field]) for field in required_fields)
    )
    compat_supported = compat_vector_supported or compat_readback_supported
    compat_best = compat if compat_vector_supported else readback
    if public_search_supported and public_recall_supported:
        status = "supported"
    elif compat_supported:
        status = "partially_supported"
    else:
        status = f"""unsupported_by_cognee_{cognee_version .replace ('.','_')}"""
    limitations = [
        message
        for message in (
            f"""public search: {search_error }""" if search_error else None,
            f"""public recall: {recall_error }""" if recall_error else None,
            (
                f"""public recall observation: {recall_observation_error }"""
                if recall_observation_error
                else None
            ),
            f"""compat: {compat_error }""" if compat_error else None,
        )
        if message
    ]
    if not public_search_supported:
        limitations.append(
            "public search does not preserve the required custom type identity and canonical chunk provenance"
        )
    if not public_recall_supported:
        limitations.append(
            "public recall does not preserve the required custom type identity and canonical chunk provenance"
        )
    return {
        "capability": spec.name,
        "status": status,
        "cognee_version": cognee_version,
        "dataset_name": dataset,
        "dataset_id": dataset_id,
        "search_type": spec.public_search_type,
        "compat_search_type": spec.compat_search_type,
        "datapoint_type": spec.datapoint_type,
        "representative_canonical_id": str(node["canonical_id"]),
        "query": query,
        "public_search_supported": public_search_supported,
        "public_recall_supported": public_recall_supported,
        "compat_supported": compat_supported,
        "custom_node_type_preserved": bool(compat_best["custom_node_type_preserved"]),
        "canonical_id_preserved": bool(compat_best["canonical_id_preserved"]),
        "source_chunk_ids_preserved": bool(compat_best["source_chunk_ids_preserved"]),
        "dataset_scope_supported": bool(compat_best["dataset_scope_supported"]),
        "ranking_usable": bool(compat_best["ranking_usable"]),
        "graph_provenance_recoverable": bool(
            compat_best["graph_provenance_recoverable"]
        ),
        "public_api_result_count": len(searched) + len(recalled),
        "compat_result_count": len(compatible) + len(readback_hits),
        "error_or_limitation": limitations,
        "surfaces": {
            "public_search": {**public_search, "error": search_error},
            "public_recall": {**public_recall, "error": recall_error},
            "public_recall_observation": {
                **recall_observation,
                "error": recall_observation_error,
            },
            "compat_vector_search": {
                **compat,
                "supported": compat_vector_supported,
                "error": compat_error,
            },
            "compat_graph_node_provenance_readback": {
                **readback,
                "supported": compat_readback_supported,
                "error": None,
            },
        },
    }


async def contract___graph_case(
    adapter: CogneeSearchAdapter,
    compat: CogneeCompatibilityAdapter,
    *,
    node: dict[str, Any],
    query: str,
    dataset: str,
    dataset_id: str | None,
    allowed_canonical_ids: set[str],
    top_k: int,
    cognee_version: str,
) -> dict[str, Any]:
    searched, search_error = await contract___safe_surface(
        adapter,
        query=query,
        dataset=dataset,
        top_k=top_k,
        search_type="GRAPH_COMPLETION_CONTEXT_EXTENSION",
        recall=False,
    )
    recalled, recall_error = await contract___safe_surface(
        adapter,
        query=query,
        dataset=dataset,
        top_k=top_k,
        search_type="GRAPH_COMPLETION_CONTEXT_EXTENSION",
        recall=True,
    )
    seed_hits, seed_error = await contract___safe_surface(
        adapter,
        query=query,
        dataset=dataset,
        top_k=top_k,
        search_type="PAPEROS_GRAPH_SEEDS",
        recall=False,
    )
    resolved = await compat.resolve_graph_nodes([hit.node_id for hit in seed_hits])
    seeds = [
        CogneeVectorHit(
            cognee_id=hit.node_id,
            canonical_id=hit.canonical_id,
            object_type=hit.result_type,
            text=hit.text,
            score=hit.score,
            source_chunk_ids=tuple(
                contract___string_list(
                    resolved.get(hit.node_id, {}).get("source_chunk_ids")
                )
            ),
            derived_from_ids=tuple(
                contract___string_list(
                    resolved.get(hit.node_id, {}).get("derived_from_ids")
                )
            ),
            canonical_snapshot_id=None,
        )
        for hit in seed_hits
    ]
    traversal_error: str | None = None
    try:
        traversed = await compat.typed_traverse(
            [seed for seed in seeds if seed.source_chunk_ids],
            depth=2,
            edge_types={item.value for item in RelationType},
        )
    except Exception as exc:
        traversed = []
        traversal_error = f"""{type (exc ).__name__ }: {exc }"""
    public_metrics = contract___surface_metrics(
        searched + recalled,
        datapoint_type="EntityDataPoint",
        allowed_canonical_ids=allowed_canonical_ids,
    )
    public_graph_provenance = False
    compat_supported = bool(traversed) and all(
        (
            item.source_canonical_id
            and item.target_canonical_id
            and item.relation_type
            and item.source_chunk_ids
            for item in traversed
        )
    )
    status = (
        "supported"
        if public_graph_provenance
        else (
            "partially_supported"
            if compat_supported
            else f"""unsupported_by_cognee_{cognee_version .replace ('.','_')}"""
        )
    )
    limitations = [
        message
        for message in (
            f"""public search: {search_error }""" if search_error else None,
            f"""public recall: {recall_error }""" if recall_error else None,
            f"""compat seed search: {seed_error }""" if seed_error else None,
            f"""compat traversal: {traversal_error }""" if traversal_error else None,
        )
        if message
    ]
    if not public_graph_provenance:
        limitations.append(
            "public context does not expose typed edge endpoints and edge provenance"
        )
    return {
        "capability": "graph_associative_context",
        "status": status,
        "cognee_version": cognee_version,
        "dataset_name": dataset,
        "dataset_id": dataset_id,
        "search_type": "GRAPH_COMPLETION_CONTEXT_EXTENSION",
        "compat_search_type": "PAPEROS_GRAPH_SEEDS + typed_traverse",
        "datapoint_type": "typed graph context",
        "representative_canonical_id": str(node["canonical_id"]),
        "query": query,
        "public_search_supported": bool(searched),
        "public_recall_supported": bool(recalled),
        "compat_supported": compat_supported,
        "custom_node_type_preserved": bool(
            public_metrics["custom_node_type_preserved"]
        ),
        "canonical_id_preserved": bool(public_metrics["canonical_id_preserved"]),
        "source_chunk_ids_preserved": bool(
            public_metrics["source_chunk_ids_preserved"]
        ),
        "dataset_scope_supported": bool(public_metrics["dataset_scope_supported"]),
        "ranking_usable": bool(public_metrics["ranking_usable"]),
        "graph_provenance_recoverable": compat_supported,
        "public_api_result_count": len(searched) + len(recalled),
        "compat_result_count": len(traversed),
        "error_or_limitation": limitations,
        "surfaces": {
            "public_graph_context": {
                **public_metrics,
                "typed_edge_provenance_count": 0,
                "search_error": search_error,
                "recall_error": recall_error,
            },
            "compat_typed_traversal": {
                "seed_count": len(seeds),
                "relation_count": len(traversed),
                "error": traversal_error or seed_error,
                "relations": [
                    {
                        "source_canonical_id": item.source_canonical_id,
                        "target_canonical_id": item.target_canonical_id,
                        "relation_type": item.relation_type,
                        "source_chunk_ids": list(item.source_chunk_ids),
                        "derived_from_ids": list(item.derived_from_ids),
                        "score": item.score,
                    }
                    for item in traversed[:top_k]
                ],
            },
        },
    }


def contract__write_contract_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f""".{path .name }.tmp""")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)
    return path


async def contract__run_live_retrieval_contract(
    application: Application,
    *,
    dataset: str,
    output_path: Path | None = None,
    query_override: str | None = None,
    top_k: int = 12,
) -> dict[str, Any]:
    """Run all contract surfaces against the application's real Cognee dataset."""
    nodes = contract___load_graph_nodes(application.paths.cognee / "graphs")
    contract___require(
        nodes, "No real Cognee graph snapshots are available for live contract."
    )
    allowed, dataset_id = contract___manifest_context(
        application.settings.data_dir, dataset
    )
    adapter = CogneeSearchAdapter(
        application.paths, application.knowledge_pipeline.compat
    )
    cognee_version = version("cognee")
    cases: list[dict[str, Any]] = []
    for spec in contract___CAPABILITIES:
        node = contract___representative(nodes, spec.datapoint_type)
        cases.append(
            await contract___datapoint_case(
                adapter,
                spec=spec,
                node=node,
                query=contract___query_from_node(node, query_override),
                dataset=dataset,
                dataset_id=dataset_id,
                allowed_canonical_ids=allowed,
                top_k=top_k,
                cognee_version=cognee_version,
            )
        )
    graph_node = contract___representative(nodes, "EntityDataPoint")
    cases.append(
        await contract___graph_case(
            adapter,
            application.knowledge_pipeline.compat,
            node=graph_node,
            query=contract___query_from_node(graph_node, query_override),
            dataset=dataset,
            dataset_id=dataset_id,
            allowed_canonical_ids=allowed,
            top_k=top_k,
            cognee_version=cognee_version,
        )
    )
    hard_failures = [
        case["capability"] for case in cases if not case["compat_supported"]
    ]
    report: dict[str, Any] = {
        "status": "partially_supported" if not hard_failures else "failed",
        "generated_at": datetime.now(UTC).isoformat(),
        "cognee_version": cognee_version,
        "dataset_name": dataset,
        "dataset_id": dataset_id,
        "case_count": len(cases),
        "hard_failures": hard_failures,
        "fallback_required": any(
            (
                not case["public_search_supported"]
                or not case["public_recall_supported"]
                or (not case["graph_provenance_recoverable"])
                for case in cases
            )
        ),
        "retrieval_fallback_types_used": sorted(
            application.knowledge_pipeline.compat.retrieval_fallback_types_used
        ),
        "cases": cases,
    }
    if output_path is not None:
        contract__write_contract_report(report, output_path)
    return report


"Fair live benchmark of Cognee public graph retrieval and PaperOS retrieval.\n\nRun directly; this project intentionally does not use pytest, mocks, or seeded\nretrieval results. The benchmark is resumable and writes after every real case.\n"
import argparse
import asyncio
import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

quality__REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(quality__REPOSITORY_ROOT))
from paperos_core.application import Application, create_application
from paperos_core.config import load_settings
from paperos_core.retrieval.candidates import QueryResponse
from paperos_core.retrieval.corpus import CorpusView

quality__UNAVAILABLE = "unavailable"
quality___QUERY_FILES = ("truth.jsonl", "associative.jsonl", "comprehensive.jsonl")


@dataclass(frozen=True, slots=True)
class quality__BenchmarkConfiguration:
    id: str
    retrieval_method: Literal["cognee_public_search", "cognee_public_recall", "paperos"]
    search_type: str | None
    top_k: int | None
    wide_search_top_k: int | None
    neighborhood_depth: int | None
    neighborhood_seed_top_k: int | None
    triplet_distance_penalty: float | str
    notes: str


quality__CONFIGURATIONS = (
    quality__BenchmarkConfiguration(
        "A",
        "cognee_public_search",
        "GRAPH_COMPLETION",
        12,
        100,
        1,
        10,
        6.5,
        "Near the previous public baseline; top_k limits graph triplets.",
    ),
    quality__BenchmarkConfiguration(
        "B",
        "cognee_public_search",
        "GRAPH_COMPLETION",
        40,
        100,
        1,
        40,
        6.5,
        "Candidate-budget comparison at one hop.",
    ),
    quality__BenchmarkConfiguration(
        "C",
        "cognee_public_search",
        "GRAPH_COMPLETION",
        40,
        100,
        2,
        40,
        6.5,
        "Primary fair comparison with PaperOS pool=40 and graph depth=2.",
    ),
    quality__BenchmarkConfiguration(
        "C80",
        "cognee_public_search",
        "GRAPH_COMPLETION",
        40,
        100,
        2,
        80,
        6.5,
        "Limited seed-budget sensitivity test.",
    ),
    quality__BenchmarkConfiguration(
        "D",
        "cognee_public_search",
        "GRAPH_COMPLETION",
        40,
        200,
        2,
        40,
        6.5,
        "Wide-recall sensitivity test; directly comparable with C.",
    ),
    quality__BenchmarkConfiguration(
        "E",
        "cognee_public_search",
        "GRAPH_COMPLETION",
        40,
        100,
        3,
        40,
        6.5,
        "Deep-neighborhood sensitivity test.",
    ),
    quality__BenchmarkConfiguration(
        "F",
        "cognee_public_search",
        "GRAPH_COMPLETION_CONTEXT_EXTENSION",
        40,
        100,
        2,
        40,
        6.5,
        "Associative/exploratory public comparison.",
    ),
    quality__BenchmarkConfiguration(
        "R",
        "cognee_public_recall",
        "GRAPH_COMPLETION",
        40,
        100,
        2,
        40,
        6.5,
        "Recall formatting/provenance comparison using C's retrieval budget.",
    ),
    quality__BenchmarkConfiguration(
        "G",
        "paperos",
        None,
        12,
        None,
        2,
        None,
        quality__UNAVAILABLE,
        "Production baseline: pool=40, final evidence=12, all configured channels.",
    ),
)


def quality___require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def quality___load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def quality___load_cases(data_dir: Path) -> list[dict[str, Any]]:
    query_root = data_dir / "validation" / "corpus" / "queries"
    cases = [
        case
        for name in quality___QUERY_FILES
        for case in quality___load_jsonl(query_root / name)
    ]
    quality___require(
        cases, f"""No genuine benchmark queries found under {query_root }"""
    )
    quality___require(
        len({str(case["case_id"]) for case in cases}) == len(cases),
        "Duplicate case IDs",
    )
    return cases


def quality___configuration_cases(
    configuration: quality__BenchmarkConfiguration, cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Select real cases relevant to this retrieval surface."""
    if configuration.id == "F":
        return [case for case in cases if case["profile"] != "truth"]
    return cases


def quality___graph_index(
    graph_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    canonical: dict[str, dict[str, Any]] = {}
    cognee_to_canonical: dict[str, str] = {}
    for path in sorted(graph_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for node in payload.get("nodes", []):
            if not isinstance(node, dict) or not node.get("canonical_id"):
                continue
            canonical_id = str(node["canonical_id"])
            canonical[canonical_id] = node
            if node.get("id"):
                cognee_to_canonical[str(node["id"])] = canonical_id
    quality___require(
        canonical, f"""No genuine graph snapshots found under {graph_root }"""
    )
    return (canonical, cognee_to_canonical)


def quality___contains_concept(searchable: str, concept: str) -> bool:
    normalized = concept.casefold()
    aliases = {"weak coupling": ("weak coupling", "弱耦合")}
    if any((alias in searchable for alias in aliases.get(normalized, (normalized,)))):
        return True
    tokens = re.findall("[a-z0-9]+", normalized)
    long_tokens = [token for token in tokens if len(token) >= 4]
    return bool(long_tokens) and all((token[:5] in searchable for token in long_tokens))


def quality___string_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return []


def quality___node_attributes(node: Any) -> dict[str, Any]:
    attributes = getattr(node, "attributes", None)
    if isinstance(attributes, dict):
        return attributes
    if isinstance(node, dict):
        nested = node.get("attributes")
        return nested if isinstance(nested, dict) else node
    return {}


def quality___node_id(node: Any, attributes: dict[str, Any]) -> str:
    value = getattr(node, "id", None) or attributes.get("id")
    return str(value) if value else ""


def quality___node_text(attributes: dict[str, Any]) -> str:
    return " ".join(
        (
            str(attributes[field])
            for field in ("name", "title", "text", "description")
            if isinstance(attributes.get(field), str) and attributes[field].strip()
        )
    )


def quality___edge_payload(
    edge: Any,
    *,
    canonical_nodes: dict[str, dict[str, Any]],
    cognee_to_canonical: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    edge_attributes = getattr(edge, "attributes", None)
    if not isinstance(edge_attributes, dict):
        edge_attributes = edge.get("attributes", {}) if isinstance(edge, dict) else {}
    node_payloads: list[dict[str, Any]] = []
    canonical_ids: list[str] = []
    for endpoint in (getattr(edge, "node1", None), getattr(edge, "node2", None)):
        attributes = quality___node_attributes(endpoint)
        cognee_id = quality___node_id(endpoint, attributes)
        public_canonical_id = attributes.get("canonical_id")
        public_source_chunk_ids = quality___string_list(
            attributes.get("source_chunk_ids")
        )
        canonical_id = str(
            public_canonical_id or cognee_to_canonical.get(cognee_id) or ""
        )
        stored = canonical_nodes.get(canonical_id, {})
        source_chunk_ids = quality___string_list(
            public_source_chunk_ids or stored.get("source_chunk_ids")
        )
        payload = {
            "node_id": cognee_id or None,
            "canonical_id": canonical_id or None,
            "node_type": str(
                attributes.get("type")
                or attributes.get("__type__")
                or stored.get("type")
                or stored.get("__type__")
                or "unavailable"
            ),
            "text": quality___node_text(attributes),
            "source_chunk_ids": source_chunk_ids,
            "public_canonical_id_preserved": bool(public_canonical_id),
            "public_source_chunk_ids_preserved": bool(public_source_chunk_ids),
            "evaluation_readback_used": bool(
                canonical_id
                and (not public_canonical_id)
                or (source_chunk_ids and (not public_source_chunk_ids))
            ),
        }
        node_payloads.append(payload)
        canonical_ids.append(canonical_id)
    relation = str(
        edge_attributes.get("relationship_name")
        or edge_attributes.get("relation_type")
        or edge_attributes.get("edge_text")
        or "unavailable"
    )
    edge_source_chunks = quality___string_list(edge_attributes.get("source_chunk_ids"))
    return (
        {
            "source": canonical_ids[0] or node_payloads[0]["node_id"],
            "target": canonical_ids[1] or node_payloads[1]["node_id"],
            "relation": relation,
            "source_chunk_ids": edge_source_chunks,
        },
        node_payloads,
    )


def quality___public_objects(
    results: object,
    *,
    canonical_nodes: dict[str, dict[str, Any]],
    cognee_to_canonical: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, int]:
    raw_items = results if isinstance(results, list) else []
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    contexts: list[str] = []
    for result in raw_items:
        if isinstance(result, dict):
            context = result.get("context_result")
            if isinstance(context, str):
                contexts.append(context)
            objects = result.get("objects_result")
        else:
            objects = None
        if not isinstance(objects, list):
            continue
        for edge in objects:
            edge_payload, endpoint_payloads = quality___edge_payload(
                edge,
                canonical_nodes=canonical_nodes,
                cognee_to_canonical=cognee_to_canonical,
            )
            edges.append(edge_payload)
            for node in endpoint_payloads:
                key = str(node["canonical_id"] or node["node_id"] or len(nodes))
                existing = nodes.get(key)
                if existing is None or (
                    not existing["source_chunk_ids"] and node["source_chunk_ids"]
                ):
                    nodes[key] = node
    return (list(nodes.values()), edges, "\n".join(contexts), len(raw_items))


def quality___recall_surface(
    entries: object,
) -> tuple[str, int, int, list[dict[str, Any]]]:
    items = entries if isinstance(entries, list) else []
    texts: list[str] = []
    structured: list[dict[str, Any]] = []
    for entry in items:
        if hasattr(entry, "model_dump"):
            payload = entry.model_dump(mode="json")
        elif isinstance(entry, dict):
            payload = entry
        else:
            payload = {"value": str(entry)}
        text = payload.get("text") or payload.get("content")
        if isinstance(text, str) and text.strip():
            texts.append(text)
        raw = payload.get("raw")
        if isinstance(raw, dict) and any(
            (
                key in raw
                for key in ("id", "node_id", "canonical_id", "source_chunk_ids")
            )
        ):
            structured.append(raw)
    return ("\n".join(texts), len(items), len(structured), structured)


def quality___chunk_ids_from_nodes_edges(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> set[str]:
    chunks = {
        chunk_id
        for node in nodes
        for chunk_id in quality___string_list(node.get("source_chunk_ids"))
    }
    chunks.update(
        (
            str(node["canonical_id"])
            for node in nodes
            if node.get("node_type") == "ChunkDataPoint" and node.get("canonical_id")
        )
    )
    chunks.update(
        (
            chunk_id
            for edge in edges
            for chunk_id in quality___string_list(edge.get("source_chunk_ids"))
        )
    )
    return chunks


def quality___quality_metrics(
    case: dict[str, Any],
    *,
    context: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    chunk_ids: set[str],
    corpus: CorpusView,
) -> dict[str, Any]:
    resolvable_chunks = {
        chunk_id for chunk_id in chunk_ids if chunk_id in corpus.chunks
    }
    filenames = {
        corpus.source_filenames[corpus.chunk_bundles[chunk_id].document.source_file_id]
        for chunk_id in resolvable_chunks
    }
    for node in nodes:
        canonical_id = node.get("canonical_id")
        if node.get("node_type") == "DocumentDataPoint":
            for bundle in corpus.bundles.values():
                if bundle.document.id == canonical_id:
                    filenames.add(
                        corpus.source_filenames[bundle.document.source_file_id]
                    )
    node_text = " ".join((str(node.get("text") or "") for node in nodes))
    relation_texts = [
        " ".join(
            [
                str(edge.get("relation") or ""),
                *(
                    str(node.get("text") or "")
                    for node in nodes
                    if node.get("canonical_id")
                    in {edge.get("source"), edge.get("target")}
                ),
            ]
        )
        for edge in edges
    ]
    resolved_chunk_text = " ".join(
        (corpus.chunks[item].text for item in resolvable_chunks)
    )
    searchable = " ".join([context, node_text, " ".join(relation_texts)]).casefold()
    expected_documents = {str(item) for item in case.get("expected_documents", [])}
    document_hits = expected_documents & filenames
    concepts = [str(item) for item in case.get("required_concepts", [])]
    concept_hits = [
        concept
        for concept in concepts
        if quality___contains_concept(searchable, concept)
    ]
    node_concept_hits = [
        concept
        for concept in concepts
        if quality___contains_concept(node_text.casefold(), concept)
    ]
    relation_concept_hits = [
        concept
        for concept in concepts
        if any(
            (
                quality___contains_concept(text.casefold(), concept)
                for text in relation_texts
            )
        )
    ]
    groups = list(case.get("required_evidence_groups", []))
    group_hits = [
        group
        for group in groups
        if any((str(term).casefold() in searchable for term in group.get("any_of", [])))
    ]
    chunk_group_hits = [
        group
        for group in groups
        if any(
            (
                str(term).casefold() in resolved_chunk_text.casefold()
                for term in group.get("any_of", [])
            )
        )
    ]
    page_count = sum(
        (corpus.chunks[item].page_start is not None for item in resolvable_chunks)
    )
    relevant_ranks: list[int] = []
    for rank, relation_text in enumerate(relation_texts, 1):
        edge_chunks = set(
            quality___string_list(edges[rank - 1].get("source_chunk_ids"))
        )
        edge_documents = {
            corpus.source_filenames[corpus.chunk_bundles[item].document.source_file_id]
            for item in edge_chunks
            if item in corpus.chunks
        }
        if expected_documents & edge_documents or any(
            (
                quality___contains_concept(relation_text.casefold(), concept)
                for concept in concepts
            )
        ):
            relevant_ranks.append(rank)
    return {
        "expected_document_hit": bool(document_hits),
        "expected_document_recall": (
            len(document_hits) / len(expected_documents) if expected_documents else None
        ),
        "expected_concept_hit": bool(concept_hits) if concepts else None,
        "expected_concept_recall": (
            len(concept_hits) / len(concepts) if concepts else None
        ),
        "relevant_node_hit": bool(node_concept_hits) if concepts else None,
        "relevant_node_recall": (
            len(node_concept_hits) / len(concepts) if concepts else None
        ),
        "relevant_relation_hit": (
            bool(relation_concept_hits) if case.get("requires_graph_relation") else None
        ),
        "relevant_relation_recall": (
            len(relation_concept_hits) / len(concepts)
            if case.get("requires_graph_relation") and concepts
            else None
        ),
        "expected_evidence_group_recall": (
            len(group_hits) / len(groups) if groups else None
        ),
        "evidence_chunk_coverage": (
            len(chunk_group_hits) / len(groups) if groups else None
        ),
        "page_provenance_available": (
            page_count / len(resolvable_chunks) if resolvable_chunks else 0.0
        ),
        "canonical_chunk_resolvable": (
            len(resolvable_chunks) / len(chunk_ids) if chunk_ids else 0.0
        ),
        "ranking_usefulness": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
        "matched_documents": sorted(document_hits),
        "matched_concepts": concept_hits,
        "resolvable_chunk_ids": sorted(resolvable_chunks),
    }


async def quality___run_public(
    configuration: quality__BenchmarkConfiguration,
    case: dict[str, Any],
    *,
    dataset: str,
    corpus: CorpusView,
    canonical_nodes: dict[str, dict[str, Any]],
    cognee_to_canonical: dict[str, str],
) -> dict[str, Any]:
    import cognee

    kwargs = {
        "query_text": str(case["query"]),
        "query_type": cognee.SearchType(str(configuration.search_type)),
        "datasets": [dataset],
        "top_k": configuration.top_k,
        "wide_search_top_k": configuration.wide_search_top_k,
        "neighborhood_depth": configuration.neighborhood_depth,
        "neighborhood_seed_top_k": configuration.neighborhood_seed_top_k,
        "triplet_distance_penalty": configuration.triplet_distance_penalty,
        "only_context": True,
    }
    started = time.perf_counter()
    error: str | None = None
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    context = ""
    raw_result_count = 0
    structured_object_count = 0
    try:
        if configuration.retrieval_method == "cognee_public_recall":
            entries = await cognee.recall(auto_route=False, **kwargs)
            context, raw_result_count, structured_object_count, raw_objects = (
                quality___recall_surface(entries)
            )
            nodes = [
                {
                    "node_id": item.get("node_id") or item.get("id"),
                    "canonical_id": item.get("canonical_id"),
                    "node_type": item.get("type") or "unavailable",
                    "text": str(item.get("text") or item.get("name") or ""),
                    "source_chunk_ids": quality___string_list(
                        item.get("source_chunk_ids")
                    ),
                }
                for item in raw_objects
            ]
        else:
            results = await cognee.search(verbose=True, node_type=None, **kwargs)
            nodes, edges, context, raw_result_count = quality___public_objects(
                results,
                canonical_nodes=canonical_nodes,
                cognee_to_canonical=cognee_to_canonical,
            )
            structured_object_count = len(edges)
    except Exception as exc:
        error = f"""{type (exc ).__name__ }: {exc }"""
    latency_ms = (time.perf_counter() - started) * 1000
    chunk_ids = quality___chunk_ids_from_nodes_edges(nodes, edges)
    public_provenance_nodes = [
        node
        for node in nodes
        if node.get("public_canonical_id_preserved")
        and (
            node.get("public_source_chunk_ids_preserved")
            or node.get("node_type") == "ChunkDataPoint"
        )
    ]
    quality = quality___quality_metrics(
        case,
        context=context,
        nodes=nodes,
        edges=edges,
        chunk_ids=chunk_ids,
        corpus=corpus,
    )
    is_context_extension = (
        configuration.search_type == "GRAPH_COMPLETION_CONTEXT_EXTENSION"
    )
    return {
        "case_id": str(case["case_id"]),
        "profile": str(case["profile"]),
        "query": str(case["query"]),
        "expected_documents": list(case.get("expected_documents", [])),
        "expected_concepts": list(case.get("required_concepts", [])),
        "expected_evidence_groups": list(case.get("required_evidence_groups", [])),
        "configuration_id": configuration.id,
        "retrieval_method": configuration.retrieval_method,
        "parameters": asdict(configuration),
        "raw_result_count": raw_result_count,
        "context_returned": bool(context.strip()),
        "structured_objects_returned": structured_object_count,
        "provenance_preserved": bool(nodes)
        and len(public_provenance_nodes) == len(nodes),
        "evaluation_readback_used": any(
            (bool(node.get("evaluation_readback_used")) for node in nodes)
        ),
        "public_canonical_id_preservation_rate": (
            sum((bool(node.get("public_canonical_id_preserved")) for node in nodes))
            / len(nodes)
            if nodes
            else 0.0
        ),
        "public_source_chunk_ids_preservation_rate": (
            sum((bool(node.get("public_source_chunk_ids_preserved")) for node in nodes))
            / len(nodes)
            if nodes
            else 0.0
        ),
        "structured_node_count": len(nodes),
        "edge_triplet_count": len(edges),
        "evidence_chunk_count": len(chunk_ids),
        "quality": quality,
        "runtime": {
            "latency_ms": round(latency_ms, 3),
            "llm_call_count": quality__UNAVAILABLE if is_context_extension else 0,
            "embedding_call_count": quality__UNAVAILABLE,
            "returned_context_size": len(context),
            "seed_count": quality__UNAVAILABLE,
            "candidate_count": len(nodes),
            "final_triplet_count": len(edges),
            "final_evidence_count": len(chunk_ids),
        },
        "nodes": nodes,
        "relations": edges,
        "context": context,
        "error": error,
    }


def quality___run_paperos(
    configuration: quality__BenchmarkConfiguration,
    case: dict[str, Any],
    *,
    run_root: Path,
    corpus: CorpusView,
    canonical_nodes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path = run_root / "logs" / "acceptance" / f"""query-{case ['case_id']}.json"""
    quality___require(path.is_file(), f"""Production response is missing: {path }""")
    payload = json.loads(path.read_text(encoding="utf-8"))
    response = QueryResponse.model_validate(payload["response"])
    chunk_ids = {item.chunk_id for item in response.evidence}
    derived_ids = {
        item
        for candidate in response.candidates
        for item in [candidate.object_id, *candidate.derived_from_ids]
    }
    nodes = [
        {
            "node_id": canonical_nodes[item].get("id"),
            "canonical_id": item,
            "node_type": canonical_nodes[item].get("type")
            or canonical_nodes[item].get("__type__"),
            "text": quality___node_text(canonical_nodes[item]),
            "source_chunk_ids": quality___string_list(
                canonical_nodes[item].get("source_chunk_ids")
            ),
        }
        for item in sorted(derived_ids)
        if item in canonical_nodes
    ]
    relation_count = sum(
        (
            candidate.knowledge_kind == "structured_relation"
            or "graph" in candidate.channels
            for candidate in response.candidates
        )
    )
    context = "\n".join((item.text for item in response.evidence))
    quality = quality___quality_metrics(
        case, context=context, nodes=nodes, edges=[], chunk_ids=chunk_ids, corpus=corpus
    )
    quality["relevant_relation_recall"] = None
    quality["relevant_relation_hit"] = (
        bool(relation_count) if case.get("requires_graph_relation") else None
    )
    return {
        "case_id": str(case["case_id"]),
        "profile": str(case["profile"]),
        "query": str(case["query"]),
        "expected_documents": list(case.get("expected_documents", [])),
        "expected_concepts": list(case.get("required_concepts", [])),
        "expected_evidence_groups": list(case.get("required_evidence_groups", [])),
        "configuration_id": configuration.id,
        "retrieval_method": configuration.retrieval_method,
        "parameters": {**asdict(configuration), "candidate_pool_size": 40},
        "raw_result_count": len(response.candidates),
        "context_returned": bool(context),
        "structured_objects_returned": len(nodes),
        "provenance_preserved": response.provenance_complete,
        "structured_node_count": len(nodes),
        "edge_triplet_count": relation_count,
        "evidence_chunk_count": len(chunk_ids),
        "quality": quality,
        "runtime": {
            "latency_ms": quality__UNAVAILABLE,
            "llm_call_count": quality__UNAVAILABLE,
            "embedding_call_count": quality__UNAVAILABLE,
            "returned_context_size": len(context),
            "seed_count": quality__UNAVAILABLE,
            "candidate_count": len(response.candidates),
            "final_triplet_count": relation_count,
            "final_evidence_count": len(response.evidence),
        },
        "nodes": nodes,
        "relations": [],
        "relation_metric_limitation": "Production response retains graph-derived IDs but not the exact ranked typed edge list; relevant_relation_recall is unavailable for G.",
        "context": context,
        "fallback_types": [
            "custom_datapoint_vector_search",
            "graph_node_provenance_readback",
            "typed_graph_traversal",
        ],
        "error": None,
    }


def quality___mean(cases: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for case in cases:
        value: Any = case
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and (not isinstance(value, bool)):
            values.append(float(value))
    return fmean(values) if values else None


def quality___aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["configuration_id"])].append(case)
    return {
        config_id: {
            "case_count": len(items),
            "error_count": sum((bool(item["error"]) for item in items)),
            "context_return_rate": fmean(
                (bool(item["context_returned"]) for item in items)
            ),
            "provenance_preservation_rate": fmean(
                (bool(item["provenance_preserved"]) for item in items)
            ),
            "expected_document_recall": quality___mean(
                items, ("quality", "expected_document_recall")
            ),
            "expected_concept_recall": quality___mean(
                items, ("quality", "expected_concept_recall")
            ),
            "relevant_node_recall": quality___mean(
                items, ("quality", "relevant_node_recall")
            ),
            "relevant_relation_recall": quality___mean(
                items, ("quality", "relevant_relation_recall")
            ),
            "evidence_chunk_coverage": quality___mean(
                items, ("quality", "evidence_chunk_coverage")
            ),
            "page_provenance_available": quality___mean(
                items, ("quality", "page_provenance_available")
            ),
            "canonical_chunk_resolvable": quality___mean(
                items, ("quality", "canonical_chunk_resolvable")
            ),
            "ranking_usefulness": quality___mean(
                items, ("quality", "ranking_usefulness")
            ),
            "latency_ms": quality___mean(items, ("runtime", "latency_ms")),
            "average_context_size": quality___mean(
                items, ("runtime", "returned_context_size")
            ),
            "average_triplet_count": quality___mean(
                items, ("runtime", "final_triplet_count")
            ),
            "average_evidence_count": quality___mean(
                items, ("runtime", "final_evidence_count")
            ),
            "by_profile": {
                profile: {
                    "case_count": len(profile_items),
                    "expected_document_recall": quality___mean(
                        profile_items, ("quality", "expected_document_recall")
                    ),
                    "expected_concept_recall": quality___mean(
                        profile_items, ("quality", "expected_concept_recall")
                    ),
                    "relevant_relation_recall": quality___mean(
                        profile_items, ("quality", "relevant_relation_recall")
                    ),
                }
                for profile in ("truth", "associative", "comprehensive")
                if (
                    profile_items := [
                        item for item in items if item["profile"] == profile
                    ]
                )
            },
        }
        for config_id, items in grouped.items()
    }


def quality___relative_change(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return right - left


def quality___conclusions(aggregate: dict[str, Any]) -> dict[str, Any]:

    def metric(config: str, name: str) -> float | None:
        value = aggregate.get(config, {}).get(name)
        return float(value) if isinstance(value, (int, float)) else None

    c_document = metric("C", "expected_document_recall")
    c_concept = metric("C", "expected_concept_recall")
    g_document = metric("G", "expected_document_recall")
    g_concept = metric("G", "expected_concept_recall")
    relevance_close = all(
        (
            left is not None and right is not None and (left >= right - 0.05)
            for left, right in ((c_document, g_document), (c_concept, g_concept))
        )
    )
    public_provenance = metric("C", "provenance_preservation_rate") or 0.0
    return {
        "fair_public_graph_relevance_close_to_paperos": relevance_close,
        "depth_1_to_2": {
            "document_recall_delta": quality___relative_change(
                metric("B", "expected_document_recall"), c_document
            ),
            "concept_recall_delta": quality___relative_change(
                metric("B", "expected_concept_recall"), c_concept
            ),
        },
        "depth_2_to_3": {
            "document_recall_delta": quality___relative_change(
                c_document, metric("E", "expected_document_recall")
            ),
            "concept_recall_delta": quality___relative_change(
                c_concept, metric("E", "expected_concept_recall")
            ),
        },
        "top_k_12_to_40_at_depth_1": {
            "document_recall_delta": quality___relative_change(
                metric("A", "expected_document_recall"),
                metric("B", "expected_document_recall"),
            ),
            "concept_recall_delta": quality___relative_change(
                metric("A", "expected_concept_recall"),
                metric("B", "expected_concept_recall"),
            ),
        },
        "wide_100_to_200_at_depth_2": {
            "document_recall_delta": quality___relative_change(
                c_document, metric("D", "expected_document_recall")
            ),
            "concept_recall_delta": quality___relative_change(
                c_concept, metric("D", "expected_concept_recall")
            ),
        },
        "seed_40_to_80_at_depth_2": {
            "document_recall_delta": quality___relative_change(
                c_document, metric("C80", "expected_document_recall")
            ),
            "concept_recall_delta": quality___relative_change(
                c_concept, metric("C80", "expected_concept_recall")
            ),
        },
        "context_extension": {
            "document_recall": metric("F", "expected_document_recall"),
            "concept_recall": metric("F", "expected_concept_recall"),
            "provenance_preservation_rate": metric("F", "provenance_preservation_rate"),
            "average_latency_ms": metric("F", "latency_ms"),
            "assessment": "Improves concept recall over standard C for associative/comprehensive queries, but remains below G, is LLM-dependent and slow, and preserves no public canonical provenance.",
        },
        "public_recall_interpretation": "R returns formatted context. Its zero structured document/evidence metrics mean document identity is unobservable without provenance, not that relevant context was absent.",
        "decisions": {
            "custom_chunk_direct_vector_search": "KEEP",
            "custom_datapoint_graph_seed_search": (
                "REDUCE" if relevance_close else "KEEP"
            ),
            "typed_graph_traversal": (
                "COMPAT_PROVENANCE_ONLY" if relevance_close else "KEEP"
            ),
            "graph_node_provenance_readback": (
                "REDUCE" if public_provenance >= 0.95 else "KEEP"
            ),
            "semantic_graph_retrieval": (
                "PUBLIC_API_PRIMARY" if relevance_close else "KEEP"
            ),
            "removed_fallbacks": [],
        },
        "decision_rule": "Public GRAPH_COMPLETION is close only when aggregate document and concept recall are each within 0.05 of PaperOS G. Provenance is evaluated separately.",
    }


def quality___write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f""".{output_path .name }.tmp""")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output_path)


async def quality__run_benchmark(
    *,
    run_root: Path,
    dataset: str,
    output_path: Path,
    resume: bool,
    configuration_ids: set[str] | None,
    profiles: set[str] | None,
    retry_errors: bool,
) -> dict[str, Any]:
    configured = load_settings()
    settings = configured.model_copy(
        update={
            "data": configured.data.model_copy(
                update={"directory": run_root, "dataset": dataset}
            )
        }
    )
    application: Application = create_application(settings)
    all_cases = quality___load_cases(configured.data_dir)
    cases = [
        case
        for case in all_cases
        if profiles is None or str(case["profile"]) in profiles
    ]
    quality___require(cases, "No benchmark cases selected.")
    configurations = [
        item
        for item in quality__CONFIGURATIONS
        if configuration_ids is None or item.id in configuration_ids
    ]
    quality___require(configurations, "No benchmark configurations selected.")
    existing: dict[tuple[str, str], dict[str, Any]] = {}
    previous_configurations: list[dict[str, Any]] = []
    if resume and output_path.is_file():
        previous = json.loads(output_path.read_text(encoding="utf-8"))
        previous_configurations = list(previous.get("configurations", []))
        existing = {
            (str(item["configuration_id"]), str(item["case_id"])): item
            for item in previous.get("cases", [])
        }
        if retry_errors:
            selected_configuration_ids = {item.id for item in configurations}
            selected_profiles = {str(case["profile"]) for case in cases}
            existing = {
                key: item
                for key, item in existing.items()
                if not (
                    key[0] in selected_configuration_ids
                    and str(item.get("profile")) in selected_profiles
                    and item.get("error")
                )
            }
    configuration_map = {
        str(item["id"]): item for item in previous_configurations if item.get("id")
    }
    configuration_map.update({item.id: asdict(item) for item in configurations})
    selected_case_keys = {
        (configuration.id, str(case["case_id"]))
        for configuration in configurations
        for case in quality___configuration_cases(configuration, cases)
    }
    all_case_keys = set(existing) | selected_case_keys
    report: dict[str, Any] = {
        "status": "running",
        "benchmark_kind": "retrieval_quality",
        "capability_contract_path": "logs/contracts/cognee-retrieval-boundary.json",
        "cognee_version": version("cognee"),
        "dataset": dataset,
        "run_root": ".",
        "generated_at": datetime.now(UTC).isoformat(),
        "source_notes": {
            "graph_completion": "Cognee 1.4.0 GraphCompletionRetriever discovers index_fields from all loaded DataPoint subclasses; node_type is explicitly None for mixed graphs.",
            "built_in_retrievers": "CHUNKS/SUMMARIES/HYBRID retain Cognee-native schema assumptions and are capability observations, not tuned quality failures.",
            "recall": "only_context recall returns normalized formatted graph context; context, structured objects, and provenance are counted separately.",
            "runtime_counts": "Standard GraphCompletion with only_context performs zero LLM completion calls. Calls not exposed reliably by public APIs are unavailable, not guessed.",
            "production_relation_metric": "G retains graph-derived IDs but not the exact ranked typed edge list; its relation recall is unavailable and is not used for fallback deletion.",
        },
        "configurations": [configuration_map[key] for key in sorted(configuration_map)],
        "invocation_scope": {
            "configuration_ids": [item.id for item in configurations],
            "profiles": sorted(profiles) if profiles is not None else "all",
            "context_extension_profiles": ["associative", "comprehensive"],
        },
        "cases": list(existing.values()),
        "aggregate_metrics": {},
        "conclusions": {},
    }
    await application.start()
    try:
        corpus = CorpusView.load(
            application.paths, application.canonical_repository, application.registry
        )
        canonical_nodes, cognee_to_canonical = quality___graph_index(
            application.paths.cognee / "graphs"
        )
        total = len(all_case_keys)
        completed = len(existing)
        for configuration in configurations:
            for case in quality___configuration_cases(configuration, cases):
                key = (configuration.id, str(case["case_id"]))
                if key in existing:
                    continue
                print(
                    f"""benchmark {completed +1 }/{total } {configuration .id } {case ['profile']} {case ['case_id']}""",
                    flush=True,
                )
                if configuration.retrieval_method == "paperos":
                    result = quality___run_paperos(
                        configuration,
                        case,
                        run_root=run_root,
                        corpus=corpus,
                        canonical_nodes=canonical_nodes,
                    )
                else:
                    result = await quality___run_public(
                        configuration,
                        case,
                        dataset=dataset,
                        corpus=corpus,
                        canonical_nodes=canonical_nodes,
                        cognee_to_canonical=cognee_to_canonical,
                    )
                existing[key] = result
                completed += 1
                report["cases"] = list(existing.values())
                report["aggregate_metrics"] = quality___aggregate(report["cases"])
                report["conclusions"] = quality___conclusions(
                    report["aggregate_metrics"]
                )
                report["completed_case_runs"] = completed
                report["total_case_runs"] = total
                quality___write_report(report, output_path)
        report["status"] = "completed"
        report["completed_at"] = datetime.now(UTC).isoformat()
        report["cases"] = sorted(
            existing.values(),
            key=lambda item: (item["configuration_id"], item["case_id"]),
        )
        report["aggregate_metrics"] = quality___aggregate(report["cases"])
        report["conclusions"] = quality___conclusions(report["aggregate_metrics"])
        report["completed_case_runs"] = len(report["cases"])
        report["total_case_runs"] = len(all_case_keys)
        quality___write_report(report, output_path)
        return report
    finally:
        await application.aclose()


def quality__main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profiles", help="Comma-separated profiles; F always excludes truth cases."
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="With --resume, rerun selected cases whose retained result has an error.",
    )
    parser.add_argument(
        "--configurations",
        help="Comma-separated configuration IDs for a bounded diagnostic run.",
    )
    args = parser.parse_args()
    profiles = (
        {item.strip() for item in args.profiles.split(",") if item.strip()}
        if args.profiles
        else None
    )
    run_root = args.run_root.expanduser().resolve()
    output = (
        args.output
        or run_root / "logs" / "contracts" / "cognee-retrieval-quality-benchmark.json"
    )
    selected = (
        {item.strip() for item in args.configurations.split(",") if item.strip()}
        if args.configurations
        else None
    )
    report = asyncio.run(
        quality__run_benchmark(
            run_root=run_root,
            dataset=args.dataset,
            output_path=output,
            resume=args.resume,
            configuration_ids=selected,
            profiles=profiles,
            retry_errors=args.retry_errors,
        )
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "completed_case_runs": report["completed_case_runs"],
                "aggregate_metrics": report["aggregate_metrics"],
                "conclusions": report["conclusions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


"Run the cumulative PaperOS acceptance path using only genuine papers.\n\nThis is the project's executable validation entry. It does not use pytest,\nmocks, fabricated parser output, precomputed embeddings, or fixed LLM output.\nEvery run starts from the user-supplied PDF corpus and calls live MinerU,\nCognee's configured LLM/embedding providers, the graph/vector stores, FTS, and\nall three PaperOS retrieval profiles.\n"
import argparse
import asyncio
import hashlib
import json
import re
import sqlite3
import sys
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

pipeline__REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(pipeline__REPOSITORY_ROOT))
pipeline__VALIDATION_ROOT_NAME = "validation"
pipeline__CORPUS_DIRECTORY_NAME = "corpus"
pipeline__RUNS_DIRECTORY_NAME = "runs"
from paperos_core.application import create_application
from paperos_core.config import RuntimeSettings, load_settings
from paperos_core.errors import PaperOSError
from paperos_core.retrieval.candidates import QueryRequest, QueryResponse


def pipeline___require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def pipeline___sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pipeline___file_hashes(roots: list[Path]) -> dict[str, str]:
    return {
        str(path): pipeline___sha256(path)
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }


def pipeline___load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pipeline___load_corpus(
    data_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corpus = data_dir / pipeline__VALIDATION_ROOT_NAME / pipeline__CORPUS_DIRECTORY_NAME
    manifest_path = corpus / "manifest.json"
    pipeline___require(
        manifest_path.is_file(),
        f"""Real corpus manifest is missing: {manifest_path }""",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    papers = manifest.get("papers")
    pipeline___require(
        isinstance(papers, list) and papers, "Real corpus contains no papers."
    )
    for paper in papers:
        pdf = corpus / "pdfs" / paper["pdf_file"]
        pipeline___require(pdf.is_file(), f"""Real PDF is missing: {pdf }""")
        pipeline___require(
            pipeline___sha256(pdf) == paper["sha256"],
            f"""Real PDF checksum mismatch: {pdf }""",
        )
    queries: list[dict[str, Any]] = []
    for name in ("truth.jsonl", "associative.jsonl", "comprehensive.jsonl"):
        query_path = corpus / "queries" / name
        pipeline___require(
            query_path.is_file(), f"""Real query cases are missing: {query_path }"""
        )
        cases = pipeline___load_jsonl(query_path)
        expected_profile = name.removesuffix(".jsonl")
        pipeline___require(
            cases, f"""Retrieval profile has no real case: {expected_profile }"""
        )
        pipeline___require(
            all((case.get("profile") == expected_profile for case in cases)),
            f"""Query file contains the wrong profile: {query_path }""",
        )
        queries.extend(cases)
    pipeline___require(queries, "Real corpus contains no query cases.")
    return (papers, queries)


def pipeline___load_expected_cases(data_dir: Path) -> dict[str, dict[str, Any]]:
    expected_root = (
        data_dir
        / pipeline__VALIDATION_ROOT_NAME
        / pipeline__CORPUS_DIRECTORY_NAME
        / "expected"
    )
    cases: dict[str, dict[str, Any]] = {}
    for path in sorted(expected_root.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        filename = str(case["pdf_file"])
        pipeline___require(
            filename not in cases, f"""Duplicate real expectation: {filename }"""
        )
        cases[filename] = case
    pipeline___require(
        cases, f"""Real ingestion expectations are missing: {expected_root }"""
    )
    return cases


def pipeline___normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub("[\\u2010-\\u2015\\u2212]", "-", value)
    return " ".join(value.split())


def pipeline___all_element_text(element: Any) -> str:
    return "\n".join(
        (
            value
            for value in (
                element.text,
                element.raw_text,
                element.markdown,
                element.latex,
                element.html,
            )
            if value
        )
    )


def pipeline___canonical_element_text(element: Any) -> str:
    if element.element_type.value == "table":
        return element.markdown or element.text or element.html or ""
    if element.element_type.value == "formula":
        return element.latex or element.text or element.markdown or ""
    return element.text if element.text is not None else element.markdown or ""


def pipeline___validate_real_ingestion(
    *,
    expected: dict[str, Any],
    bundle: Any,
    projection: Any,
    chunk_target_tokens: int,
    chunk_hard_max_tokens: int,
    cognee_manifest_path: Path,
    index_manifest_path: Path,
) -> dict[str, Any]:
    """Validate live MinerU/Cognee output against one genuine paper."""
    filename = str(expected["pdf_file"])
    document = bundle.document
    expected_document = expected["document"]
    pipeline___require(
        pipeline___normalized(document.title)
        == pipeline___normalized(expected_document["expected_title"]),
        f"""Canonical title mismatch: {filename } / {document .title }""",
    )
    pipeline___require(
        document.language == expected_document["language"],
        f"""Language mismatch: {filename }""",
    )
    pipeline___require(
        document.document_type == expected_document["document_type"],
        f"""Document type mismatch: {filename }""",
    )
    structure = expected["structure"]
    pipeline___require(
        len(bundle.sections) >= structure["minimum_section_count"],
        f"""Too few real sections: {filename }""",
    )
    pipeline___require(
        len(projection.chunks) >= structure["minimum_chunk_count"],
        f"""Too few real chunks: {filename }""",
    )
    pipeline___require(
        len(bundle.references) >= structure["minimum_reference_count"],
        f"""Too few real references: {filename }""",
    )
    section_titles = [
        pipeline___normalized(section.title) for section in bundle.sections
    ]
    for requirement in structure["required_sections"]:
        title = pipeline___normalized(requirement["title"])
        if requirement["match"] == "normalized_exact":
            found = title in section_titles
        else:
            compact_title = title.replace(" ", "")
            found = any(
                (
                    compact_title in candidate.replace(" ", "")
                    for candidate in section_titles
                )
            )
        pipeline___require(
            found,
            f"""Required real section absent: {filename } / {requirement ['title']}""",
        )
    element_counts = Counter(
        (element.element_type.value for element in bundle.elements)
    )
    element_requirements = expected["elements"]
    for element_type in element_requirements["must_contain"]:
        pipeline___require(
            element_counts[element_type] > 0,
            f"""Missing {element_type }: {filename }""",
        )
    pipeline___require(
        element_counts["figure"] >= element_requirements["minimum_figure_count"],
        f"""Too few figures: {filename }""",
    )
    pipeline___require(
        element_counts["formula"] >= element_requirements["minimum_formula_count"],
        f"""Too few formulas: {filename }""",
    )
    if element_requirements["require_figure_captions"]:
        pipeline___require(
            element_counts["caption"] > 0, f"""Figure captions absent: {filename }"""
        )
    if element_requirements["require_reference_entries"]:
        pipeline___require(
            bundle.references, f"""Reference entries absent: {filename }"""
        )
    searchable = pipeline___normalized(
        "\n".join(
            [
                document.title,
                document.abstract or "",
                *(pipeline___all_element_text(element) for element in bundle.elements),
            ]
        )
    )
    for check in expected["content_checks"]:
        if not check.get("required", True):
            continue
        pipeline___require(
            any(
                (
                    pipeline___normalized(value) in searchable
                    for value in check["any_of"]
                )
            ),
            f"""Required real text absent: {filename } / {check ['any_of']}""",
        )
    elements = {element.id: element for element in bundle.elements}
    chunks = {chunk.id: chunk for chunk in projection.chunks}
    for chunk in projection.chunks:
        pipeline___require(
            chunk.token_count is not None, f"""Chunk token count absent: {chunk .id }"""
        )
        pipeline___require(
            chunk.token_count <= chunk_hard_max_tokens,
            f"""Chunk exceeds token target: {chunk .id } / {chunk .token_count }""",
        )
        pipeline___require(chunk.spans, f"""Chunk spans absent: {chunk .id }""")
        pipeline___require(
            chunk.element_span_ids == [span.id for span in chunk.spans],
            f"""Chunk span IDs diverge: {chunk .id }""",
        )
        span_sections: set[str | None] = set()
        for span in chunk.spans:
            element = elements.get(span.element_id)
            if element is None:
                raise RuntimeError(f"""Chunk references unknown element: {span .id }""")
            source = pipeline___canonical_element_text(element)
            pipeline___require(
                source[span.character_start_in_element : span.character_end_in_element]
                == span.text,
                f"""Element character span is not exact: {span .id }""",
            )
            pipeline___require(
                span.token_start < span.token_end,
                f"""Invalid token span: {span .id }""",
            )
            span_sections.add(element.section_id)
        non_null_sections = {value for value in span_sections if value is not None}
        pipeline___require(
            non_null_sections <= {chunk.section_id},
            f"""Chunk crosses section boundary: {chunk .id }""",
        )
        for source_chunk_id in chunk.overlap_source_chunk_ids:
            source_chunk = chunks.get(source_chunk_id)
            if source_chunk is None:
                raise RuntimeError(f"""Unknown overlap chunk: {source_chunk_id }""")
            pipeline___require(
                source_chunk.section_id == chunk.section_id,
                f"""Overlap crosses section boundary: {chunk .id }""",
            )
    cognee_manifest = json.loads(cognee_manifest_path.read_text(encoding="utf-8"))
    mapped_ids = set(cognee_manifest["canonical_to_cognee_id"])
    pipeline___require(
        set(chunks) <= mapped_ids,
        f"""Cognee is missing canonical chunks: {filename }""",
    )
    pipeline___require(
        cognee_manifest["node_count"] > 0, f"""Cognee has no nodes: {filename }"""
    )
    pipeline___require(
        cognee_manifest["relation_count"] > 0,
        f"""Cognee has no relations: {filename }""",
    )
    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    projection_ids = set(index_manifest["chunk_projection_ids"])
    lexical_ids = set(index_manifest["lexical_object_ids"])
    pipeline___require(
        projection_ids == set(chunks), f"""Chunk projection mismatch: {filename }"""
    )
    pipeline___require(
        set(chunks) <= lexical_ids, f"""FTS is missing canonical chunks: {filename }"""
    )
    searchable_types: set[str] = set()
    for chunk in projection.chunks:
        if chunk.id not in lexical_ids:
            continue
        searchable_types.update(
            (
                elements[element_id].element_type.value
                for element_id in chunk.element_ids
                if element_id in elements
            )
        )
    return {
        "filename": filename,
        "sections": len(bundle.sections),
        "chunks": len(projection.chunks),
        "references": len(bundle.references),
        "element_counts": dict(sorted(element_counts.items())),
        "searchable_element_types": sorted(searchable_types),
    }


def pipeline___settings_for_run(
    configured: RuntimeSettings, run_root: Path, dataset: str
) -> RuntimeSettings:
    return configured.model_copy(
        update={
            "data": configured.data.model_copy(
                update={"directory": run_root.resolve(), "dataset": dataset}
            )
        }
    )


def pipeline___contains_concept(searchable: str, concept: str) -> bool:
    normalized = concept.casefold()
    aliases = {"weak coupling": ("weak coupling", "弱耦合")}
    if any((alias in searchable for alias in aliases.get(normalized, (normalized,)))):
        return True
    tokens = re.findall("[a-z0-9]+", normalized)
    long_tokens = [token for token in tokens if len(token) >= 4]
    return bool(long_tokens) and all((token[:5] in searchable for token in long_tokens))


def pipeline___validate_query(
    case: dict[str, Any], response: QueryResponse
) -> dict[str, Any]:
    """Enforce model-independent integrity and measure semantic quality softly."""
    case_id = str(case["case_id"])
    quality_warnings: list[str] = []
    pipeline___require(
        response.profile.value == case["profile"], f"""Profile mismatch: {case_id }"""
    )
    pipeline___require(
        set(case.get("required_channels", [])) <= set(response.channels_used),
        f"""Missing retrieval channel: {case_id }""",
    )
    pipeline___require(
        set(case.get("required_stages", [])) <= set(response.stages),
        f"""Missing retrieval stage: {case_id }""",
    )
    pipeline___require(
        response.provenance_complete, f"""Incomplete provenance: {case_id }"""
    )
    pipeline___require(
        len(response.evidence) == len(response.candidates) > 0,
        f"""No evidence-bound candidates: {case_id }""",
    )
    pipeline___require(
        all((item.chunk_id for item in response.evidence)),
        f"""Evidence lacks chunk IDs: {case_id }""",
    )
    cited_evidence = [
        item for item in response.evidence if item.evidence_id in response.answer
    ]
    pipeline___require(
        cited_evidence, f"""Answer lacks evidence citations: {case_id }"""
    )
    if case.get("requires_page"):
        pipeline___require(
            all((item.page_start is not None for item in response.evidence)),
            f"""Evidence lacks page coordinates: {case_id }""",
        )
    if case.get("requires_graph_relation"):
        pipeline___require(
            "graph" in response.channels_used, f"""Graph channel absent: {case_id }"""
        )
        if not any(
            (
                "graph" in candidate.channels
                or candidate.knowledge_kind == "structured_relation"
                for candidate in response.candidates
            )
        ):
            quality_warnings.append(
                f"""{case_id }: graph stage ran but returned no structured relation evidence"""
            )
    filenames = {item.source_filename for item in response.evidence}
    expected_documents = set(case.get("expected_documents", []))
    document_hits = expected_documents & filenames
    missing_documents = sorted(expected_documents - filenames)
    if missing_documents:
        quality_warnings.append(
            f"""{case_id }: expected documents absent from ranked evidence: {missing_documents }"""
        )
    minimum_documents = int(case.get("minimum_distinct_documents", 1))
    if response.distinct_documents < minimum_documents:
        quality_warnings.append(
            f"""{case_id }: document diversity {response .distinct_documents } < {minimum_documents }"""
        )
    searchable = " ".join(
        [response.answer, *(item.text for item in response.evidence)]
    ).casefold()
    evidence_groups = list(case.get("required_evidence_groups", []))
    evidence_group_hits = 0
    for group in evidence_groups:
        if any((term.casefold() in searchable for term in group["any_of"])):
            evidence_group_hits += 1
        else:
            quality_warnings.append(
                f"""{case_id }: expected evidence terms absent: {group ['any_of']}"""
            )
    concepts = [str(item) for item in case.get("required_concepts", [])]
    concept_hits = 0
    for concept in concepts:
        if pipeline___contains_concept(searchable, concept):
            concept_hits += 1
        else:
            quality_warnings.append(
                f"""{case_id }: expected concept absent: {concept }"""
            )
    evidence_count = len(response.evidence)
    page_count = sum((item.page_start is not None for item in response.evidence))
    return {
        "case_id": case_id,
        "profile": response.profile.value,
        "channels_used": response.channels_used,
        "stages": response.stages,
        "candidate_count": len(response.candidates),
        "distinct_documents": response.distinct_documents,
        "provenance_complete": response.provenance_complete,
        "expected_document_hit_rate": (
            len(document_hits) / len(expected_documents) if expected_documents else None
        ),
        "expected_concept_hit_rate": concept_hits / len(concepts) if concepts else None,
        "expected_evidence_group_hit_rate": (
            evidence_group_hits / len(evidence_groups) if evidence_groups else None
        ),
        "evidence_precision_indicators": {
            "chunk_provenance_ratio": sum(
                (bool(item.chunk_id) for item in response.evidence)
            )
            / evidence_count,
            "page_provenance_ratio": page_count / evidence_count,
            "citation_ratio": len(cited_evidence) / evidence_count,
        },
        "quality_warnings": quality_warnings,
    }


def pipeline___validate_enrichment(path: Path, filename: str) -> None:
    enrichment = json.loads(path.read_text(encoding="utf-8"))
    pipeline___require(
        enrichment["coverage_ratio"] == 1.0, f"""Incomplete enrichment: {filename }"""
    )
    pipeline___require(
        not enrichment["uncovered_chunk_ids"], f"""Uncovered chunks: {filename }"""
    )
    pipeline___require(
        enrichment["prompt_sha256"], f"""Missing prompt SHA: {filename }"""
    )


async def pipeline___run(args: argparse.Namespace) -> dict[str, Any]:
    configured = load_settings()
    if args.local_inference_port is not None:
        pipeline___require(
            1 <= args.local_inference_port <= 65535,
            "--local-inference-port must be between 1 and 65535.",
        )
        configured = configured.model_copy(
            update={
                "local_inference": configured.local_inference.model_copy(
                    update={"port": args.local_inference_port}
                )
            }
        )
    pipeline___require(
        configured.mineru.api_key_value(),
        "mineru.api_key must be configured in config/paperos.toml.",
    )
    papers, queries = pipeline___load_corpus(configured.data_dir)
    expected_cases = pipeline___load_expected_cases(configured.data_dir)
    pipeline___require(
        set(expected_cases) == {str(paper["pdf_file"]) for paper in papers},
        "The real manifest and ingestion expectations describe different papers.",
    )
    run_root = args.run_root.resolve()
    logs = run_root / "logs" / "acceptance"
    logs.mkdir(parents=True, exist_ok=True)
    settings = pipeline___settings_for_run(configured, run_root, args.dataset)
    application = create_application(settings)
    local_pid: int | None = None
    local_process: asyncio.subprocess.Process | None = None
    started_at = datetime.now(UTC)
    ingestions: list[dict[str, Any]] = []
    structural_results: list[dict[str, Any]] = []
    responses: list[QueryResponse] = []
    quality_warnings: list[str] = []
    quality_results: list[dict[str, Any]] = []
    print(f"""run_root={run_root }""", flush=True)
    print(f"""dataset={args .dataset }""", flush=True)
    await application.start()
    local_process = application.runtime.local_inference.process
    local_pid = application.runtime.local_inference.pid
    try:
        existing = {
            application.registry.get_source(
                bundle.document.source_file_id
            ).original_filename: bundle
            for bundle in application.canonical_repository.list_bundles()
        }
        for position, paper in enumerate(papers, 1):
            filename = str(paper["pdf_file"])
            payload: dict[str, Any]
            if args.resume and filename in existing:
                bundle = existing[filename]
                enrichment_path = (
                    application.paths.cognee
                    / "enrichment"
                    / f"""{bundle .snapshot .id }.json"""
                )
                cognee_manifest = (
                    application.paths.cognee
                    / "manifests"
                    / f"""{bundle .snapshot .id }.json"""
                )
                index_manifest = (
                    application.paths.indexes
                    / "manifests"
                    / f"""{bundle .snapshot .id }.json"""
                )
                if all(
                    (
                        path.is_file()
                        for path in (enrichment_path, cognee_manifest, index_manifest)
                    )
                ):
                    pipeline___validate_enrichment(enrichment_path, filename)
                    projection = application.canonical_repository.get_chunk_projection(
                        bundle.snapshot.id
                    )
                    structural_results.append(
                        pipeline___validate_real_ingestion(
                            expected=expected_cases[filename],
                            bundle=bundle,
                            projection=projection,
                            chunk_target_tokens=settings.ingestion.chunk_target_tokens,
                            chunk_hard_max_tokens=settings.ingestion.chunk_hard_max_tokens,
                            cognee_manifest_path=cognee_manifest,
                            index_manifest_path=index_manifest,
                        )
                    )
                    print(
                        f"""ingest {position }/{len (papers )} reused {filename }""",
                        flush=True,
                    )
                    continue
                print(
                    f"""ingest {position }/{len (papers )} resume-knowledge {filename }""",
                    flush=True,
                )
                indexing_report, enrichment_path = (
                    await application.knowledge_pipeline.ingest_bundle(bundle)
                )
                projection = application.canonical_repository.get_chunk_projection(
                    bundle.snapshot.id
                )
                parse_run = application.parser_artifacts.get_parse_run(
                    bundle.snapshot.parse_run_id
                )
                payload = {
                    "resumed_snapshot_id": bundle.snapshot.id,
                    "parse_run": parse_run.model_dump(
                        mode="json", exclude={"artifact_manifest_path"}
                    ),
                    "counts": {"chunks": len(projection.chunks)},
                    "knowledge": indexing_report.public_dict(),
                }
            else:
                print(
                    f"""ingest {position }/{len (papers )} live {filename }""",
                    flush=True,
                )
                result = await application.services.ingestion.ingest_pdf_to_knowledge(
                    pipeline__CORPUS_ROOT / "papers" / str(paper["pool_file"]),
                    dataset=args.dataset,
                )
                payload = result.public_dict()
            parse_run_payload = cast(dict[str, Any], payload["parse_run"])
            counts_payload = cast(dict[str, Any], payload["counts"])
            knowledge = cast(dict[str, Any], payload["knowledge"])
            pipeline___require(
                parse_run_payload["provider"] == "mineru_cloud",
                "MinerU provider mismatch.",
            )
            pipeline___require(
                counts_payload["chunks"] > 0, f"""No chunks produced for {filename }"""
            )
            pipeline___require(
                knowledge["consistency_valid"],
                f"""Index inconsistency for {filename }""",
            )
            snapshot_id = (
                bundle.snapshot.id
                if args.resume and filename in existing
                else str(cast(dict[str, Any], payload["canonical_snapshot"])["id"])
            )
            pipeline___validate_enrichment(
                application.paths.cognee / "enrichment" / f"""{snapshot_id }.json""",
                filename,
            )
            projection = application.canonical_repository.get_chunk_projection(
                snapshot_id
            )
            current_bundle = application.canonical_repository.get_bundle(
                projection.snapshot_id
            )
            structural_results.append(
                pipeline___validate_real_ingestion(
                    expected=expected_cases[filename],
                    bundle=current_bundle,
                    projection=projection,
                    chunk_target_tokens=settings.ingestion.chunk_target_tokens,
                    chunk_hard_max_tokens=settings.ingestion.chunk_hard_max_tokens,
                    cognee_manifest_path=application.paths.cognee
                    / "manifests"
                    / f"""{snapshot_id }.json""",
                    index_manifest_path=application.paths.indexes
                    / "manifests"
                    / f"""{snapshot_id }.json""",
                )
            )
            output_path = logs / f"""ingest-{paper ['case_id']}.json"""
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            ingestions.append(payload)
        bundles = application.canonical_repository.list_bundles()
        active_filenames = {
            application.registry.get_source(
                bundle.document.source_file_id
            ).original_filename
            for bundle in bundles
        }
        pipeline___require(
            {str(paper["pdf_file"]) for paper in papers} <= active_filenames,
            "The cumulative run does not contain every genuine paper.",
        )
        searchable_element_types = {
            element_type
            for result in structural_results
            for element_type in result["searchable_element_types"]
        }
        pipeline___require(
            "table" in searchable_element_types, "No genuine table reached FTS5."
        )
        pipeline___require(
            "formula" in searchable_element_types, "No genuine formula reached FTS5."
        )
        protected = pipeline___file_hashes(
            [
                application.paths.raw,
                application.paths.parsed,
                application.paths.canonical,
            ]
        )
        for position, case in enumerate(queries, 1):
            print(
                f"""query {position }/{len (queries )} {case ['profile']} {case ['case_id']}""",
                flush=True,
            )
            response = await application.services.retrieval.query(
                QueryRequest(query=case["query"], profile=case["profile"])
            )
            quality = pipeline___validate_query(case, response)
            quality_results.append(quality)
            quality_warnings.extend(quality["quality_warnings"])
            query_report = {**quality, "response": response.model_dump(mode="json")}
            (logs / f"""query-{case ['case_id']}.json""").write_text(
                json.dumps(query_report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            responses.append(response)
        pipeline___require(
            pipeline___file_hashes(
                [
                    application.paths.raw,
                    application.paths.parsed,
                    application.paths.canonical,
                ]
            )
            == protected,
            "Retrieval mutated immutable source/canonical evidence.",
        )
        health = await application.services.health.report()
        pipeline___require(
            health["status"] == "healthy", f"""Final health is not healthy: {health }"""
        )
        executed_profiles = {response.profile.value for response in responses}
        pipeline___require(
            executed_profiles == {"truth", "associative", "comprehensive"},
            "Not every retrieval profile completed a real query.",
        )
        retrieval_contract_path = (
            run_root / "logs" / "contracts" / "cognee-retrieval-boundary.json"
        )
        retrieval_contract = await contract__run_live_retrieval_contract(
            application, dataset=args.dataset, output_path=retrieval_contract_path
        )
        pipeline___require(
            not retrieval_contract["hard_failures"],
            "Public and compatibility retrieval both failed for: "
            + ", ".join(retrieval_contract["hard_failures"]),
        )
        runtime_config = application.knowledge_pipeline.compat.runtime_config_snapshot()
        cognee_version = str(retrieval_contract["cognee_version"])
        profile_counts = Counter((response.profile.value for response in responses))

        def average_metric(name: str) -> float | None:
            values = [
                result[name]
                for result in quality_results
                if isinstance(result.get(name), (int, float))
            ]
            return (
                sum((float(value) for value in values)) / len(values)
                if values
                else None
            )

        quality_status = "reasonable" if not quality_warnings else "weak"
        quality_metrics = {
            "warning_count": len(quality_warnings),
            "average_expected_document_hit_rate": average_metric(
                "expected_document_hit_rate"
            ),
            "average_expected_concept_hit_rate": average_metric(
                "expected_concept_hit_rate"
            ),
            "average_expected_evidence_group_hit_rate": average_metric(
                "expected_evidence_group_hit_rate"
            ),
            "queries": quality_results,
        }
        health_summary = {
            "status": health["status"],
            "components": {
                name: component.get("status", "unknown")
                for name, component in health["components"].items()
            },
        }
        acceptance_report: dict[str, Any] = {
            "status": "passed",
            "pipeline_status": "passed",
            "quality_status": quality_status,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "run_root": ".",
            "dataset": args.dataset,
            "paper_count": len(papers),
            "new_ingestion_count": len(ingestions),
            "structural_results": structural_results,
            "query_count": len(responses),
            "profiles": sorted(executed_profiles),
            "truth_case_count": profile_counts["truth"],
            "associative_case_count": profile_counts["associative"],
            "comprehensive_case_count": profile_counts["comprehensive"],
            "llm_provider": runtime_config["llm_provider"],
            "llm_model": runtime_config["llm_model"],
            "embedding_provider": runtime_config["embedding_provider"],
            "embedding_model": runtime_config["embedding_model"],
            "cognee_version": cognee_version,
            "retrieval_fallback_types_used": sorted(
                application.knowledge_pipeline.compat.retrieval_fallback_types_used
            ),
            "retrieval_contract_status": retrieval_contract["status"],
            "retrieval_contract_path": retrieval_contract_path.relative_to(
                run_root
            ).as_posix(),
            "quality_warnings": quality_warnings,
            "quality_metrics": quality_metrics,
            "health": health_summary,
        }
        (logs / "acceptance-report.json").write_text(
            json.dumps(acceptance_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return acceptance_report
    finally:
        await application.aclose()
        if local_process is not None:
            await local_process.wait()
            pipeline___require(
                local_process.returncode is not None,
                f"""Local inference child process {local_pid } survived shutdown.""",
            )
            print(f"""local inference process {local_pid } cleaned""", flush=True)


def pipeline__main() -> None:
    configured = load_settings()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Run cumulative acceptance against the genuine four-paper corpus."
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("data/validation/retrieval/output")
    )
    parser.add_argument("--dataset", default=f"""paperos-real-{timestamp .lower ()}""")
    parser.add_argument(
        "--local-inference-port",
        type=int,
        help="Override the machine-local inference port for this acceptance run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already ingested genuine papers in the selected run root.",
    )
    args = parser.parse_args()
    try:
        report = asyncio.run(pipeline___run(args))
    except Exception as exc:
        failure_report = {
            "status": "failed",
            "pipeline_status": "failed",
            "quality_status": "unevaluated",
            "completed_at": datetime.now(UTC).isoformat(),
            "run_root": ".",
            "dataset": args.dataset,
            "quality_warnings": [],
            "quality_metrics": {},
            "failure_type": type(exc).__name__,
        }
        report_path = (
            args.run_root.resolve() / "logs" / "acceptance" / "acceptance-report.json"
        )
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(failure_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as report_error:
            print(
                f"""Unable to persist failed acceptance report: {report_error }""",
                file=sys.stderr,
            )
        if isinstance(exc, PaperOSError):
            print(
                json.dumps(exc.as_dict(), ensure_ascii=False, indent=2), file=sys.stderr
            )
        else:
            print(
                json.dumps(failure_report, ensure_ascii=False, indent=2),
                file=sys.stderr,
            )
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


pipeline__CORPUS_ROOT = Path("data/validation/corpus")
pipeline__CONFIG_ROOT = Path("data/validation/retrieval/config")
pipeline__RUN_ROOT = Path("data/validation/retrieval/output")


def _retained_filenames_by_sha() -> dict[str, str]:
    retained: dict[str, str] = {}
    registry = pipeline__RUN_ROOT / "jobs" / "registry.sqlite3"
    if not registry.is_file():
        return retained
    with sqlite3.connect(registry) as connection:
        sources = connection.execute(
            "SELECT id, original_filename FROM source_files"
        ).fetchall()
    for source_id, original_filename in sources:
        source_pdf = pipeline__RUN_ROOT / "raw" / str(source_id) / "source.pdf"
        if source_pdf.is_file():
            retained[pipeline___sha256(source_pdf)] = str(original_filename)
    return retained


def _retrieval_inputs() -> (
    tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]
):
    pool = json.loads(
        (pipeline__CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8")
    )
    selected = json.loads(
        (pipeline__CONFIG_ROOT / "papers.json").read_text(encoding="utf-8")
    )["papers"]
    spec = json.loads(
        (pipeline__CONFIG_ROOT / "corpus_spec.json").read_text(encoding="utf-8")
    )
    by_id = {str(item["paper_id"]): dict(item) for item in spec["papers"]}
    expected: dict[str, dict[str, Any]] = {}
    old_to_new: dict[str, str] = {}
    sha_to_id = {str(value["sha256"]): key for key, value in pool.items()}
    retained_by_sha = _retained_filenames_by_sha()
    for path in sorted((pipeline__CONFIG_ROOT / "expected").glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        digest = str(case["source_integrity"]["sha256"])
        paper_id = sha_to_id[digest]
        pool_file = Path(str(pool[paper_id]["file"])).name
        filename = retained_by_sha.get(digest, pool_file)
        old_to_new[str(case["pdf_file"])] = filename
        old_to_new[pool_file] = filename
        case["paper_id"] = paper_id
        case["pdf_file"] = filename
        expected[filename] = case
    papers = []
    for paper_id in selected:
        item = by_id[paper_id]
        entry = pool[paper_id]
        path = pipeline__CORPUS_ROOT / str(entry["file"])
        if not path.is_file() or pipeline___sha256(path) != entry["sha256"]:
            raise RuntimeError(
                f"""Retrieval corpus integrity failure: {paper_id } / {path }"""
            )
        item["pool_file"] = Path(str(entry["file"])).name
        item["pdf_file"] = retained_by_sha.get(entry["sha256"], item["pool_file"])
        item["sha256"] = entry["sha256"]
        papers.append(item)
    queries = []
    for name in ("truth.jsonl", "associative.jsonl", "comprehensive.jsonl"):
        for case in pipeline___load_jsonl(pipeline__CONFIG_ROOT / "queries" / name):
            case["expected_documents"] = [
                old_to_new.get(str(value), str(value))
                for value in case.get("expected_documents", [])
            ]
            queries.append(case)
    return papers, queries, expected


def pipeline___load_corpus(
    data_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    papers, queries, _ = _retrieval_inputs()
    return papers, queries


def pipeline___load_expected_cases(data_dir: Path) -> dict[str, dict[str, Any]]:
    _, _, expected = _retrieval_inputs()
    return expected


def quality___load_cases(data_dir: Path) -> list[dict[str, Any]]:
    _, queries, _ = _retrieval_inputs()
    return queries


def _dispatch_main() -> int:
    commands = {
        "run": pipeline__main,
        "pipeline": pipeline__main,
        "quality": quality__main,
    }
    command = "run"
    if len(sys.argv) > 1 and sys.argv[1] in commands:
        command = sys.argv.pop(1)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", type=Path, default=Path("data/validation/corpus"))
    common.add_argument(
        "--config", type=Path, default=Path("data/validation/retrieval/config")
    )
    common.add_argument(
        "--output", type=Path, default=Path("data/validation/retrieval/output")
    )
    common_args, remaining = common.parse_known_args(sys.argv[1:])
    sys.argv[1:] = remaining
    global pipeline__CORPUS_ROOT, pipeline__CONFIG_ROOT, pipeline__RUN_ROOT
    pipeline__CORPUS_ROOT = common_args.corpus.resolve()
    pipeline__CONFIG_ROOT = common_args.config.resolve()
    pipeline__RUN_ROOT = common_args.output.resolve()
    if command in {"run", "pipeline"}:
        sys.argv.extend(["--run-root", str(common_args.output)])
    elif command == "quality":
        sys.argv.extend(
            ["--output", str(common_args.output / "retrieval-quality-benchmark.json")]
        )
    result = commands[command]()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(_dispatch_main())
