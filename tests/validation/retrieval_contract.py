"""Shared live Cognee retrieval contract for direct scripts and acceptance."""

from __future__ import annotations

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
class CapabilitySpec:
    name: str
    datapoint_type: str
    public_search_type: str
    compat_search_type: str


_CAPABILITIES = (
    CapabilitySpec("chunk", "ChunkDataPoint", "CHUNKS", "PAPEROS_CHUNKS"),
    CapabilitySpec(
        "entity",
        "EntityDataPoint",
        "GRAPH_COMPLETION",
        "PAPEROS_ENTITIES",
    ),
    CapabilitySpec(
        "claim",
        "ClaimDataPoint",
        "GRAPH_COMPLETION",
        "PAPEROS_CLAIMS",
    ),
    CapabilitySpec(
        "summary",
        "SummaryDataPoint",
        "GRAPH_COMPLETION",
        "PAPEROS_SUMMARIES",
    ),
)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _load_graph_nodes(graph_root: Path) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for path in sorted(graph_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_nodes = payload.get("nodes", [])
        if isinstance(raw_nodes, list):
            nodes.extend(item for item in raw_nodes if isinstance(item, dict))
    return nodes


def _representative(
    nodes: list[dict[str, Any]], datapoint_type: str
) -> dict[str, Any]:
    matching = [
        node
        for node in nodes
        if str(node.get("__type__") or node.get("type") or "") == datapoint_type
        and node.get("canonical_id")
    ]
    _require(matching, f"No real {datapoint_type} exists in retained graph snapshots.")
    return min(matching, key=lambda node: str(node["canonical_id"]))


def _query_from_node(node: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    for field in ("name", "title", "text", "description"):
        value = node.get(field)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())[:500]
    return str(node["canonical_id"])


def _manifest_context(
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
            canonical_ids.update(str(item) for item in mapping)
    _require(canonical_ids, "No canonical IDs are registered for the selected dataset.")
    _require(len(dataset_ids) <= 1, f"Dataset name maps to multiple IDs: {dataset_ids}")
    return canonical_ids, next(iter(dataset_ids), None)


def _surface_metrics(
    hits: list[CogneeSearchHit],
    *,
    datapoint_type: str,
    allowed_canonical_ids: set[str],
) -> dict[str, Any]:
    typed = [hit for hit in hits if hit.result_type == datapoint_type]
    canonical = bool(typed) and all(
        hit.canonical_id in allowed_canonical_ids for hit in typed
    )
    source_chunks = bool(typed) and all(
        hit.source_chunk_ids
        or (
            datapoint_type == "ChunkDataPoint"
            and hit.canonical_id.startswith("chunk_")
        )
        for hit in typed
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
        and all(score > 0 for score in scores)
        and scores == sorted(scores, reverse=True),
        "graph_provenance_recoverable": bool(typed)
        and all(bool(hit.references) for hit in typed),
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


async def _safe_surface(
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
                    query,
                    dataset=dataset,
                    top_k=top_k,
                    search_type=search_type,
                ),
                None,
            )
        return (
            await adapter.graph_search(
                query,
                dataset=dataset,
                top_k=top_k,
                search_type=search_type,
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - public limitation is contract data.
        return [], f"{type(exc).__name__}: {exc}"


async def _datapoint_case(
    adapter: CogneeSearchAdapter,
    *,
    spec: CapabilitySpec,
    node: dict[str, Any],
    query: str,
    dataset: str,
    dataset_id: str | None,
    allowed_canonical_ids: set[str],
    top_k: int,
    cognee_version: str,
) -> dict[str, Any]:
    searched, search_error = await _safe_surface(
        adapter,
        query=query,
        dataset=dataset,
        top_k=top_k,
        search_type=spec.public_search_type,
        recall=False,
    )
    recalled, recall_error = await _safe_surface(
        adapter,
        query=query,
        dataset=dataset,
        top_k=top_k,
        search_type=spec.public_search_type,
        recall=True,
    )
    compatible, compat_error = await _safe_surface(
        adapter,
        query=query,
        dataset=dataset,
        top_k=top_k,
        search_type=spec.compat_search_type,
        recall=False,
    )
    public_search = _surface_metrics(
        searched,
        datapoint_type=spec.datapoint_type,
        allowed_canonical_ids=allowed_canonical_ids,
    )
    public_recall = _surface_metrics(
        recalled,
        datapoint_type=spec.datapoint_type,
        allowed_canonical_ids=allowed_canonical_ids,
    )
    compat = _surface_metrics(
        compatible,
        datapoint_type=spec.datapoint_type,
        allowed_canonical_ids=allowed_canonical_ids,
    )
    public_typed_hits = [
        hit
        for hit in [*searched, *recalled]
        if hit.result_type == spec.datapoint_type
    ]
    resolved = await adapter.compat.resolve_graph_nodes(
        [hit.node_id for hit in public_typed_hits]
    )
    readback_hits = [
        CogneeSearchHit(
            node_id=hit.node_id,
            canonical_id=hit.canonical_id,
            source_chunk_ids=(
                hit.source_chunk_ids
                or tuple(
                    _string_list(
                        resolved.get(hit.node_id, {}).get("source_chunk_ids")
                    )
                )
            ),
            references=(
                hit.references
                or tuple(
                    _string_list(
                        resolved.get(hit.node_id, {}).get("derived_from_ids")
                    )
                )
            ),
            result_type=hit.result_type,
            text=hit.text,
            score=hit.score,
        )
        for hit in public_typed_hits
    ]
    readback = _surface_metrics(
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
        bool(public_search[field]) for field in required_fields
    )
    public_recall_supported = all(
        bool(public_recall[field]) for field in required_fields
    )
    compat_vector_supported = all(bool(compat[field]) for field in required_fields)
    compat_readback_supported = all(
        bool(readback[field]) for field in required_fields
    )
    compat_supported = compat_vector_supported or compat_readback_supported
    compat_best = compat if compat_vector_supported else readback
    if public_search_supported and public_recall_supported:
        status = "supported"
    elif compat_supported:
        status = "partially_supported"
    else:
        status = f"unsupported_by_cognee_{cognee_version.replace('.', '_')}"
    limitations = [
        message
        for message in (
            f"public search: {search_error}" if search_error else None,
            f"public recall: {recall_error}" if recall_error else None,
            f"compat: {compat_error}" if compat_error else None,
        )
        if message
    ]
    if not public_search_supported:
        limitations.append(
            "public search does not preserve the required custom type identity "
            "and canonical chunk provenance"
        )
    if not public_recall_supported:
        limitations.append(
            "public recall does not preserve the required custom type identity "
            "and canonical chunk provenance"
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
        "custom_node_type_preserved": bool(
            compat_best["custom_node_type_preserved"]
        ),
        "canonical_id_preserved": bool(compat_best["canonical_id_preserved"]),
        "source_chunk_ids_preserved": bool(
            compat_best["source_chunk_ids_preserved"]
        ),
        "dataset_scope_supported": bool(
            compat_best["dataset_scope_supported"]
        ),
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


async def _graph_case(
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
    searched, search_error = await _safe_surface(
        adapter,
        query=query,
        dataset=dataset,
        top_k=top_k,
        search_type="GRAPH_COMPLETION_CONTEXT_EXTENSION",
        recall=False,
    )
    recalled, recall_error = await _safe_surface(
        adapter,
        query=query,
        dataset=dataset,
        top_k=top_k,
        search_type="GRAPH_COMPLETION_CONTEXT_EXTENSION",
        recall=True,
    )
    seed_hits, seed_error = await _safe_surface(
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
                _string_list(resolved.get(hit.node_id, {}).get("source_chunk_ids"))
            ),
            derived_from_ids=tuple(
                _string_list(resolved.get(hit.node_id, {}).get("derived_from_ids"))
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
    except Exception as exc:  # noqa: BLE001 - limitation is contract data.
        traversed = []
        traversal_error = f"{type(exc).__name__}: {exc}"
    public_metrics = _surface_metrics(
        searched + recalled,
        datapoint_type="EntityDataPoint",
        allowed_canonical_ids=allowed_canonical_ids,
    )
    public_graph_provenance = False
    compat_supported = bool(traversed) and all(
        item.source_canonical_id
        and item.target_canonical_id
        and item.relation_type
        and item.source_chunk_ids
        for item in traversed
    )
    status = (
        "supported"
        if public_graph_provenance
        else "partially_supported"
        if compat_supported
        else f"unsupported_by_cognee_{cognee_version.replace('.', '_')}"
    )
    limitations = [
        message
        for message in (
            f"public search: {search_error}" if search_error else None,
            f"public recall: {recall_error}" if recall_error else None,
            f"compat seed search: {seed_error}" if seed_error else None,
            f"compat traversal: {traversal_error}" if traversal_error else None,
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


def write_contract_report(report: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


async def run_live_retrieval_contract(
    application: Application,
    *,
    dataset: str,
    output_path: Path | None = None,
    query_override: str | None = None,
    top_k: int = 12,
) -> dict[str, Any]:
    """Run all contract surfaces against the application's real Cognee dataset."""

    nodes = _load_graph_nodes(application.paths.cognee / "graphs")
    _require(nodes, "No real Cognee graph snapshots are available for live contract.")
    allowed, dataset_id = _manifest_context(application.settings.data_dir, dataset)
    adapter = CogneeSearchAdapter(
        application.paths,
        application.knowledge_pipeline.compat,
    )
    cognee_version = version("cognee")
    cases: list[dict[str, Any]] = []
    for spec in _CAPABILITIES:
        node = _representative(nodes, spec.datapoint_type)
        cases.append(
            await _datapoint_case(
                adapter,
                spec=spec,
                node=node,
                query=_query_from_node(node, query_override),
                dataset=dataset,
                dataset_id=dataset_id,
                allowed_canonical_ids=allowed,
                top_k=top_k,
                cognee_version=cognee_version,
            )
        )
    graph_node = _representative(nodes, "EntityDataPoint")
    cases.append(
        await _graph_case(
            adapter,
            application.knowledge_pipeline.compat,
            node=graph_node,
            query=_query_from_node(graph_node, query_override),
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
            not case["public_search_supported"]
            or not case["public_recall_supported"]
            or not case["graph_provenance_recoverable"]
            for case in cases
        ),
        "retrieval_fallback_types_used": sorted(
            application.knowledge_pipeline.compat.retrieval_fallback_types_used
        ),
        "cases": cases,
    }
    if output_path is not None:
        write_contract_report(report, output_path)
    return report
