"""Fair live benchmark of Cognee public graph retrieval and PaperOS retrieval.

Run directly; this project intentionally does not use pytest, mocks, or seeded
retrieval results. The benchmark is resumable and writes after every real case.
"""

from __future__ import annotations

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

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.application import Application, create_application
from paperos_core.config import load_settings
from paperos_core.retrieval.candidates import QueryResponse
from paperos_core.retrieval.corpus import CorpusView

UNAVAILABLE = "unavailable"
_QUERY_FILES = ("truth.jsonl", "associative.jsonl", "comprehensive.jsonl")


@dataclass(frozen=True, slots=True)
class BenchmarkConfiguration:
    id: str
    retrieval_method: Literal["cognee_public_search", "cognee_public_recall", "paperos"]
    search_type: str | None
    top_k: int | None
    wide_search_top_k: int | None
    neighborhood_depth: int | None
    neighborhood_seed_top_k: int | None
    triplet_distance_penalty: float | str
    notes: str


CONFIGURATIONS = (
    BenchmarkConfiguration(
        "A", "cognee_public_search", "GRAPH_COMPLETION", 12, 100, 1, 10,
        6.5, "Near the previous public baseline; top_k limits graph triplets.",
    ),
    BenchmarkConfiguration(
        "B", "cognee_public_search", "GRAPH_COMPLETION", 40, 100, 1, 40,
        6.5, "Candidate-budget comparison at one hop.",
    ),
    BenchmarkConfiguration(
        "C", "cognee_public_search", "GRAPH_COMPLETION", 40, 100, 2, 40,
        6.5, "Primary fair comparison with PaperOS pool=40 and graph depth=2.",
    ),
    BenchmarkConfiguration(
        "C80", "cognee_public_search", "GRAPH_COMPLETION", 40, 100, 2, 80,
        6.5, "Limited seed-budget sensitivity test.",
    ),
    BenchmarkConfiguration(
        "D", "cognee_public_search", "GRAPH_COMPLETION", 40, 200, 2, 40,
        6.5, "Wide-recall sensitivity test; directly comparable with C.",
    ),
    BenchmarkConfiguration(
        "E", "cognee_public_search", "GRAPH_COMPLETION", 40, 100, 3, 40,
        6.5, "Deep-neighborhood sensitivity test.",
    ),
    BenchmarkConfiguration(
        "F", "cognee_public_search", "GRAPH_COMPLETION_CONTEXT_EXTENSION", 40,
        100, 2, 40, 6.5, "Associative/exploratory public comparison.",
    ),
    BenchmarkConfiguration(
        "R", "cognee_public_recall", "GRAPH_COMPLETION", 40, 100, 2, 40,
        6.5, "Recall formatting/provenance comparison using C's retrieval budget.",
    ),
    BenchmarkConfiguration(
        "G", "paperos", None, 12, None, 2, None, UNAVAILABLE,
        "Production baseline: pool=40, final evidence=12, all configured channels.",
    ),
)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_cases(data_dir: Path) -> list[dict[str, Any]]:
    query_root = data_dir / "validation" / "corpus" / "queries"
    cases = [case for name in _QUERY_FILES for case in _load_jsonl(query_root / name)]
    _require(cases, f"No genuine benchmark queries found under {query_root}")
    _require(len({str(case["case_id"]) for case in cases}) == len(cases), "Duplicate case IDs")
    return cases


def _configuration_cases(
    configuration: BenchmarkConfiguration,
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select real cases relevant to this retrieval surface."""

    if configuration.id == "F":
        return [case for case in cases if case["profile"] != "truth"]
    return cases


def _graph_index(graph_root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
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
    _require(canonical, f"No genuine graph snapshots found under {graph_root}")
    return canonical, cognee_to_canonical


def _contains_concept(searchable: str, concept: str) -> bool:
    normalized = concept.casefold()
    aliases = {"weak coupling": ("weak coupling", "弱耦合")}
    if any(alias in searchable for alias in aliases.get(normalized, (normalized,))):
        return True
    tokens = re.findall(r"[a-z0-9]+", normalized)
    long_tokens = [token for token in tokens if len(token) >= 4]
    return bool(long_tokens) and all(token[:5] in searchable for token in long_tokens)


def _string_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return []


def _node_attributes(node: Any) -> dict[str, Any]:
    attributes = getattr(node, "attributes", None)
    if isinstance(attributes, dict):
        return attributes
    if isinstance(node, dict):
        nested = node.get("attributes")
        return nested if isinstance(nested, dict) else node
    return {}


def _node_id(node: Any, attributes: dict[str, Any]) -> str:
    value = getattr(node, "id", None) or attributes.get("id")
    return str(value) if value else ""


def _node_text(attributes: dict[str, Any]) -> str:
    return " ".join(
        str(attributes[field])
        for field in ("name", "title", "text", "description")
        if isinstance(attributes.get(field), str) and attributes[field].strip()
    )


def _edge_payload(
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
        attributes = _node_attributes(endpoint)
        cognee_id = _node_id(endpoint, attributes)
        public_canonical_id = attributes.get("canonical_id")
        public_source_chunk_ids = _string_list(attributes.get("source_chunk_ids"))
        canonical_id = str(
            public_canonical_id
            or cognee_to_canonical.get(cognee_id)
            or ""
        )
        stored = canonical_nodes.get(canonical_id, {})
        source_chunk_ids = _string_list(
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
            "text": _node_text(attributes),
            "source_chunk_ids": source_chunk_ids,
            "public_canonical_id_preserved": bool(public_canonical_id),
            "public_source_chunk_ids_preserved": bool(public_source_chunk_ids),
            "evaluation_readback_used": bool(
                (canonical_id and not public_canonical_id)
                or (source_chunk_ids and not public_source_chunk_ids)
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
    edge_source_chunks = _string_list(edge_attributes.get("source_chunk_ids"))
    return (
        {
            "source": canonical_ids[0] or node_payloads[0]["node_id"],
            "target": canonical_ids[1] or node_payloads[1]["node_id"],
            "relation": relation,
            "source_chunk_ids": edge_source_chunks,
        },
        node_payloads,
    )


def _public_objects(
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
            edge_payload, endpoint_payloads = _edge_payload(
                edge,
                canonical_nodes=canonical_nodes,
                cognee_to_canonical=cognee_to_canonical,
            )
            edges.append(edge_payload)
            for node in endpoint_payloads:
                key = str(node["canonical_id"] or node["node_id"] or len(nodes))
                existing = nodes.get(key)
                if existing is None or (not existing["source_chunk_ids"] and node["source_chunk_ids"]):
                    nodes[key] = node
    return list(nodes.values()), edges, "\n".join(contexts), len(raw_items)


def _recall_surface(entries: object) -> tuple[str, int, int, list[dict[str, Any]]]:
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
            key in raw for key in ("id", "node_id", "canonical_id", "source_chunk_ids")
        ):
            structured.append(raw)
    return "\n".join(texts), len(items), len(structured), structured


def _chunk_ids_from_nodes_edges(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> set[str]:
    chunks = {
        chunk_id
        for node in nodes
        for chunk_id in _string_list(node.get("source_chunk_ids"))
    }
    chunks.update(
        str(node["canonical_id"])
        for node in nodes
        if node.get("node_type") == "ChunkDataPoint" and node.get("canonical_id")
    )
    chunks.update(
        chunk_id
        for edge in edges
        for chunk_id in _string_list(edge.get("source_chunk_ids"))
    )
    return chunks


def _quality_metrics(
    case: dict[str, Any],
    *,
    context: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    chunk_ids: set[str],
    corpus: CorpusView,
) -> dict[str, Any]:
    resolvable_chunks = {chunk_id for chunk_id in chunk_ids if chunk_id in corpus.chunks}
    filenames = {
        corpus.source_filenames[corpus.chunk_bundles[chunk_id].document.source_file_id]
        for chunk_id in resolvable_chunks
    }
    for node in nodes:
        canonical_id = node.get("canonical_id")
        if node.get("node_type") == "DocumentDataPoint":
            for bundle in corpus.bundles.values():
                if bundle.document.id == canonical_id:
                    filenames.add(corpus.source_filenames[bundle.document.source_file_id])
    node_text = " ".join(str(node.get("text") or "") for node in nodes)
    relation_texts = [
        " ".join(
            [
                str(edge.get("relation") or ""),
                *(str(node.get("text") or "") for node in nodes if node.get("canonical_id") in {edge.get("source"), edge.get("target")}),
            ]
        )
        for edge in edges
    ]
    resolved_chunk_text = " ".join(corpus.chunks[item].text for item in resolvable_chunks)
    searchable = " ".join([context, node_text, " ".join(relation_texts)]).casefold()
    expected_documents = {str(item) for item in case.get("expected_documents", [])}
    document_hits = expected_documents & filenames
    concepts = [str(item) for item in case.get("required_concepts", [])]
    concept_hits = [concept for concept in concepts if _contains_concept(searchable, concept)]
    node_concept_hits = [concept for concept in concepts if _contains_concept(node_text.casefold(), concept)]
    relation_concept_hits = [
        concept
        for concept in concepts
        if any(_contains_concept(text.casefold(), concept) for text in relation_texts)
    ]
    groups = list(case.get("required_evidence_groups", []))
    group_hits = [
        group
        for group in groups
        if any(str(term).casefold() in searchable for term in group.get("any_of", []))
    ]
    chunk_group_hits = [
        group
        for group in groups
        if any(str(term).casefold() in resolved_chunk_text.casefold() for term in group.get("any_of", []))
    ]
    page_count = sum(corpus.chunks[item].page_start is not None for item in resolvable_chunks)
    relevant_ranks: list[int] = []
    for rank, relation_text in enumerate(relation_texts, 1):
        edge_chunks = set(_string_list(edges[rank - 1].get("source_chunk_ids")))
        edge_documents = {
            corpus.source_filenames[corpus.chunk_bundles[item].document.source_file_id]
            for item in edge_chunks
            if item in corpus.chunks
        }
        if expected_documents & edge_documents or any(
            _contains_concept(relation_text.casefold(), concept) for concept in concepts
        ):
            relevant_ranks.append(rank)
    return {
        "expected_document_hit": bool(document_hits),
        "expected_document_recall": (
            len(document_hits) / len(expected_documents) if expected_documents else None
        ),
        "expected_concept_hit": bool(concept_hits) if concepts else None,
        "expected_concept_recall": len(concept_hits) / len(concepts) if concepts else None,
        "relevant_node_hit": bool(node_concept_hits) if concepts else None,
        "relevant_node_recall": len(node_concept_hits) / len(concepts) if concepts else None,
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


async def _run_public(
    configuration: BenchmarkConfiguration,
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
            context, raw_result_count, structured_object_count, raw_objects = _recall_surface(entries)
            nodes = [
                {
                    "node_id": item.get("node_id") or item.get("id"),
                    "canonical_id": item.get("canonical_id"),
                    "node_type": item.get("type") or "unavailable",
                    "text": str(item.get("text") or item.get("name") or ""),
                    "source_chunk_ids": _string_list(item.get("source_chunk_ids")),
                }
                for item in raw_objects
            ]
        else:
            results = await cognee.search(verbose=True, node_type=None, **kwargs)
            nodes, edges, context, raw_result_count = _public_objects(
                results,
                canonical_nodes=canonical_nodes,
                cognee_to_canonical=cognee_to_canonical,
            )
            structured_object_count = len(edges)
    except Exception as exc:  # noqa: BLE001 - a public limitation is benchmark data.
        error = f"{type(exc).__name__}: {exc}"
    latency_ms = (time.perf_counter() - started) * 1000
    chunk_ids = _chunk_ids_from_nodes_edges(nodes, edges)
    public_provenance_nodes = [
        node
        for node in nodes
        if node.get("public_canonical_id_preserved")
        and (
            node.get("public_source_chunk_ids_preserved")
            or node.get("node_type") == "ChunkDataPoint"
        )
    ]
    quality = _quality_metrics(
        case,
        context=context,
        nodes=nodes,
        edges=edges,
        chunk_ids=chunk_ids,
        corpus=corpus,
    )
    is_context_extension = configuration.search_type == "GRAPH_COMPLETION_CONTEXT_EXTENSION"
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
        "provenance_preserved": bool(nodes) and len(public_provenance_nodes) == len(nodes),
        "evaluation_readback_used": any(
            bool(node.get("evaluation_readback_used")) for node in nodes
        ),
        "public_canonical_id_preservation_rate": (
            sum(bool(node.get("public_canonical_id_preserved")) for node in nodes) / len(nodes)
            if nodes
            else 0.0
        ),
        "public_source_chunk_ids_preservation_rate": (
            sum(bool(node.get("public_source_chunk_ids_preserved")) for node in nodes) / len(nodes)
            if nodes
            else 0.0
        ),
        "structured_node_count": len(nodes),
        "edge_triplet_count": len(edges),
        "evidence_chunk_count": len(chunk_ids),
        "quality": quality,
        "runtime": {
            "latency_ms": round(latency_ms, 3),
            "llm_call_count": UNAVAILABLE if is_context_extension else 0,
            "embedding_call_count": UNAVAILABLE,
            "returned_context_size": len(context),
            "seed_count": UNAVAILABLE,
            "candidate_count": len(nodes),
            "final_triplet_count": len(edges),
            "final_evidence_count": len(chunk_ids),
        },
        "nodes": nodes,
        "relations": edges,
        "context": context,
        "error": error,
    }


def _run_paperos(
    configuration: BenchmarkConfiguration,
    case: dict[str, Any],
    *,
    run_root: Path,
    corpus: CorpusView,
    canonical_nodes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    path = run_root / "logs" / "acceptance" / f"query-{case['case_id']}.json"
    _require(path.is_file(), f"Production response is missing: {path}")
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
            "node_type": canonical_nodes[item].get("type") or canonical_nodes[item].get("__type__"),
            "text": _node_text(canonical_nodes[item]),
            "source_chunk_ids": _string_list(canonical_nodes[item].get("source_chunk_ids")),
        }
        for item in sorted(derived_ids)
        if item in canonical_nodes
    ]
    relation_count = sum(
        candidate.knowledge_kind == "structured_relation" or "graph" in candidate.channels
        for candidate in response.candidates
    )
    context = "\n".join(item.text for item in response.evidence)
    quality = _quality_metrics(
        case,
        context=context,
        nodes=nodes,
        edges=[],
        chunk_ids=chunk_ids,
        corpus=corpus,
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
            "latency_ms": UNAVAILABLE,
            "llm_call_count": UNAVAILABLE,
            "embedding_call_count": UNAVAILABLE,
            "returned_context_size": len(context),
            "seed_count": UNAVAILABLE,
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


def _mean(cases: list[dict[str, Any]], path: tuple[str, ...]) -> float | None:
    values: list[float] = []
    for case in cases:
        value: Any = case
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return fmean(values) if values else None


def _aggregate(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case["configuration_id"])].append(case)
    return {
        config_id: {
            "case_count": len(items),
            "error_count": sum(bool(item["error"]) for item in items),
            "context_return_rate": fmean(bool(item["context_returned"]) for item in items),
            "provenance_preservation_rate": fmean(bool(item["provenance_preserved"]) for item in items),
            "expected_document_recall": _mean(items, ("quality", "expected_document_recall")),
            "expected_concept_recall": _mean(items, ("quality", "expected_concept_recall")),
            "relevant_node_recall": _mean(items, ("quality", "relevant_node_recall")),
            "relevant_relation_recall": _mean(items, ("quality", "relevant_relation_recall")),
            "evidence_chunk_coverage": _mean(items, ("quality", "evidence_chunk_coverage")),
            "page_provenance_available": _mean(items, ("quality", "page_provenance_available")),
            "canonical_chunk_resolvable": _mean(items, ("quality", "canonical_chunk_resolvable")),
            "ranking_usefulness": _mean(items, ("quality", "ranking_usefulness")),
            "latency_ms": _mean(items, ("runtime", "latency_ms")),
            "average_context_size": _mean(items, ("runtime", "returned_context_size")),
            "average_triplet_count": _mean(items, ("runtime", "final_triplet_count")),
            "average_evidence_count": _mean(items, ("runtime", "final_evidence_count")),
            "by_profile": {
                profile: {
                    "case_count": len(profile_items),
                    "expected_document_recall": _mean(profile_items, ("quality", "expected_document_recall")),
                    "expected_concept_recall": _mean(profile_items, ("quality", "expected_concept_recall")),
                    "relevant_relation_recall": _mean(profile_items, ("quality", "relevant_relation_recall")),
                }
                for profile in ("truth", "associative", "comprehensive")
                if (profile_items := [item for item in items if item["profile"] == profile])
            },
        }
        for config_id, items in grouped.items()
    }


def _relative_change(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return right - left


def _conclusions(aggregate: dict[str, Any]) -> dict[str, Any]:
    def metric(config: str, name: str) -> float | None:
        value = aggregate.get(config, {}).get(name)
        return float(value) if isinstance(value, (int, float)) else None

    c_document = metric("C", "expected_document_recall")
    c_concept = metric("C", "expected_concept_recall")
    g_document = metric("G", "expected_document_recall")
    g_concept = metric("G", "expected_concept_recall")
    relevance_close = all(
        left is not None and right is not None and left >= right - 0.05
        for left, right in ((c_document, g_document), (c_concept, g_concept))
    )
    public_provenance = metric("C", "provenance_preservation_rate") or 0.0
    return {
        "fair_public_graph_relevance_close_to_paperos": relevance_close,
        "depth_1_to_2": {
            "document_recall_delta": _relative_change(metric("B", "expected_document_recall"), c_document),
            "concept_recall_delta": _relative_change(metric("B", "expected_concept_recall"), c_concept),
        },
        "depth_2_to_3": {
            "document_recall_delta": _relative_change(c_document, metric("E", "expected_document_recall")),
            "concept_recall_delta": _relative_change(c_concept, metric("E", "expected_concept_recall")),
        },
        "top_k_12_to_40_at_depth_1": {
            "document_recall_delta": _relative_change(metric("A", "expected_document_recall"), metric("B", "expected_document_recall")),
            "concept_recall_delta": _relative_change(metric("A", "expected_concept_recall"), metric("B", "expected_concept_recall")),
        },
        "wide_100_to_200_at_depth_2": {
            "document_recall_delta": _relative_change(c_document, metric("D", "expected_document_recall")),
            "concept_recall_delta": _relative_change(c_concept, metric("D", "expected_concept_recall")),
        },
        "seed_40_to_80_at_depth_2": {
            "document_recall_delta": _relative_change(c_document, metric("C80", "expected_document_recall")),
            "concept_recall_delta": _relative_change(c_concept, metric("C80", "expected_concept_recall")),
        },
        "context_extension": {
            "document_recall": metric("F", "expected_document_recall"),
            "concept_recall": metric("F", "expected_concept_recall"),
            "provenance_preservation_rate": metric("F", "provenance_preservation_rate"),
            "average_latency_ms": metric("F", "latency_ms"),
            "assessment": (
                "Improves concept recall over standard C for associative/comprehensive "
                "queries, but remains below G, is LLM-dependent and slow, and preserves "
                "no public canonical provenance."
            ),
        },
        "public_recall_interpretation": (
            "R returns formatted context. Its zero structured document/evidence metrics "
            "mean document identity is unobservable without provenance, not that relevant "
            "context was absent."
        ),
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
        "decision_rule": (
            "Public GRAPH_COMPLETION is close only when aggregate document and concept "
            "recall are each within 0.05 of PaperOS G. Provenance is evaluated separately."
        ),
    }


def _write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)


async def run_benchmark(
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
    all_cases = _load_cases(configured.data_dir)
    cases = [
        case for case in all_cases
        if profiles is None or str(case["profile"]) in profiles
    ]
    _require(cases, "No benchmark cases selected.")
    configurations = [
        item for item in CONFIGURATIONS
        if configuration_ids is None or item.id in configuration_ids
    ]
    _require(configurations, "No benchmark configurations selected.")
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
        for case in _configuration_cases(configuration, cases)
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
            "graph_completion": (
                "Cognee 1.4.0 GraphCompletionRetriever discovers index_fields from all "
                "loaded DataPoint subclasses; node_type is explicitly None for mixed graphs."
            ),
            "built_in_retrievers": (
                "CHUNKS/SUMMARIES/HYBRID retain Cognee-native schema assumptions and are "
                "capability observations, not tuned quality failures."
            ),
            "recall": (
                "only_context recall returns normalized formatted graph context; context, "
                "structured objects, and provenance are counted separately."
            ),
            "runtime_counts": (
                "Standard GraphCompletion with only_context performs zero LLM completion "
                "calls. Calls not exposed reliably by public APIs are unavailable, not guessed."
            ),
            "production_relation_metric": (
                "G retains graph-derived IDs but not the exact ranked typed edge list; its "
                "relation recall is unavailable and is not used for fallback deletion."
            ),
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
        canonical_nodes, cognee_to_canonical = _graph_index(
            application.paths.cognee / "graphs"
        )
        total = len(all_case_keys)
        completed = len(existing)
        for configuration in configurations:
            for case in _configuration_cases(configuration, cases):
                key = (configuration.id, str(case["case_id"]))
                if key in existing:
                    continue
                print(
                    f"benchmark {completed + 1}/{total} {configuration.id} "
                    f"{case['profile']} {case['case_id']}",
                    flush=True,
                )
                if configuration.retrieval_method == "paperos":
                    result = _run_paperos(
                        configuration,
                        case,
                        run_root=run_root,
                        corpus=corpus,
                        canonical_nodes=canonical_nodes,
                    )
                else:
                    result = await _run_public(
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
                report["aggregate_metrics"] = _aggregate(report["cases"])
                report["conclusions"] = _conclusions(report["aggregate_metrics"])
                report["completed_case_runs"] = completed
                report["total_case_runs"] = total
                _write_report(report, output_path)
        report["status"] = "completed"
        report["completed_at"] = datetime.now(UTC).isoformat()
        report["cases"] = sorted(
            existing.values(), key=lambda item: (item["configuration_id"], item["case_id"])
        )
        report["aggregate_metrics"] = _aggregate(report["cases"])
        report["conclusions"] = _conclusions(report["aggregate_metrics"])
        report["completed_case_runs"] = len(report["cases"])
        report["total_case_runs"] = len(all_case_keys)
        _write_report(report, output_path)
        return report
    finally:
        await application.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--profiles",
        help="Comma-separated profiles; F always excludes truth cases.",
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
    output = args.output or (
        run_root / "logs" / "contracts" / "cognee-retrieval-quality-benchmark.json"
    )
    selected = (
        {item.strip() for item in args.configurations.split(",") if item.strip()}
        if args.configurations
        else None
    )
    report = asyncio.run(
        run_benchmark(
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


if __name__ == "__main__":
    main()
