"""Deterministic JSON/SVG renderer for real retrieval graph validation."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

from paperos_core.retrieval.candidates import QueryResponse

_NODE_TYPES = {
    "ChunkDataPoint",
    "EntityDataPoint",
    "ClaimDataPoint",
    "SummaryDataPoint",
    "DocumentDataPoint",
    "ConceptRelationDataPoint",
    "TripletDataPoint",
}
_COLORS = {
    "ChunkDataPoint": "#dbeafe",
    "EntityDataPoint": "#dcfce7",
    "ClaimDataPoint": "#fef3c7",
    "SummaryDataPoint": "#f3e8ff",
    "DocumentDataPoint": "#fee2e2",
    "ConceptRelationDataPoint": "#e0e7ff",
    "TripletDataPoint": "#cffafe",
}


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _short_label(node: dict[str, Any]) -> str:
    for field in ("name", "title", "text", "description"):
        value = node.get(field)
        if isinstance(value, str) and value.strip():
            normalized = " ".join(value.split())
            return normalized if len(normalized) <= 46 else normalized[:43] + "..."
    return str(node.get("canonical_id") or node.get("id") or "unavailable")


def _load_graphs(graph_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for path in sorted(graph_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_nodes = payload.get("nodes", [])
        if isinstance(raw_nodes, list):
            for node in raw_nodes:
                if not isinstance(node, dict):
                    continue
                canonical_id = node.get("canonical_id")
                if canonical_id:
                    nodes[str(canonical_id)] = node
        raw_edges = payload.get("relations", [])
        if isinstance(raw_edges, list):
            edges.extend(edge for edge in raw_edges if isinstance(edge, dict))
    return nodes, edges


def _node_payload(
    canonical_id: str,
    node: dict[str, Any],
    scores: dict[str, float],
) -> dict[str, Any]:
    node_type = str(node.get("__type__") or node.get("type") or "unavailable")
    return {
        "id": canonical_id,
        "cognee_id": str(node.get("id") or "unavailable"),
        "node_type": node_type,
        "label": _short_label(node),
        "canonical_id": canonical_id,
        "source_chunk_ids": _string_list(node.get("source_chunk_ids")),
        "retrieval_score": scores.get(canonical_id),
    }


def _edge_payload(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": str(edge.get("source_id") or "unavailable"),
        "target": str(edge.get("target_id") or "unavailable"),
        "relation": str(edge.get("relation_type") or "unavailable"),
        "provenance": {
            "source_chunk_ids": _string_list(edge.get("source_chunk_ids")),
            "derived_from_ids": _string_list(edge.get("derived_from_ids")),
        },
    }


def _render_svg(snapshot: dict[str, Any], path: Path) -> None:
    nodes = snapshot["nodes"]
    edges = snapshot["edges"]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        grouped[str(node["node_type"])].append(node)
    ordered_types = sorted(grouped)
    column_width = 245
    margin_x = 40
    margin_y = 145
    row_height = 94
    width = max(1000, margin_x * 2 + column_width * max(len(ordered_types), 1))
    height = max(
        720,
        margin_y + row_height * max((len(items) for items in grouped.values()), default=1)
        + 130,
    )
    positions: dict[str, tuple[float, float]] = {}
    blocks: list[str] = []
    for column, node_type in enumerate(ordered_types):
        x = margin_x + column * column_width
        blocks.append(
            f'<text x="{x}" y="112" class="column">{escape(node_type)}</text>'
        )
        for row, node in enumerate(sorted(grouped[node_type], key=lambda item: item["id"])):
            y = margin_y + row * row_height
            positions[str(node["id"])] = (x + 90, y + 30)
            color = _COLORS.get(node_type, "#f1f5f9")
            score = node.get("retrieval_score")
            score_text = "unavailable" if score is None else f"{float(score):.3f}"
            blocks.extend(
                [
                    (
                        f'<rect x="{x}" y="{y}" width="180" height="62" rx="9" '
                        f'fill="{color}" stroke="#334155"/>'
                    ),
                    (
                        f'<text x="{x + 8}" y="{y + 20}" class="node-label">'
                        f'{escape(str(node["label"]))}</text>'
                    ),
                    (
                        f'<text x="{x + 8}" y="{y + 39}" class="node-meta">'
                        f'score: {escape(score_text)}</text>'
                    ),
                    (
                        f'<text x="{x + 8}" y="{y + 54}" class="node-meta">'
                        f'{escape(str(node["id"])[:25])}</text>'
                    ),
                ]
            )
    edge_blocks: list[str] = []
    for edge in edges:
        source = positions.get(str(edge["source"]))
        target = positions.get(str(edge["target"]))
        if source is None or target is None:
            continue
        x1, y1 = source
        x2, y2 = target
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        relation = escape(str(edge["relation"]))
        edge_blocks.extend(
            [
                (
                    f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                    'stroke="#94a3b8" stroke-width="1.2" marker-end="url(#arrow)"/>'
                ),
                (
                    f'<text x="{mid_x}" y="{mid_y - 3}" class="edge-label">'
                    f'{relation}</text>'
                ),
            ]
        )
    metadata = snapshot["metadata"]
    truncation = (
        f'truncated={str(snapshot["truncated"]).lower()}, '
        f'nodes={snapshot["rendered_node_count"]}/{snapshot["original_node_count"]}, '
        f'edges={snapshot["rendered_edge_count"]}/{snapshot["original_edge_count"]}'
    )
    legend = " | ".join(ordered_types)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
 viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3"
   orient="auto"><path d="M0,0 L8,3 L0,6 z" fill="#64748b"/></marker>
  <style>
    text {{ font-family: system-ui, sans-serif; fill: #0f172a; }}
    .title {{ font-size: 22px; font-weight: 700; }}
    .subtitle {{ font-size: 13px; fill: #475569; }}
    .column {{ font-size: 14px; font-weight: 700; }}
    .node-label {{ font-size: 11px; font-weight: 600; }}
    .node-meta {{ font-size: 9px; fill: #475569; }}
    .edge-label {{ font-size: 8px; fill: #475569; paint-order: stroke;
      stroke: white; stroke-width: 3px; }}
  </style>
</defs>
<rect width="100%" height="100%" fill="#ffffff"/>
<text x="40" y="34" class="title">{escape(str(snapshot["case_id"]))}
 - {escape(str(snapshot["profile"]))}</text>
<text x="40" y="58" class="subtitle">{escape(str(snapshot["query"])[:180])}</text>
<text x="40" y="79" class="subtitle">dataset={escape(str(metadata["dataset"]))};
 cognee={escape(str(metadata["cognee_version"]))}; {escape(truncation)}</text>
<text x="40" y="{height - 40}" class="subtitle">Legend: {escape(legend)}</text>
{"".join(edge_blocks)}
{"".join(blocks)}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def generate_retrieval_graph(
    *,
    case_id: str,
    profile: str,
    response: QueryResponse,
    graph_root: Path,
    output_root: Path,
    dataset: str,
    cognee_version: str,
    max_nodes: int = 80,
    max_edges: int = 160,
) -> dict[str, Any]:
    """Write one graph using only real response provenance and stored graph edges."""

    all_nodes, all_edges = _load_graphs(graph_root)
    if not all_nodes or not all_edges:
        raise RuntimeError("Real Cognee graph snapshots are missing nodes or relations.")
    priority: dict[str, int] = {}
    scores: dict[str, float] = {}
    for candidate in response.candidates:
        priority[candidate.object_id] = min(priority.get(candidate.object_id, 9), 0)
        priority[candidate.chunk_id] = min(priority.get(candidate.chunk_id, 9), 1)
        score = candidate.rerank_score or candidate.fused_score
        scores[candidate.object_id] = max(scores.get(candidate.object_id, 0.0), score)
        scores[candidate.chunk_id] = max(scores.get(candidate.chunk_id, 0.0), score)
        for derived_id in candidate.derived_from_ids:
            priority[derived_id] = min(priority.get(derived_id, 9), 2)
    for evidence in response.evidence:
        priority[evidence.chunk_id] = min(priority.get(evidence.chunk_id, 9), 1)
        for derived_id in evidence.derived_from_ids:
            priority[derived_id] = min(priority.get(derived_id, 9), 2)
    direct_ids = {node_id for node_id in priority if node_id in all_nodes}
    relevant_edges = [
        edge
        for edge in all_edges
        if str(edge.get("source_id")) in direct_ids
        or str(edge.get("target_id")) in direct_ids
    ]
    relevant_ids = set(direct_ids)
    for edge in relevant_edges:
        relevant_ids.add(str(edge.get("source_id")))
        relevant_ids.add(str(edge.get("target_id")))
    relevant_ids = {
        node_id
        for node_id in relevant_ids
        if node_id in all_nodes
        and str(
            all_nodes[node_id].get("__type__")
            or all_nodes[node_id].get("type")
            or ""
        )
        in _NODE_TYPES
    }
    for node_id in relevant_ids:
        priority.setdefault(node_id, 3)
    ordered_ids = sorted(
        relevant_ids,
        key=lambda node_id: (
            priority[node_id],
            str(
                all_nodes[node_id].get("__type__")
                or all_nodes[node_id].get("type")
                or ""
            ),
            node_id,
        ),
    )
    original_node_count = len(ordered_ids)
    selected_ids = set(ordered_ids[:max_nodes])
    selected_edges = [
        edge
        for edge in relevant_edges
        if str(edge.get("source_id")) in selected_ids
        and str(edge.get("target_id")) in selected_ids
    ]
    selected_edges.sort(
        key=lambda edge: (
            str(edge.get("source_id")),
            str(edge.get("target_id")),
            str(edge.get("relation_type")),
        )
    )
    original_edge_count = len(selected_edges)
    selected_edges = selected_edges[:max_edges]
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / f"{case_id}.graph.json"
    svg_path = output_root / f"{case_id}.graph.svg"
    snapshot: dict[str, Any] = {
        "case_id": case_id,
        "profile": profile,
        "query": response.query,
        "nodes": [
            _node_payload(node_id, all_nodes[node_id], scores)
            for node_id in ordered_ids[:max_nodes]
        ],
        "edges": [_edge_payload(edge) for edge in selected_edges],
        "truncated": original_node_count > max_nodes or original_edge_count > max_edges,
        "original_node_count": original_node_count,
        "rendered_node_count": min(original_node_count, max_nodes),
        "original_edge_count": original_edge_count,
        "rendered_edge_count": len(selected_edges),
        "metadata": {
            "dataset": dataset,
            "cognee_version": cognee_version,
            "retrieval_paths": [*response.channels_used, *response.stages],
            "generated_at": datetime.now(UTC).isoformat(),
        },
    }
    json_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _render_svg(snapshot, svg_path)
    return {
        "case_id": case_id,
        "profile": profile,
        "json": str(json_path),
        "svg": str(svg_path),
        "node_count": snapshot["rendered_node_count"],
        "edge_count": snapshot["rendered_edge_count"],
        "truncated": snapshot["truncated"],
    }
