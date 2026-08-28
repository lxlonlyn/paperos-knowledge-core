"""PaperOS-owned, read-only graph visualization payload."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Query

from paperos_core.api.dependencies import ApplicationDep

router = APIRouter(tags=["visualize"])


@router.get("/api/v1/visualize")
async def visualize_dataset(
    application: ApplicationDep,
    dataset: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return only graph projections for PaperOS's selected dataset."""
    dataset_name = (dataset or application.settings.dataset).strip()
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    active_snapshot_ids: list[str] = []
    for bundle in application.canonical_repository.list_active_bundles():
        if bundle.snapshot.dataset_id != dataset_name:
            continue
        path = application.paths.cognee / "graphs" / f"{bundle.snapshot.id}.json"
        if not path.is_file():
            continue
        try:
            graph = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        active_snapshot_ids.append(bundle.snapshot.id)
        for raw in graph.get("nodes", []):
            if not isinstance(raw, dict):
                continue
            canonical_id = str(raw.get("canonical_id") or "")
            if not canonical_id:
                continue
            nodes[canonical_id] = {
                "id": canonical_id,
                "type": str(raw.get("__type__") or "unknown"),
                "label": str(raw.get("title") or raw.get("name") or raw.get("text") or canonical_id),
                "source_chunk_ids": raw.get("source_chunk_ids") or [],
            }
        for raw in graph.get("relations", []):
            if not isinstance(raw, dict):
                continue
            edges.append({
                "source": raw.get("source_id"),
                "target": raw.get("target_id"),
                "type": raw.get("relation_type"),
                "source_chunk_ids": raw.get("source_chunk_ids") or [],
            })
    return {
        "dataset": dataset_name,
        "active_snapshot_ids": active_snapshot_ids,
        "nodes": list(nodes.values()),
        "edges": edges,
    }
