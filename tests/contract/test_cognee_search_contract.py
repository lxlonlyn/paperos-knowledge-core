"""Contract tests for parsing Cognee's real search return structures."""

from __future__ import annotations

import json
from pathlib import Path

import cognee
import pytest
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge, Node
from cognee.modules.recall.types.SearchResultItem import SearchResultItem
from cognee.modules.search.types import SearchType

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.adapters.cognee.search import CogneeSearchAdapter
from paperos_core.errors import CogneeStorageError
from paperos_core.paths import build_data_paths


def _node(
    node_id: str,
    *,
    object_type: str,
    text: str,
    distance: float | None,
) -> Node:
    node = Node(node_id=node_id, attributes={"type": object_type, "text": text})
    if distance is not None:
        node.update_distance_for_query(0, distance, 1, 6.5)
    return node


def _write_manifest(root: Path) -> None:
    manifest = {
        "canonical_to_cognee_id": {
            "chunk_alpha": "11111111-1111-1111-1111-111111111111",
            "entity_alpha": "22222222-2222-2222-2222-222222222222",
        }
    }
    path = root / "cognee" / "manifests" / "snapshot_contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.asyncio
async def test_graph_search_parses_real_cognee_edge_structures(
    gate1_run_dir: Path, monkeypatch
) -> None:
    root = gate1_run_dir / "search-contract"
    _write_manifest(root)
    captured: dict[str, object] = {}

    async def fake_search(**kwargs):
        captured.update(kwargs)
        chunk = _node(
            "11111111-1111-1111-1111-111111111111",
            object_type="ChunkDataPoint",
            text="alpha chunk text",
            distance=0.5,
        )
        entity = _node(
            "22222222-2222-2222-2222-222222222222",
            object_type="EntityDataPoint",
            text="alpha entity",
            distance=None,
        )
        edge = Edge(chunk, entity, attributes={"edge_type_id": "edge_1"})
        return [{"text_result": None, "context_result": "ctx", "objects_result": [edge]}]

    monkeypatch.setattr(cognee, "search", fake_search)
    paths = build_data_paths(root)
    adapter = CogneeSearchAdapter(paths, CogneeCompatibilityAdapter(paths))
    hits = await adapter.graph_search(
        "alpha",
        dataset="papers",
        top_k=10,
        search_type="GRAPH_COMPLETION",
    )
    assert captured["query_type"] is cognee.SearchType.GRAPH_COMPLETION
    assert captured["datasets"] == ["papers"]
    by_id = {hit.canonical_id: hit for hit in hits}
    assert by_id["chunk_alpha"].object_type == "ChunkDataPoint"
    assert by_id["chunk_alpha"].text == "alpha chunk text"
    assert abs(by_id["chunk_alpha"].score - (1.0 / 1.5)) < 1e-9
    assert by_id["entity_alpha"].object_type == "EntityDataPoint"
    assert by_id["entity_alpha"].score == 0.0


@pytest.mark.asyncio
async def test_graph_search_supports_profile_specific_search_types(
    gate1_run_dir: Path, monkeypatch
) -> None:
    root = gate1_run_dir / "search-contract-types"
    _write_manifest(root)
    captured: dict[str, object] = {}

    async def fake_search(**kwargs):
        captured.update(kwargs)
        node = _node(
            "11111111-1111-1111-1111-111111111111",
            object_type="ChunkDataPoint",
            text="alpha chunk text",
            distance=1.0,
        )
        return [{"text_result": None, "context_result": "ctx", "objects_result": [Edge(node, node)]}]

    monkeypatch.setattr(cognee, "search", fake_search)
    paths = build_data_paths(root)
    adapter = CogneeSearchAdapter(paths, CogneeCompatibilityAdapter(paths))
    await adapter.graph_search(
        "alpha",
        dataset="papers",
        top_k=5,
        search_type="GRAPH_COMPLETION_DECOMPOSITION",
    )
    assert captured["query_type"] is cognee.SearchType.GRAPH_COMPLETION_DECOMPOSITION
    with pytest.raises(CogneeStorageError, match="Unsupported graph search type"):
        await adapter.graph_search(
            "alpha",
            dataset="papers",
            top_k=5,
            search_type="CHUNKS",
        )


@pytest.mark.asyncio
async def test_recall_context_reads_normalized_graph_entries(
    gate1_run_dir: Path, monkeypatch
) -> None:
    root = gate1_run_dir / "recall-contract"
    entry = SearchResultItem(
        kind="graph_completion",
        search_type=SearchType.GRAPH_COMPLETION,
        text="recall context passage",
    )
    entry = entry.model_copy(update={"source": "graph"})

    async def fake_recall(**kwargs):
        return [entry]

    monkeypatch.setattr(cognee, "recall", fake_recall)
    paths = build_data_paths(root)
    adapter = CogneeSearchAdapter(paths, CogneeCompatibilityAdapter(paths))
    contexts = await adapter.recall_context("alpha", dataset="papers", top_k=5)
    assert contexts == ["recall context passage"]
