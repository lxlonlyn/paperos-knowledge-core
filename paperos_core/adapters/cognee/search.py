"""CogneeSearchAdapter: public ``cognee.search`` / ``cognee.recall`` calls.

PaperOS never generates query embeddings, opens vector collections, or calls
the graph engine for ranking. This adapter only calls Cognee's public search
surface and resolves returned Cognee node ids back to canonical ids through the
PaperOS-owned manifests. Chunk provenance backtracking for non-chunk hits is
performed by the narrow ``CogneeCompatibilityAdapter`` reader.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.errors import CogneeStorageError
from paperos_core.paths import DataPaths

_PUBLIC_GRAPH_SEARCH_TYPES = {
    "CHUNKS",
    "GRAPH_COMPLETION",
    "GRAPH_COMPLETION_DECOMPOSITION",
    "GRAPH_COMPLETION_CONTEXT_EXTENSION",
    "GRAPH_SUMMARY_COMPLETION",
}

_PAPEROS_VECTOR_SEARCH_TYPES: dict[str, tuple[tuple[str, str, str], ...]] = {
    "PAPEROS_CHUNKS": (
        ("ChunkDataPoint_text", "text", "ChunkDataPoint"),
    ),
    "PAPEROS_ENTITY_CLAIM": (
        ("EntityDataPoint_name", "name", "EntityDataPoint"),
        ("EntityDataPoint_description", "description", "EntityDataPoint"),
        ("ClaimDataPoint_text", "text", "ClaimDataPoint"),
    ),
    "PAPEROS_ASSOCIATIVE_SEEDS": (
        ("ChunkDataPoint_text", "text", "ChunkDataPoint"),
        ("EntityDataPoint_name", "name", "EntityDataPoint"),
        ("EntityDataPoint_description", "description", "EntityDataPoint"),
        ("ClaimDataPoint_text", "text", "ClaimDataPoint"),
    ),
    "PAPEROS_GRAPH_SEEDS": (
        ("EntityDataPoint_name", "name", "EntityDataPoint"),
        ("EntityDataPoint_description", "description", "EntityDataPoint"),
        ("ClaimDataPoint_text", "text", "ClaimDataPoint"),
        ("TripletDataPoint_text", "text", "TripletDataPoint"),
        (
            "ConceptRelationDataPoint_description",
            "description",
            "ConceptRelationDataPoint",
        ),
    ),
    "PAPEROS_SUMMARIES": (
        ("SummaryDataPoint_text", "text", "SummaryDataPoint"),
    ),
}


@dataclass(frozen=True, slots=True)
class CogneeSearchHit:
    node_id: str
    canonical_id: str
    source_chunk_ids: tuple[str, ...]
    references: tuple[str, ...]
    result_type: str
    text: str
    score: float

    @property
    def cognee_id(self) -> str:
        return self.node_id

    @property
    def object_type(self) -> str:
        return self.result_type


class CogneeSearchAdapter:
    def __init__(
        self,
        paths: DataPaths,
        compat: CogneeCompatibilityAdapter,
    ) -> None:
        self.paths = paths
        self.compat = compat

    def _manifest_index(self) -> dict[str, str]:
        """Return ``cognee UUID -> canonical id`` from PaperOS-owned manifests."""
        mapping: dict[str, str] = {}
        manifests = self.paths.cognee / "manifests"
        if not manifests.is_dir():
            return mapping
        for manifest_path in manifests.glob("*.json"):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            canonical_to_cognee = payload.get("canonical_to_cognee_id")
            if isinstance(canonical_to_cognee, dict):
                for canonical_id, cognee_id in canonical_to_cognee.items():
                    mapping[str(cognee_id)] = str(canonical_id)
        return mapping

    async def graph_search(
        self,
        query: str,
        *,
        dataset: str,
        top_k: int,
        search_type: str = "GRAPH_COMPLETION",
    ) -> list[CogneeSearchHit]:
        """Run Cognee's graph/context search and normalize node hits."""
        if top_k <= 0:
            return []
        mapping = self._manifest_index()
        vector_collections = _PAPEROS_VECTOR_SEARCH_TYPES.get(search_type)
        if vector_collections is not None:
            vector_hits = await self.compat.search_datapoint_vectors(
                query,
                dataset_name=dataset,
                collections=vector_collections,
                canonical_ids=mapping,
                top_k=top_k,
            )
            resolved = await self.compat.resolve_graph_nodes(
                [
                    hit.cognee_id
                    for hit in vector_hits
                    if hit.object_type != "ChunkDataPoint"
                ]
            )
            return [
                CogneeSearchHit(
                    node_id=hit.cognee_id,
                    canonical_id=hit.canonical_id,
                    source_chunk_ids=(
                        hit.source_chunk_ids
                        or tuple(
                            _string_list(
                                resolved.get(hit.cognee_id, {}).get(
                                    "source_chunk_ids"
                                )
                            )
                        )
                    ),
                    references=(
                        hit.derived_from_ids
                        or tuple(
                            _string_list(
                                resolved.get(hit.cognee_id, {}).get(
                                    "derived_from_ids"
                                )
                            )
                        )
                    ),
                    result_type=hit.object_type,
                    text=hit.text,
                    score=hit.score,
                )
                for hit in vector_hits
            ]
        if search_type not in _PUBLIC_GRAPH_SEARCH_TYPES:
            raise CogneeStorageError(
                f"Unsupported graph search type: {search_type}",
                affected=dataset,
            )
        import cognee  # type: ignore[import-untyped]

        try:
            results = await cognee.search(
                query_text=query,
                query_type=cognee.SearchType(search_type),
                only_context=True,
                verbose=True,
                datasets=[dataset],
                top_k=top_k,
            )
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee search failed: {exc}",
                affected=dataset,
            ) from exc
        best: dict[str, CogneeSearchHit] = {}
        for result_rank, result in enumerate(_result_items(results), 1):
            normalized = _normalized_item_hit(result, mapping, rank=result_rank)
            if normalized is not None:
                existing = best.get(normalized.canonical_id)
                if existing is None or normalized.score > existing.score:
                    best[normalized.canonical_id] = normalized
            objects = result.get("objects_result") if isinstance(result, dict) else []
            if not isinstance(objects, list):
                objects = []
            for edge_rank, edge in enumerate(objects, 1):
                for node in (
                    getattr(edge, "node1", None),
                    getattr(edge, "node2", None),
                ):
                    hit = _node_hit(node, mapping, rank=result_rank + edge_rank - 1)
                    if hit is None:
                        continue
                    existing = best.get(hit.canonical_id)
                    if existing is None or hit.score > existing.score:
                        best[hit.canonical_id] = hit
        return sorted(
            best.values(),
            key=lambda hit: (-hit.score, hit.canonical_id),
        )[:top_k]

    async def recall_context(
        self,
        query: str,
        *,
        dataset: str,
        top_k: int,
        search_type: str = "GRAPH_COMPLETION",
    ) -> list[CogneeSearchHit]:
        """Return only recall entries with stable node/canonical provenance."""
        if top_k <= 0:
            return []
        if search_type in _PAPEROS_VECTOR_SEARCH_TYPES:
            vector_hits = await self.graph_search(
                query,
                dataset=dataset,
                top_k=top_k,
                search_type=search_type,
            )
            return [
                hit
                for hit in vector_hits
                if hit.source_chunk_ids or hit.references
            ]
        import cognee

        try:
            entries = await cognee.recall(
                query_text=query,
                query_type=cognee.SearchType(search_type),
                datasets=[dataset],
                only_context=True,
                top_k=top_k,
            )
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee recall failed: {exc}",
                affected=dataset,
            ) from exc
        mapping = self._manifest_index()
        hits: list[CogneeSearchHit] = []
        for rank, entry in enumerate(_result_items(entries), 1):
            hit = _normalized_item_hit(entry, mapping, rank=rank)
            if hit is not None and (hit.source_chunk_ids or hit.references):
                hits.append(hit)
        return hits[:top_k]


def _node_hit(node: Any, mapping: dict[str, str], *, rank: int) -> CogneeSearchHit | None:
    attributes = getattr(node, "attributes", None)
    if not isinstance(attributes, dict):
        return None
    cognee_id = str(node.id or attributes.get("id") or "")
    if not cognee_id:
        return None
    canonical_id = mapping.get(cognee_id) or attributes.get("canonical_id")
    if not canonical_id:
        return None
    object_type = str(attributes.get("type") or "unknown")
    text = str(attributes.get("text") or attributes.get("name") or "")
    if not text.strip():
        return None
    return CogneeSearchHit(
        node_id=cognee_id,
        canonical_id=str(canonical_id),
        source_chunk_ids=tuple(_string_list(attributes.get("source_chunk_ids"))),
        references=tuple(_string_list(attributes.get("references"))),
        result_type=object_type,
        text=text,
        score=_distance_score(attributes.get("vector_distance"), rank=rank),
    )


def _distance_score(value: Any, *, rank: int) -> float:
    """Map distance when present, otherwise preserve Cognee's result rank."""
    distances = value if isinstance(value, (list, tuple)) else [value]
    numeric = [
        float(item) for item in distances if isinstance(item, (int, float))
    ]
    if not numeric:
        return 1.0 / max(rank, 1)
    return 1.0 / (1.0 + max(min(numeric), 0.0))


def _result_items(results: Any) -> list[Any]:
    if isinstance(results, list):
        return results
    items = getattr(results, "items", None)
    return list(items) if isinstance(items, list) else []


def _normalized_item_hit(
    entry: Any, mapping: dict[str, str], *, rank: int
) -> CogneeSearchHit | None:
    metadata = getattr(entry, "metadata", None)
    raw = getattr(entry, "raw", None)
    text = getattr(entry, "text", None)
    score = getattr(entry, "score", None)
    kind = getattr(entry, "kind", None)
    if isinstance(entry, dict) and "objects_result" not in entry:
        metadata = entry.get("metadata")
        raw = entry.get("raw")
        text = entry.get("text")
        score = entry.get("score")
        kind = entry.get("kind") or entry.get("result_type")
    metadata = metadata if isinstance(metadata, dict) else {}
    raw = raw if isinstance(raw, dict) else {}
    combined = {**raw, **metadata}
    node_id = str(combined.get("node_id") or combined.get("id") or "")
    canonical_id = str(combined.get("canonical_id") or mapping.get(node_id) or "")
    source_chunk_ids = _string_list(combined.get("source_chunk_ids"))
    if not source_chunk_ids and combined.get("chunk_id"):
        source_chunk_ids = [str(combined["chunk_id"])]
    if not canonical_id and len(source_chunk_ids) == 1:
        canonical_id = source_chunk_ids[0]
    if not node_id or not canonical_id or not isinstance(text, str) or not text.strip():
        return None
    return CogneeSearchHit(
        node_id=node_id,
        canonical_id=canonical_id,
        source_chunk_ids=tuple(source_chunk_ids),
        references=tuple(_string_list(combined.get("references"))),
        result_type=str(combined.get("object_type") or combined.get("type") or kind or "unknown"),
        text=text,
        score=float(score) if isinstance(score, (int, float)) else 1.0 / max(rank, 1),
    )


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item]
    return []
