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


@dataclass(frozen=True, slots=True)
class CogneeSearchHit:
    cognee_id: str
    canonical_id: str
    object_type: str
    text: str
    score: float


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
    ) -> list[CogneeSearchHit]:
        """Run Cognee's graph/context search and normalize node hits."""
        if top_k <= 0:
            return []
        import cognee  # type: ignore[import-untyped]

        try:
            results = await cognee.search(
                query_text=query,
                query_type=cognee.SearchType.GRAPH_COMPLETION,
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
        mapping = self._manifest_index()
        best: dict[str, CogneeSearchHit] = {}
        for result in results:
            objects = result.get("objects_result") if isinstance(result, dict) else []
            for edge in objects:
                for node in (
                    getattr(edge, "node1", None),
                    getattr(edge, "node2", None),
                ):
                    hit = _node_hit(node, mapping)
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
    ) -> list[str]:
        """Run Cognee's public recall and return its normalized context texts."""
        if top_k <= 0:
            return []
        import cognee  # type: ignore[import-untyped]

        try:
            entries = await cognee.recall(
                query_text=query,
                query_type=cognee.SearchType.GRAPH_COMPLETION,
                datasets=[dataset],
                only_context=True,
                top_k=top_k,
            )
        except Exception as exc:
            raise CogneeStorageError(
                f"Cognee recall failed: {exc}",
                affected=dataset,
            ) from exc
        texts: list[str] = []
        for entry in entries:
            text = getattr(entry, "text", None)
            if isinstance(text, str) and text.strip():
                texts.append(text)
        return texts


def _node_hit(node: Any, mapping: dict[str, str]) -> CogneeSearchHit | None:
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
        cognee_id=cognee_id,
        canonical_id=str(canonical_id),
        object_type=object_type,
        text=text,
        score=_distance_score(attributes.get("vector_distance")),
    )


def _distance_score(value: Any) -> float:
    distances = value if isinstance(value, (list, tuple)) else [value]
    numeric = [
        float(item) for item in distances if isinstance(item, (int, float))
    ]
    if not numeric:
        return 0.0
    return 1.0 / (1.0 + max(min(numeric), 0.0))
