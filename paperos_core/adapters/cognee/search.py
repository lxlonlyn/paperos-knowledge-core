"""Cognee boundary for the single production Chunk vector collection."""

from __future__ import annotations

import json
from dataclasses import dataclass

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.errors import CogneeStorageError
from paperos_core.paths import DataPaths
from paperos_core.retrieval.candidates import VectorSearchDiagnostics

_CHUNK_SEARCH_TYPE = "PAPEROS_CHUNKS"
_INITIAL_VECTOR_OVERFETCH = 32
_MIN_VECTOR_SAFETY_LIMIT = 1_024
_MAX_VECTOR_OVERFETCH = 10_000
_VECTOR_OVERFETCH_FACTOR = 32


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

    def _manifest_index(self, active_snapshot_ids: set[str]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        manifests = self.paths.cognee / "manifests"
        if not manifests.is_dir():
            return mapping
        for snapshot_id in sorted(active_snapshot_ids):
            manifest_path = manifests / f"{snapshot_id}.json"
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            canonical_to_cognee = payload.get("canonical_to_cognee_id")
            if not isinstance(canonical_to_cognee, dict):
                continue
            for canonical_id, cognee_id in canonical_to_cognee.items():
                mapping[str(cognee_id)] = str(canonical_id)
        return mapping

    async def graph_search(
        self,
        query: str,
        *,
        dataset: str,
        top_k: int,
        active_snapshot_ids: set[str],
        search_type: str = _CHUNK_SEARCH_TYPE,
        diagnostics: VectorSearchDiagnostics | None = None,
    ) -> list[CogneeSearchHit]:
        """Search PaperOSChunkDataPoint vectors and preserve canonical chunk identity."""
        if top_k <= 0 or not active_snapshot_ids:
            if diagnostics is not None:
                diagnostics.backend_exhausted = True
            return []
        if search_type != _CHUNK_SEARCH_TYPE:
            raise CogneeStorageError(
                f"Unsupported production search type: {search_type}",
                affected=dataset,
            )
        canonical_ids = self._manifest_index(active_snapshot_ids)
        if not canonical_ids:
            if diagnostics is not None:
                diagnostics.backend_exhausted = True
            return []
        safety_limit = min(
            _MAX_VECTOR_OVERFETCH,
            max(top_k * _VECTOR_OVERFETCH_FACTOR, _MIN_VECTOR_SAFETY_LIMIT),
        )
        request_limit = min(
            safety_limit,
            max(top_k * 2, _INITIAL_VECTOR_OVERFETCH),
        )
        vector_hits = []
        backend_exhausted = False
        while True:
            boundary_diagnostics: dict[str, int | bool] = {}
            vector_hits = await self.compat.search_datapoint_vectors(
                query,
                dataset_name=dataset,
                search_type=_CHUNK_SEARCH_TYPE,
                canonical_ids=canonical_ids,
                active_snapshot_ids=active_snapshot_ids,
                top_k=request_limit,
                diagnostics=boundary_diagnostics,
            )
            backend_exhausted = bool(
                boundary_diagnostics.get("backend_exhausted", False)
            )
            if diagnostics is not None:
                diagnostics.request_limits.append(request_limit)
                diagnostics.raw_hit_counts.append(
                    int(boundary_diagnostics.get("raw_hit_count", 0))
                )
                diagnostics.filtered_hit_counts.append(len(vector_hits))
            if (
                len(vector_hits) >= top_k
                or backend_exhausted
                or request_limit >= safety_limit
            ):
                break
            request_limit = min(safety_limit, request_limit * 2)
        if diagnostics is not None:
            diagnostics.backend_exhausted = backend_exhausted
            diagnostics.safety_limit_reached = (
                request_limit >= safety_limit and len(vector_hits) < top_k
            )
        return [
            CogneeSearchHit(
                node_id=hit.cognee_id,
                canonical_id=hit.canonical_id,
                source_chunk_ids=hit.source_chunk_ids,
                references=hit.derived_from_ids,
                result_type=hit.object_type,
                text=hit.text,
                score=hit.score,
            )
            for hit in vector_hits
            if hit.object_type == "PaperOSChunkDataPoint"
        ][:top_k]
