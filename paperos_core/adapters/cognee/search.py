"""Cognee boundary for the single production Chunk vector collection."""

from __future__ import annotations

import json
from dataclasses import dataclass

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.errors import CogneeStorageError
from paperos_core.paths import DataPaths

_CHUNK_SEARCH_TYPE = "PAPEROS_CHUNKS"


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
        search_type: str = _CHUNK_SEARCH_TYPE,
    ) -> list[CogneeSearchHit]:
        """Search ChunkDataPoint vectors and preserve canonical chunk identity."""
        if top_k <= 0:
            return []
        if search_type != _CHUNK_SEARCH_TYPE:
            raise CogneeStorageError(
                f"Unsupported production search type: {search_type}",
                affected=dataset,
            )
        vector_hits = await self.compat.search_datapoint_vectors(
            query,
            dataset_name=dataset,
            search_type=_CHUNK_SEARCH_TYPE,
            canonical_ids=self._manifest_index(),
            top_k=top_k,
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
            if hit.object_type == "ChunkDataPoint"
        ]
