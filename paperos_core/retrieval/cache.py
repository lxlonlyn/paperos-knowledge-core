"""Versioned query-response cache keyed by request and retained knowledge state."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from paperos_core.domain.ids import stable_id
from paperos_core.feedback.service import FeedbackService
from paperos_core.paths import DataPaths
from paperos_core.retrieval.candidates import QueryRequest, QueryResponse
from paperos_core.retrieval.corpus import CorpusView

QUERY_CACHE_VERSIONS = {
    "truth": "13",
    "associative": "25",
    "comprehensive": "25",
}


class QueryCache:
    def __init__(self, paths: DataPaths, feedback: FeedbackService) -> None:
        self.root = paths.cache / "query"
        self.feedback = feedback

    def key(self, request: QueryRequest, corpus: CorpusView) -> str:
        from paperos_core.retrieval.ablation import current_ablation_policy

        snapshot_ids = sorted(
            bundle.snapshot.id for bundle in corpus.bundles.values()
        )
        improvement_ids = sorted(
            item.id for item in self.feedback.confirmed_improvements()
        )
        policy = current_ablation_policy()
        ablation_parts: list[str] = []
        if policy is not None:
            ablation_parts = [
                f"ablation:{policy.configuration_id}",
                f"pool:{policy.candidate_pool_size or ''}",
                f"topk:{policy.final_top_k or ''}",
            ]
        return stable_id(
            "answer",
            request.model_dump_json(),
            *snapshot_ids,
            *improvement_ids,
            *ablation_parts,
            id_version=QUERY_CACHE_VERSIONS[request.profile.value],
        )

    def get(self, key: str) -> QueryResponse | None:
        path = self.root / f"{key}.json"
        try:
            response = QueryResponse.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return None
        return response if response.id == key else None

    def put(self, response: QueryResponse) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{response.id}.json"
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=self.root
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(response.model_dump_json(indent=2))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return target
