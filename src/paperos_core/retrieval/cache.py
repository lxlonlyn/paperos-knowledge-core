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

QUERY_CACHE_VERSION = "4"


class QueryCache:
    def __init__(self, paths: DataPaths, feedback: FeedbackService) -> None:
        self.root = paths.cache / "query"
        self.feedback = feedback

    def key(self, request: QueryRequest, corpus: CorpusView) -> str:
        snapshot_ids = sorted(
            bundle.snapshot.id for bundle in corpus.bundles.values()
        )
        improvement_ids = sorted(
            item.id for item in self.feedback.confirmed_improvements()
        )
        return stable_id(
            "answer",
            request.model_dump_json(),
            *snapshot_ids,
            *improvement_ids,
            id_version=QUERY_CACHE_VERSION,
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
