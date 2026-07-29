"""Typed graph traversal with backtracking to canonical chunks."""

from __future__ import annotations

import json
import re

from paperos_core.adapters.cognee.repository import CogneeRepository
from paperos_core.retrieval.candidates import Candidate
from paperos_core.retrieval.corpus import CorpusView

_TOKEN = re.compile(r"[\w-]{2,}", re.UNICODE)


async def graph_retrieve(
    repository: CogneeRepository,
    corpus: CorpusView,
    queries: list[str],
    *,
    limit: int,
    document_ids: set[str],
) -> list[Candidate]:
    query_tokens = {
        token.casefold() for query in queries for token in _TOKEN.findall(query)
    }
    candidates: dict[str, Candidate] = {}
    verified_objects: set[str] = set()
    for document_id, bundle in corpus.bundles.items():
        if document_id not in document_ids:
            continue
        manifest_path = (
            corpus.paths.cognee / "manifests" / f"{bundle.snapshot.id}.json"
        )
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relation in payload["relations"]:
            source_chunks = relation.get("source_chunk_ids") or []
            if not source_chunks:
                continue
            searchable = (
                f"{relation['relation_type']} "
                f"{relation['source_id']} {relation['target_id']}"
            ).casefold()
            overlap = sum(token in searchable for token in query_tokens)
            score = 1.0 + float(overlap)
            source_id = str(relation["source_id"])
            if (
                source_id in payload["canonical_to_cognee_id"]
                and source_id not in verified_objects
            ):
                await repository.get_datapoint(source_id)
                verified_objects.add(source_id)
            for chunk_id in source_chunks:
                if chunk_id not in corpus.chunks:
                    continue
                candidate = corpus.candidate_for_chunk(
                    chunk_id,
                    channel="graph",
                    score=score,
                    object_id=source_id,
                    object_type="graph_relation",
                    knowledge_kind="structured_relation",
                    derived_from_ids=[
                        source_id,
                        str(relation["target_id"]),
                        *(relation.get("derived_from_ids") or []),
                    ],
                )
                existing = candidates.get(chunk_id)
                if existing is None or score > existing.channel_scores["graph"]:
                    candidates[chunk_id] = candidate
    return sorted(
        candidates.values(),
        key=lambda item: (-item.channel_scores["graph"], item.id),
    )[:limit]
