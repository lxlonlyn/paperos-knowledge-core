"""1-hop Claim --ABOUT--> ScholarlyWork retrieval for subject scope."""

from __future__ import annotations

import re

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.adapters.cognee.search import CogneeSearchAdapter
from paperos_core.domain.provenance import RelationType
from paperos_core.retrieval.candidates import Candidate, ResolvedQueryScope
from paperos_core.retrieval.corpus import CorpusView
from paperos_core.retrieval.scope import filter_candidates_by_scope, residual_query_text

_TOKEN = re.compile(r"[a-z0-9]{3,}|[\u4e00-\u9fff]{2,}", re.UNICODE)


async def subject_claim_retrieve(
    search: CogneeSearchAdapter,
    compat: CogneeCompatibilityAdapter,
    corpus: CorpusView,
    query: str,
    *,
    dataset_name: str,
    scope: ResolvedQueryScope,
    limit: int,
) -> list[Candidate]:
    if not scope.subject_work_ids or limit <= 0:
        return []
    relations = await compat.incoming_typed_relations(
        list(scope.subject_work_ids),
        dataset_name=dataset_name,
        relation_type=RelationType.ABOUT.value,
        depth=1,
        limit=500,
    )
    residual_query = residual_query_text(query, list(corpus.work_titles.values()))
    vector_scores = await _claim_vector_scores(
        search,
        query=residual_query or query,
        topic_queries=scope.topic_queries,
        dataset_name=dataset_name,
        limit=max(limit * 4, 40),
    )
    topic_blob = " ".join([residual_query or query, *scope.topic_queries])
    limitation_query = any(
        token in topic_blob.casefold()
        for token in ("limit", "限制", "drawback", "problem", "问题")
    )
    candidates: dict[str, Candidate] = {}
    for relation in relations:
        if relation.target_canonical_id not in scope.subject_work_ids:
            continue
        for chunk_id in relation.source_chunk_ids:
            chunk = corpus.chunks.get(chunk_id)
            if chunk is None:
                continue
            source_work_id = relation.source_work_id or corpus.work_id_by_document.get(
                chunk.document_id
            )
            text = relation.text or chunk.text
            section = (chunk.section_path or "").casefold()
            self_about = source_work_id == relation.target_canonical_id
            score = _about_rank_score(
                claim_text=relation.text or "",
                chunk_text=chunk.text,
                topic_blob=topic_blob,
                vector_score=vector_scores.get(relation.source_canonical_id, 0.0),
                section=section,
                limitation_query=limitation_query,
                self_about=self_about,
            )
            candidate = corpus.candidate_for_chunk(
                chunk_id,
                channel="subject_claim",
                score=score,
                object_id=relation.source_canonical_id,
                object_type="claim_about",
                knowledge_kind="structured_relation",
                derived_from_ids=[
                    relation.source_canonical_id,
                    relation.target_canonical_id,
                    *relation.derived_from_ids,
                    *[f"about_role:{role}" for role in relation.roles],
                ],
                text=text,
                source_work_id=source_work_id,
                subject_work_ids=[relation.target_canonical_id],
                candidate_id=relation.source_canonical_id,
            )
            existing = candidates.get(candidate.id)
            if (
                existing is None
                or score > existing.channel_scores["subject_claim"]
            ):
                candidates[candidate.id] = candidate
    ranked = sorted(
        candidates.values(),
        key=lambda item: (-item.channel_scores["subject_claim"], item.id),
    )
    return filter_candidates_by_scope(ranked, scope)[:limit]


async def _claim_vector_scores(
    search: CogneeSearchAdapter,
    *,
    query: str,
    topic_queries: list[str],
    dataset_name: str,
    limit: int,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    phrases = [query, *[item for item in topic_queries if item.strip()]]
    for phrase in phrases[:4]:
        try:
            hits = await search.graph_search(
                phrase,
                dataset=dataset_name,
                top_k=limit,
                search_type="PAPEROS_CLAIMS",
            )
        except Exception:  # noqa: BLE001 - vector boost is optional ranking.
            continue
        for hit in hits:
            if hit.object_type != "ClaimDataPoint":
                continue
            current = scores.get(hit.canonical_id, 0.0)
            if hit.score > current:
                scores[hit.canonical_id] = float(hit.score)
    return scores


def _about_rank_score(
    *,
    claim_text: str,
    chunk_text: str,
    topic_blob: str,
    vector_score: float,
    section: str = "",
    limitation_query: bool = False,
    self_about: bool = False,
) -> float:
    claim_haystack = claim_text.casefold()
    support_haystack = " ".join(
        part for part in (claim_text, chunk_text, section) if part
    ).casefold()
    tokens = list(dict.fromkeys(_TOKEN.findall(topic_blob.casefold())))
    overlap = sum(1 for token in tokens if token in claim_haystack)
    score = 1.0 + overlap + max(vector_score, 0.0)
    if self_about:
        score += 1.0
    if any(
        token in section
        for token in ("limit", "限制", "discussion", "结论", "conclusion")
    ):
        score += 2.0
    if limitation_query and any(
        token in support_haystack
        for token in (
            "limit",
            "restrict",
            "drawback",
            "cannot",
            "fail",
            "however",
            "artifact",
            "problem",
            "weakness",
            "shortcoming",
            "although",
            "error",
            "noise",
        )
    ):
        score += 3.0
    elif limitation_query:
        # Implementation / method prose without limitation language should not
        # outrank real self-reported failure modes via vector similarity alone.
        score -= 3.5
    if limitation_query and any(
        token in section for token in ("result", "experiment", "evaluation")
    ):
        score += 1.0
    if limitation_query and any(
        phrase in support_haystack
        for phrase in (
            "may appear",
            "local minimum",
            "local minima",
            "nonzero",
            "tends to",
            "artifact",
        )
    ):
        score += 2.0
    if limitation_query and self_about:
        score += 1.5
        if any(
            token in section
            for token in ("appendix", "proof", "theorem", "related work")
        ):
            score -= 3.0
    return score
