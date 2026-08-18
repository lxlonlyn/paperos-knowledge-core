"""Deterministic + bounded LLM query scope planning over ScholarlyWork."""

from __future__ import annotations

import re
from typing import Any

from paperos_core.adapters.cognee.llm import LLMClient
from paperos_core.domain.scholarly import ScholarlyWork
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry
from paperos_core.retrieval.candidates import (
    QueryRequest,
    QueryScopeInput,
    QueryScopeTrace,
    ResolvedQueryScope,
)
from paperos_core.retrieval.corpus import CorpusView

_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "via",
    "with",
}

_SOURCE_ONLY_CUES = (
    "只根据",
    "仅根据",
    "只依据",
    "仅依据",
    "原文",
    "according to",
    "based only on",
    "only from",
    "only using",
)

_EXTERNAL_SUBJECT_CUES = (
    "后来论文",
    "其他论文",
    "别的论文",
    "后续论文",
    "指出的问题",
    "later papers",
    "other papers",
    "subsequent papers",
    "critic",
    "limitation pointed",
    "problems pointed",
)

_SELF_CUES = (
    "自己报告",
    "自己承认",
    "自身",
    "itself",
    "self-reported",
    "self reported",
    "own limitation",
    "own limitations",
)

_COMPARE_CUES = (
    "比较",
    "对比",
    "差异",
    "compare",
    "versus",
    " vs ",
    "difference",
    "differences",
)

_DISCUSSION_SUBJECT_CUES = (
    "现有论文",
    "有哪些评价",
    "评价或讨论",
    "评价",
    "讨论",
    "existing papers",
    "what do papers say",
    "how do papers discuss",
    "discussion of",
    "comments on",
)

_TOPIC_STOP = {
    "哪些",
    "什么",
    "如何",
    "说明",
    "问题",
    "后来",
    "论文",
    "指出",
    "比较",
    "差异",
    "方面",
    "根据",
    "原文",
    "自己",
    "报告",
    "which",
    "what",
    "how",
    "problem",
    "problems",
    "paper",
    "papers",
    "later",
    "compare",
    "difference",
    "differences",
    "according",
    "only",
    "from",
    "itself",
}


def resolve_query_scope(
    request: QueryRequest,
    corpus: CorpusView,
    scholarly_registry: ScholarlyRegistry,
    *,
    llm: LLMClient | None = None,
) -> tuple[ResolvedQueryScope, QueryScopeTrace]:
    """Resolve scope: explicit request beats planner; planner failure stays unscoped."""
    all_works, ingested_works = _work_catalogs(scholarly_registry, corpus)
    mentioned = _mention_work_ids(request.query, all_works, corpus)
    if request.scope is not None:
        resolved = _from_explicit(
            request.scope,
            ingested_ids=set(ingested_works),
            all_ids=set(all_works),
        )
        return resolved, QueryScopeTrace(
            resolution="explicit",
            mentioned_work_ids=sorted(mentioned),
            warnings=[],
        )

    deterministic = _deterministic_scope(request.query, mentioned, all_works)
    if deterministic is not None:
        return deterministic, QueryScopeTrace(
            resolution="deterministic",
            mentioned_work_ids=sorted(mentioned),
        )

    if llm is None or not all_works:
        return ResolvedQueryScope(), QueryScopeTrace(
            resolution="fallback_unscoped",
            mentioned_work_ids=sorted(mentioned),
            warnings=["No LLM planner available; left query unscoped."],
        )
    return ResolvedQueryScope(), QueryScopeTrace(
        resolution="fallback_unscoped",
        mentioned_work_ids=sorted(mentioned),
        warnings=["Synchronous resolver does not run the LLM planner."],
    )


async def resolve_query_scope_async(
    request: QueryRequest,
    corpus: CorpusView,
    scholarly_registry: ScholarlyRegistry,
    *,
    llm: LLMClient,
) -> tuple[ResolvedQueryScope, QueryScopeTrace]:
    all_works, ingested_works = _work_catalogs(scholarly_registry, corpus)
    mentioned = _mention_work_ids(request.query, all_works, corpus)

    if request.scope is not None:
        return _from_explicit(
            request.scope,
            ingested_ids=set(ingested_works),
            all_ids=set(all_works),
        ), QueryScopeTrace(
            resolution="explicit",
            mentioned_work_ids=sorted(mentioned),
        )

    deterministic = _deterministic_scope(request.query, mentioned, all_works)
    if deterministic is not None:
        return deterministic, QueryScopeTrace(
            resolution="deterministic",
            mentioned_work_ids=sorted(mentioned),
        )

    catalog = _build_scope_catalog(all_works)
    if not catalog["key_to_work_id"]:
        return ResolvedQueryScope(), QueryScopeTrace(
            resolution="fallback_unscoped",
            mentioned_work_ids=sorted(mentioned),
            warnings=["Empty Work catalog; left query unscoped."],
        )

    try:
        planned = await llm.plan_query_scope(
            query=request.query,
            catalog_entries=catalog["entries"],
        )
    except Exception as exc:  # noqa: BLE001 - planner must never hard-fail retrieval.
        return ResolvedQueryScope(), QueryScopeTrace(
            resolution="fallback_unscoped",
            mentioned_work_ids=sorted(mentioned),
            warnings=[f"Scope planner failed: {type(exc).__name__}: {exc}"],
        )

    if planned is None or not planned.confident:
        return ResolvedQueryScope(), QueryScopeTrace(
            resolution="fallback_unscoped",
            mentioned_work_ids=sorted(mentioned),
            warnings=["Scope planner was not confident; left query unscoped."],
            planner_notes=planned.notes if planned else None,
        )

    key_map: dict[str, str] = catalog["key_to_work_id"]
    try:
        resolved = ResolvedQueryScope(
            source_work_ids=[
                work_id
                for work_id in _map_keys(planned.source_work_keys, key_map)
                if work_id in ingested_works
            ],
            exclude_source_work_ids=_map_keys(
                planned.exclude_source_work_keys, key_map
            ),
            subject_work_ids=_map_keys(planned.subject_work_keys, key_map),
            work_set_work_ids=[
                work_id
                for work_id in _map_keys(planned.work_set_work_keys, key_map)
                if work_id in ingested_works
            ],
            topic_queries=[
                item.strip()
                for item in planned.topic_queries
                if isinstance(item, str) and item.strip()
            ][:8],
        )
    except ValueError as exc:
        return ResolvedQueryScope(), QueryScopeTrace(
            resolution="fallback_unscoped",
            mentioned_work_ids=sorted(mentioned),
            warnings=[str(exc)],
            planner_notes=planned.notes,
        )

    if not resolved.has_hard_work_scope and not resolved.topic_queries:
        return ResolvedQueryScope(), QueryScopeTrace(
            resolution="fallback_unscoped",
            mentioned_work_ids=sorted(mentioned),
            warnings=["Planner returned empty scope; left query unscoped."],
            planner_notes=planned.notes,
        )
    return resolved, QueryScopeTrace(
        resolution="llm",
        mentioned_work_ids=sorted(mentioned),
        planner_notes=planned.notes,
    )


def apply_scope_to_document_ids(
    corpus: CorpusView,
    base_document_ids: set[str],
    scope: ResolvedQueryScope,
) -> set[str]:
    """Restrict document universe for chunk lanes from source / work-set scope."""
    selected = set(base_document_ids)
    if scope.source_work_ids:
        selected &= corpus.document_ids_for_works(scope.source_work_ids)
    if scope.work_set_work_ids:
        selected &= corpus.document_ids_for_works(scope.work_set_work_ids)
    if scope.exclude_source_work_ids:
        selected -= corpus.document_ids_for_works(scope.exclude_source_work_ids)
    return selected


def filter_candidates_by_scope(
    candidates: list[Any],
    scope: ResolvedQueryScope,
) -> list[Any]:
    """Drop candidates that violate source / exclude / work-set constraints."""
    if not scope.has_hard_work_scope:
        return candidates
    allowed_source = set(scope.source_work_ids) if scope.source_work_ids else None
    excluded = set(scope.exclude_source_work_ids)
    work_set = set(scope.work_set_work_ids) if scope.work_set_work_ids else None
    kept = []
    for candidate in candidates:
        source_work_id = getattr(candidate, "source_work_id", None)
        if source_work_id is None:
            # Without provenance, hard scopes cannot safely keep the hit.
            if allowed_source is not None or work_set is not None or excluded:
                continue
        if allowed_source is not None and source_work_id not in allowed_source:
            continue
        if source_work_id in excluded:
            continue
        if work_set is not None and source_work_id not in work_set:
            continue
        kept.append(candidate)
    return kept


def filter_candidates_by_subject(
    candidates: list[Any],
    scope: ResolvedQueryScope,
    mention_index: dict[str, tuple[str, ...]],
) -> list[Any]:
    """Keep only candidates with a provable link to the subject Work."""
    if not scope.subject_work_ids:
        return candidates
    kept = []
    for candidate in candidates:
        proven = proven_subject_work_ids(
            text=getattr(candidate, "text", "") or "",
            structured_subject_ids=list(
                getattr(candidate, "subject_work_ids", None) or []
            ),
            derived_from_ids=list(getattr(candidate, "derived_from_ids", None) or []),
            subject_work_ids=scope.subject_work_ids,
            mention_index=mention_index,
        )
        if not proven:
            continue
        updater = getattr(candidate, "model_copy", None)
        if updater is not None:
            merged = list(
                dict.fromkeys(
                    [
                        *list(getattr(candidate, "subject_work_ids", None) or []),
                        *proven,
                    ]
                )
            )
            kept.append(updater(update={"subject_work_ids": merged}))
        else:
            kept.append(candidate)
    return kept


def apply_scope_filters(
    candidates: list[Any],
    scope: ResolvedQueryScope,
    mention_index: dict[str, tuple[str, ...]],
) -> list[Any]:
    """Apply source/exclude/work-set then subject-relevance filters."""
    return filter_candidates_by_subject(
        filter_candidates_by_scope(candidates, scope),
        scope,
        mention_index,
    )


def proven_subject_work_ids(
    *,
    text: str,
    structured_subject_ids: list[str],
    derived_from_ids: list[str],
    subject_work_ids: list[str],
    mention_index: dict[str, tuple[str, ...]],
) -> list[str]:
    """Return subject Work IDs that this evidence can actually prove."""
    structured = set(structured_subject_ids)
    derived = set(derived_from_ids)
    normalized = _normalize(text)
    proven: list[str] = []
    for work_id in dict.fromkeys(subject_work_ids):
        if work_id in structured or work_id in derived:
            proven.append(work_id)
            continue
        if any(
            _contains_phrase(normalized, phrase)
            for phrase in mention_index.get(work_id, ())
        ):
            proven.append(work_id)
    return proven


def build_mention_index(
    works: dict[str, ScholarlyWork],
) -> dict[str, tuple[str, ...]]:
    """Title plus globally unique aliases/DOI/arXiv for textual subject proof."""
    alias_owners: dict[str, set[str]] = {}
    for work_id, work in works.items():
        for alias in _work_aliases(work):
            if alias:
                alias_owners.setdefault(alias, set()).add(work_id)
        if work.doi:
            alias_owners.setdefault(_normalize(work.doi), set()).add(work_id)
        if work.arxiv_id:
            alias_owners.setdefault(_normalize(work.arxiv_id), set()).add(work_id)
    unique_aliases = {
        alias: next(iter(owners))
        for alias, owners in alias_owners.items()
        if len(owners) == 1
    }
    index: dict[str, tuple[str, ...]] = {}
    for work_id, work in works.items():
        phrases = [_normalize(work.title)]
        phrases.extend(
            alias for alias, owner in unique_aliases.items() if owner == work_id
        )
        index[work_id] = tuple(dict.fromkeys(item for item in phrases if item))
    return index


def should_apply_explicit_document_scope(
    *,
    scope: ResolvedQueryScope,
    explicit_document_ids: set[str],
    comparative_query: bool,
) -> bool:
    """Avoid locking subject queries onto the mentioned paper's own Document."""
    if not explicit_document_ids:
        return False
    if scope.has_hard_work_scope:
        return False
    if len(explicit_document_ids) == 1 and comparative_query:
        return False
    return True


def _from_explicit(
    scope: QueryScopeInput,
    *,
    ingested_ids: set[str],
    all_ids: set[str],
) -> ResolvedQueryScope:
    def clean(values: list[str] | None, allowed: set[str]) -> list[str]:
        if not values:
            return []
        return [
            item
            for item in dict.fromkeys(values)
            if item in allowed or not allowed
        ]

    return ResolvedQueryScope(
        source_work_ids=clean(scope.source_work_ids, ingested_ids),
        exclude_source_work_ids=clean(scope.exclude_source_work_ids, all_ids),
        subject_work_ids=clean(scope.subject_work_ids, all_ids),
        work_set_work_ids=clean(scope.work_set_work_ids, ingested_ids),
        topic_queries=[
            item.strip()
            for item in (scope.topic_queries or [])
            if item and item.strip()
        ],
    )


def _deterministic_scope(
    query: str,
    mentioned: set[str],
    works: dict[str, ScholarlyWork],
) -> ResolvedQueryScope | None:
    normalized = query.casefold()
    topics_stripped = _topic_queries_from_text(query, works)
    topics_raw = _topic_queries_from_text(query, {})
    if not mentioned:
        return None

    if any(cue in normalized for cue in _COMPARE_CUES) and len(mentioned) >= 2:
        return ResolvedQueryScope(
            work_set_work_ids=sorted(mentioned),
            topic_queries=topics_raw or topics_stripped,
        )

    if any(cue in normalized for cue in _SELF_CUES) and len(mentioned) == 1:
        work_id = next(iter(mentioned))
        return ResolvedQueryScope(
            source_work_ids=[work_id],
            subject_work_ids=[work_id],
            topic_queries=topics_stripped or ["limitations"],
        )

    if any(cue in normalized for cue in _DISCUSSION_SUBJECT_CUES) and mentioned:
        return ResolvedQueryScope(
            subject_work_ids=sorted(mentioned),
            topic_queries=topics_stripped or ["discussion"],
        )

    if any(cue in normalized for cue in _EXTERNAL_SUBJECT_CUES) and len(mentioned) == 1:
        work_id = next(iter(mentioned))
        return ResolvedQueryScope(
            subject_work_ids=[work_id],
            exclude_source_work_ids=[work_id],
            topic_queries=topics_stripped or ["limitations", "problems"],
        )

    if any(cue in normalized for cue in _SOURCE_ONLY_CUES) and len(mentioned) == 1:
        work_id = next(iter(mentioned))
        return ResolvedQueryScope(
            source_work_ids=[work_id],
            topic_queries=topics_stripped,
        )

    # Source + subject: one work is the evidence source, another is the subject.
    if (
        any(cue in normalized for cue in _SOURCE_ONLY_CUES)
        and len(mentioned) >= 2
    ):
        # Prefer the work whose full title appears as the "only according to" source.
        source_id = _source_work_from_source_cue(query, mentioned, works)
        subject_ids = sorted(mentioned - {source_id}) if source_id else []
        if source_id and subject_ids:
            return ResolvedQueryScope(
                source_work_ids=[source_id],
                subject_work_ids=subject_ids,
                topic_queries=list(dict.fromkeys([*topics_raw, *topics_stripped]))
                or ["limitations", "problems"],
            )
    return None


def _source_work_from_source_cue(
    query: str,
    mentioned: set[str],
    works: dict[str, ScholarlyWork],
) -> str | None:
    normalized = _normalize(query)
    best: tuple[int, str] | None = None
    for work_id in mentioned:
        title = _normalize(works[work_id].title)
        if title and title in normalized:
            score = len(title)
            if best is None or score > best[0]:
                best = (score, work_id)
    if best is not None:
        return best[1]
    # Fall back to the longest unique acronym/title token mention near source cues.
    return sorted(mentioned)[0] if len(mentioned) == 1 else None


def _work_catalogs(
    scholarly_registry: ScholarlyRegistry, corpus: CorpusView
) -> tuple[dict[str, ScholarlyWork], dict[str, ScholarlyWork]]:
    all_works = {work.id: work for work in scholarly_registry.list_works()}
    ingested = {
        work_id: work
        for work_id, work in all_works.items()
        if work_id in corpus.document_ids_by_work
    }
    return all_works, ingested


def _mention_work_ids(
    query: str,
    works: dict[str, ScholarlyWork],
    corpus: CorpusView,
) -> set[str]:
    normalized_query = _normalize(query)
    title_owners: dict[str, set[str]] = {}
    alias_owners: dict[str, set[str]] = {}
    for work_id, work in works.items():
        title = _normalize(work.title)
        if title:
            title_owners.setdefault(title, set()).add(work_id)
        for alias in _work_aliases(work):
            if alias:
                alias_owners.setdefault(alias, set()).add(work_id)
        if work.doi:
            alias_owners.setdefault(_normalize(work.doi), set()).add(work_id)
        if work.arxiv_id:
            alias_owners.setdefault(_normalize(work.arxiv_id), set()).add(work_id)
    matched: set[str] = set()
    # Full titles may have duplicate identities; still treat the mention as real.
    for title, owners in title_owners.items():
        if _contains_phrase(normalized_query, title):
            matched.update(owners)
    for alias, owners in alias_owners.items():
        if len(owners) != 1:
            continue
        if _contains_phrase(normalized_query, alias):
            matched.update(owners)
    for document_id in corpus.explicitly_mentioned_document_ids(query):
        work_id = corpus.work_id_by_document.get(document_id)
        if work_id and work_id in works:
            matched.add(work_id)
    return matched


def _work_aliases(work: ScholarlyWork) -> list[str]:
    aliases: list[str] = []
    tokens = [
        token
        for token in re.findall(r"[A-Za-z0-9]+", work.title)
        if token.casefold() not in _STOPWORDS
    ]
    for token in re.findall(r"[A-Za-z]{3,}(?:-[A-Za-z0-9]+)+|[A-Z]{3,}", work.title):
        aliases.append(_normalize(token))
    initials = [token[0] for token in tokens if token[:1].isalpha()]
    # Prefix acronyms ignore trailing author-name tokens commonly appended to titles.
    for length in range(3, min(len(initials), 5) + 1):
        aliases.append(_normalize("".join(initials[:length])))
    for token in tokens:
        if len(token) >= 4 and token.isalpha() and token.upper() == token:
            aliases.append(_normalize(token))
    return list(dict.fromkeys(item for item in aliases if len(item) >= 3))


def residual_query_text(query: str, titles: list[str] | tuple[str, ...]) -> str:
    """Strip Work titles so paper names do not leak into topic / ranking text."""
    residual = query
    for title in sorted({item for item in titles if item}, key=len, reverse=True):
        residual = re.sub(re.escape(title), " ", residual, flags=re.IGNORECASE)
    return " ".join(residual.split())


def _topic_queries_from_text(
    query: str, works: dict[str, ScholarlyWork]
) -> list[str]:
    residual = residual_query_text(query, [work.title for work in works.values()])
    lowered = residual.casefold()
    topics: list[str] = []
    topic_patterns = (
        ("volume", ("volume", "体积")),
        ("intermediate shape", ("intermediate", "中间形")),
        ("topology", ("topology", "拓扑")),
        ("smoothness", ("smooth", "光滑")),
        ("limitations", ("limitation", "限制", "问题", "drawback")),
    )
    for label, cues in topic_patterns:
        if any(cue in lowered for cue in cues):
            topics.append(label)
    tokens = [
        token
        for token in re.findall(r"[A-Za-z]{4,}|[\u4e00-\u9fff]{2,}", residual)
        if token.casefold() not in _TOPIC_STOP and token.casefold() not in _STOPWORDS
    ]
    for token in tokens[:4]:
        if token.casefold() not in {item.casefold() for item in topics}:
            topics.append(token)
    return topics[:6]


def _build_scope_catalog(works: dict[str, ScholarlyWork]) -> dict[str, Any]:
    ordered = sorted(works.values(), key=lambda item: item.id)
    key_to_work_id: dict[str, str] = {}
    entries: list[dict[str, Any]] = []
    for index, work in enumerate(ordered, start=1):
        key = f"WORK_{index:03d}"
        key_to_work_id[key] = work.id
        entries.append(
            {
                "work_key": key,
                "title": work.title,
                "authors": list(work.authors),
                "year": work.year,
                "doi": work.doi,
                "arxiv_id": work.arxiv_id,
                "aliases": _work_aliases(work),
            }
        )
    return {"key_to_work_id": key_to_work_id, "entries": entries}


def _map_keys(keys: list[str], key_map: dict[str, str]) -> list[str]:
    mapped: list[str] = []
    for key in keys:
        normalized = key.strip().upper()
        if normalized not in key_map:
            raise ValueError(f"Planner returned unknown work_key: {key}")
        mapped.append(key_map[normalized])
    return list(dict.fromkeys(mapped))


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _contains_phrase(normalized_query: str, phrase: str) -> bool:
    if not phrase:
        return False
    return f" {phrase} " in f" {normalized_query} "
