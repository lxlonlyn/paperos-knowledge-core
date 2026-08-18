"""Deterministic Task 03 query-scope contracts. This project does not use pytest.

    python tests/contract/test_query_scope.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.domain.scholarly import ScholarlyWork, WorkIdentityStatus
from paperos_core.paths import build_data_paths
from paperos_core.retrieval.candidates import (
    Candidate,
    QueryRequest,
    QueryScopeInput,
    ResolvedQueryScope,
)
from paperos_core.retrieval.corpus import CorpusView
from paperos_core.retrieval.scope import (
    apply_scope_to_document_ids,
    build_mention_index,
    filter_candidates_by_scope,
    filter_candidates_by_subject,
    resolve_query_scope,
    should_apply_explicit_document_scope,
)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


class _FakeRegistry:
    def __init__(self, works: list[ScholarlyWork]) -> None:
        self._works = works

    def list_works(self, *, include_redirected: bool = False) -> list[ScholarlyWork]:
        return list(self._works)


def _work(work_id: str, title: str) -> ScholarlyWork:
    return ScholarlyWork(
        id=work_id,
        title=title,
        normalized_title=title.casefold(),
        identity_status=WorkIdentityStatus.INGESTED,
        identity_confidence=1.0,
        year=2024,
        authors=["Author"],
    )


def _corpus(
    works: list[ScholarlyWork],
    *,
    ingested_ids: set[str] | None = None,
) -> CorpusView:
    selected = ingested_ids if ingested_ids is not None else {work.id for work in works}
    work_id_by_document = {}
    document_ids_by_work: dict[str, set[str]] = {}
    work_titles = {work.id: work.title for work in works}
    for work in works:
        if work.id not in selected:
            continue
        document_id = f"doc_{work.id}"
        work_id_by_document[document_id] = work.id
        document_ids_by_work[work.id] = {document_id}
    return CorpusView(
        paths=build_data_paths(REPOSITORY_ROOT / "data"),
        bundles={},
        chunks={},
        chunk_bundles={},
        source_filenames={},
        work_id_by_document=work_id_by_document,
        document_ids_by_work=document_ids_by_work,
        work_titles=work_titles,
    )


def _candidate(
    candidate_id: str,
    source_work_id: str,
    *,
    text: str = "text",
    subject_work_ids: list[str] | None = None,
) -> Candidate:
    return Candidate(
        id=candidate_id,
        object_id=candidate_id,
        object_type="chunk",
        document_id=f"doc_{source_work_id}",
        source_file_id="src",
        source_filename="paper.pdf",
        canonical_snapshot_id="snapshot",
        chunk_id=candidate_id,
        text=text,
        channels=["lexical"],
        source_work_id=source_work_id,
        subject_work_ids=list(subject_work_ids or []),
    )


def _works() -> tuple[list[ScholarlyWork], dict[str, str]]:
    works = [
        _work("work_nise", "Neural Implicit Surface Evolution"),
        _work("work_adadiv", "Volume Preserving Neural Shape Morphing"),
        _work(
            "work_efis",
            "Explicit flows for implicit surfaces Camille Buonomo, Julie Digne, Raphaëlle Chaine",
        ),
        _work("work_lipmlp", "Learning Smooth Neural Functions via Lipschitz Regularization"),
    ]
    return works, {work.id: work.id for work in works}


def deterministic_scope_contract() -> dict[str, object]:
    works, _ids = _works()
    corpus = _corpus(works)
    registry = _FakeRegistry(works)
    cases = [
        (
            "A",
            "只根据 NISE 原文，说明它的方法和限制。",
            {"source_work_ids": ["work_nise"], "subject_work_ids": []},
        ),
        (
            "B",
            "NISE 有哪些后来论文指出的问题？",
            {
                "subject_work_ids": ["work_nise"],
                "exclude_source_work_ids": ["work_nise"],
            },
        ),
        (
            "C",
            "只根据 Volume Preserving Neural Shape Morphing，NISE 有哪些问题？",
            {
                "source_work_ids": ["work_adadiv"],
                "subject_work_ids": ["work_nise"],
            },
        ),
        (
            "D",
            "Volume Preserving Neural Shape Morphing 自己报告了哪些限制？",
            {
                "source_work_ids": ["work_adadiv"],
                "subject_work_ids": ["work_adadiv"],
            },
        ),
        (
            "E",
            "比较 NISE、Volume Preserving Neural Shape Morphing 和 EFIS 在 volume preservation / intermediate shape 方面的差异。",
            {
                "work_set_work_ids": [
                    "work_adadiv",
                    "work_efis",
                    "work_nise",
                ]
            },
        ),
    ]
    resolved_cases = {}
    for case_id, query, expected in cases:
        scope, trace = resolve_query_scope(
            QueryRequest(query=query), corpus, registry
        )
        _require(
            trace.resolution == "deterministic",
            f"{case_id} should resolve deterministically, got {trace.resolution}: {scope}",
        )
        for field, values in expected.items():
            actual = getattr(scope, field)
            _require(
                actual == values,
                f"{case_id} {field} expected {values}, got {actual}",
            )
        if case_id == "C":
            _require(
                any("volume" in item.casefold() for item in scope.topic_queries),
                f"C topic should keep volume from the source title, got {scope.topic_queries}",
            )
        if case_id == "D":
            _require(
                not any(item.casefold() == "volume" for item in scope.topic_queries),
                f"D topics must not inherit volume from the title, got {scope.topic_queries}",
            )
        if case_id == "E":
            _require(
                "work_lipmlp" not in scope.work_set_work_ids,
                "E must not include LipMLP in the work-set",
            )
            _require(
                any("volume" in item.casefold() for item in scope.topic_queries),
                f"E topic should mention volume, got {scope.topic_queries}",
            )
        resolved_cases[case_id] = scope.model_dump()
    return {"status": "passed", "cases": resolved_cases}


def explicit_scope_precedence_contract() -> dict[str, object]:
    works, _ids = _works()
    corpus = _corpus(works)
    registry = _FakeRegistry(works)
    scope, trace = resolve_query_scope(
        QueryRequest(
            query="只根据 NISE 原文，说明它的方法和限制。",
            scope=QueryScopeInput(source_work_ids=["work_efis"]),
        ),
        corpus,
        registry,
    )
    _require(trace.resolution == "explicit", "Explicit scope must win.")
    _require(scope.source_work_ids == ["work_efis"], scope.source_work_ids)
    return {"status": "passed", "source_work_ids": scope.source_work_ids}


def unscoped_fallback_contract() -> dict[str, object]:
    works, _ids = _works()
    corpus = _corpus(works)
    registry = _FakeRegistry(works)
    scope, trace = resolve_query_scope(
        QueryRequest(query="What is a neural implicit surface in general?"),
        corpus,
        registry,
    )
    _require(trace.resolution == "fallback_unscoped", trace.resolution)
    _require(not scope.has_hard_work_scope, scope)
    return {"status": "passed", "warnings": trace.warnings}


def source_filter_before_fusion_contract() -> dict[str, object]:
    scope = ResolvedQueryScope(
        source_work_ids=["work_nise"],
        exclude_source_work_ids=["work_efis"],
    )
    kept = filter_candidates_by_scope(
        [
            _candidate("c1", "work_nise"),
            _candidate("c2", "work_adadiv"),
            _candidate("c3", "work_efis"),
        ],
        scope,
    )
    _require([item.id for item in kept] == ["c1"], [item.id for item in kept])
    works, _ids = _works()
    corpus = _corpus(works)
    docs = apply_scope_to_document_ids(
        corpus, {f"doc_{work.id}" for work in works}, scope
    )
    _require(docs == {"doc_work_nise"}, docs)
    apply = should_apply_explicit_document_scope(
        scope=scope,
        explicit_document_ids={"doc_work_nise"},
        comparative_query=False,
    )
    _require(not apply, "Hard work scope must disable title-document lock.")
    return {"status": "passed", "kept": [item.id for item in kept]}


def compat_incoming_api_contract() -> dict[str, object]:
    from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter

    source = Path(
        CogneeCompatibilityAdapter.incoming_typed_relations.__code__.co_filename
    ).read_text(encoding="utf-8")
    _require(
        "incoming_typed_relations" in source
        and "depth: int = 1" in source
        and "limit: int = 200" in source,
        "compat must expose bounded incoming typed relations",
    )
    retrieval_root = REPOSITORY_ROOT / "paperos_core" / "retrieval"
    for path in retrieval_root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        _require(
            "get_graph_engine" not in text,
            f"{path.name} must not access the graph engine",
        )
    return {"status": "passed"}


def external_subject_without_document_contract() -> dict[str, object]:
    ingested, _ids = _works()
    external = ScholarlyWork(
        id="work_nfgp",
        title="Geometry Processing with Neural Fields",
        normalized_title="geometry processing with neural fields",
        identity_status=WorkIdentityStatus.PROVISIONAL,
        identity_confidence=0.5,
        year=2022,
        authors=["Author"],
    )
    registry = _FakeRegistry([*ingested, external])
    corpus = _corpus(ingested)
    _require(
        "work_nfgp" not in corpus.document_ids_by_work,
        "external Work must not have a Document",
    )
    scope, trace = resolve_query_scope(
        QueryRequest(
            query="现有论文对 Geometry Processing with Neural Fields 有哪些评价或讨论？"
        ),
        corpus,
        registry,
    )
    _require(trace.resolution == "deterministic", trace.resolution)
    _require(scope.subject_work_ids == ["work_nfgp"], scope.subject_work_ids)
    _require(not scope.source_work_ids, scope.source_work_ids)
    explicit, explicit_trace = resolve_query_scope(
        QueryRequest(
            query="ignored because explicit scope wins",
            scope=QueryScopeInput(subject_work_ids=["work_nfgp"]),
        ),
        corpus,
        registry,
    )
    _require(explicit_trace.resolution == "explicit", explicit_trace.resolution)
    _require(explicit.subject_work_ids == ["work_nfgp"], explicit.subject_work_ids)
    return {
        "status": "passed",
        "subject_work_ids": scope.subject_work_ids,
        "explicit_subject_work_ids": explicit.subject_work_ids,
    }


def subject_evidence_filter_contract() -> dict[str, object]:
    works, _ids = _works()
    mention_index = build_mention_index({work.id: work for work in works})
    scope = ResolvedQueryScope(subject_work_ids=["work_nise"])
    kept = filter_candidates_by_subject(
        [
            _candidate(
                "about",
                "work_adadiv",
                text="A structured claim about NISE.",
                subject_work_ids=["work_nise"],
            ),
            _candidate(
                "mention",
                "work_adadiv",
                text="Neural Implicit Surface Evolution cannot preserve volume.",
            ),
            _candidate(
                "unrelated",
                "work_adadiv",
                text="AdaDiv uses an adaptive divergence velocity field.",
            ),
        ],
        scope,
        mention_index,
    )
    _require(
        [item.id for item in kept] == ["about", "mention"],
        [item.id for item in kept],
    )
    return {"status": "passed", "kept": [item.id for item in kept]}


def no_corpus_specific_ranking_contract() -> dict[str, object]:
    source = (
        REPOSITORY_ROOT / "paperos_core" / "retrieval" / "subject_claim.py"
    ).read_text(encoding="utf-8")
    forbidden = (
        '"blob"',
        "'blob'",
        '"detach"',
        "'detach'",
        '"reattach"',
        "'reattach'",
        "handcrafted",
        "adadiv",
        "lipmlp",
    )
    hits = [token for token in forbidden if token in source.casefold()]
    _require(not hits, f"corpus-specific ranking tokens in production: {hits}")
    return {"status": "passed"}


def main() -> None:
    report = {
        "deterministic_scope": deterministic_scope_contract(),
        "explicit_scope": explicit_scope_precedence_contract(),
        "unscoped_fallback": unscoped_fallback_contract(),
        "source_filter": source_filter_before_fusion_contract(),
        "external_subject": external_subject_without_document_contract(),
        "subject_filter": subject_evidence_filter_contract(),
        "no_corpus_ranking_hardcode": no_corpus_specific_ranking_contract(),
        "compat_boundary": compat_incoming_api_contract(),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
