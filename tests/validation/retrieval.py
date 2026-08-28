"""Real PDF-to-LLM acceptance for the single Chunk-first Search architecture."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.application import create_application
from paperos_core.config import load_settings
from paperos_core.domain.provenance import SEMANTIC_RELATION_TYPES, RelationType
from paperos_core.errors import ConfigurationError
from paperos_core.retrieval.candidates import QueryRequest, QueryResponse
from paperos_core.retrieval.corpus import CorpusView
from paperos_core.retrieval.expansion import (
    local_neighbor_expand,
    semantic_post_hit_expand,
)

_DEFAULT_CONFIG_ROOT = Path("data/validation/search_graph_acceptance/config")
_DEFAULT_CORPUS_ROOT = Path("data/validation/corpus/papers")
_DEFAULT_OUTPUT_ROOT = Path("data/validation/search_graph_acceptance/output")
_REQUIRED_STAGES = {
    "explicit_filters",
    "lexical_chunk_retrieval",
    "vector_chunk_retrieval",
    "rrf",
    "chunk_id_dedup",
    "final_selection",
    "source_grounded_evidence",
    "synthesis",
}
_FORBIDDEN_SEARCH_STAGES = {
    "citation_expansion",
    "citation_post_hit_expansion",
    "typed_traversal",
    "graph_traversal",
    "entity_claim_search",
    "global_context",
    "confirmed_knowledge_retrieval",
    "subject_about_retrieval",
    "query_scope",
    "profile_mapping",
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _graph_records(graph_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: list[dict[str, Any]] = []
    relations: list[dict[str, Any]] = []
    for path in sorted(graph_root.glob("*.json")):
        payload = _read_json(path)
        nodes.extend(item for item in payload.get("nodes", []) if isinstance(item, dict))
        relations.extend(item for item in payload.get("relations", []) if isinstance(item, dict))
    return nodes, relations


def _contains_any(text: str, snippets: list[str]) -> bool:
    folded = text.casefold()
    return any(str(snippet).casefold() in folded for snippet in snippets)


def _grounded(response: QueryResponse, chunks: dict[str, Any]) -> bool:
    return response.provenance_complete and all(
        item.chunk_id in chunks
        and item.document_id == chunks[item.chunk_id].document_id
        and item.text == chunks[item.chunk_id].text
        for item in response.evidence
    )


def _response_review(
    case: dict[str, Any],
    request: QueryRequest,
    response: QueryResponse,
    *,
    symbolic_by_work: dict[str, str],
    facts_by_id: dict[str, dict[str, Any]],
    all_document_ids: set[str],
) -> dict[str, Any]:
    local_requested = bool(case["expansion"]["local"])
    semantic_requested = bool(case["expansion"]["semantic"])
    source_work_ids = list(
        dict.fromkeys(
            item.source_work_id for item in response.evidence if item.source_work_id is not None
        )
    )
    source_symbols = [symbolic_by_work.get(work_id, work_id) for work_id in source_work_ids]
    expected_facts = [
        facts_by_id[fact_id]
        for fact_id in case.get("expected_fact_ids_any_of", [])
        if fact_id in facts_by_id
    ]
    evidence_text = "\n".join(item.text for item in response.evidence)
    matched_facts = [
        fact["id"]
        for fact in expected_facts
        if _contains_any(evidence_text, list(fact.get("evidence_any_of", [])))
    ]
    expected_any = set(case.get("expected_source_works_any_of", []))
    expected_exact = set(case.get("expected_source_works_exact", []))
    source_set = set(source_symbols)
    source_ok = (
        source_set == expected_exact
        if expected_exact
        else bool(source_set.intersection(expected_any))
    )
    minimum_sources = int(case.get("minimum_source_works", 0))
    source_ok = source_ok and len(source_set) >= minimum_sources
    minimum_external = int(case.get("minimum_external_source_works", 0))
    if minimum_external:
        source_ok = source_ok and len(source_set.intersection(expected_any)) >= minimum_external
    natural_language_unfiltered = (
        not case.get("assert_natural_language_work_name_does_not_hard_filter")
        or set(response.trace.applied_document_ids) == all_document_ids
    )
    architecture_ok = (
        _REQUIRED_STAGES.issubset(response.stages)
        and not _FORBIDDEN_SEARCH_STAGES.intersection(response.stages)
        and ("semantic_relation_expansion" in response.stages) == semantic_requested
        and ("local_post_hit_expansion" in response.stages) == local_requested
    )
    local_new = response.trace.local_new_chunk_ids
    semantic_new = response.trace.semantic_new_chunk_ids
    new_ids = list(dict.fromkeys([*local_new, *semantic_new]))
    second_candidates = response.trace.second_rerank_candidate_ids
    entered_second_rerank = [item for item in new_ids if item in second_candidates]
    second_rerank_executed = "second_rerank" in response.stages
    if local_requested:
        expansion_status = (
            "PASS"
            if local_new
            and set(local_new).issubset(entered_second_rerank)
            and second_rerank_executed
            else "FAIL"
        )
    elif semantic_requested:
        if semantic_new:
            expansion_status = (
                "PASS"
                if set(semantic_new).issubset(entered_second_rerank)
                and second_rerank_executed
                else "FAIL"
            )
        elif response.trace.semantic_expanded_chunk_ids:
            expansion_status = "NO_NEW_CHUNK"
        else:
            expansion_status = "NO_CASE"
    else:
        expansion_status = "NOT_REQUESTED"
    passed = (
        source_ok
        and bool(matched_facts)
        and natural_language_unfiltered
        and architecture_ok
        and bool(response.answer.strip())
        and expansion_status != "FAIL"
    )
    status = "PASS" if passed else ("FAIL" if case.get("hard") else "WARN")
    return {
        "id": case["id"],
        "mode": case["mode"],
        "query": case["query"],
        "explicit_filters": request.model_dump(
            mode="json",
            include={"document_ids", "work_ids"},
            exclude_none=True,
        ),
        "first_stage_top_chunk_ids": response.trace.first_stage_chunk_ids,
        "first_reranked_chunk_ids": response.trace.first_reranked_chunk_ids,
        "first_reranked_chunk_count": len(response.trace.first_reranked_chunk_ids),
        "local_expanded_chunk_ids": response.trace.local_expanded_chunk_ids,
        "local_expanded_chunk_count": len(response.trace.local_expanded_chunk_ids),
        "local_new_chunk_ids": local_new,
        "local_new_chunk_count": len(local_new),
        "semantic_expanded_chunk_ids": response.trace.semantic_expanded_chunk_ids,
        "semantic_expanded_chunk_count": len(response.trace.semantic_expanded_chunk_ids),
        "semantic_new_chunk_ids": semantic_new,
        "semantic_new_chunk_count": len(semantic_new),
        "second_rerank_executed": second_rerank_executed,
        "second_rerank_candidate_ids": second_candidates,
        "second_rerank_candidate_count": len(second_candidates),
        "expanded_chunk_ids_entered_second_rerank": entered_second_rerank,
        "expansion_integration_status": expansion_status,
        "final_evidence_chunk_ids": [item.chunk_id for item in response.evidence],
        "source_work_ids": source_work_ids,
        "source_work_symbols": source_symbols,
        "matched_ground_truth_fact_ids": matched_facts,
        "source_expectation_met": source_ok,
        "natural_language_did_not_hard_filter": natural_language_unfiltered,
        "architecture_ok": architecture_ok,
        "stages": response.stages,
        "answer": response.answer,
        "status": status,
        "hard": bool(case.get("hard")),
    }


def _find_anchor_chunk(
    corpus: CorpusView,
    work_id: str,
    anchors: list[str],
) -> Any | None:
    document_ids = corpus.document_ids_for_works({work_id})
    return next(
        (
            chunk
            for chunk in sorted(corpus.chunks.values(), key=lambda item: item.order)
            if chunk.document_id in document_ids
            and _contains_any(
                "\n".join(filter(None, [chunk.text, chunk.retrieval_text])),
                anchors,
            )
        ),
        None,
    )


def _local_probes(
    probes: list[dict[str, Any]],
    corpus: CorpusView,
    work_by_symbol: dict[str, str],
    all_document_ids: set[str],
) -> dict[str, Any]:
    reviews: list[dict[str, Any]] = []
    for probe in probes:
        anchor = _find_anchor_chunk(
            corpus,
            work_by_symbol[probe["work"]],
            list(probe["seed_anchor_any_of"]),
        )
        if anchor is None:
            reviews.append(
                {
                    **probe,
                    "status": "FAIL",
                    "boundary_guard_status": "FAIL",
                    "error": "seed not found",
                }
            )
            continue
        seed = corpus.candidate_for_chunk(
            anchor.id,
            channel="validation_seed",
            score=1.0,
        )
        expanded = local_neighbor_expand(
            corpus,
            [seed],
            document_ids=all_document_ids,
        )
        adjacent_ids = [
            chunk_id
            for chunk_id in (anchor.previous_chunk_id, anchor.next_chunk_id)
            if chunk_id is not None and chunk_id in corpus.chunks
        ]
        eligible_ids = [
            chunk_id
            for chunk_id in adjacent_ids
            if corpus.chunks[chunk_id].document_id == anchor.document_id
            and corpus.chunks[chunk_id].document_region == anchor.document_region
            and corpus.chunks[chunk_id].major_section_id == anchor.major_section_id
        ]
        actual_ids = [item.chunk_id for item in expanded]
        constraints_ok = len(actual_ids) == len(set(actual_ids)) and set(actual_ids) == set(
            eligible_ids
        )
        blocked_ids = [chunk_id for chunk_id in adjacent_ids if chunk_id not in eligible_ids]
        operator_status = (
            "PASS"
            if constraints_ok and eligible_ids
            else ("NO_NEW_CHUNK" if constraints_ok else "FAIL")
        )
        boundary_status = (
            "PASS"
            if constraints_ok and blocked_ids
            else ("NOT_APPLICABLE" if constraints_ok else "FAIL")
        )
        reviews.append(
            {
                "id": probe["id"],
                "seed_chunk_id": anchor.id,
                "adjacent_chunk_ids": adjacent_ids,
                "eligible_chunk_ids": eligible_ids,
                "boundary_blocked_chunk_ids": blocked_ids,
                "boundary_blocked": bool(blocked_ids),
                "expanded_chunk_ids": actual_ids,
                "document_region": anchor.document_region,
                "major_section_id": anchor.major_section_id,
                "note": (
                    "No adjacent Chunk is eligible after the required document/region/"
                    "major-section boundary checks."
                    if not eligible_ids
                    else "Expansion exactly matches the eligible canonical ±1 neighbors."
                ),
                "status": operator_status,
                "boundary_guard_status": boundary_status,
            }
        )
    return {
        "status": (
            "PASS"
            if any(item["status"] == "PASS" for item in reviews)
            and not any(item["status"] == "FAIL" for item in reviews)
            else "FAIL"
        ),
        "boundary_guard_status": (
            "PASS"
            if any(item["boundary_guard_status"] == "PASS" for item in reviews)
            and not any(item["boundary_guard_status"] == "FAIL" for item in reviews)
            else "FAIL"
        ),
        "probes": reviews,
    }


async def _semantic_probes(
    probes: list[dict[str, Any]],
    application: Any,
    corpus: CorpusView,
    work_by_symbol: dict[str, str],
    all_document_ids: set[str],
    dataset_name: str,
) -> dict[str, Any]:
    reviews: list[dict[str, Any]] = []
    for probe in probes:
        anchor = _find_anchor_chunk(
            corpus,
            work_by_symbol[probe["work"]],
            list(probe["seed_anchor_any_of"]),
        )
        if anchor is None:
            reviews.append({**probe, "status": "NO_CASE", "error": "seed not found"})
            continue
        seed = corpus.candidate_for_chunk(
            anchor.id,
            channel="validation_seed",
            score=1.0,
        )
        try:
            expanded = await semantic_post_hit_expand(
                application.knowledge_pipeline.compat,
                corpus,
                [seed],
                dataset_name=dataset_name,
                document_ids=all_document_ids,
                limit=200,
            )
            valid = all(
                item.chunk_id in corpus.chunks
                and set(item.relation_types).issubset(
                    {relation.value for relation in SEMANTIC_RELATION_TYPES}
                )
                and item.text == corpus.chunks[item.chunk_id].text
                for item in expanded
            )
            new_ids = [item.chunk_id for item in expanded if item.chunk_id != anchor.id]
            if not valid:
                status = "FAIL"
            elif new_ids:
                status = "PASS"
            elif expanded:
                status = "NO_NEW_CHUNK"
            else:
                status = "NO_CASE"
            reviews.append(
                {
                    "id": probe["id"],
                    "seed_chunk_id": anchor.id,
                    "expanded_chunk_ids": [item.chunk_id for item in expanded],
                    "new_chunk_ids": new_ids,
                    "relation_types": sorted(
                        {relation for item in expanded for relation in item.relation_types}
                    ),
                    "derived_from_ids": list(
                        dict.fromkeys(
                            derived for item in expanded for derived in item.derived_from_ids
                        )
                    ),
                    "dataset_name": dataset_name,
                    "performance_trace": dict(
                        application.knowledge_pipeline.compat.last_semantic_relation_trace
                    ),
                    "status": status,
                }
            )
        except Exception as exc:  # noqa: BLE001 - acceptance records boundary failure
            reviews.append(
                {
                    "id": probe["id"],
                    "seed_chunk_id": anchor.id,
                    "status": "FAIL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "status": "FAIL" if any(item["status"] == "FAIL" for item in reviews) else "PASS",
        "probes": reviews,
    }


async def _citation_checks(
    ground_truth: dict[str, Any],
    relations: list[dict[str, Any]],
    application: Any,
    corpus: CorpusView,
    work_by_symbol: dict[str, str],
    symbolic_by_work: dict[str, str],
    dataset_name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = ground_truth["citation_edges_within_corpus"]
    forbidden = ground_truth["forbidden_reverse_edges_within_corpus"]
    contexts = {
        (item["source_work"], item["target_work"]): item
        for item in ground_truth["citation_contexts"]
    }
    cites = [item for item in relations if item.get("relation_type") == RelationType.CITES]
    edge_reviews: list[dict[str, Any]] = []
    incoming_reviews: list[dict[str, Any]] = []
    for item in expected:
        source_id = work_by_symbol[item["source"]]
        target_id = work_by_symbol[item["target"]]
        matching = [
            edge
            for edge in cites
            if edge.get("source_id") == source_id and edge.get("target_id") == target_id
        ]
        chunk_ids = list(
            dict.fromkeys(
                chunk_id for edge in matching for chunk_id in edge.get("source_chunk_ids", [])
            )
        )
        context = contexts.get((item["source"], item["target"]), {})
        valid_chunks = [
            chunk_id
            for chunk_id in chunk_ids
            if chunk_id in corpus.chunks
            and corpus.work_id_by_document.get(corpus.chunks[chunk_id].document_id) == source_id
            and str(corpus.chunks[chunk_id].document_region).casefold() != "references"
            and target_id in corpus.cited_work_ids_by_chunk.get(chunk_id, set())
        ]
        context_matched = any(
            _contains_any(
                "\n".join(
                    filter(
                        None,
                        [
                            corpus.chunks[chunk_id].text,
                            corpus.chunks[chunk_id].retrieval_text,
                        ],
                    )
                ),
                list(context.get("evidence_any_of", [])),
            )
            for chunk_id in valid_chunks
        )
        edge_reviews.append(
            {
                "source": item["source"],
                "source_work_id": source_id,
                "relation_type": "CITES",
                "target": item["target"],
                "target_work_id": target_id,
                "source_chunk_ids": chunk_ids,
                "valid_body_source_chunk_ids": valid_chunks,
                "matched_context_id": context.get("id"),
                "context_matched": context_matched,
                "status": ("PASS" if matching and valid_chunks and context_matched else "FAIL"),
            }
        )
        incoming = await application.knowledge_pipeline.compat.incoming_typed_relations(
            [target_id],
            dataset_name=dataset_name,
            relation_type=RelationType.CITES.value,
            limit=200,
        )
        incoming_match = next(
            (
                relation
                for relation in incoming
                if relation.source_canonical_id == source_id
                and relation.target_canonical_id == target_id
            ),
            None,
        )
        same_chunks = (
            incoming_match is not None
            and set(incoming_match.source_chunk_ids) == set(chunk_ids)
            and bool(chunk_ids)
        )
        incoming_reviews.append(
            {
                "target": item["target"],
                "target_work_id": target_id,
                "incoming_source": (
                    symbolic_by_work.get(
                        incoming_match.source_canonical_id,
                        incoming_match.source_canonical_id,
                    )
                    if incoming_match is not None
                    else None
                ),
                "incoming_source_work_id": (
                    incoming_match.source_canonical_id if incoming_match is not None else None
                ),
                "source_chunk_ids": (
                    list(incoming_match.source_chunk_ids) if incoming_match is not None else []
                ),
                "matches_outgoing_source_chunk_ids": same_chunks,
                "status": "PASS" if same_chunks else "FAIL",
            }
        )
    forbidden_reviews = [
        {
            **item,
            "status": (
                "PASS"
                if not any(
                    edge.get("source_id") == work_by_symbol[item["source"]]
                    and edge.get("target_id") == work_by_symbol[item["target"]]
                    for edge in cites
                )
                else "FAIL"
            ),
        }
        for item in forbidden
    ]
    citation_status = (
        "PASS"
        if all(item["status"] == "PASS" for item in [*edge_reviews, *forbidden_reviews])
        else "FAIL"
    )
    incoming_status = (
        "PASS" if all(item["status"] == "PASS" for item in incoming_reviews) else "FAIL"
    )
    return (
        {
            "status": citation_status,
            "edges": edge_reviews,
            "forbidden_reverse_edges": forbidden_reviews,
            "trace_example": edge_reviews[0] if edge_reviews else None,
        },
        {
            "status": incoming_status,
            "queries": incoming_reviews,
            "trace_example": incoming_reviews[0] if incoming_reviews else None,
        },
    )


def _element_payload(element: Any, bundle: Any) -> dict[str, Any]:
    elements = {item.id: item for item in bundle.elements}
    return {
        "element_id": element.id,
        "element_type": element.element_type.value,
        "text": element.text,
        "markdown": element.markdown,
        "latex": element.latex,
        "html": element.html,
        "asset_path": str(element.asset_path) if element.asset_path else None,
        "page": element.page,
        "bounding_box": element.bounding_box,
        "source_span": (
            element.source_span.model_dump(mode="json") if element.source_span is not None else None
        ),
        "caption_elements": [
            elements[item].model_dump(mode="json")
            for item in element.caption_element_ids
            if item in elements
        ],
        "footnote_elements": [
            elements[item].model_dump(mode="json")
            for item in element.footnote_element_ids
            if item in elements
        ],
    }


def _structure_checks(
    probes: list[dict[str, Any]],
    corpus: CorpusView,
    work_by_symbol: dict[str, str],
) -> dict[str, Any]:
    reviews: list[dict[str, Any]] = []
    for probe in probes:
        work_id = work_by_symbol[probe["work"]]
        document_ids = corpus.document_ids_for_works({work_id})
        expected_types = set(probe["expected_element_type_any_of"])
        matches: list[tuple[Any, Any, Any, Any]] = []
        for document_id in document_ids:
            bundle = corpus.bundles[document_id]
            elements = {element.id: element for element in bundle.elements}
            for chunk in corpus.chunks.values():
                if chunk.document_id != document_id:
                    continue
                anchor_matches = _contains_any(
                    "\n".join(filter(None, [chunk.text, chunk.retrieval_text])),
                    [probe["anchor"]],
                )
                for element_id in chunk.element_ids:
                    source_element = elements.get(element_id)
                    if source_element is None:
                        continue
                    element = (
                        elements.get(source_element.parent_element_id)
                        if source_element.parent_element_id is not None
                        else source_element
                    )
                    if element is None:
                        continue
                    element_page = element.page or (
                        element.source_span.page if element.source_span is not None else None
                    )
                    type_matches = element.element_type.value in expected_types
                    page_matches = element_page == probe["pdf_page"]
                    source_element_anchor = _contains_any(
                        "\n".join(
                            filter(
                                None,
                                [
                                    source_element.text,
                                    source_element.markdown,
                                    source_element.latex,
                                    source_element.html,
                                ],
                            )
                        ),
                        [probe["anchor"]],
                    )
                    element_anchor = _contains_any(
                        "\n".join(
                            filter(
                                None,
                                [element.text, element.markdown, element.latex, element.html],
                            )
                        ),
                        [probe["anchor"]],
                    )
                    if type_matches and page_matches and (
                        anchor_matches or source_element_anchor or element_anchor
                    ):
                        matches.append((chunk, source_element, element, bundle))
        if matches:
            chunk, source_element, element, bundle = matches[0]
            containment_path = [chunk.id, source_element.id]
            if element.id != source_element.id:
                containment_path.append(element.id)
            reviews.append(
                {
                    "id": probe["id"],
                    "work": probe["work"],
                    "chunk_id": chunk.id,
                    "section_id": chunk.section_id,
                    "section_path": chunk.section_path,
                    "chunk_page_start": chunk.page_start,
                    "chunk_page_end": chunk.page_end,
                    "chunk_element_ids": chunk.element_ids,
                    "source_containment_path": containment_path,
                    "source_element": _element_payload(source_element, bundle),
                    "element": _element_payload(element, bundle),
                    "required": probe["required"],
                    "status": "PASS",
                }
            )
        else:
            reviews.append(
                {
                    "id": probe["id"],
                    "work": probe["work"],
                    "required": probe["required"],
                    "status": "FAIL" if probe["required"] else "NO_CASE",
                    "error": "No matching Chunk to source Element provenance path.",
                }
            )
    return {
        "status": "PASS"
        if all(item["status"] == "PASS" for item in reviews if item["required"])
        else "FAIL",
        "probes": reviews,
        "trace_example": next(
            (item for item in reviews if item["status"] == "PASS"),
            None,
        ),
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Search / Cognee Graph Acceptance",
        "",
        f"Overall: **{report['overall_status']}**",
        "",
        f"PDF to LLM completed: **{report['pipeline_completed_pdf_to_llm']}**",
        "",
        "## Acceptance status",
        "",
    ]
    for key in (
        "default_search",
        "explicit_filter",
        "claim_off",
        "local_expansion",
        "local_boundary_guard",
        "semantic_relation_expansion",
        "citation_provenance",
        "incoming_cites_query",
        "source_grounding",
        "structure_provenance",
        "expansion_requires_reranker",
        "vector_index_scope",
    ):
        lines.append(f"- {key}: **{report[key]['status']}**")
    lines.extend(["", "## Query cases", ""])
    for review in report["queries"]:
        lines.extend(
            [
                f"### {review['id']}",
                "",
                f"- Status: **{review['status']}**",
                f"- Query: {review['query']}",
                "- Explicit filters: " + json.dumps(review["explicit_filters"], ensure_ascii=False),
                "- First-stage Chunks: " + ", ".join(review["first_stage_top_chunk_ids"][:12]),
                "- First-reranked Chunks: " + ", ".join(review["first_reranked_chunk_ids"][:12]),
                (
                    "- Local expanded/new: "
                    f"{review['local_expanded_chunk_count']}/{review['local_new_chunk_count']}"
                ),
                (
                    "- Semantic expanded/new: "
                    f"{review['semantic_expanded_chunk_count']}/"
                    f"{review['semantic_new_chunk_count']}"
                ),
                "- Second-rerank candidates: "
                + ", ".join(review["second_rerank_candidate_ids"][:12]),
                "- Expanded Chunks entering second rerank: "
                + ", ".join(review["expanded_chunk_ids_entered_second_rerank"]),
                f"- Expansion integration: {review['expansion_integration_status']}",
                "- Final Evidence Chunks: " + ", ".join(review["final_evidence_chunk_ids"]),
                "- Source Works: " + ", ".join(review["source_work_symbols"]),
                "- Matched facts: " + ", ".join(review["matched_ground_truth_fact_ids"]),
                "",
            ]
        )
    lines.extend(
        [
            "## Citation trace",
            "",
            "    "
            + json.dumps(
                report["citation_provenance"]["trace_example"],
                ensure_ascii=False,
            ),
            "",
            "## Incoming CITES trace",
            "",
            "    "
            + json.dumps(
                report["incoming_cites_query"]["trace_example"],
                ensure_ascii=False,
            ),
            "",
            "## Structure trace",
            "",
            "    "
            + json.dumps(
                report["structure_provenance"]["trace_example"],
                ensure_ascii=False,
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _package_review(output_root: Path) -> Path:
    result_path = output_root / "result.zip"
    review_paths = [
        output_root / "acceptance.json",
        output_root / "acceptance.md",
        *sorted((output_root / "review").glob("*.json")),
    ]
    with zipfile.ZipFile(
        result_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in review_paths:
            if path.is_file():
                archive.write(path, path.relative_to(output_root))
    return result_path


async def run(args: argparse.Namespace) -> dict[str, Any]:
    config_root = args.acceptance_config.resolve()
    output_root = args.output.resolve()
    runtime_root = output_root / "runtime"
    if args.rebuild and runtime_root.exists():
        shutil.rmtree(runtime_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rebuild_review_path = output_root / "review" / "derived_rebuild.json"
    previous_acceptance_path = output_root / "acceptance.json"
    previous_acceptance = (
        _read_json(previous_acceptance_path) if previous_acceptance_path.is_file() else {}
    )
    retained_rebuild_review = (
        _read_json(rebuild_review_path)
        if rebuild_review_path.is_file()
        else previous_acceptance.get("derived_rebuild")
    )

    corpus_spec = _read_json(config_root / "corpus_spec.json")
    papers_config = _read_json(config_root / "papers.json")
    queries_config = _read_json(config_root / "queries.json")
    ground_truth = _read_json(config_root / "ground_truth.json")
    paper_by_symbol = {item["id"]: item for item in corpus_spec["papers"]}
    requested_stems = set(papers_config["papers"])
    ingest_symbols = [
        symbol
        for symbol in corpus_spec["recommended_ingest_order"]
        if paper_by_symbol[symbol]["paper_id"] in requested_stems
    ]
    if len(ingest_symbols) != 4:
        raise RuntimeError("Acceptance config must resolve exactly four papers.")

    base = load_settings(args.config)
    settings = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={"directory": runtime_root, "dataset": args.dataset}
            ),
            "ingestion": base.ingestion.model_copy(update={"claim_enrichment_enabled": False}),
        }
    )
    application = create_application(settings)
    await application.start()
    try:
        ingestion_results: dict[str, Any] = {}
        ingestion_seconds: dict[str, float] = {}
        work_by_symbol: dict[str, str] = {}
        document_by_symbol: dict[str, str] = {}
        retained_by_filename = {
            application.registry.get_source(
                bundle.document.source_file_id
            ).original_filename: bundle
            for bundle in application.canonical_repository.list_active_bundles()
            if bundle.snapshot.dataset_id == args.dataset
        }
        for symbol in ingest_symbols:
            paper = paper_by_symbol[symbol]
            pdf = args.corpus.resolve() / f"{paper['paper_id']}.pdf"
            if not pdf.is_file():
                raise RuntimeError(f"Authoritative corpus PDF is missing: {pdf}")
            retained = retained_by_filename.get(pdf.name)
            if retained is None:
                started = time.perf_counter()
                result = await application.services.ingestion.ingest_pdf_to_knowledge(
                    pdf,
                    dataset=args.dataset,
                )
                ingestion_seconds[symbol] = round(time.perf_counter() - started, 3)
                ingestion_results[symbol] = result
                document_id = result.canonical_result.canonical.document.id
            else:
                ingestion_seconds[symbol] = 0.0
                ingestion_results[symbol] = retained
                document_id = retained.document.id
            work = application.scholarly_registry.work_for_document(document_id)
            if work is None:
                raise RuntimeError(f"No runtime Work for {symbol}: {document_id}")
            work_by_symbol[symbol] = work.id
            document_by_symbol[symbol] = document_id

        rebuild_report = None
        if args.rebuild_derived:
            rebuild_report = await application.services.rebuilder.rebuild()
            retained_rebuild_review = rebuild_report.public_dict()
            _write_json(rebuild_review_path, retained_rebuild_review)

        symbolic_by_work = {work_id: symbol for symbol, work_id in work_by_symbol.items()}
        corpus = CorpusView.load(
            application.paths,
            application.canonical_repository,
            application.registry,
            application.scholarly_registry,
        )
        all_document_ids = set(document_by_symbol.values())
        facts_by_id = {item["id"]: item for item in ground_truth["retrieval_facts"]}
        query_reviews: list[dict[str, Any]] = []
        responses: list[QueryResponse] = []
        for case in queries_config["cases"]:
            filters = case.get("filters", {})
            work_ids = [work_by_symbol[symbol] for symbol in filters.get("work_ids", [])] or None
            request = QueryRequest(
                query=case["query"],
                dataset=args.dataset,
                top_k=case.get("top_k"),
                work_ids=work_ids,
                expand_context=bool(case["expansion"]["local"]),
                expand_graph=bool(case["expansion"]["semantic"]),
            )
            retrieval_service = application.services.retrieval
            original_config = retrieval_service.config
            acceptance_pool_size = case.get("acceptance_candidate_pool_size")
            if acceptance_pool_size is not None:
                retrieval_service.config = original_config.model_copy(
                    update={
                        "retrieval": original_config.retrieval.model_copy(
                            update={"candidate_pool_size": int(acceptance_pool_size)}
                        )
                    }
                )
            try:
                response = await retrieval_service.query(request)
            finally:
                retrieval_service.config = original_config
            responses.append(response)
            review = _response_review(
                case,
                request,
                response,
                symbolic_by_work=symbolic_by_work,
                facts_by_id=facts_by_id,
                all_document_ids=all_document_ids,
            )
            review["acceptance_candidate_pool_size"] = acceptance_pool_size
            if case["expansion"]["semantic"]:
                review["semantic_performance_trace"] = dict(
                    application.knowledge_pipeline.compat.last_semantic_relation_trace
                )
            query_reviews.append(review)

        reranker_rejections: list[dict[str, Any]] = []
        retrieval_service = application.services.retrieval
        original_config = retrieval_service.config
        retrieval_service.config = original_config.model_copy(
            update={
                "retrieval": original_config.retrieval.model_copy(
                    update={"rerank_enabled": False}
                )
            }
        )
        try:
            for expansion_field in ("expand_context", "expand_graph"):
                try:
                    await retrieval_service.query(
                        QueryRequest(query="reranker invariant", **{expansion_field: True})
                    )
                except ConfigurationError as exc:
                    reranker_rejections.append(
                        {
                            "request": expansion_field,
                            "error_code": exc.code,
                            "message": exc.message,
                            "status": "PASS",
                        }
                    )
                else:
                    reranker_rejections.append(
                        {"request": expansion_field, "status": "FAIL"}
                    )
        finally:
            retrieval_service.config = original_config
        reranker_invariant = {
            "status": (
                "PASS"
                if len(reranker_rejections) == 2
                and all(item["status"] == "PASS" for item in reranker_rejections)
                else "FAIL"
            ),
            "rule": "Expansion requests are rejected unless reranking is enabled.",
            "cases": reranker_rejections,
        }

        nodes, relations = _graph_records(application.paths.cognee / "graphs")
        claim_count = sum(item.get("__type__") == "ClaimDataPoint" for item in nodes)
        about_count = sum(item.get("relation_type") == RelationType.ABOUT for item in relations)
        triplet_count = sum(item.get("__type__") == "TripletDataPoint" for item in nodes)
        local = _local_probes(
            ground_truth["operator_probes"]["local"],
            corpus,
            work_by_symbol,
            all_document_ids,
        )
        semantic = await _semantic_probes(
            ground_truth["operator_probes"]["semantic"],
            application,
            corpus,
            work_by_symbol,
            all_document_ids,
            args.dataset,
        )
        citation, incoming = await _citation_checks(
            ground_truth,
            relations,
            application,
            corpus,
            work_by_symbol,
            symbolic_by_work,
            args.dataset,
        )
        structure = _structure_checks(
            ground_truth["structure_probes"],
            corpus,
            work_by_symbol,
        )
        vector_runtime = await application.knowledge_pipeline.compat.vector_status(
            dataset_name=args.dataset
        )
        vector_collections = dict(vector_runtime.get("collections") or {})
        paperos_vector_collections = sorted(
            name for name in vector_collections if "DataPoint_" in name
        )
        vector_scope = {
            "status": (
                "PASS"
                if paperos_vector_collections == ["PaperOSChunkDataPoint_text"]
                else "FAIL"
            ),
            "paperos_collections": paperos_vector_collections,
            "all_runtime_collections": vector_collections,
            "production_consumers": {
                "PaperOSChunkDataPoint_text": [
                    "CogneeSearchAdapter.search_chunks",
                    "semantic_retrieve",
                ]
            },
        }
        local_query_reviews = [
            item for item in query_reviews if item["mode"] == "local_expansion"
        ]
        semantic_query_reviews = [
            item for item in query_reviews if item["mode"] == "semantic_expansion"
        ]
        local_e2e_status = (
            "PASS"
            if local_query_reviews
            and all(
                item["expansion_integration_status"] == "PASS"
                for item in local_query_reviews
            )
            else "FAIL"
        )
        semantic_e2e_statuses = [
            item["expansion_integration_status"] for item in semantic_query_reviews
        ]
        semantic_e2e_status = (
            "FAIL"
            if not semantic_e2e_statuses or "FAIL" in semantic_e2e_statuses
            else (
                "PASS"
                if "PASS" in semantic_e2e_statuses
                else (
                    "NO_NEW_CHUNK"
                    if "NO_NEW_CHUNK" in semantic_e2e_statuses
                    else "NO_CASE"
                )
            )
        )
        local["end_to_end_status"] = local_e2e_status
        local["status"] = (
            "PASS"
            if local["status"] == "PASS" and local_e2e_status == "PASS"
            else "FAIL"
        )
        semantic["end_to_end_status"] = semantic_e2e_status
        if semantic_e2e_status == "FAIL":
            semantic["status"] = "FAIL"
        default_hard = [
            item for item in query_reviews if item["mode"] == "default" and item["hard"]
        ]
        explicit_hard = [
            item for item in query_reviews if item["mode"] == "explicit_filter" and item["hard"]
        ]
        default_status = (
            "PASS"
            if default_hard and all(item["status"] == "PASS" for item in default_hard)
            else "FAIL"
        )
        explicit_status = (
            "PASS"
            if explicit_hard and all(item["status"] == "PASS" for item in explicit_hard)
            else "FAIL"
        )
        grounding_status = (
            "PASS"
            if responses and all(_grounded(item, corpus.chunks) for item in responses)
            else "FAIL"
        )
        pipeline_completed = (
            len(ingestion_results) == 4
            and len(responses) == len(queries_config["cases"])
            and all(response.answer.strip() for response in responses)
        )
        report: dict[str, Any] = {
            "overall_status": "PENDING",
            "pipeline_completed_pdf_to_llm": pipeline_completed,
            "dataset": args.dataset,
            "ingest_order": ingest_symbols,
            "runtime_work_ids": work_by_symbol,
            "runtime_document_ids": document_by_symbol,
            "pdf_to_active_seconds": ingestion_seconds,
            "semantic_relation_types": sorted(item.value for item in SEMANTIC_RELATION_TYPES),
            "default_search": {"status": default_status},
            "explicit_filter": {"status": explicit_status},
            "claim_off": {
                "status": (
                    "PASS"
                    if claim_count == 0 and about_count == 0 and triplet_count == 0
                    else "FAIL"
                ),
                "prompt_name": "semantic_enrichment_without_claims",
                "claim_count": claim_count,
                "about_edge_count": about_count,
                "triplet_node_count": triplet_count,
            },
            "local_expansion": local,
            "local_boundary_guard": {"status": local["boundary_guard_status"]},
            "semantic_relation_expansion": semantic,
            "citation_provenance": citation,
            "incoming_cites_query": incoming,
            "source_grounding": {"status": grounding_status},
            "structure_provenance": structure,
            "expansion_requires_reranker": reranker_invariant,
            "vector_index_scope": vector_scope,
            "derived_rebuild": retained_rebuild_review,
            "queries": query_reviews,
            "counts": {
                "ingested_papers": len(ingestion_results),
                "chunks": len(corpus.chunks),
                "graph_nodes": len(nodes),
                "graph_relations": len(relations),
                "claims": claim_count,
                "about_edges": about_count,
                "triplet_nodes": triplet_count,
                "cites_edges": sum(
                    item.get("relation_type") == RelationType.CITES for item in relations
                ),
            },
            "search_architecture": (
                "Chunk lexical + Chunk vector -> RRF -> chunk_id dedup -> "
                "rerank -> optional local/direct-semantic expansion -> "
                "canonical Evidence -> LLM"
            ),
            "automatic_citation_expansion": False,
        }
        hard_statuses = [
            report[key]["status"]
            for key in (
                "default_search",
                "explicit_filter",
                "claim_off",
                "local_expansion",
                "local_boundary_guard",
                "semantic_relation_expansion",
                "citation_provenance",
                "incoming_cites_query",
                "source_grounding",
                "structure_provenance",
                "expansion_requires_reranker",
                "vector_index_scope",
            )
        ]
        report["overall_status"] = (
            "PASS" if pipeline_completed and "FAIL" not in hard_statuses else "FAIL"
        )
        _write_json(output_root / "acceptance.json", report)
        (output_root / "acceptance.md").write_text(
            _markdown(report),
            encoding="utf-8",
        )
        _write_json(output_root / "review" / "queries.json", query_reviews)
        _write_json(
            output_root / "review" / "citation.json",
            {"outgoing": citation, "incoming": incoming},
        )
        _write_json(output_root / "review" / "structure.json", structure)
        _write_json(
            output_root / "review" / "expansions.json",
            {
                "local": local,
                "local_boundary_guard": report["local_boundary_guard"],
                "semantic": semantic,
                "expansion_requires_reranker": report["expansion_requires_reranker"],
            },
        )
        _write_json(output_root / "review" / "vector_scope.json", vector_scope)
        report["result_zip"] = str(_package_review(output_root))
        _write_json(output_root / "acceptance.json", report)
        _package_review(output_root)
        return report
    finally:
        await application.aclose()


def _failure_report(output_root: Path, exc: Exception) -> dict[str, Any]:
    report = {
        "overall_status": "FAIL",
        "pipeline_completed_pdf_to_llm": False,
        "blocked_stage": "external_pipeline",
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_json(output_root / "acceptance.json", report)
    (output_root / "acceptance.md").write_text(
        "# Search / Cognee Graph Acceptance\n\n"
        "Overall: **FAIL**\n\n"
        f"Error: {report['error_type']}: {report['error']}\n",
        encoding="utf-8",
    )
    _package_review(output_root)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/paperos.toml"))
    parser.add_argument(
        "--acceptance-config",
        type=Path,
        default=_DEFAULT_CONFIG_ROOT,
    )
    parser.add_argument("--corpus", type=Path, default=_DEFAULT_CORPUS_ROOT)
    parser.add_argument("--output", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset", default="search_graph_acceptance")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--rebuild-derived",
        action="store_true",
        help="Rebuild graph/vector/lexical projections while reusing retained enrichment.",
    )
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001 - always persist an acceptance artifact
        report = _failure_report(args.output.resolve(), exc)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["overall_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
