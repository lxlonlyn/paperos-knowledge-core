"""Real PDF→Chunk-first retrieval→LLM acceptance with review artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.application import create_application
from paperos_core.config import load_settings
from paperos_core.retrieval.candidates import QueryRequest, QueryResponse

_DEFAULT_PAPERS = (
    "volume_preserving.pdf",
    "explicit_flows.pdf",
    "nise.pdf",
    "gaussian_splatting.pdf",
)
_QUERY = (
    "How do neural implicit surface methods control geometric evolution, "
    "regularity, and shape preservation?"
)


def _graph_stats(graph_root: Path) -> tuple[dict[str, int], list[dict[str, Any]]]:
    counts = {"claims": 0, "about": 0, "cites": 0}
    relations: list[dict[str, Any]] = []
    for path in sorted(graph_root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts["claims"] += sum(
            node.get("__type__") == "ClaimDataPoint"
            for node in payload.get("nodes", [])
        )
        for relation in payload.get("relations", []):
            relations.append(relation)
            kind = relation.get("relation_type")
            counts["about"] += kind == "ABOUT"
            counts["cites"] += kind == "CITES"
    return counts, relations


def _grounded(response: QueryResponse, chunks: dict[str, Any]) -> bool:
    return response.provenance_complete and all(
        evidence.chunk_id in chunks
        and evidence.document_id == chunks[evidence.chunk_id].document_id
        and evidence.text == chunks[evidence.chunk_id].text
        for evidence in response.evidence
    )


def _response_review(response: QueryResponse) -> dict[str, Any]:
    return {
        "query": response.query,
        "stages": response.stages,
        "channels": response.channels_used,
        "top_retrieved_chunk_ids": response.trace.first_reranked_chunk_ids[:12],
        "final_evidence": [item.model_dump(mode="json") for item in response.evidence],
        "expansion_trace": response.trace.model_dump(mode="json"),
        "answer": response.answer,
    }


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Chunk-first Search Acceptance",
        "",
        f"Overall: **{report['overall_status']}**",
        "",
        "## Counts",
        "",
        f"- Ingested papers: {report['counts']['ingested_papers']}",
        f"- Canonical chunks: {report['counts']['chunks']}",
        f"- Claims: {report['counts']['claims']}",
        f"- ABOUT edges: {report['counts']['about_edges']}",
        f"- CITES edges: {report['counts']['cites_edges']}",
        "",
        "## Status",
        "",
    ]
    for key, value in report["status"].items():
        lines.append(f"- {key.replace('_', ' ')}: **{value}**")
    lines.extend(["", "## Citation anchor", "", "```json"])
    lines.append(json.dumps(report["citation_example"], ensure_ascii=False, indent=2))
    lines.extend(["```", "", "## Queries", ""])
    for name, review in report["queries"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"Query: {review['query']}",
                "",
                "Top Chunks: " + ", ".join(review["top_retrieved_chunk_ids"][:8]),
                "",
                "```json",
                json.dumps(review["expansion_trace"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


async def run(args: argparse.Namespace) -> dict[str, Any]:
    base = load_settings(args.config)
    output_root = args.output.resolve()
    runtime_root = output_root / "runtime"
    if args.rebuild and runtime_root.exists():
        shutil.rmtree(runtime_root)
    output_root.mkdir(parents=True, exist_ok=True)
    settings = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={"directory": runtime_root, "dataset": args.dataset}
            ),
            "ingestion": base.ingestion.model_copy(
                update={"claim_enrichment_enabled": False}
            ),
        }
    )
    application = create_application(settings)
    await application.start()
    try:
        corpus_root = args.corpus.resolve()
        pdfs = [corpus_root / name for name in args.papers]
        ingestion_results = []
        for pdf in pdfs:
            if not pdf.is_file():
                raise RuntimeError(f"Authoritative corpus PDF is missing: {pdf}")
            ingestion_results.append(
                await application.services.ingestion.ingest_pdf_to_knowledge(
                    pdf, dataset=args.dataset
                )
            )

        corpus = application.services.retrieval.canonical_repository
        projections = {
            result.canonical_result.canonical.snapshot.id:
            corpus.get_chunk_projection(
                result.canonical_result.canonical.snapshot.id
            )
            for result in ingestion_results
        }
        chunks = {
            chunk.id: chunk
            for projection in projections.values()
            for chunk in projection.chunks
        }
        first_document_id = ingestion_results[0].canonical_result.canonical.document.id
        default = await application.services.retrieval.query(QueryRequest(query=_QUERY))
        explicit = await application.services.retrieval.query(
            QueryRequest(query=_QUERY, document_ids=[first_document_id])
        )
        local = await application.services.retrieval.query(
            QueryRequest(query=_QUERY, expand_context=True)
        )

        citation_seed = next(
            (chunk for chunk in chunks.values() if chunk.citation_reference_entry_ids),
            None,
        )
        graph_query = (
            " ".join((citation_seed.retrieval_text or citation_seed.text).split()[:24])
            if citation_seed is not None
            else _QUERY
        )
        graph = await application.services.retrieval.query(
            QueryRequest(query=graph_query, expand_graph=True)
        )

        counts, relations = _graph_stats(application.paths.cognee / "graphs")
        citation_example: dict[str, Any] | None = None
        for chunk in chunks.values():
            for reference_id in chunk.citation_reference_entry_ids:
                work = application.scholarly_registry.work_for_reference(reference_id)
                source_work = application.scholarly_registry.work_for_document(
                    chunk.document_id
                )
                if work is None or source_work is None:
                    continue
                edge = next(
                    (
                        relation
                        for relation in relations
                        if relation.get("relation_type") == "CITES"
                        and relation.get("source_id") == source_work.id
                        and relation.get("target_id") == work.id
                        and chunk.id in relation.get("source_chunk_ids", [])
                    ),
                    None,
                )
                if edge is not None:
                    citation_example = {
                        "chunk_id": chunk.id,
                        "document_region": chunk.document_region,
                        "reference_entry_id": reference_id,
                        "source_work_id": source_work.id,
                        "cited_work_id": work.id,
                        "cites_source_chunk_ids": edge["source_chunk_ids"],
                    }
                    break
            if citation_example is not None:
                break

        responses = [default, explicit, local, graph]
        forbidden = {
            "entity_claim_search",
            "typed_traversal",
            "global_context",
            "confirmed_knowledge_retrieval",
            "subject_about_retrieval",
            "cognee_recall",
            "profile_mapping",
        }
        default_ok = (
            {"lexical", "vector"}.issuperset(default.channels_used)
            and not forbidden.intersection(default.stages)
            and "rrf" in default.stages
            and "chunk_id_dedup" in default.stages
            and "first_rerank" in default.stages
            and bool(default.answer)
        )
        explicit_ok = bool(explicit.evidence) and all(
            item.document_id == first_document_id for item in explicit.evidence
        )
        local_ok = bool(local.trace.local_expanded_chunk_ids)
        graph_new = (
            set(graph.trace.citation_expanded_chunk_ids)
            | set(graph.trace.graph_expanded_chunk_ids)
        ) - set(graph.trace.first_reranked_chunk_ids)
        graph_status = "PASS" if graph_new else "NO_CASE"
        grounding_ok = all(_grounded(response, chunks) for response in responses)
        claim_off_ok = counts["claims"] == 0 and counts["about"] == 0
        citation_ok = (
            citation_example is not None
            and citation_example["document_region"] != "REFERENCES"
        )
        status = {
            "default_query": "PASS" if default_ok else "FAIL",
            "explicit_filter": "PASS" if explicit_ok else "FAIL",
            "claim_off": "PASS" if claim_off_ok else "FAIL",
            "citation_anchor": "PASS" if citation_ok else "FAIL",
            "local_expansion": "PASS" if local_ok else "FAIL",
            "graph_expansion": graph_status,
            "source_grounding": "PASS" if grounding_ok else "FAIL",
        }
        hard_failures = [value for value in status.values() if value == "FAIL"]
        report = {
            "overall_status": "PASS" if not hard_failures else "FAIL",
            "pipeline_completed_pdf_to_llm": all(response.answer for response in responses),
            "papers": [pdf.name for pdf in pdfs],
            "counts": {
                "ingested_papers": len(ingestion_results),
                "chunks": len(chunks),
                "claims": counts["claims"],
                "about_edges": counts["about"],
                "cites_edges": counts["cites"],
            },
            "status": status,
            "claim_off": {
                "prompt_name": "semantic_enrichment_without_claims",
                "response_schema": "_SectionExtractionWithoutClaims",
                "claim_count": counts["claims"],
                "about_edge_count": counts["about"],
            },
            "citation_example": citation_example,
            "graph_expansion_note": (
                None
                if graph_new
                else "No effective cross-paper expansion case was present in this run."
            ),
            "queries": {
                "default": _response_review(default),
                "explicit_filter": _response_review(explicit),
                "local_expansion": _response_review(local),
                "graph_expansion": _response_review(graph),
            },
        }
        (output_root / "acceptance.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_root / "acceptance.md").write_text(
            _markdown(report), encoding="utf-8"
        )
        return report
    finally:
        await application.aclose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/paperos.toml"))
    parser.add_argument(
        "--corpus", type=Path, default=Path("data/validation/corpus/papers")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/validation/retrieval/output")
    )
    parser.add_argument("--dataset", default="validation_chunk_first")
    parser.add_argument("--papers", nargs="+", default=list(_DEFAULT_PAPERS))
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001 - persist any external-stage failure report
        output_root = args.output.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        report = {
            "overall_status": "FAIL",
            "pipeline_completed_pdf_to_llm": False,
            "blocked_stage": "external_pipeline",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        (output_root / "acceptance.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (output_root / "acceptance.md").write_text(
            "# Chunk-first Search Acceptance\n\n"
            "Overall: **FAIL**\n\n"
            f"Blocked stage: `{report['blocked_stage']}`\n\n"
            f"Error: `{report['error_type']}: {report['error']}`\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["overall_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
