"""Run the cumulative PaperOS acceptance path using only genuine papers.

This is the project's executable validation entry. It does not use pytest,
mocks, fabricated parser output, precomputed embeddings, or fixed LLM output.
Every run starts from the user-supplied PDF corpus and calls live MinerU,
Cognee's configured LLM/embedding providers, the graph/vector stores, FTS, and
all three PaperOS retrieval profiles.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))
VALIDATION_ROOT_NAME = "validation"
CORPUS_DIRECTORY_NAME = "corpus"
RUNS_DIRECTORY_NAME = "runs"

from paperos_core.application import create_application
from paperos_core.config import RuntimeSettings, load_settings
from paperos_core.errors import PaperOSError
from paperos_core.retrieval.candidates import QueryRequest, QueryResponse


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _file_hashes(roots: list[Path]) -> dict[str, str]:
    return {
        str(path): _sha256(path)
        for root in roots
        for path in root.rglob("*")
        if path.is_file()
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_corpus(data_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corpus = data_dir / VALIDATION_ROOT_NAME / CORPUS_DIRECTORY_NAME
    manifest_path = corpus / "manifest.json"
    _require(manifest_path.is_file(), f"Real corpus manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    papers = manifest.get("papers")
    _require(isinstance(papers, list) and papers, "Real corpus contains no papers.")
    for paper in papers:
        pdf = corpus / "pdfs" / paper["pdf_file"]
        _require(pdf.is_file(), f"Real PDF is missing: {pdf}")
        _require(
            _sha256(pdf) == paper["sha256"],
            f"Real PDF checksum mismatch: {pdf}",
        )
    queries: list[dict[str, Any]] = []
    for name in ("truth.jsonl", "associative.jsonl", "comprehensive.jsonl"):
        query_path = corpus / "queries" / name
        _require(query_path.is_file(), f"Real query cases are missing: {query_path}")
        queries.extend(_load_jsonl(query_path))
    _require(queries, "Real corpus contains no query cases.")
    return papers, queries


def _load_expected_cases(data_dir: Path) -> dict[str, dict[str, Any]]:
    expected_root = (
        data_dir / VALIDATION_ROOT_NAME / CORPUS_DIRECTORY_NAME / "expected"
    )
    cases: dict[str, dict[str, Any]] = {}
    for path in sorted(expected_root.glob("*.json")):
        case = json.loads(path.read_text(encoding="utf-8"))
        filename = str(case["pdf_file"])
        _require(filename not in cases, f"Duplicate real expectation: {filename}")
        cases[filename] = case
    _require(cases, f"Real ingestion expectations are missing: {expected_root}")
    return cases


def _normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[\u2010-\u2015\u2212]", "-", value)
    return " ".join(value.split())


def _all_element_text(element: Any) -> str:
    return "\n".join(
        value
        for value in (
            element.text,
            element.raw_text,
            element.markdown,
            element.latex,
            element.html,
        )
        if value
    )


def _canonical_element_text(element: Any) -> str:
    if element.element_type.value == "table":
        return element.markdown or element.text or element.html or ""
    if element.element_type.value == "formula":
        return element.latex or element.text or element.markdown or ""
    return element.text if element.text is not None else (element.markdown or "")


def _validate_real_ingestion(
    *,
    expected: dict[str, Any],
    bundle: Any,
    projection: Any,
    chunk_target_tokens: int,
    cognee_manifest_path: Path,
    index_manifest_path: Path,
) -> dict[str, Any]:
    """Validate live MinerU/Cognee output against one genuine paper."""
    filename = str(expected["pdf_file"])
    document = bundle.document
    expected_document = expected["document"]
    _require(
        _normalized(document.title) == _normalized(expected_document["expected_title"]),
        f"Canonical title mismatch: {filename} / {document.title}",
    )
    _require(document.language == expected_document["language"], f"Language mismatch: {filename}")
    _require(
        document.document_type == expected_document["document_type"],
        f"Document type mismatch: {filename}",
    )

    structure = expected["structure"]
    _require(
        len(bundle.sections) >= structure["minimum_section_count"],
        f"Too few real sections: {filename}",
    )
    _require(
        len(projection.chunks) >= structure["minimum_chunk_count"],
        f"Too few real chunks: {filename}",
    )
    _require(
        len(bundle.references) >= structure["minimum_reference_count"],
        f"Too few real references: {filename}",
    )
    section_titles = [_normalized(section.title) for section in bundle.sections]
    for requirement in structure["required_sections"]:
        title = _normalized(requirement["title"])
        if requirement["match"] == "normalized_exact":
            found = title in section_titles
        else:
            compact_title = title.replace(" ", "")
            found = any(
                compact_title in candidate.replace(" ", "")
                for candidate in section_titles
            )
        _require(found, f"Required real section absent: {filename} / {requirement['title']}")

    element_counts = Counter(element.element_type.value for element in bundle.elements)
    element_requirements = expected["elements"]
    for element_type in element_requirements["must_contain"]:
        _require(element_counts[element_type] > 0, f"Missing {element_type}: {filename}")
    _require(
        element_counts["figure"] >= element_requirements["minimum_figure_count"],
        f"Too few figures: {filename}",
    )
    _require(
        element_counts["formula"] >= element_requirements["minimum_formula_count"],
        f"Too few formulas: {filename}",
    )
    if element_requirements["require_figure_captions"]:
        _require(element_counts["caption"] > 0, f"Figure captions absent: {filename}")
    if element_requirements["require_reference_entries"]:
        _require(bundle.references, f"Reference entries absent: {filename}")

    searchable = _normalized(
        "\n".join(
            [
                document.title,
                document.abstract or "",
                *(_all_element_text(element) for element in bundle.elements),
            ]
        )
    )
    for check in expected["content_checks"]:
        if not check.get("required", True):
            continue
        _require(
            any(_normalized(value) in searchable for value in check["any_of"]),
            f"Required real text absent: {filename} / {check['any_of']}",
        )

    elements = {element.id: element for element in bundle.elements}
    chunks = {chunk.id: chunk for chunk in projection.chunks}
    for chunk in projection.chunks:
        _require(chunk.token_count is not None, f"Chunk token count absent: {chunk.id}")
        _require(
            chunk.token_count <= chunk_target_tokens,
            f"Chunk exceeds token target: {chunk.id} / {chunk.token_count}",
        )
        _require(chunk.spans, f"Chunk spans absent: {chunk.id}")
        _require(
            chunk.element_span_ids == [span.id for span in chunk.spans],
            f"Chunk span IDs diverge: {chunk.id}",
        )
        span_sections: set[str | None] = set()
        for span in chunk.spans:
            element = elements.get(span.element_id)
            if element is None:
                raise RuntimeError(f"Chunk references unknown element: {span.id}")
            source = _canonical_element_text(element)
            _require(
                source[span.character_start_in_element : span.character_end_in_element]
                == span.text,
                f"Element character span is not exact: {span.id}",
            )
            _require(span.token_start < span.token_end, f"Invalid token span: {span.id}")
            span_sections.add(element.section_id)
        non_null_sections = {value for value in span_sections if value is not None}
        _require(
            non_null_sections <= {chunk.section_id},
            f"Chunk crosses section boundary: {chunk.id}",
        )
        for source_chunk_id in chunk.overlap_source_chunk_ids:
            source_chunk = chunks.get(source_chunk_id)
            if source_chunk is None:
                raise RuntimeError(f"Unknown overlap chunk: {source_chunk_id}")
            _require(
                source_chunk.section_id == chunk.section_id,
                f"Overlap crosses section boundary: {chunk.id}",
            )

    cognee_manifest = json.loads(cognee_manifest_path.read_text(encoding="utf-8"))
    mapped_ids = set(cognee_manifest["canonical_to_cognee_id"])
    _require(set(chunks) <= mapped_ids, f"Cognee is missing canonical chunks: {filename}")
    _require(cognee_manifest["node_count"] > 0, f"Cognee has no nodes: {filename}")
    _require(cognee_manifest["relation_count"] > 0, f"Cognee has no relations: {filename}")

    index_manifest = json.loads(index_manifest_path.read_text(encoding="utf-8"))
    projection_ids = set(index_manifest["chunk_projection_ids"])
    lexical_ids = set(index_manifest["lexical_object_ids"])
    _require(projection_ids == set(chunks), f"Chunk projection mismatch: {filename}")
    _require(set(chunks) <= lexical_ids, f"FTS is missing canonical chunks: {filename}")
    searchable_types: set[str] = set()
    for chunk in projection.chunks:
        if chunk.id not in lexical_ids:
            continue
        searchable_types.update(
            elements[element_id].element_type.value
            for element_id in chunk.element_ids
            if element_id in elements
        )
    return {
        "filename": filename,
        "sections": len(bundle.sections),
        "chunks": len(projection.chunks),
        "references": len(bundle.references),
        "element_counts": dict(sorted(element_counts.items())),
        "searchable_element_types": sorted(searchable_types),
    }


def _settings_for_run(
    configured: RuntimeSettings, run_root: Path, dataset: str
) -> RuntimeSettings:
    return configured.model_copy(
        update={
            "data": configured.data.model_copy(
                update={"directory": run_root.resolve(), "dataset": dataset}
            )
        }
    )


def _contains_concept(searchable: str, concept: str) -> bool:
    normalized = concept.casefold()
    aliases = {
        "weak coupling": ("weak coupling", "弱耦合"),
    }
    if any(alias in searchable for alias in aliases.get(normalized, (normalized,))):
        return True
    tokens = re.findall(r"[a-z0-9]+", normalized)
    long_tokens = [token for token in tokens if len(token) >= 4]
    return bool(long_tokens) and all(token[:5] in searchable for token in long_tokens)


def _validate_query(case: dict[str, Any], response: QueryResponse) -> list[str]:
    """Enforce pipeline integrity and report model/retrieval quality separately."""
    quality_warnings: list[str] = []
    _require(response.profile.value == case["profile"], f"Profile mismatch: {case['case_id']}")
    _require(
        set(case.get("required_channels", [])) <= set(response.channels_used),
        f"Missing retrieval channel: {case['case_id']}",
    )
    _require(
        set(case.get("required_stages", [])) <= set(response.stages),
        f"Missing retrieval stage: {case['case_id']}",
    )
    _require(response.provenance_complete, f"Incomplete provenance: {case['case_id']}")
    _require(
        len(response.evidence) == len(response.candidates) > 0,
        f"No evidence-bound candidates: {case['case_id']}",
    )
    _require(
        all(item.chunk_id for item in response.evidence),
        f"Evidence lacks chunk IDs: {case['case_id']}",
    )
    _require(
        any(item.evidence_id in response.answer for item in response.evidence),
        f"Answer lacks evidence citations: {case['case_id']}",
    )
    filenames = {item.source_filename for item in response.evidence}
    missing_documents = sorted(set(case["expected_documents"]) - filenames)
    if missing_documents:
        quality_warnings.append(
            f"{case['case_id']}: expected documents absent from ranked evidence: "
            f"{missing_documents}"
        )
    if response.distinct_documents < case["minimum_distinct_documents"]:
        quality_warnings.append(
            f"{case['case_id']}: document diversity "
            f"{response.distinct_documents} < {case['minimum_distinct_documents']}"
        )
    if case.get("requires_page"):
        _require(
            all(item.page_start is not None for item in response.evidence),
            f"Evidence lacks page coordinates: {case['case_id']}",
        )
    if case.get("requires_graph_relation"):
        _require("graph" in response.channels_used, f"Graph channel absent: {case['case_id']}")
        _require(
            any(
                "graph" in candidate.channels
                or candidate.knowledge_kind == "structured_relation"
                for candidate in response.candidates
            ),
            f"Graph evidence absent: {case['case_id']}",
        )
    searchable = " ".join(
        [response.answer, *(item.text for item in response.evidence)]
    ).casefold()
    for group in case.get("required_evidence_groups", []):
        if not any(term.casefold() in searchable for term in group["any_of"]):
            quality_warnings.append(
                f"{case['case_id']}: expected evidence terms absent: {group['any_of']}"
            )
    for concept in case.get("required_concepts", []):
        if not _contains_concept(searchable, concept):
            quality_warnings.append(
                f"{case['case_id']}: expected concept absent: {concept}"
            )
    return quality_warnings


def _validate_enrichment(path: Path, filename: str) -> None:
    enrichment = json.loads(path.read_text(encoding="utf-8"))
    _require(enrichment["coverage_ratio"] == 1.0, f"Incomplete enrichment: {filename}")
    _require(not enrichment["uncovered_chunk_ids"], f"Uncovered chunks: {filename}")
    _require(enrichment["prompt_sha256"], f"Missing prompt SHA: {filename}")


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    configured = load_settings()
    if args.local_inference_port is not None:
        _require(
            1 <= args.local_inference_port <= 65535,
            "--local-inference-port must be between 1 and 65535.",
        )
        configured = configured.model_copy(
            update={
                "local_inference": configured.local_inference.model_copy(
                    update={"port": args.local_inference_port}
                )
            }
        )
    _require(
        configured.mineru.api_key_value(),
        "mineru.api_key must be configured in config/paperos.toml.",
    )
    papers, queries = _load_corpus(configured.data_dir)
    expected_cases = _load_expected_cases(configured.data_dir)
    _require(
        set(expected_cases) == {str(paper["pdf_file"]) for paper in papers},
        "The real manifest and ingestion expectations describe different papers.",
    )
    run_root = args.run_root.resolve()
    logs = run_root / "logs" / "acceptance"
    logs.mkdir(parents=True, exist_ok=True)
    settings = _settings_for_run(configured, run_root, args.dataset)
    application = create_application(settings)
    local_pid: int | None = None
    local_process: asyncio.subprocess.Process | None = None
    started_at = datetime.now(UTC)
    ingestions: list[dict[str, Any]] = []
    structural_results: list[dict[str, Any]] = []
    responses: list[QueryResponse] = []
    quality_warnings: list[str] = []
    print(f"run_root={run_root}", flush=True)
    print(f"dataset={args.dataset}", flush=True)
    await application.start()
    local_process = application.runtime.local_inference.process
    local_pid = application.runtime.local_inference.pid
    try:
        existing = {
            application.registry.get_source(bundle.document.source_file_id).original_filename: bundle
            for bundle in application.canonical_repository.list_bundles()
        }
        for position, paper in enumerate(papers, 1):
            filename = str(paper["pdf_file"])
            payload: dict[str, Any]
            if args.resume and filename in existing:
                bundle = existing[filename]
                enrichment_path = (
                    application.paths.cognee / "enrichment" / f"{bundle.snapshot.id}.json"
                )
                cognee_manifest = (
                    application.paths.cognee / "manifests" / f"{bundle.snapshot.id}.json"
                )
                index_manifest = (
                    application.paths.indexes / "manifests" / f"{bundle.snapshot.id}.json"
                )
                if all(
                    path.is_file()
                    for path in (enrichment_path, cognee_manifest, index_manifest)
                ):
                    _validate_enrichment(enrichment_path, filename)
                    projection = application.canonical_repository.get_chunk_projection(
                        bundle.snapshot.id
                    )
                    structural_results.append(
                        _validate_real_ingestion(
                            expected=expected_cases[filename],
                            bundle=bundle,
                            projection=projection,
                            chunk_target_tokens=settings.ingestion.chunk_target_tokens,
                            cognee_manifest_path=cognee_manifest,
                            index_manifest_path=index_manifest,
                        )
                    )
                    print(f"ingest {position}/{len(papers)} reused {filename}", flush=True)
                    continue
                print(
                    f"ingest {position}/{len(papers)} resume-knowledge {filename}",
                    flush=True,
                )
                indexing_report, enrichment_path = (
                    await application.knowledge_pipeline.ingest_bundle(bundle)
                )
                projection = application.canonical_repository.get_chunk_projection(
                    bundle.snapshot.id
                )
                parse_run = application.parser_artifacts.get_parse_run(
                    bundle.snapshot.parse_run_id
                )
                payload = {
                    "resumed_snapshot_id": bundle.snapshot.id,
                    "parse_run": parse_run.model_dump(mode="json"),
                    "counts": {"chunks": len(projection.chunks)},
                    "knowledge": {
                        **indexing_report.model_dump(mode="json"),
                        "enrichment_path": str(enrichment_path),
                    },
                }
            else:
                print(f"ingest {position}/{len(papers)} live {filename}", flush=True)
                result = await application.services.ingestion.ingest_pdf_to_knowledge(
                    configured.data_dir
                    / VALIDATION_ROOT_NAME
                    / CORPUS_DIRECTORY_NAME
                    / "pdfs"
                    / filename,
                    dataset=args.dataset,
                )
                payload = result.public_dict()
            parse_run_payload = cast(dict[str, Any], payload["parse_run"])
            counts_payload = cast(dict[str, Any], payload["counts"])
            knowledge = cast(dict[str, Any], payload["knowledge"])
            _require(
                parse_run_payload["provider"] == "mineru_cloud",
                "MinerU provider mismatch.",
            )
            _require(counts_payload["chunks"] > 0, f"No chunks produced for {filename}")
            _require(knowledge["consistency_valid"], f"Index inconsistency for {filename}")
            _validate_enrichment(Path(knowledge["enrichment_path"]), filename)
            projection = application.canonical_repository.get_chunk_projection(
                bundle.snapshot.id
                if args.resume and filename in existing
                else cast(dict[str, Any], payload["canonical_snapshot"])["id"]
            )
            current_bundle = application.canonical_repository.get_bundle(
                projection.snapshot_id
            )
            structural_results.append(
                _validate_real_ingestion(
                    expected=expected_cases[filename],
                    bundle=current_bundle,
                    projection=projection,
                    chunk_target_tokens=settings.ingestion.chunk_target_tokens,
                    cognee_manifest_path=Path(knowledge["cognee_manifest_path"]),
                    index_manifest_path=Path(knowledge["manifest_path"]),
                )
            )
            output_path = logs / f"ingest-{paper['case_id']}.json"
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            ingestions.append(payload)

        bundles = application.canonical_repository.list_bundles()
        active_filenames = {
            application.registry.get_source(bundle.document.source_file_id).original_filename
            for bundle in bundles
        }
        _require(
            {str(paper["pdf_file"]) for paper in papers} <= active_filenames,
            "The cumulative run does not contain every genuine paper.",
        )
        searchable_element_types = {
            element_type
            for result in structural_results
            for element_type in result["searchable_element_types"]
        }
        _require("table" in searchable_element_types, "No genuine table reached FTS5.")
        _require("formula" in searchable_element_types, "No genuine formula reached FTS5.")
        protected = _file_hashes(
            [application.paths.raw, application.paths.parsed, application.paths.canonical]
        )
        for position, case in enumerate(queries, 1):
            print(
                f"query {position}/{len(queries)} {case['profile']} {case['case_id']}",
                flush=True,
            )
            response = await application.services.retrieval.query(
                QueryRequest(query=case["query"], profile=case["profile"])
            )
            (logs / f"query-{case['case_id']}.json").write_text(
                response.model_dump_json(indent=2), encoding="utf-8"
            )
            quality_warnings.extend(_validate_query(case, response))
            responses.append(response)
        _require(
            _file_hashes(
                [application.paths.raw, application.paths.parsed, application.paths.canonical]
            )
            == protected,
            "Retrieval mutated immutable source/canonical evidence.",
        )
        health = await application.services.health.report()
        _require(health["status"] == "healthy", f"Final health is not healthy: {health}")
        acceptance_report: dict[str, Any] = {
            "status": "passed",
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "run_root": str(run_root),
            "dataset": args.dataset,
            "paper_count": len(papers),
            "new_ingestion_count": len(ingestions),
            "structural_results": structural_results,
            "query_count": len(responses),
            "profiles": sorted({response.profile.value for response in responses}),
            "quality_warnings": quality_warnings,
            "local_inference_pid": local_pid,
            "health": health,
        }
        (logs / "acceptance-report.json").write_text(
            json.dumps(acceptance_report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return acceptance_report
    finally:
        await application.aclose()
        if local_process is not None:
            await local_process.wait()
            _require(
                local_process.returncode is not None,
                f"Local inference child process {local_pid} survived shutdown.",
            )
            print(f"local inference process {local_pid} cleaned", flush=True)


def main() -> None:
    configured = load_settings()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    parser = argparse.ArgumentParser(
        description="Run cumulative acceptance against the genuine four-paper corpus."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=(
            configured.data_dir
            / VALIDATION_ROOT_NAME
            / RUNS_DIRECTORY_NAME
            / timestamp
        ),
    )
    parser.add_argument("--dataset", default=f"paperos-real-{timestamp.lower()}")
    parser.add_argument(
        "--local-inference-port",
        type=int,
        help="Override the machine-local inference port for this acceptance run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already ingested genuine papers in the selected run root.",
    )
    args = parser.parse_args()
    try:
        report = asyncio.run(_run(args))
    except PaperOSError as exc:
        print(json.dumps(exc.as_dict(), ensure_ascii=False, indent=2), file=sys.stderr)
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
