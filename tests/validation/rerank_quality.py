"""Task 6B retained-corpus rerank quality benchmark.

The benchmark never ingests PDFs or calls MinerU, enrichment, or synthesis. It
freezes one production first-stage parent pool per query, scores controlled
rerank representations, selects on dev, and only then evaluates holdout.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.application import Application, create_application
from paperos_core.config import load_settings
from paperos_core.domain.canonical import Chunk, RerankSpan
from paperos_core.errors import PaperOSError
from paperos_core.ingestion.chunking import build_chunks
from paperos_core.ingestion.rerank_projection import build_rerank_projection
from paperos_core.ingestion.tokenization import AUTHORITATIVE_CHUNK_TOKENIZER
from paperos_core.retrieval.candidates import Candidate, VectorSearchDiagnostics
from paperos_core.retrieval.corpus import CorpusView
from paperos_core.retrieval.evidence import format_evidence
from paperos_core.retrieval.expansion import local_neighbor_expand
from paperos_core.retrieval.fusion import deduplicate_candidates_by_chunk, weighted_rrf
from paperos_core.retrieval.lexical import lexical_retrieve
from paperos_core.retrieval.semantic import semantic_retrieve

_DEFAULT_CONFIG_ROOT = Path("tests/validation/fixtures/rerank_quality")
_DEFAULT_OUTPUT_ROOT = Path("data/validation/rerank_quality/output")
_RETAINED_ROOT = Path(
    "data/validation/rerank_projection_acceptance/output/runtime"
)
_PAPERS_CONFIG = Path(
    "data/validation/rerank_projection_acceptance/config/papers.json"
)
_CANDIDATE_POOL_SIZE = 40
_SCORE_BATCH_SIZE = 128
_ALLOWED_CUDA_DEVICES = "6"
_BASELINE_ID = "structured_256_384_maxp"

StrategyKind = Literal["full", "legacy", "structured", "hybrid"]
AggregationKind = Literal["maxp", "top2_mean", "rrf"]


@dataclass(frozen=True, slots=True)
class Strategy:
    id: str
    kind: StrategyKind
    aggregation: AggregationKind
    target_tokens: int | None = None
    hard_max_tokens: int | None = None
    overlap_tokens: int = 0
    production_eligible: bool = False

    @property
    def representation_key(self) -> str:
        if self.kind == "structured" or self.kind == "hybrid":
            return f"structured_{self.target_tokens}_{self.hard_max_tokens}"
        return self.kind


@dataclass(slots=True)
class PreparedCase:
    config: dict[str, Any]
    gold: dict[str, Any]
    candidates: list[Candidate]
    document_ids: set[str]
    snapshot_ids: set[str]
    first_stage_seconds: float
    lexical_request_limits: list[int]
    vector_request_limits: list[int]


@dataclass(slots=True)
class ComponentScore:
    values_by_parent: dict[str, list[float]]
    scoring_count: int
    latency_seconds: float


@dataclass(slots=True)
class RankedPass:
    candidates: list[Candidate]
    scoring_count: int
    latency_seconds: float


_LEGACY_TOKENIZER_JAVASCRIPT = r"""
import {getLlama} from "node-llama-cpp";

const WINDOW_TOKENS = 96;
const OVERLAP_TOKENS = 16;
const SENTENCE_END = new Set([".", "!", "?", "。", "！", "？"]);
const NON_TERMINAL = new Set([
  "al", "dr", "eq", "eqs", "e.g", "fig", "figs", "i.e", "mr", "mrs",
  "prof", "sec", "secs", "vs",
]);

function startsSentence(character) {
  return /[A-ZÀ-ÖØ-Þ\u3400-\u9fff"“‘([]/u.test(character);
}

function isNonTerminalAbbreviation(text, periodIndex) {
  if (text[periodIndex] !== ".") return false;
  const prefix = text.slice(0, periodIndex).toLowerCase();
  const match = prefix.match(/([a-z](?:[a-z.]*)?)$/u);
  return match !== null && NON_TERMINAL.has(match[1]);
}

function splitSentences(text) {
  const sentences = [];
  let start = 0;
  for (let index = 0; index < text.length; index += 1) {
    if (!SENTENCE_END.has(text[index])) continue;
    let next = index + 1;
    while (next < text.length && /\s/u.test(text[next])) next += 1;
    if (next >= text.length) {
      sentences.push(text.slice(start).trim());
      start = text.length;
      break;
    }
    if (!startsSentence(text[next]) || isNonTerminalAbbreviation(text, index)) continue;
    sentences.push(text.slice(start, index + 1).trim());
    start = next;
    index = next - 1;
  }
  if (start < text.length) sentences.push(text.slice(start).trim());
  return sentences.filter((sentence) => sentence.length > 0);
}

let inputText = "";
for await (const chunk of process.stdin) inputText += chunk;
const input = JSON.parse(inputText);
const modelPath = process.env.PAPEROS_BENCHMARK_RERANKER_MODEL;
if (!modelPath) throw new Error("PAPEROS_BENCHMARK_RERANKER_MODEL is required");
const llama = await getLlama({gpu: false});
const model = await llama.loadModel({modelPath});
const tokenCount = (text) => model.tokenize(text).length;

function tokenWindows(text) {
  const tokens = model.tokenize(text);
  const windows = [];
  let start = 0;
  while (start < tokens.length) {
    const end = Math.min(tokens.length, start + WINDOW_TOKENS);
    windows.push(model.detokenize(tokens.slice(start, end)));
    if (end >= tokens.length) break;
    start = Math.max(start + 1, end - OVERLAP_TOKENS);
  }
  return windows;
}

function windowsForBlock(block) {
  if (tokenCount(block) <= WINDOW_TOKENS) return [block];
  const sentences = splitSentences(block);
  const windows = [];
  let start = 0;
  while (start < sentences.length) {
    let end = start;
    while (
      end < sentences.length &&
      tokenCount(sentences.slice(start, end + 1).join(" ")) <= WINDOW_TOKENS
    ) {
      end += 1;
    }
    if (end === start) {
      windows.push(...tokenWindows(sentences[start]));
      start += 1;
      continue;
    }
    windows.push(sentences.slice(start, end).join(" "));
    if (end >= sentences.length) break;
    start = end - start > 1 ? end - 1 : end;
  }
  return windows;
}

function windows(text) {
  const blocks = text.split(/\n\n+/u).filter((block) => block.length > 0);
  return (blocks.length > 0 ? blocks : [text]).flatMap(windowsForBlock);
}

const output = Object.fromEntries(input.map((item) => [item.id, windows(item.text)]));
process.stdout.write(JSON.stringify(output));
await model.dispose();
await llama.dispose();
"""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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


def _git_head() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _latency_summary(values: list[float]) -> dict[str, Any]:
    return {
        "sample_count": len(values),
        "mean_seconds": round(statistics.fmean(values), 4) if values else 0.0,
        "p50_seconds": round(_percentile(values, 0.5), 4),
        "p95_seconds": round(_percentile(values, 0.95), 4),
        "raw_seconds": [round(value, 4) for value in values],
    }


def _strategy_catalog() -> dict[str, Strategy]:
    strategies = [
        Strategy("full_chunk", "full", "maxp"),
        Strategy("legacy96_maxp", "legacy", "maxp", overlap_tokens=16),
        Strategy(
            _BASELINE_ID,
            "structured",
            "maxp",
            256,
            384,
            production_eligible=True,
        ),
        Strategy(
            "structured_256_384_top2_mean",
            "structured",
            "top2_mean",
            256,
            384,
            production_eligible=True,
        ),
        Strategy(
            "hybrid_full_structured_256_384_rrf",
            "hybrid",
            "rrf",
            256,
            384,
            production_eligible=True,
        ),
    ]
    return {strategy.id: strategy for strategy in strategies}


def _strategy_for_size(template: Strategy, target: int, hard_max: int) -> Strategy:
    if template.kind == "hybrid":
        identifier = f"hybrid_full_structured_{target}_{hard_max}_rrf"
    else:
        identifier = f"structured_{target}_{hard_max}_{template.aggregation}"
    return Strategy(
        id=identifier,
        kind=template.kind,
        aggregation=template.aggregation,
        target_tokens=target,
        hard_max_tokens=hard_max,
        production_eligible=True,
    )


def _work_maps(
    application: Application,
    corpus: CorpusView,
    papers: list[dict[str, Any]],
    dataset: str,
) -> tuple[dict[str, str], dict[str, str]]:
    active = [
        bundle
        for bundle in application.canonical_repository.list_active_bundles()
        if bundle.snapshot.dataset_id == dataset
    ]
    _require(
        len(active) == len(papers) == 5,
        "BLOCKED: retained Task 6A corpus does not contain exactly five active papers",
    )
    by_filename = {
        application.registry.get_source(bundle.document.source_file_id).original_filename: bundle
        for bundle in active
    }
    work_by_symbol: dict[str, str] = {}
    document_by_symbol: dict[str, str] = {}
    for paper in papers:
        bundle = by_filename.get(str(paper["filename"]))
        _require(bundle is not None, f"BLOCKED: retained paper missing: {paper['id']}")
        work = application.scholarly_registry.work_for_document(bundle.document.id)
        _require(work is not None, f"BLOCKED: active Work missing: {paper['id']}")
        work_by_symbol[str(paper["id"])] = work.id
        document_by_symbol[str(paper["id"])] = bundle.document.id
    _require(set(corpus.bundles) == set(document_by_symbol.values()), "Corpus is not the five-paper active set")
    return work_by_symbol, document_by_symbol


def _validate_gold(
    cases: list[dict[str, Any]],
    gold_by_case: dict[str, dict[str, Any]],
    corpus: CorpusView,
    work_by_symbol: dict[str, str],
) -> dict[str, Any]:
    case_ids = [str(case["id"]) for case in cases]
    _require(len(case_ids) == len(set(case_ids)), "Query IDs are not unique")
    _require(set(case_ids) == set(gold_by_case), "Query and Gold case IDs differ")
    split_counts = {
        split: sum(case.get("split") == split for case in cases)
        for split in ("dev", "holdout")
    }
    _require(split_counts == {"dev": 10, "holdout": 5}, "Expected frozen 10/5 dev/holdout split")
    paper_counts: dict[str, int] = {}
    checked_anchors = 0
    for case in cases:
        case_id = str(case["id"])
        gold = gold_by_case[case_id]
        work_symbol = str(gold["expected_work"])
        _require(work_symbol in work_by_symbol, f"Unknown expected Work: {work_symbol}")
        paper_counts[work_symbol] = paper_counts.get(work_symbol, 0) + 1
        gold_chunks = [str(item) for item in gold["gold_chunk_ids"]]
        anchors = [str(item) for item in gold["anchors"]]
        _require(gold_chunks and anchors, f"Gold is incomplete: {case_id}")
        combined = "\n".join(corpus.chunks[chunk_id].text for chunk_id in gold_chunks)
        for chunk_id in gold_chunks:
            chunk = corpus.chunks.get(chunk_id)
            _require(chunk is not None, f"Gold Chunk is not active: {chunk_id}")
            actual_work = corpus.work_id_by_document.get(chunk.document_id)
            _require(
                actual_work == work_by_symbol[work_symbol],
                f"Gold Chunk Work mismatch: {case_id}/{chunk_id}",
            )
        for anchor in anchors:
            _require(anchor.casefold() in combined.casefold(), f"Gold anchor is not exact source text: {case_id}: {anchor}")
            checked_anchors += 1
    _require(all(count == 3 for count in paper_counts.values()), "Each paper must contribute exactly three queries")
    return {
        "split_counts": split_counts,
        "paper_query_counts": dict(sorted(paper_counts.items())),
        "checked_anchor_count": checked_anchors,
    }


def _refresh_retained_default_projections(
    application: Application,
    *,
    dataset: str,
) -> list[dict[str, Any]]:
    """Reproject scoring spans only; never rewrite canonical Chunks or indexes."""

    reports: list[dict[str, Any]] = []
    repository = application.canonical_repository
    for bundle in repository.list_active_bundles():
        if bundle.snapshot.dataset_id != dataset:
            continue
        chunk_path = repository.chunk_store_path(bundle.snapshot.id)
        stored_chunks = [
            Chunk.model_validate_json(line)
            for line in chunk_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        spans: list[RerankSpan] = []
        rebuilt, _mentions = build_chunks(
            document=bundle.document,
            snapshot_id=bundle.snapshot.id,
            sections=bundle.sections,
            elements=bundle.elements,
            references=bundle.references,
            target_tokens=application.settings.ingestion.chunk_target_tokens,
            hard_max_tokens=application.settings.ingestion.chunk_hard_max_tokens,
            overlap_tokens=application.settings.ingestion.chunk_overlap_tokens,
            tokenizer=AUTHORITATIVE_CHUNK_TOKENIZER,
            rerank_span_sink=spans,
        )
        _require(
            sorted((chunk.id, chunk.text) for chunk in rebuilt)
            == sorted((chunk.id, chunk.text) for chunk in stored_chunks),
            f"Default reprojection changed retained Chunks: {bundle.snapshot.id}",
        )
        repository.save_rerank_projection(
            build_rerank_projection(bundle.snapshot.id, spans)
        )
        reports.append(
            {
                "snapshot_id": bundle.snapshot.id,
                "chunk_count": len(stored_chunks),
                "span_count": len(spans),
            }
        )
    _require(len(reports) == 5, "BLOCKED: default reprojection did not cover five papers")
    return reports


def _build_projection_map(
    application: Application,
    corpus: CorpusView,
    *,
    dataset: str,
    target_tokens: int,
    hard_max_tokens: int,
) -> dict[str, list[RerankSpan]]:
    spans_by_chunk: dict[str, list[RerankSpan]] = {}
    for bundle in application.canonical_repository.list_active_bundles():
        if bundle.snapshot.dataset_id != dataset:
            continue
        spans: list[RerankSpan] = []
        rebuilt, _mentions = build_chunks(
            document=bundle.document,
            snapshot_id=bundle.snapshot.id,
            sections=bundle.sections,
            elements=bundle.elements,
            references=bundle.references,
            target_tokens=application.settings.ingestion.chunk_target_tokens,
            hard_max_tokens=application.settings.ingestion.chunk_hard_max_tokens,
            overlap_tokens=application.settings.ingestion.chunk_overlap_tokens,
            tokenizer=AUTHORITATIVE_CHUNK_TOKENIZER,
            rerank_span_sink=spans,
            rerank_target_tokens=target_tokens,
            rerank_hard_max_tokens=hard_max_tokens,
        )
        active = [
            chunk
            for chunk in corpus.chunks.values()
            if chunk.canonical_snapshot_id == bundle.snapshot.id
        ]
        active_signature = sorted((chunk.id, chunk.text) for chunk in active)
        rebuilt_signature = sorted((chunk.id, chunk.text) for chunk in rebuilt)
        _require(
            active_signature == rebuilt_signature,
            f"Retained reprojection changed authoritative Chunks: {bundle.snapshot.id}",
        )
        for span in spans:
            spans_by_chunk.setdefault(span.parent_chunk_id, []).append(span)
    _audit_projection_map(
        spans_by_chunk,
        corpus,
        hard_max_tokens=hard_max_tokens,
    )
    return spans_by_chunk


def _audit_projection_map(
    spans_by_chunk: dict[str, list[RerankSpan]],
    corpus: CorpusView,
    *,
    hard_max_tokens: int,
) -> None:
    _require(set(spans_by_chunk) == set(corpus.chunks), "Benchmark projection misses active parent Chunks")
    for chunk_id, spans in spans_by_chunk.items():
        chunk = corpus.chunks[chunk_id]
        ordered = sorted(spans, key=lambda item: item.ordinal)
        _require(ordered and ordered[0].character_start_in_chunk == 0, f"Projection prefix missing: {chunk_id}")
        cursor = 0
        for ordinal, span in enumerate(ordered):
            _require(span.ordinal == ordinal, f"Projection ordinal mismatch: {span.id}")
            _require(span.character_start_in_chunk == cursor, f"Projection gap/overlap: {span.id}")
            text = span.scoring_text(chunk)
            actual = AUTHORITATIVE_CHUNK_TOKENIZER.count_tokens(text)
            _require(actual == span.token_count, f"Projection token mismatch: {span.id}")
            _require(actual <= hard_max_tokens, f"Projection hard-max violation: {span.id}")
            cursor = span.character_end_in_chunk
        _require(cursor == len(chunk.text), f"Projection suffix missing: {chunk_id}")


def _legacy_windows(
    chunks: dict[str, Chunk],
    *,
    model_path: Path,
) -> dict[str, list[str]]:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = _ALLOWED_CUDA_DEVICES
    environment["NODE_LLAMA_CPP_SKIP_DOWNLOAD"] = "true"
    environment["PAPEROS_BENCHMARK_RERANKER_MODEL"] = str(model_path)
    payload = [{"id": chunk.id, "text": chunk.text} for chunk in chunks.values()]
    completed = subprocess.run(
        ["node", "--input-type=module", "-e", _LEGACY_TOKENIZER_JAVASCRIPT],
        cwd=REPOSITORY_ROOT / "services" / "local_models",
        env=environment,
        input=json.dumps(payload, ensure_ascii=False),
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    if completed.returncode != 0:
        raise RuntimeError("Legacy tokenizer process failed: " + completed.stderr[-1000:])
    parsed = json.loads(completed.stdout)
    _require(isinstance(parsed, dict), "Legacy tokenizer output is invalid")
    result = {
        str(chunk_id): [str(window) for window in windows]
        for chunk_id, windows in parsed.items()
    }
    _require(set(result) == set(chunks), "Legacy tokenizer omitted parent Chunks")
    _require(all(result.values()), "Legacy tokenizer emitted an empty window set")
    return result


async def _prepare_case(
    application: Application,
    corpus: CorpusView,
    case: dict[str, Any],
    gold: dict[str, Any],
    work_by_symbol: dict[str, str],
    *,
    dataset: str,
) -> PreparedCase:
    requested_symbols = [str(item) for item in case.get("work_ids", [])]
    requested_work_ids = [work_by_symbol[item] for item in requested_symbols]
    document_ids = corpus.filtered_document_ids(None, dataset)
    if requested_work_ids:
        document_ids.intersection_update(corpus.document_ids_for_works(requested_work_ids))
    snapshot_ids = corpus.snapshot_ids_for_documents(document_ids)
    _require(document_ids and snapshot_ids, f"Explicit filter resolved empty: {case['id']}")
    lexical_diagnostics: dict[str, list[int]] = {}
    vector_diagnostics = VectorSearchDiagnostics()
    started = time.perf_counter()
    lexical = lexical_retrieve(
        application.services.retrieval.index_manager.lexical,
        corpus,
        [str(case["query"])],
        limit=_CANDIDATE_POOL_SIZE,
        document_ids=document_ids,
        active_snapshot_ids=snapshot_ids,
        diagnostics=lexical_diagnostics,
    )
    vector = await semantic_retrieve(
        application.services.retrieval.search,
        corpus,
        str(case["query"]),
        dataset_name=dataset,
        limit=_CANDIDATE_POOL_SIZE,
        document_ids=document_ids,
        active_snapshot_ids=snapshot_ids,
        diagnostics=vector_diagnostics,
    )
    fused = weighted_rrf({"lexical": lexical, "vector": vector}, {"lexical": 1.0, "vector": 1.0})
    fused = deduplicate_candidates_by_chunk(fused)[:_CANDIDATE_POOL_SIZE]
    elapsed = time.perf_counter() - started
    _require(fused, f"First-stage retrieval returned no candidates: {case['id']}")
    return PreparedCase(
        config=case,
        gold=gold,
        candidates=fused,
        document_ids=document_ids,
        snapshot_ids=snapshot_ids,
        first_stage_seconds=elapsed,
        lexical_request_limits=lexical_diagnostics.get("request_limits", []),
        vector_request_limits=vector_diagnostics.request_limits,
    )


class BenchmarkRunner:
    def __init__(
        self,
        application: Application,
        corpus: CorpusView,
        *,
        work_by_symbol: dict[str, str],
        structured: dict[str, dict[str, list[RerankSpan]]],
        legacy_windows: dict[str, list[str]],
    ) -> None:
        self.application = application
        self.corpus = corpus
        self.work_by_symbol = work_by_symbol
        self.structured = structured
        self.legacy_windows = legacy_windows
        self.score_cache: dict[tuple[str, str, tuple[str, ...]], ComponentScore] = {}

    async def warmup(self) -> None:
        chunk_id = min(self.corpus.chunks)
        chunk = self.corpus.chunks[chunk_id]
        await self.application.local_inference_client.rerank(
            "PaperOS reranker benchmark warmup",
            [f"warmup:{chunk_id}"],
            [chunk.text[: min(len(chunk.text), 256)]],
            limit=1,
        )

    async def evaluate(
        self,
        prepared: PreparedCase,
        strategy: Strategy,
    ) -> dict[str, Any]:
        first = await self._rank_pass(prepared, strategy, prepared.candidates)
        top_k = int(prepared.config["top_k"])
        seeds = first.candidates[:top_k]
        first_ids = {candidate.chunk_id for candidate in first.candidates}
        expanded = []
        expansion_seconds = 0.0
        if bool(prepared.config.get("expand_context")):
            started = time.perf_counter()
            expanded = local_neighbor_expand(
                self.corpus,
                seeds,
                document_ids=prepared.document_ids,
            )
            expansion_seconds = time.perf_counter() - started
        genuinely_new = [item for item in expanded if item.chunk_id not in first_ids]
        second_candidates: list[Candidate] = []
        second = RankedPass([], 0, 0.0)
        if genuinely_new:
            second_candidates = deduplicate_candidates_by_chunk(
                [*first.candidates, *genuinely_new]
            )
            second = await self._rank_pass(prepared, strategy, second_candidates)
            final_ranking = second.candidates
        else:
            final_ranking = first.candidates
        selected = deduplicate_candidates_by_chunk(final_ranking)[:top_k]
        evidence = format_evidence(selected, self.corpus)
        invariant = self._pipeline_invariant(
            prepared,
            selected,
            evidence,
            expansion_requested=bool(prepared.config.get("expand_context")),
            expanded=expanded,
            genuinely_new=genuinely_new,
            second_executed=bool(genuinely_new),
        )
        _require(invariant["passed"], f"Pipeline invariant failed: {prepared.config['id']}/{strategy.id}")

        gold_chunk_ids = {str(item) for item in prepared.gold["gold_chunk_ids"]}
        gold_rank = next(
            (
                rank
                for rank, candidate in enumerate(final_ranking, start=1)
                if candidate.chunk_id in gold_chunk_ids
            ),
            None,
        )
        evidence_text = "\n".join(item.text for item in evidence).casefold()
        anchors = [str(item) for item in prepared.gold["anchors"]]
        matched = [anchor for anchor in anchors if anchor.casefold() in evidence_text]
        expected_work = self.work_by_symbol[str(prepared.gold["expected_work"])]
        evidence_work_ids = {item.source_work_id for item in evidence}
        top1_work = (
            self.corpus.work_id_by_document.get(selected[0].document_id)
            if selected
            else None
        )
        total_seconds = (
            prepared.first_stage_seconds
            + first.latency_seconds
            + expansion_seconds
            + second.latency_seconds
        )
        return {
            "case_id": prepared.config["id"],
            "split": prepared.config["split"],
            "query": prepared.config["query"],
            "strategy_id": strategy.id,
            "pre_rerank_candidate_ids": [item.chunk_id for item in prepared.candidates],
            "first_reranked_candidate_ids": [item.chunk_id for item in first.candidates],
            "expansion_candidate_ids": [item.chunk_id for item in expanded],
            "new_expansion_candidate_ids": [item.chunk_id for item in genuinely_new],
            "second_rerank_candidate_ids": [item.chunk_id for item in second_candidates],
            "final_evidence_ids": [item.evidence_id for item in evidence],
            "final_evidence_chunk_ids": [item.chunk_id for item in evidence],
            "gold_chunk_ids": sorted(gold_chunk_ids),
            "first_gold_rank": gold_rank,
            "reciprocal_rank": 1.0 / gold_rank if gold_rank else 0.0,
            "hit_at_5": bool(gold_rank and gold_rank <= 5),
            "hit_at_10": bool(gold_rank and gold_rank <= 10),
            "matched_anchors": matched,
            "matched_anchor_count": len(matched),
            "total_anchor_count": len(anchors),
            "expected_work_in_evidence": expected_work in evidence_work_ids,
            "top1_expected_work": top1_work == expected_work,
            "reranker_document_count": first.scoring_count + second.scoring_count,
            "first_rerank_document_count": first.scoring_count,
            "second_rerank_document_count": second.scoring_count,
            "first_stage_latency_seconds": round(prepared.first_stage_seconds, 4),
            "first_rerank_latency_seconds": round(first.latency_seconds, 4),
            "second_rerank_latency_seconds": round(second.latency_seconds, 4),
            "total_query_latency_seconds": round(total_seconds, 4),
            "lexical_request_limits": prepared.lexical_request_limits,
            "vector_request_limits": prepared.vector_request_limits,
            "pipeline_invariant": invariant,
        }

    async def _rank_pass(
        self,
        prepared: PreparedCase,
        strategy: Strategy,
        candidates: list[Candidate],
    ) -> RankedPass:
        if strategy.kind == "hybrid":
            full = await self._component(prepared, "full", candidates)
            local = await self._component(
                prepared,
                strategy.representation_key,
                candidates,
            )
            full_scores = {key: max(values) for key, values in full.values_by_parent.items()}
            local_scores = {key: max(values) for key, values in local.values_by_parent.items()}
            full_rank = _rank_positions(candidates, full_scores)
            local_rank = _rank_positions(candidates, local_scores)
            parent_scores = {
                candidate.chunk_id: (
                    1.0 / (60 + full_rank[candidate.chunk_id])
                    + 1.0 / (60 + local_rank[candidate.chunk_id])
                )
                for candidate in candidates
            }
            scoring_count = full.scoring_count + local.scoring_count
            latency = full.latency_seconds + local.latency_seconds
        else:
            component = await self._component(
                prepared,
                strategy.representation_key,
                candidates,
            )
            if strategy.aggregation == "top2_mean":
                parent_scores = {
                    key: statistics.fmean(sorted(values, reverse=True)[:2])
                    for key, values in component.values_by_parent.items()
                }
            else:
                parent_scores = {
                    key: max(values)
                    for key, values in component.values_by_parent.items()
                }
            scoring_count = component.scoring_count
            latency = component.latency_seconds
        ranked = _rank_candidates(candidates, parent_scores)
        return RankedPass(ranked, scoring_count, latency)

    async def _component(
        self,
        prepared: PreparedCase,
        representation: str,
        candidates: list[Candidate],
    ) -> ComponentScore:
        parent_ids = tuple(candidate.chunk_id for candidate in candidates)
        cache_key = (str(prepared.config["id"]), representation, parent_ids)
        cached = self.score_cache.get(cache_key)
        if cached is not None:
            return cached

        scoring_ids: list[str] = []
        scoring_texts: list[str] = []
        parent_by_scoring_id: dict[str, str] = {}
        if representation == "full":
            for candidate in candidates:
                scoring_id = f"full:{candidate.chunk_id}"
                scoring_ids.append(scoring_id)
                scoring_texts.append(self.corpus.chunks[candidate.chunk_id].text)
                parent_by_scoring_id[scoring_id] = candidate.chunk_id
        elif representation == "legacy":
            for candidate in candidates:
                for index, text in enumerate(self.legacy_windows[candidate.chunk_id]):
                    scoring_id = f"legacy:{candidate.chunk_id}:{index}"
                    scoring_ids.append(scoring_id)
                    scoring_texts.append(text)
                    parent_by_scoring_id[scoring_id] = candidate.chunk_id
        else:
            spans_by_chunk = self.structured[representation]
            for candidate in candidates:
                chunk = self.corpus.chunks[candidate.chunk_id]
                for span in spans_by_chunk[candidate.chunk_id]:
                    scoring_ids.append(span.id)
                    scoring_texts.append(span.scoring_text(chunk))
                    parent_by_scoring_id[span.id] = candidate.chunk_id
        _require(len(scoring_ids) == len(set(scoring_ids)), "Benchmark scoring IDs are not unique")
        scores, latency = await self._score_texts(
            str(prepared.config["query"]),
            scoring_ids,
            scoring_texts,
        )
        values_by_parent: dict[str, list[float]] = {
            candidate.chunk_id: [] for candidate in candidates
        }
        for scoring_id, score in scores.items():
            values_by_parent[parent_by_scoring_id[scoring_id]].append(score)
        _require(all(values_by_parent.values()), "A parent candidate has no scoring text")
        component = ComponentScore(values_by_parent, len(scoring_ids), latency)
        self.score_cache[cache_key] = component
        return component

    async def _score_texts(
        self,
        query: str,
        scoring_ids: list[str],
        scoring_texts: list[str],
    ) -> tuple[dict[str, float], float]:
        scores: dict[str, float] = {}
        elapsed = 0.0
        for start in range(0, len(scoring_ids), _SCORE_BATCH_SIZE):
            batch_ids = scoring_ids[start : start + _SCORE_BATCH_SIZE]
            batch_texts = scoring_texts[start : start + _SCORE_BATCH_SIZE]
            started = time.perf_counter()
            results = await self.application.local_inference_client.rerank(
                query,
                batch_ids,
                batch_texts,
                limit=len(batch_ids),
            )
            elapsed += time.perf_counter() - started
            returned = [result.candidate_id for result in results]
            _require(len(returned) == len(set(returned)), "Reranker returned duplicate IDs")
            _require(set(returned) == set(batch_ids), "Reranker returned a mismatched ID set")
            scores.update(
                {result.candidate_id: result.relevance_score for result in results}
            )
        _require(set(scores) == set(scoring_ids), "Scoring response is incomplete")
        return scores, elapsed

    def _pipeline_invariant(
        self,
        prepared: PreparedCase,
        selected: list[Candidate],
        evidence: list[Any],
        *,
        expansion_requested: bool,
        expanded: list[Candidate],
        genuinely_new: list[Candidate],
        second_executed: bool,
    ) -> dict[str, Any]:
        parents_only = all(item.chunk_id in self.corpus.chunks for item in selected)
        canonical_evidence = all(
            item.chunk_id in self.corpus.chunks
            and item.text == self.corpus.chunks[item.chunk_id].text
            for item in evidence
        )
        filters_respected = all(
            item.document_id in prepared.document_ids for item in selected
        )
        local_expansion_ok = (
            not expansion_requested
            or bool(expanded)
            and (not genuinely_new or second_executed)
        )
        passed = parents_only and canonical_evidence and filters_respected and local_expansion_ok
        return {
            "passed": passed,
            "final_object_is_parent_chunk": parents_only,
            "evidence_is_canonical_chunk": canonical_evidence,
            "explicit_filters_respected": filters_respected,
            "local_expansion_works": local_expansion_ok,
            "second_rerank_executed_for_new_candidates": (
                not genuinely_new or second_executed
            ),
        }


def _rank_positions(
    candidates: list[Candidate],
    scores: dict[str, float],
) -> dict[str, int]:
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (-scores[item[1].chunk_id], item[0]),
    )
    return {
        candidate.chunk_id: rank
        for rank, (_index, candidate) in enumerate(ranked, start=1)
    }


def _rank_candidates(
    candidates: list[Candidate],
    scores: dict[str, float],
) -> list[Candidate]:
    ranked = sorted(
        enumerate(candidates),
        key=lambda item: (-scores[item[1].chunk_id], item[0]),
    )
    output: list[Candidate] = []
    for rank, (_index, candidate) in enumerate(ranked, start=1):
        selected = candidate.model_copy(deep=True)
        selected.rerank_score = scores[candidate.chunk_id]
        selected.final_rank = rank
        output.append(selected)
    return output


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    _require(results, "Cannot summarize an empty benchmark result")
    anchor_total = sum(int(item["total_anchor_count"]) for item in results)
    anchor_matched = sum(int(item["matched_anchor_count"]) for item in results)
    latencies = [float(item["total_query_latency_seconds"]) for item in results]
    first_latencies = [float(item["first_rerank_latency_seconds"]) for item in results]
    second_latencies = [float(item["second_rerank_latency_seconds"]) for item in results]
    return {
        "query_count": len(results),
        "mrr": round(statistics.fmean(float(item["reciprocal_rank"]) for item in results), 6),
        "hit_at_5": round(statistics.fmean(bool(item["hit_at_5"]) for item in results), 6),
        "hit_at_10": round(statistics.fmean(bool(item["hit_at_10"]) for item in results), 6),
        "anchor_coverage": round(anchor_matched / anchor_total, 6),
        "matched_anchor_count": anchor_matched,
        "total_anchor_count": anchor_total,
        "expected_work_accuracy": round(
            statistics.fmean(bool(item["expected_work_in_evidence"]) for item in results),
            6,
        ),
        "top1_expected_work_accuracy": round(
            statistics.fmean(bool(item["top1_expected_work"]) for item in results),
            6,
        ),
        "total_reranker_document_count": sum(
            int(item["reranker_document_count"]) for item in results
        ),
        "first_rerank_latency": _latency_summary(first_latencies),
        "second_rerank_latency": _latency_summary(second_latencies),
        "total_query_latency": _latency_summary(latencies),
        "pipeline_invariants_passed": all(
            bool(item["pipeline_invariant"]["passed"]) for item in results
        ),
    }


def _quality_key(summary: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(summary["anchor_coverage"]),
        float(summary["mrr"]),
        float(summary["hit_at_5"]),
        float(summary["hit_at_10"]),
        float(summary["expected_work_accuracy"]),
        -float(summary["total_query_latency"]["mean_seconds"]),
        -float(summary["total_reranker_document_count"]),
    )


def _candidate_improves(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    candidate_results: list[dict[str, Any]],
    baseline_results: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    quality_fields = ("anchor_coverage", "mrr", "hit_at_5", "hit_at_10")
    for field in quality_fields:
        if float(candidate[field]) + 1e-9 < float(baseline[field]):
            reasons.append(f"{field}_regressed")
    improved = any(
        float(candidate[field]) > float(baseline[field]) + 1e-9
        for field in quality_fields
    )
    if not improved:
        reasons.append("no_observable_quality_improvement")
    baseline_by_case = {str(item["case_id"]): item for item in baseline_results}
    for result in candidate_results:
        previous = baseline_by_case[str(result["case_id"])]
        if previous["hit_at_5"] and not result["hit_at_10"]:
            reasons.append(f"key_query_regression:{result['case_id']}")
    candidate_latency = float(candidate["total_query_latency"]["mean_seconds"])
    baseline_latency = float(baseline["total_query_latency"]["mean_seconds"])
    if candidate_latency > baseline_latency * 2.25:
        reasons.append("latency_growth_over_2.25x")
    candidate_count = int(candidate["total_reranker_document_count"])
    baseline_count = int(baseline["total_reranker_document_count"])
    if candidate_count > baseline_count * 2.25:
        reasons.append("scoring_count_growth_over_2.25x")
    return not reasons, reasons


async def _evaluate_strategies(
    runner: BenchmarkRunner,
    prepared_cases: list[PreparedCase],
    strategies: list[Strategy],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        results = [
            await runner.evaluate(prepared, strategy)
            for prepared in prepared_cases
        ]
        output[strategy.id] = {
            "strategy": asdict(strategy),
            "summary": _summary(results),
            "queries": results,
        }
        print(
            json.dumps(
                {
                    "strategy": strategy.id,
                    "summary": output[strategy.id]["summary"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return output


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Task 6B — Rerank Quality Optimization",
        "",
        f"Overall: **{report['overall_status']}**",
        "",
        f"Dev winner: `{report['selection']['dev_winner']}`",
        f"Final recommendation: **{report['selection']['final_recommendation']}**",
        f"Benchmark execution HEAD: `{report['validated_head']}`",
        "",
        "The benchmark used the retained five-paper active corpus. It did not run PDF ingestion, MinerU, semantic enrichment, or LLM synthesis.",
        "",
        "## Dev aggregation comparison",
        "",
        "| Strategy | Anchor coverage | MRR | Hit@5 | Hit@10 | Mean latency (s) | Scoring count |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["aggregation_dev"].values():
        summary = item["summary"]
        lines.append(
            "| {id} | {coverage:.3f} | {mrr:.3f} | {hit5:.3f} | {hit10:.3f} | {latency:.3f} | {count} |".format(
                id=item["strategy"]["id"],
                coverage=summary["anchor_coverage"],
                mrr=summary["mrr"],
                hit5=summary["hit_at_5"],
                hit10=summary["hit_at_10"],
                latency=summary["total_query_latency"]["mean_seconds"],
                count=summary["total_reranker_document_count"],
            )
        )
    lines.extend(
        [
            "",
            "## Selection",
            "",
            f"- Span-size candidates: {', '.join(report['selection']['span_size_candidates'])}",
            f"- Holdout compared: {', '.join(report['selection']['holdout_compared'])}",
            f"- Holdout reversal: {report['selection']['holdout_reversal']}",
            f"- Production changed: {report['selection']['production_change_recommended']}",
            "",
            "Gold anchors are exact substrings of active canonical parent Chunks. Expected-Work accuracy means the expected Work appears in the final canonical Evidence set.",
            "",
        ]
    )
    return "\n".join(lines)


def _validation_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.validation_root is not None:
        validation_root = args.validation_root.resolve()
        return validation_root / "config", validation_root / "output"
    return args.config_root.resolve(), args.output_root.resolve()


async def run(args: argparse.Namespace) -> dict[str, Any]:
    config_root, output_root = _validation_paths(args)
    runtime_root = args.retained_runtime.resolve()
    _require(runtime_root.is_dir(), "BLOCKED: retained Task 6A runtime is unavailable")
    _require(os.environ.get("CUDA_VISIBLE_DEVICES") == _ALLOWED_CUDA_DEVICES, "Task 6B requires CUDA_VISIBLE_DEVICES=6")

    query_payload = _read_json(config_root / "queries.json")
    truth_payload = _read_json(config_root / "ground_truth.json")
    papers_payload = _read_json(args.papers_config.resolve())
    cases = list(query_payload["cases"])
    gold_by_case = {str(item["case_id"]): item for item in truth_payload["gold"]}
    dataset = str(query_payload["dataset"])

    base = load_settings(args.config)
    settings = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={"directory": runtime_root, "dataset": dataset}
            ),
            "local_inference": base.local_inference.model_copy(
                update={"cuda_devices": [6]}
            ),
            "ingestion": base.ingestion.model_copy(
                update={
                    "semantic_enrichment_enabled": False,
                    "claim_enrichment_enabled": False,
                }
            ),
        }
    )
    _require(settings.retrieval.rerank_enabled, "Task 6B requires reranking")
    application = create_application(settings)
    await application.start()
    try:
        health = await application.local_inference_client.health()
        _require(health.get("cuda_visible_devices") == _ALLOWED_CUDA_DEVICES, "Local inference escaped CUDA device 6")
        retained_reprojection = _refresh_retained_default_projections(
            application,
            dataset=dataset,
        )
        corpus = CorpusView.load(
            application.paths,
            application.canonical_repository,
            application.registry,
            application.scholarly_registry,
        )
        work_by_symbol, document_by_symbol = _work_maps(
            application,
            corpus,
            list(papers_payload["papers"]),
            dataset,
        )
        gold_audit = _validate_gold(cases, gold_by_case, corpus, work_by_symbol)

        persisted_key = "structured_256_384"
        structured: dict[str, dict[str, list[RerankSpan]]] = {
            persisted_key: corpus.rerank_spans_by_chunk
        }
        _audit_projection_map(
            structured[persisted_key],
            corpus,
            hard_max_tokens=384,
        )
        for target, hard_max in ((192, 288), (384, 576)):
            key = f"structured_{target}_{hard_max}"
            structured[key] = _build_projection_map(
                application,
                corpus,
                dataset=dataset,
                target_tokens=target,
                hard_max_tokens=hard_max,
            )
        legacy = _legacy_windows(
            corpus.chunks,
            model_path=settings.local_inference.reranker_model_path,
        )
        runner = BenchmarkRunner(
            application,
            corpus,
            work_by_symbol=work_by_symbol,
            structured=structured,
            legacy_windows=legacy,
        )
        await runner.warmup()

        dev_cases = [case for case in cases if case["split"] == "dev"]
        holdout_cases = [case for case in cases if case["split"] == "holdout"]
        prepared_dev = [
            await _prepare_case(
                application,
                corpus,
                case,
                gold_by_case[str(case["id"])],
                work_by_symbol,
                dataset=dataset,
            )
            for case in dev_cases
        ]
        catalog = _strategy_catalog()
        aggregation_strategies = list(catalog.values())
        aggregation_dev = await _evaluate_strategies(
            runner,
            prepared_dev,
            aggregation_strategies,
        )

        structured_aggregation = [
            catalog[_BASELINE_ID],
            catalog["structured_256_384_top2_mean"],
            catalog["hybrid_full_structured_256_384_rrf"],
        ]
        selected_aggregations = sorted(
            structured_aggregation,
            key=lambda strategy: _quality_key(
                aggregation_dev[strategy.id]["summary"]
            ),
            reverse=True,
        )[:2]
        span_strategies: dict[str, Strategy] = {}
        for template in selected_aggregations:
            for target, hard_max in ((192, 288), (256, 384), (384, 576)):
                strategy = _strategy_for_size(template, target, hard_max)
                span_strategies[strategy.id] = strategy
        span_size_dev = await _evaluate_strategies(
            runner,
            prepared_dev,
            list(span_strategies.values()),
        )

        baseline_dev = aggregation_dev[_BASELINE_ID]
        eligible: list[tuple[Strategy, dict[str, Any], list[str]]] = []
        eligibility: dict[str, dict[str, Any]] = {}
        for strategy in span_strategies.values():
            candidate = span_size_dev[strategy.id]
            improves, reasons = _candidate_improves(
                candidate["summary"],
                baseline_dev["summary"],
                candidate["queries"],
                baseline_dev["queries"],
            )
            eligibility[strategy.id] = {
                "eligible": improves,
                "reasons": reasons,
            }
            if improves:
                eligible.append((strategy, candidate, reasons))
        if eligible:
            selected_strategy = max(
                eligible,
                key=lambda item: _quality_key(item[1]["summary"]),
            )[0]
        else:
            selected_strategy = catalog[_BASELINE_ID]

        # Holdout is intentionally prepared and scored only after dev selection.
        prepared_holdout = [
            await _prepare_case(
                application,
                corpus,
                case,
                gold_by_case[str(case["id"])],
                work_by_symbol,
                dataset=dataset,
            )
            for case in holdout_cases
        ]
        holdout_strategies = {catalog[_BASELINE_ID].id: catalog[_BASELINE_ID]}
        holdout_strategies[selected_strategy.id] = selected_strategy
        holdout = await _evaluate_strategies(
            runner,
            prepared_holdout,
            list(holdout_strategies.values()),
        )
        holdout_reversal = False
        final_strategy = selected_strategy
        if selected_strategy.id != _BASELINE_ID:
            baseline_summary = holdout[_BASELINE_ID]["summary"]
            selected_summary = holdout[selected_strategy.id]["summary"]
            holdout_reversal = any(
                float(selected_summary[field]) + 1e-9 < float(baseline_summary[field])
                for field in ("anchor_coverage", "mrr", "hit_at_5", "hit_at_10")
            )
            if holdout_reversal:
                final_strategy = catalog[_BASELINE_ID]

        production_change = final_strategy.id != _BASELINE_ID
        recommendation = (
            f"adopt {final_strategy.id}"
            if production_change
            else "retain structured 256/384 MaxP; no better production candidate found"
        )
        first_stage = {
            prepared.config["id"]: {
                "split": prepared.config["split"],
                "candidate_chunk_ids_before_rerank": [
                    item.chunk_id for item in prepared.candidates
                ],
                "latency_seconds": round(prepared.first_stage_seconds, 4),
                "lexical_request_limits": prepared.lexical_request_limits,
                "vector_request_limits": prepared.vector_request_limits,
            }
            for prepared in [*prepared_dev, *prepared_holdout]
        }
        report = {
            "overall_status": "PASS",
            "scope": "Task 6B controlled rerank quality benchmark",
            "validated_head": _git_head(),
            "dataset": dataset,
            "retained_runtime": "rerank_projection_acceptance/output/runtime",
            "cuda_visible_devices": _ALLOWED_CUDA_DEVICES,
            "corpus": {
                "paper_count": len(papers_payload["papers"]),
                "active_chunk_count": len(corpus.chunks),
                "work_ids": work_by_symbol,
                "document_ids": document_by_symbol,
            },
            "gold_audit": gold_audit,
            "retained_default_reprojection": retained_reprojection,
            "benchmark_order": [
                "prepare dev first-stage pools",
                "aggregation comparison on dev",
                "span-size comparison for top two structured aggregations",
                "freeze one dev candidate",
                "prepare and run holdout baseline vs selected candidate once",
            ],
            "first_stage": first_stage,
            "aggregation_dev": aggregation_dev,
            "span_size_dev": span_size_dev,
            "holdout": holdout,
            "selection": {
                "aggregation_shortlist": [item.id for item in selected_aggregations],
                "span_size_candidates": list(span_strategies),
                "eligibility_against_baseline": eligibility,
                "dev_winner": selected_strategy.id,
                "holdout_compared": list(holdout_strategies),
                "holdout_executed_after_dev_selection": True,
                "holdout_reversal": holdout_reversal,
                "final_strategy": final_strategy.id,
                "production_change_recommended": production_change,
                "final_recommendation": recommendation,
            },
            "pipeline_invariants_passed": all(
                item["summary"]["pipeline_invariants_passed"]
                for stage in (aggregation_dev, span_size_dev, holdout)
                for item in stage.values()
            ),
            "no_ingestion_or_synthesis": True,
        }
        _require(report["pipeline_invariants_passed"], "A benchmark strategy violated parent/Evidence/filter invariants")
        _write_json(output_root / "benchmark.json", report)
        (output_root / "README.md").write_text(
            _markdown(report),
            encoding="utf-8",
        )
        return report
    finally:
        await application.aclose()


def _external_boundary_failure(exc: BaseException) -> bool:
    external_codes = {
        "local_inference_unavailable",
        "local_inference_response_error",
        "cognee_storage_error",
    }
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if isinstance(current, PaperOSError) and current.code in external_codes:
            return True
        current = current.__cause__ or current.__context__
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/paperos.toml"))
    parser.add_argument("--config-root", type=Path, default=_DEFAULT_CONFIG_ROOT)
    parser.add_argument("--output-root", type=Path, default=_DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--validation-root",
        type=Path,
        default=None,
        help="Legacy combined root containing config/ and output/.",
    )
    parser.add_argument("--retained-runtime", type=Path, default=_RETAINED_ROOT)
    parser.add_argument("--papers-config", type=Path, default=_PAPERS_CONFIG)
    args = parser.parse_args()
    try:
        report = asyncio.run(run(args))
    except Exception as exc:  # noqa: BLE001 - validation must classify external blockers
        blocked = _external_boundary_failure(exc) or "BLOCKED:" in str(exc)
        report = {
            "overall_status": "BLOCKED" if blocked else "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        output = _validation_paths(args)[1] / "benchmark.json"
        _write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["overall_status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
