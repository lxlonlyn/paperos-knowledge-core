"""Real-PDF chunk corpus review: MinerU → Canonical → production chunk builder.

All runtime data MUST stay under a validation run root (never production ``data/``).

    PYTHONPATH=. python tests/validation/chunk_corpus_review.py \\
      --run-dir data/validation/runs/chunk
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.adapters.cognee.compat import resolve_cognee_tokenizer
from paperos_core.application import create_application
from paperos_core.config import load_settings
from paperos_core.domain.enums import ElementType
from paperos_core.ingestion.chunk_markdown import render_chunk_review_markdown
from paperos_core.ingestion.chunking import build_chunks
from paperos_core.ingestion.chunk_dp import TINY_TOKEN_THRESHOLD
from paperos_core.ingestion.sentence_units import resolve_major_section_id

DEFAULT_RUN_DIR = Path("data/validation/runs/chunk")
DEFAULT_DATASET = "paperos-chunk-corpus-review"


def _assert_validation_run_dir(run_dir: Path) -> None:
    """Refuse to write MinerU/canonical artifacts outside validation/runs."""
    resolved = run_dir.expanduser().resolve()
    parts = resolved.parts
    if "validation" not in parts or "runs" not in parts:
        raise RuntimeError(
            f"--run-dir must live under data/validation/runs/ (got {resolved})"
        )
    validation_index = parts.index("validation")
    if validation_index + 1 >= len(parts) or parts[validation_index + 1] != "runs":
        raise RuntimeError(
            f"--run-dir must be a child of validation/runs (got {resolved})"
        )


def _slugify(name: str) -> str:
    stem = Path(name).stem
    return "".join(char if char.isalnum() else "_" for char in stem).strip("_")[:120]


def _validate_chunks(
    *,
    chunks: list[Any],
    hard_max_tokens: int,
    section_by_id: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if not chunks:
        errors.append("chunk_count_zero")
    for chunk in chunks:
        if not (chunk.text or "").strip():
            errors.append(f"empty_chunk:{chunk.id}")
        if (chunk.token_count or 0) > hard_max_tokens:
            errors.append(
                f"hard_max_violation:{chunk.id}:{chunk.token_count}>{hard_max_tokens}"
            )
        if chunk.retrieval_text and chunk.text not in chunk.retrieval_text:
            errors.append(f"retrieval_missing_authoritative:{chunk.id}")
        if chunk.text.startswith("Paper:") or "\nSection:\n" in chunk.text[:80]:
            errors.append(f"authoritative_has_header:{chunk.id}")
    return {
        "pass": not errors,
        "errors": errors,
        "chunk_count": len(chunks),
    }


async def _process_pdf(
    application: Any,
    pdf_path: Path,
    *,
    settings: Any,
    output_dir: Path,
    overlap_tokens: int,
) -> dict[str, Any]:
    target = settings.ingestion.chunk_target_tokens
    hard_max = settings.ingestion.chunk_hard_max_tokens
    result: dict[str, Any] = {
        "pdf": str(pdf_path),
        "status": "pending",
    }
    try:
        canonical = await application.services.ingestion.ingest_pdf_to_canonical(
            pdf_path,
            dataset=settings.dataset,
        )
        bundle = canonical.canonical
        tokenizer = resolve_cognee_tokenizer()
        chunks, mentions = build_chunks(
            document=bundle.document,
            snapshot_id=bundle.snapshot.id,
            sections=bundle.sections,
            elements=bundle.elements,
            references=bundle.references,
            target_tokens=target,
            hard_max_tokens=hard_max,
            overlap_tokens=overlap_tokens,
            tokenizer=tokenizer,
        )
        section_by_id = {section.id: section for section in bundle.sections}
        invariants = _validate_chunks(
            chunks=chunks,
            hard_max_tokens=hard_max,
            section_by_id=section_by_id,
        )
        token_counts = [chunk.token_count or 0 for chunk in chunks]
        boundaries = Counter(chunk.metadata.get("end_boundary") for chunk in chunks)
        emergency = sum(
            int(chunk.metadata.get("emergency_oversized_sentence_splits") or 0)
            for chunk in chunks
        )
        ref_chunks = [
            element
            for element in bundle.elements
            if element.element_type == ElementType.REFERENCE
        ]
        markdown = render_chunk_review_markdown(
            bundle=bundle,
            chunks=chunks,
            mentions=mentions,
            source_pdf=pdf_path,
            target_tokens=target,
            hard_max_tokens=hard_max,
            overlap_tokens=overlap_tokens,
            invariants=invariants,
        )
        slug = _slugify(pdf_path.name)
        output_dir.mkdir(parents=True, exist_ok=True)
        md_path = output_dir / f"{slug}.chunks.md"
        md_path.write_text(markdown, encoding="utf-8")
        json_path = output_dir / f"{slug}.chunks.json"
        json_path.write_text(
            json.dumps(
                {
                    "pdf": str(pdf_path),
                    "document_id": bundle.document.id,
                    "snapshot_id": bundle.snapshot.id,
                    "chunk_count": len(chunks),
                    "invariants": invariants,
                    "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
                    "citation_mentions": [
                        mention.model_dump(mode="json") for mention in mentions
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        result.update(
            {
                "status": "PASS" if invariants["pass"] else "FAIL",
                "chunk_count": len(chunks),
                "min_tokens": min(token_counts) if token_counts else 0,
                "median_tokens": (
                    statistics.median(token_counts) if token_counts else 0
                ),
                "max_tokens": max(token_counts) if token_counts else 0,
                "tiny_chunks": sum(
                    1 for count in token_counts if count < TINY_TOKEN_THRESHOLD
                ),
                "emergency_splits": emergency,
                "citation_mentions": len(mentions),
                "reference_resolved": sum(
                    1 for mention in mentions if mention.reference_entry_id
                ),
                "reference_unresolved": sum(
                    1 for mention in mentions if not mention.reference_entry_id
                ),
                "unresolved_citations": sorted(
                    {
                        mention.surface_text
                        for mention in mentions
                        if not mention.reference_entry_id
                    }
                ),
                "work_resolved": sum(
                    1 for mention in mentions if mention.resolved_work_id
                ),
                "reference_elements": len(ref_chunks),
                "boundaries": dict(boundaries),
                "markdown_path": str(md_path),
                "json_path": str(json_path),
                "errors": invariants["errors"],
            }
        )
    except Exception as exc:  # noqa: BLE001 - report real pipeline failures
        result.update({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
    return result


async def run(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = args.run_dir.expanduser().resolve()
    _assert_validation_run_dir(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    configured = load_settings(args.config)
    settings = configured.model_copy(
        update={
            "data": configured.data.model_copy(
                update={"directory": run_dir, "dataset": args.dataset}
            )
        }
    )
    application = create_application(settings)
    application.storage.initialize()
    pdfs = sorted(args.corpus_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {args.corpus_dir}")
    rows: list[dict[str, Any]] = []
    try:
        for pdf_path in pdfs:
            rows.append(
                await _process_pdf(
                    application,
                    pdf_path,
                    settings=settings,
                    output_dir=run_dir,
                    overlap_tokens=args.overlap_tokens,
                )
            )
    finally:
        await application.mineru.aclose()
        await application.local_inference_client.aclose()
    summary = {
        "run_dir": str(run_dir),
        "corpus_dir": str(args.corpus_dir.resolve()),
        "dataset": args.dataset,
        "pdf_count": len(pdfs),
        "pass_count": sum(1 for row in rows if row.get("status") == "PASS"),
        "results": rows,
    }
    report_path = run_dir / "chunk-corpus-review.json"
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary["report_path"] = str(report_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/validation/corpus/chunk"),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Isolated DATA_DIR for this validation run (must be under validation/runs/).",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--overlap-tokens", type=int, default=0)
    args = parser.parse_args()
    summary = asyncio.run(run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["pass_count"] != summary["pdf_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
