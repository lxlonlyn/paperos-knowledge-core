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
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.config import load_settings
from paperos_core.domain.canonical import (
    CanonicalBundle,
    CanonicalSnapshot,
    Document,
    Element,
    ReferenceEntry,
    Section,
)
from paperos_core.domain.enums import ElementType
from paperos_core.ingestion.chunk_markdown import render_chunk_review_markdown
from paperos_core.ingestion.chunking import build_chunks
from paperos_core.domain.ids import CHUNKING_VERSION

from paperos_core.ingestion.chunk_dp import TINY_TOKEN_THRESHOLD


class _WhitespaceTokenizer:
    def count_tokens(self, text: str) -> int:
        if not text.strip():
            return 0
        return max(1, len(text.split()))


def _resolve_tokenizer() -> Any:
    try:
        from paperos_core.adapters.cognee.compat import resolve_cognee_tokenizer

        return resolve_cognee_tokenizer()
    except ImportError:
        return _WhitespaceTokenizer()


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


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _cleanup_review_artifacts(output_dir: Path, *, active_slugs: set[str]) -> None:
    for pattern in ("*.chunks.md", "*.chunks.json"):
        for path in output_dir.glob(pattern):
            slug = path.name.split(".chunks.", 1)[0]
            if slug not in active_slugs:
                path.unlink(missing_ok=True)


def _citation_metrics(mentions: list[Any]) -> dict[str, Any]:
    spans: dict[str, list[Any]] = defaultdict(list)
    for mention in mentions:
        spans[mention.citation_span_id].append(mention)

    span_status = Counter()
    for group in spans.values():
        span_status[group[0].span_resolution_status] += 1

    atomic_resolved_reference = sum(
        1 for mention in mentions if mention.reference_entry_id
    )
    atomic_resolved_work = sum(1 for mention in mentions if mention.resolved_work_id)
    failure_reasons = Counter(
        mention.failure_reason for mention in mentions if mention.failure_reason
    )
    unresolved_targets = [
        {
            "surface_text": mention.surface_text,
            "atomic_key": mention.atomic_key,
            "bibliography_scope_id": mention.bibliography_scope_id,
            "document_region": mention.document_region,
            "failure_reason": mention.failure_reason,
            "resolution_status": mention.resolution_status,
        }
        for mention in mentions
        if mention.resolution_status != "resolved"
    ]
    return {
        "citation_span_count": len(spans),
        "fully_resolved_span_count": span_status.get("resolved", 0),
        "partially_resolved_span_count": span_status.get("partially_resolved", 0),
        "unresolved_span_count": span_status.get("unresolved", 0),
        "atomic_target_count": len(mentions),
        "resolved_reference_target_count": atomic_resolved_reference,
        "resolved_work_target_count": atomic_resolved_work,
        "unresolved_target_count": len(mentions) - atomic_resolved_reference,
        "failure_reasons": dict(failure_reasons),
        "unresolved_targets": unresolved_targets,
    }


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


def _load_bundle_from_snapshot_dir(snapshot_dir: Path) -> CanonicalBundle:
    warnings_payload = json.loads((snapshot_dir / "warnings.json").read_text(encoding="utf-8"))
    return CanonicalBundle(
        snapshot=CanonicalSnapshot.model_validate_json(
            (snapshot_dir / "snapshot.json").read_text(encoding="utf-8")
        ),
        document=Document.model_validate_json(
            (snapshot_dir / "document.json").read_text(encoding="utf-8")
        ),
        sections=[
            Section.model_validate_json(line)
            for line in (snapshot_dir / "sections.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        elements=[
            Element.model_validate_json(line)
            for line in (snapshot_dir / "elements.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        references=[
            ReferenceEntry.model_validate_json(line)
            for line in (snapshot_dir / "references.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ],
        warnings=warnings_payload,
    )


def _guess_pdf_for_bundle(bundle: CanonicalBundle, corpus_dir: Path) -> Path:
    title_tokens = {
        token
        for token in _slugify(bundle.document.title).casefold().split("_")
        if len(token) > 2
    }
    best: tuple[int, Path] | None = None
    for pdf_path in sorted(corpus_dir.glob("*.pdf")):
        pdf_tokens = {
            token
            for token in _slugify(pdf_path.stem).casefold().split("_")
            if len(token) > 2
        }
        score = len(title_tokens & pdf_tokens)
        if best is None or score > best[0]:
            best = (score, pdf_path)
    if best is None or best[0] == 0:
        raise RuntimeError(
            f"Unable to map canonical bundle to corpus PDF: {bundle.document.title}"
        )
    return best[1]


def _process_bundle(
    *,
    bundle: CanonicalBundle,
    pdf_path: Path,
    settings: Any,
    output_dir: Path,
    overlap_tokens: int,
) -> dict[str, Any]:
    target = settings.ingestion.chunk_target_tokens
    hard_max = settings.ingestion.chunk_hard_max_tokens
    result: dict[str, Any] = {
        "pdf": str(pdf_path),
        "status": "pending",
        "snapshot_id": bundle.snapshot.id,
        "document_id": bundle.document.id,
    }
    try:
        tokenizer = _resolve_tokenizer()
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
        citation_stats = _citation_metrics(mentions)
        markdown = render_chunk_review_markdown(
            bundle=bundle,
            chunks=chunks,
            mentions=mentions,
            source_pdf=pdf_path,
            target_tokens=target,
            hard_max_tokens=hard_max,
            overlap_tokens=overlap_tokens,
            invariants=invariants,
            citation_stats=citation_stats,
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
                    "chunking_version": CHUNKING_VERSION,
                    "chunk_count": len(chunks),
                    "invariants": invariants,
                    "citation_stats": citation_stats,
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
                "reference_elements": len(ref_chunks),
                "boundaries": dict(boundaries),
                "markdown_path": str(md_path),
                "json_path": str(json_path),
                "errors": invariants["errors"],
                **citation_stats,
            }
        )
    except Exception as exc:  # noqa: BLE001 - report real pipeline failures
        result.update({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
    return result


async def _process_pdf(
    application: Any,
    pdf_path: Path,
    *,
    settings: Any,
    output_dir: Path,
    overlap_tokens: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pdf": str(pdf_path),
        "status": "pending",
    }
    try:
        canonical = await application.services.ingestion.ingest_pdf_to_canonical(
            pdf_path,
            dataset=settings.dataset,
        )
        return _process_bundle(
            bundle=canonical.canonical,
            pdf_path=pdf_path,
            settings=settings,
            output_dir=output_dir,
            overlap_tokens=overlap_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - report real pipeline failures
        result.update({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"})
    return result


def _run_rechunk_from_canonical(
    *,
    run_dir: Path,
    corpus_dir: Path,
    settings: Any,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    canonical_root = run_dir / "canonical"
    for src_dir in sorted(canonical_root.glob("src_*")):
        snapshot_dirs = sorted(src_dir.glob("snapshot_*"))
        if not snapshot_dirs:
            continue
        bundle = _load_bundle_from_snapshot_dir(snapshot_dirs[-1])
        pdf_path = _guess_pdf_for_bundle(bundle, corpus_dir)
        rows.append(
            _process_bundle(
                bundle=bundle,
                pdf_path=pdf_path,
                settings=settings,
                output_dir=run_dir,
                overlap_tokens=overlap_tokens,
            )
        )
    return rows


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
    overlap_tokens = (
        args.overlap_tokens
        if args.overlap_tokens is not None
        else settings.ingestion.chunk_overlap_tokens
    )
    pdfs = sorted(args.corpus_dir.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {args.corpus_dir}")
    active_slugs = {_slugify(pdf_path.name) for pdf_path in pdfs}
    _cleanup_review_artifacts(run_dir, active_slugs=active_slugs)

    if args.rebuild_canonical:
        from tests.validation.rebuild_canonical import rebuild_all_canonical_snapshots

        rebuild_rows = rebuild_all_canonical_snapshots(
            run_dir=run_dir,
            dataset_id=args.dataset,
        )
        print(json.dumps({"rebuild_canonical": rebuild_rows}, indent=2))
    if args.rebuild_canonical or args.rechunk_canonical:
        rows = _run_rechunk_from_canonical(
            run_dir=run_dir,
            corpus_dir=args.corpus_dir,
            settings=settings,
            overlap_tokens=overlap_tokens,
        )
    else:
        from paperos_core.application import create_application

        application = create_application(settings)
        application.storage.initialize()
        rows = []
        try:
            for pdf_path in pdfs:
                rows.append(
                    await _process_pdf(
                        application,
                        pdf_path,
                        settings=settings,
                        output_dir=run_dir,
                        overlap_tokens=overlap_tokens,
                    )
                )
        finally:
            await application.mineru.aclose()
            await application.local_inference_client.aclose()

    summary = {
        "run_dir": str(run_dir),
        "corpus_dir": str(args.corpus_dir.resolve()),
        "dataset": args.dataset,
        "git_commit": _git_commit(),
        "chunking_version": CHUNKING_VERSION,
        "overlap_tokens": overlap_tokens,
        "chunk_target_tokens": settings.ingestion.chunk_target_tokens,
        "chunk_hard_max_tokens": settings.ingestion.chunk_hard_max_tokens,
        "pdf_count": len(rows),
        "pass_count": sum(1 for row in rows if row.get("status") == "PASS"),
        "structure_only_pass_count": sum(1 for row in rows if row.get("status") == "PASS"),
        "overall_status": (
            "STRUCTURE_ONLY_PASS"
            if rows and all(row.get("status") == "PASS" for row in rows)
            else "FAIL"
        ),
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
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        default=None,
        help="Defaults to production config chunk_overlap_tokens.",
    )
    parser.add_argument(
        "--rebuild-canonical",
        action="store_true",
        help="Re-map canonical from cached parsed MinerU artifacts (no re-OCR).",
    )
    parser.add_argument(
        "--rechunk-canonical",
        action="store_true",
        help="Rebuild chunks from cached canonical snapshots (skip MinerU ingest).",
    )
    args = parser.parse_args()
    summary = asyncio.run(run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["pass_count"] != summary["pdf_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
