#!/usr/bin/env python3
"""MinerU→canonical survival and gold-occurrence source survival gates."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.domain.enums import ElementType
from paperos_core.ingestion.normalization import source_evidence_text
from tests.validation.chunk_corpus_review import _guess_pdf_for_bundle, _load_bundle_from_snapshot_dir
from tests.validation.gold_audit_candidate_builder import context_hash, norm_context
from tests.validation.rebuild_canonical import isogeometric_regression_text


ISOGEOMETRIC_PDF = "isogeometric_analysis_of_geometric_partial_differential_equations.pdf"


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_mineru_full_md(run_dir: Path, source_id: str) -> str | None:
    parsed = run_dir / "parsed" / source_id
    if not parsed.exists():
        return None
    parse_dirs = sorted(parsed.glob("parse_*"))
    if not parse_dirs:
        return None
    full_md = parse_dirs[-1] / "artifacts" / "full.md"
    if not full_md.exists():
        return None
    return full_md.read_text(encoding="utf-8", errors="replace")


def _canonical_combined_text(elements) -> str:
    return "\n".join(
        source_evidence_text(element.text or element.markdown or "")
        for element in elements
        if (element.text or element.markdown)
    )


def _check_isogeometric_regression(bundle, run_dir: Path, source_id: str) -> list[dict]:
    pdf_path = _guess_pdf_for_bundle(bundle, Path("data/validation/corpus/chunk"))
    if pdf_path.name != ISOGEOMETRIC_PDF:
        return []
    marker_status = isogeometric_regression_text(bundle.elements)
    failures = []
    for marker, present in marker_status.items():
        if not present:
            failures.append(
                {
                    "failure_type": "CANONICAL_SOURCE_LOSS",
                    "paper": "isogeometric",
                    "marker": marker,
                    "reason": "isogeometric_regression_marker_missing",
                }
            )
    mineru_text = _load_mineru_full_md(run_dir, source_id)
    canonical_text = _canonical_combined_text(bundle.elements)
    if mineru_text:
        for marker in marker_status:
            if marker in mineru_text and marker not in canonical_text:
                failures.append(
                    {
                        "failure_type": "CANONICAL_SOURCE_LOSS",
                        "paper": "isogeometric",
                        "marker": marker,
                        "reason": "mineru_present_canonical_missing",
                    }
                )
    return failures


def _locate_gold_occurrence(
    *,
    surface: str,
    locator: dict,
    elements,
) -> tuple[bool, str | None]:
    element_id = locator.get("element_id")
    left = locator.get("left_context", "")
    right = locator.get("right_context", "")
    chash = locator.get("context_hash")
    candidates = []
    for element in elements:
        if element_id and element.id != element_id:
            continue
        text = source_evidence_text(element.text or element.markdown or "")
        if surface not in text:
            continue
        for match in re.finditer(re.escape(surface), text):
            left_ctx = norm_context(text[max(0, match.start() - 80) : match.start()])[-60:]
            right_ctx = norm_context(text[match.end() : match.end() + 80])[:60]
            candidate_hash = context_hash(
                text[max(0, match.start() - 80) : match.start()],
                surface,
                text[match.end() : match.end() + 80],
            )
            if chash and candidate_hash != chash:
                continue
            if left and left_ctx and left not in left_ctx and left_ctx not in left:
                continue
            if right and right_ctx and right not in right_ctx and right_ctx not in right:
                continue
            candidates.append(element.id)
    if not candidates:
        return False, None
    return True, candidates[0]


def _check_gold_source_survival(*, paper_key: str, paper_gold: dict, bundle) -> list[dict]:
    failures = []
    spans = paper_gold.get("citation_spans", paper_gold.get("citation_groups", []))
    for span in spans:
        surface = span.get("surface_text") or span.get("surface", "")
        locator = span.get("locator") or {}
        if not surface:
            continue
        found, element_id = _locate_gold_occurrence(
            surface=surface,
            locator=locator,
            elements=bundle.elements,
        )
        if not found:
            failures.append(
                {
                    "failure_type": "GOLD_CANONICAL_SOURCE_LOSS",
                    "paper": paper_key,
                    "surface": surface,
                    "page": locator.get("page"),
                    "source_domain": locator.get("source_domain"),
                    "left_context": locator.get("left_context"),
                    "right_context": locator.get("right_context"),
                    "element_id": locator.get("element_id"),
                }
            )
        elif element_id and locator.get("element_id") and element_id != locator["element_id"]:
            failures.append(
                {
                    "failure_type": "GOLD_CANONICAL_ELEMENT_MISMATCH",
                    "paper": paper_key,
                    "surface": surface,
                    "expected_element_id": locator.get("element_id"),
                    "actual_element_id": element_id,
                }
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/validation/runs/chunk"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/validation/corpus/chunk"))
    parser.add_argument("--gold", type=Path, default=Path("tests/fixtures/chunk/citation_gold_v2.json"))
    args = parser.parse_args()

    gold = json.loads(args.gold.read_text(encoding="utf-8")) if args.gold.exists() else {"papers": {}}
    failures: list[dict] = []
    papers_report: dict[str, dict] = {}

    for src_dir in sorted((args.run_dir / "canonical").glob("src_*")):
        snapshot_dirs = sorted(src_dir.glob("snapshot_*"))
        if not snapshot_dirs:
            continue
        bundle = _load_bundle_from_snapshot_dir(snapshot_dirs[-1])
        pdf_path = _guess_pdf_for_bundle(bundle, args.corpus_dir)
        paper_key = None
        for key, paper in gold.get("papers", {}).items():
            if paper.get("pdf_basename") == pdf_path.name:
                paper_key = key
                break
        paper_failures = _check_isogeometric_regression(bundle, args.run_dir, src_dir.name)
        if paper_key:
            paper_failures.extend(
                _check_gold_source_survival(
                    paper_key=paper_key,
                    paper_gold=gold["papers"][paper_key],
                    bundle=bundle,
                )
            )
        failures.extend(paper_failures)
        papers_report[pdf_path.name] = {
            "snapshot_id": bundle.snapshot.id,
            "failure_count": len(paper_failures),
            "failures": paper_failures[:20],
        }

    report = {
        "git_commit": _git_commit(),
        "canonical_source_loss": len(
            [item for item in failures if item["failure_type"] == "CANONICAL_SOURCE_LOSS"]
        ),
        "gold_canonical_source_loss": len(
            [
                item
                for item in failures
                if item["failure_type"] in {"GOLD_CANONICAL_SOURCE_LOSS", "GOLD_CANONICAL_ELEMENT_MISMATCH"}
            ]
        ),
        "failure_count": len(failures),
        "papers": papers_report,
        "failures": failures[:100],
        "pass": not failures,
    }
    output = args.run_dir / "canonical-source-survival.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "failures": len(failures), "report": str(output)}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
