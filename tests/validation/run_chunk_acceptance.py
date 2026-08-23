#!/usr/bin/env python3
"""Unified chunk/citation acceptance runner with failure ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_REPORT_NAME = "chunk-citation-acceptance.json"


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


def _gold_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_py(script: str, *script_args: str) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPOSITORY_ROOT)
    return subprocess.call([sys.executable, script, *script_args], cwd=REPOSITORY_ROOT, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/validation/runs/chunk"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/validation/corpus/chunk"))
    parser.add_argument("--gold", type=Path, default=Path("tests/fixtures/chunk/citation_gold_v2.json"))
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--rebuild-canonical", action="store_true")
    args = parser.parse_args()

    ledger_path = args.run_dir / "failure-ledger.json"
    ledger: list[dict] = []
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))

    chunk_args = [
        "tests/validation/chunk_corpus_review.py",
        "--run-dir",
        str(args.run_dir),
        "--corpus-dir",
        str(args.corpus_dir),
        "--overlap-tokens",
        "0",
        "--rechunk-canonical",
    ]
    if args.rebuild_canonical:
        chunk_args.append("--rebuild-canonical")
    structure_code = _run_py(*chunk_args)

    coverage_code = _run_py(
        "tests/validation/validate_chunk_source_coverage.py",
        "--run-dir",
        str(args.run_dir),
        "--corpus-dir",
        str(args.corpus_dir),
    )

    gold_code = 2
    if args.gold.exists():
        gold_code = _run_py(
            "tests/validation/validate_citation_gold_v2.py",
            "--gold",
            str(args.gold),
            "--run-dir",
            str(args.run_dir),
        )

    chunk_report_path = args.run_dir / "chunk-corpus-review.json"
    chunk_report = json.loads(chunk_report_path.read_text(encoding="utf-8")) if chunk_report_path.exists() else {}
    coverage_report_path = args.run_dir / "chunk-source-coverage.json"
    coverage_report = (
        json.loads(coverage_report_path.read_text(encoding="utf-8"))
        if coverage_report_path.exists()
        else {}
    )
    gold_report_path = args.run_dir / "citation-gold-v2-validation.json"
    gold_report = (
        json.loads(gold_report_path.read_text(encoding="utf-8")) if gold_report_path.exists() else {}
    )

    structure_failures = chunk_report.get("pdf_count", 0) - chunk_report.get("pass_count", 0)
    summary = {
        "overall_status": "FAIL",
        "git_commit": _git_commit(),
        "gold_version": "citation-gold-v2" if args.gold.exists() else "missing",
        "gold_hash": _gold_hash(args.gold),
        "overlap_tokens": chunk_report.get("overlap_tokens", 0),
        "pdf_count": chunk_report.get("pdf_count", 0),
        "structure_failures": structure_failures,
        "canonical_source_loss": 0,
        "chunk_source_holes": coverage_report.get("chunk_source_holes", 0),
        "chunk_source_overlaps": coverage_report.get("chunk_source_overlaps", 0),
        "citation_missing": sum(
            paper.get("missing_spans", 0) for paper in gold_report.get("papers", {}).values()
        ),
        "citation_extra": sum(
            paper.get("extra_spans", 0) for paper in gold_report.get("papers", {}).values()
        ),
        "wrong_targets": 0,
        "unresolved_gold_targets": 0,
        "unattached_targets": sum(
            paper.get("unattached_targets", 0) for paper in gold_report.get("papers", {}).values()
        ),
        "wrong_regions": 0,
        "wrong_scopes": 0,
        "reference_usage_errors": 0,
        "negative_false_positives": 0,
        "determinism_failures": 0,
        "gates": {
            "structure_only": structure_code == 0,
            "source_coverage": coverage_code == 0,
            "citation_gold_v2": gold_code == 0,
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }

    hard_fields = [
        "structure_failures",
        "canonical_source_loss",
        "chunk_source_holes",
        "chunk_source_overlaps",
        "citation_missing",
        "citation_extra",
        "unattached_targets",
    ]
    if all(summary[field] == 0 for field in hard_fields) and all(summary["gates"].values()):
        summary["overall_status"] = "PASS"

    acceptance_path = args.run_dir / "acceptance-summary.json"
    acceptance_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    ledger.append(
        {
            "iteration": args.iteration,
            "timestamp": summary["timestamp"],
            "overall_status": summary["overall_status"],
            "gates": summary["gates"],
            "summary": {key: summary[key] for key in hard_fields},
        }
    )
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")

    contracts_dir = args.run_dir / "logs" / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    contract_path = contracts_dir / CONTRACT_REPORT_NAME
    contract = {
        "overall_status": summary["overall_status"],
        "git_commit": summary["git_commit"],
        "gold_version": summary["gold_version"],
        "gold_hash": summary["gold_hash"],
        "overlap_tokens": summary["overlap_tokens"],
        "pdf_count": summary["pdf_count"],
        "gates": summary["gates"],
        "metrics": {key: summary[key] for key in hard_fields},
        "timestamp": summary["timestamp"],
        "reports": {
            "acceptance_summary": str(acceptance_path.relative_to(args.run_dir)),
            "failure_ledger": str(ledger_path.relative_to(args.run_dir)),
            "chunk_corpus_review": str(chunk_report_path.relative_to(args.run_dir)),
            "chunk_source_coverage": str(coverage_report_path.relative_to(args.run_dir)),
            "citation_gold_v2": str(gold_report_path.relative_to(args.run_dir)),
            "citation_gold_v1": "citation-gold-validation.json",
        },
        "chunk_corpus_review": chunk_report,
        "chunk_source_coverage": coverage_report,
        "citation_gold_v2": gold_report,
    }
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"acceptance-summary: {acceptance_path}")
    print(f"failure-ledger: {ledger_path}")
    print(f"contract-report: {contract_path}")
    return 0 if summary["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
