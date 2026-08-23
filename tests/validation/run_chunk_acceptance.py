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
from typing import Any

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


def _gate_result(*, exit_code: int | None, failures: int = 0, **metrics: Any) -> dict[str, Any]:
    if exit_code is None:
        return {"status": "NOT_CHECKED", "failures": failures, **metrics}
    status = "PASS" if exit_code == 0 and failures == 0 else "FAIL"
    return {"status": status, "failures": failures, **metrics}


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/validation/runs/chunk"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/validation/corpus/chunk"))
    parser.add_argument("--gold", type=Path, default=Path("tests/fixtures/chunk/citation_gold_v2.json"))
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--rebuild-canonical", action="store_true")
    parser.add_argument("--skip-determinism", action="store_true")
    args = parser.parse_args()

    ledger_path = args.run_dir / "failure-ledger.json"
    ledger: list[dict] = _load_json(ledger_path) if ledger_path.exists() else []
    if not isinstance(ledger, list):
        ledger = []
    iteration = args.iteration if args.iteration is not None else len(ledger) + 1

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

    survival_code = _run_py(
        "tests/validation/validate_canonical_source_survival.py",
        "--run-dir",
        str(args.run_dir),
        "--corpus-dir",
        str(args.corpus_dir),
        "--gold",
        str(args.gold),
    )

    coverage_code = _run_py(
        "tests/validation/validate_chunk_source_coverage.py",
        "--run-dir",
        str(args.run_dir),
        "--corpus-dir",
        str(args.corpus_dir),
    )

    regions_code = _run_py(
        "tests/validation/validate_chunk_regions_scopes.py",
        "--run-dir",
        str(args.run_dir),
        "--corpus-dir",
        str(args.corpus_dir),
    )

    reference_usage_code = _run_py(
        "tests/validation/validate_reference_usage.py",
        "--run-dir",
        str(args.run_dir),
        "--corpus-dir",
        str(args.corpus_dir),
    )

    negative_code = _run_py(
        "tests/validation/validate_negative_citations.py",
        "--run-dir",
        str(args.run_dir),
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

    chunk_report = _load_json(args.run_dir / "chunk-corpus-review.json")
    survival_report = _load_json(args.run_dir / "canonical-source-survival.json")
    coverage_report = _load_json(args.run_dir / "chunk-source-coverage.json")
    regions_report = _load_json(args.run_dir / "chunk-regions-scopes.json")
    reference_usage_report = _load_json(args.run_dir / "reference-usage.json")
    negative_report = _load_json(args.run_dir / "negative-citations.json")
    gold_report = _load_json(args.run_dir / "citation-gold-v2-validation.json")

    structure_failures = chunk_report.get("pdf_count", 0) - chunk_report.get("pass_count", 0)
    canonical_source_loss = survival_report.get("failure_count", 0)
    chunk_source_holes = coverage_report.get("chunk_source_holes", 0)
    chunk_source_overlaps = coverage_report.get("chunk_source_overlaps", 0)
    wrong_regions = regions_report.get("wrong_regions", 0)
    wrong_scopes = regions_report.get("wrong_bibliography_scopes", 0)
    citation_missing = gold_report.get("missing_occurrences", 0)
    citation_extra = gold_report.get("extra_occurrences", 0)
    citation_wrong_targets = gold_report.get("wrong_targets", 0)
    citation_unresolved_expected = gold_report.get("unresolved_expected_targets", 0)
    unattached_targets = gold_report.get("unattached_targets", 0)
    missed_used_references = reference_usage_report.get("missed_used_references", 0)
    unexpected_used_references = reference_usage_report.get("unexpected_used_references", 0)
    negative_false_positives = negative_report.get("negative_false_positives", 0)

    gates: dict[str, dict[str, Any]] = {
        "structure": _gate_result(
            exit_code=structure_code,
            failures=structure_failures,
            structure_only_status=chunk_report.get("overall_status"),
        ),
        "canonical_source_survival": _gate_result(
            exit_code=survival_code,
            failures=canonical_source_loss,
            canonical_source_loss=survival_report.get("canonical_source_loss", 0),
            gold_canonical_source_loss=survival_report.get("gold_canonical_source_loss", 0),
        ),
        "source_coverage": _gate_result(
            exit_code=coverage_code,
            failures=chunk_source_holes + chunk_source_overlaps,
            holes=chunk_source_holes,
            overlaps=chunk_source_overlaps,
        ),
        "citation_occurrence": _gate_result(
            exit_code=0 if (citation_missing + citation_extra) == 0 and args.gold.exists() else (1 if args.gold.exists() else None),
            failures=citation_missing + citation_extra,
            missing=citation_missing,
            extra=citation_extra,
        ),
        "citation_targets": _gate_result(
            exit_code=0 if (citation_wrong_targets + citation_unresolved_expected) == 0 and args.gold.exists() else (1 if args.gold.exists() and (citation_wrong_targets + citation_unresolved_expected) > 0 else (0 if args.gold.exists() else None)),
            failures=citation_wrong_targets + citation_unresolved_expected,
            wrong=citation_wrong_targets,
            unresolved_expected=citation_unresolved_expected,
        ),
        "citation_attachment": _gate_result(
            exit_code=0 if unattached_targets == 0 and args.gold.exists() else (1 if args.gold.exists() and unattached_targets > 0 else (0 if args.gold.exists() else None)),
            failures=unattached_targets,
            unattached=unattached_targets,
        ),
        "regions": _gate_result(exit_code=0 if wrong_regions == 0 else 1, failures=wrong_regions, wrong=wrong_regions),
        "bibliography_scopes": _gate_result(
            exit_code=0 if wrong_scopes == 0 else 1,
            failures=wrong_scopes,
            wrong=wrong_scopes,
        ),
        "reference_usage": _gate_result(
            exit_code=0 if missed_used_references == 0 and unexpected_used_references == 0 else 1,
            failures=missed_used_references + unexpected_used_references,
            missed=missed_used_references,
            unexpected=unexpected_used_references,
        ),
        "negative_cases": _gate_result(
            exit_code=0 if negative_false_positives == 0 else 1,
            failures=negative_false_positives,
            false_positive=negative_false_positives,
        ),
        "determinism": _gate_result(exit_code=None, failures=0),
    }

    determinism_failures = 0
    if not args.skip_determinism and all(
        gate["status"] == "PASS" for gate in gates.values() if gate["status"] != "NOT_CHECKED"
    ):
        first_summary = json.dumps(
            {
                "gates": gates,
                "chunk_source_holes": chunk_source_holes,
                "citation_missing": citation_missing,
            },
            sort_keys=True,
        )
        second_structure = _run_py(*chunk_args)
        second_gold = _run_py(
            "tests/validation/validate_citation_gold_v2.py",
            "--gold",
            str(args.gold),
            "--run-dir",
            str(args.run_dir),
        )
        second_chunk = _load_json(args.run_dir / "chunk-corpus-review.json")
        second_gold_report = _load_json(args.run_dir / "citation-gold-v2-validation.json")
        second_summary = json.dumps(
            {
                "structure_code": second_structure,
                "gold_code": second_gold,
                "pass_count": second_chunk.get("pass_count"),
                "missing": second_gold_report.get("missing_occurrences"),
                "extra": second_gold_report.get("extra_occurrences"),
            },
            sort_keys=True,
        )
        if second_structure != 0 or second_gold != 0 or first_summary != second_summary:
            determinism_failures = 1
        gates["determinism"] = _gate_result(exit_code=0 if determinism_failures == 0 else 1, failures=determinism_failures)

    overall_pass = all(gate["status"] == "PASS" for gate in gates.values())
    summary = {
        "overall_status": "PASS" if overall_pass else "FAIL",
        "git_commit": _git_commit(),
        "gold_version": "citation-gold-v2" if args.gold.exists() else "missing",
        "gold_hash": _gold_hash(args.gold),
        "overlap_tokens": chunk_report.get("overlap_tokens", 0),
        "pdf_count": chunk_report.get("pdf_count", 0),
        "structure_failures": structure_failures,
        "canonical_source_loss": canonical_source_loss,
        "chunk_source_holes": chunk_source_holes,
        "chunk_source_overlaps": chunk_source_overlaps,
        "citation_missing_occurrences": citation_missing,
        "citation_extra_occurrences": citation_extra,
        "citation_wrong_targets": citation_wrong_targets,
        "citation_unresolved_expected_targets": citation_unresolved_expected,
        "citation_unattached_targets": unattached_targets,
        "wrong_regions": wrong_regions,
        "wrong_bibliography_scopes": wrong_scopes,
        "missed_used_references": missed_used_references,
        "unexpected_used_references": unexpected_used_references,
        "negative_false_positives": negative_false_positives,
        "determinism_failures": determinism_failures,
        "gates": gates,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    acceptance_path = args.run_dir / "acceptance-summary.json"
    acceptance_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    ledger.append(
        {
            "iteration": iteration,
            "timestamp": summary["timestamp"],
            "overall_status": summary["overall_status"],
            "gates": {name: gate["status"] for name, gate in gates.items()},
            "summary": {
                "structure_failures": structure_failures,
                "canonical_source_loss": canonical_source_loss,
                "chunk_source_holes": chunk_source_holes,
                "chunk_source_overlaps": chunk_source_overlaps,
                "citation_missing_occurrences": citation_missing,
                "citation_extra_occurrences": citation_extra,
                "citation_wrong_targets": citation_wrong_targets,
                "citation_unattached_targets": unattached_targets,
                "wrong_regions": wrong_regions,
                "wrong_bibliography_scopes": wrong_scopes,
                "determinism_failures": determinism_failures,
            },
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
        "gates": gates,
        "metrics": summary,
        "timestamp": summary["timestamp"],
        "reports": {
            "acceptance_summary": str(acceptance_path.relative_to(args.run_dir)),
            "failure_ledger": str(ledger_path.relative_to(args.run_dir)),
            "chunk_corpus_review": "chunk-corpus-review.json",
            "canonical_source_survival": "canonical-source-survival.json",
            "chunk_source_coverage": "chunk-source-coverage.json",
            "chunk_regions_scopes": "chunk-regions-scopes.json",
            "citation_gold_v2": "citation-gold-v2-validation.json",
        },
    }
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"acceptance-summary: {acceptance_path}")
    print(f"failure-ledger: {ledger_path}")
    print(f"contract-report: {contract_path}")
    return 0 if summary["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
