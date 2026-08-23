#!/usr/bin/env python3
"""Run the four authoritative six-paper chunk/citation acceptance gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
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


def _load_json(path: Path) -> Any:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_py(script: str, *script_args: str) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPOSITORY_ROOT)
    return subprocess.call(
        [sys.executable, script, *script_args],
        cwd=REPOSITORY_ROOT,
        env=env,
    )


def _gate(status: bool, failures: int, **metrics: Any) -> dict[str, Any]:
    return {"status": "PASS" if status and failures == 0 else "FAIL", "failures": failures, **metrics}


def _projection_hashes(run_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(run_dir.glob("*.chunks.json")):
        payload = _load_json(path)
        deterministic_projection = {
            "snapshot_id": payload.get("snapshot_id"),
            "chunking_version": payload.get("chunking_version"),
            "chunks": payload.get("chunks", []),
            "citation_mentions": payload.get("citation_mentions", []),
        }
        hashes[path.name] = hashlib.sha256(
            json.dumps(
                deterministic_projection,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    return hashes


def _source_anchor_hashes(gold_report: dict[str, Any]) -> dict[str, str | None]:
    return {
        paper: result.get("source_anchor_digest")
        for paper, result in sorted(gold_report.get("papers", {}).items())
    }


def _chunk_structure_metrics(run_dir: Path, hard_max: int) -> tuple[int, int]:
    hard_max_violations = 0
    empty_chunks = 0
    for path in run_dir.glob("*.chunks.json"):
        for chunk in _load_json(path).get("chunks", []):
            hard_max_violations += int((chunk.get("token_count") or 0) > hard_max)
            empty_chunks += int(not (chunk.get("text") or "").strip())
    return hard_max_violations, empty_chunks


def _write_result_package(
    *,
    run_dir: Path,
    gold: Path,
    summary: dict[str, Any],
) -> Path:
    candidates = [
        run_dir / "acceptance-summary.json",
        run_dir / "canonical-source-survival.json",
        run_dir / "chunk-source-coverage.json",
        run_dir / "chunk-regions-scopes.json",
        run_dir / "citation-gold-v3-validation.json",
        run_dir / "reference-usage.json",
        run_dir / "chunk-corpus-review.json",
        run_dir / "failure-ledger.json",
        run_dir / "logs" / "contracts" / CONTRACT_REPORT_NAME,
        gold,
        gold.with_name("gold-v3-audit.json"),
        gold.with_name("gold-v3-audit.md"),
        *sorted(run_dir.glob("*.chunks.json")),
        *sorted(run_dir.glob("*.chunks.md")),
        *sorted((run_dir / "logs").rglob("*")),
    ]
    files = sorted({path.resolve() for path in candidates if path.is_file()})
    manifest_path = run_dir / "result-manifest.json"
    manifest = {
        "git_commit": summary["git_commit"],
        "gold_version": summary["gold_version"],
        "gold_hash": summary["gold_hash"],
        "overall_status": summary["overall_status"],
        "files": [
            {
                "path": (
                    str(path.relative_to(REPOSITORY_ROOT))
                    if path.is_relative_to(REPOSITORY_ROOT)
                    else path.name
                ),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    files.append(manifest_path.resolve())
    package = run_dir / "result.zip"
    with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            if path.is_relative_to(run_dir.resolve()):
                arcname = path.relative_to(run_dir.resolve())
            elif path.is_relative_to(REPOSITORY_ROOT):
                arcname = Path("repository") / path.relative_to(REPOSITORY_ROOT)
            else:
                arcname = Path(path.name)
            archive.write(path, arcname.as_posix())
    return package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/validation/runs/chunk"))
    parser.add_argument("--corpus-dir", type=Path, default=Path("data/validation/corpus/chunk"))
    parser.add_argument("--gold", type=Path, default=Path("tests/fixtures/chunk/citation_gold_v3.json"))
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--rebuild-canonical", action="store_true")
    parser.add_argument("--skip-determinism", action="store_true")
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    args.corpus_dir = args.corpus_dir.resolve()
    args.gold = args.gold.resolve()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    chunk_args = [
        "tests/validation/chunk_corpus_review.py",
        "--run-dir", str(args.run_dir),
        "--corpus-dir", str(args.corpus_dir),
        "--overlap-tokens", "0",
        "--rechunk-canonical",
    ]
    if args.rebuild_canonical:
        chunk_args.append("--rebuild-canonical")

    structure_code = _run_py(*chunk_args)
    survival_code = _run_py(
        "tests/validation/validate_canonical_source_survival.py",
        "--run-dir", str(args.run_dir),
        "--corpus-dir", str(args.corpus_dir),
        "--gold", str(args.gold),
    )
    coverage_code = _run_py(
        "tests/validation/validate_chunk_source_coverage.py",
        "--run-dir", str(args.run_dir),
        "--corpus-dir", str(args.corpus_dir),
    )
    regions_code = _run_py(
        "tests/validation/validate_chunk_regions_scopes.py",
        "--run-dir", str(args.run_dir),
        "--corpus-dir", str(args.corpus_dir),
    )
    gold_code = _run_py(
        "tests/validation/validate_citation_gold_v3.py",
        "--gold", str(args.gold),
        "--run-dir", str(args.run_dir),
    )
    reference_code = _run_py(
        "tests/validation/validate_reference_usage.py",
        "--gold", str(args.gold),
        "--run-dir", str(args.run_dir),
        "--corpus-dir", str(args.corpus_dir),
    )

    chunk_report = _load_json(args.run_dir / "chunk-corpus-review.json")
    survival_report = _load_json(args.run_dir / "canonical-source-survival.json")
    coverage_report = _load_json(args.run_dir / "chunk-source-coverage.json")
    regions_report = _load_json(args.run_dir / "chunk-regions-scopes.json")
    gold_report = _load_json(args.run_dir / "citation-gold-v3-validation.json")
    reference_report = _load_json(args.run_dir / "reference-usage.json")

    hard_max = int(chunk_report.get("chunk_hard_max_tokens", 0))
    hard_max_violations, empty_chunks = _chunk_structure_metrics(args.run_dir, hard_max)
    structure_failures = int(chunk_report.get("pdf_count", 0)) - int(chunk_report.get("pass_count", 0))
    wrong_regions = int(regions_report.get("wrong_regions", 0))
    wrong_namespaces = int(regions_report.get("wrong_namespaces", 0))
    source_failures = int(survival_report.get("failure_count", 0))
    holes = int(coverage_report.get("chunk_source_holes", 0))
    overlaps = int(coverage_report.get("chunk_source_overlaps", 0))
    citation_metrics = {
        "missing_occurrences": int(gold_report.get("missing_occurrences", 0)),
        "extra_occurrences": int(gold_report.get("extra_occurrences", 0)),
        "wrong_targets": int(gold_report.get("wrong_targets", 0)),
        "unresolved_expected_targets": int(gold_report.get("unresolved_expected_targets", 0)),
        "unattached_targets": int(gold_report.get("unattached_targets", 0)),
        "wrong_namespaces": int(gold_report.get("wrong_namespaces", 0)),
        "source_mapping_failures": int(gold_report.get("source_mapping_failures", 0)),
        "unexpected_used_references": int(reference_report.get("unexpected_used_references", 0)),
        "missed_expected_used_references": int(reference_report.get("missed_used_references", 0)),
    }

    source_failure_count = source_failures + holes + overlaps
    structure_failure_count = structure_failures + wrong_regions + wrong_namespaces + hard_max_violations + empty_chunks
    citation_failure_count = sum(citation_metrics.values())
    gates: dict[str, dict[str, Any]] = {
        "source": _gate(
            survival_code == 0 and coverage_code == 0,
            source_failure_count,
            canonical_source_loss=survival_report.get("canonical_source_loss", 0),
            gold_canonical_source_loss=survival_report.get("gold_canonical_source_loss", 0),
            chunk_holes=holes,
            chunk_overlaps=overlaps,
        ),
        "structure": _gate(
            structure_code == 0 and regions_code == 0,
            structure_failure_count,
            pdf_failures=structure_failures,
            wrong_regions=wrong_regions,
            wrong_namespaces=wrong_namespaces,
            hard_max_violations=hard_max_violations,
            empty_chunks=empty_chunks,
        ),
        "citation_gold_v3": _gate(
            gold_code == 0 and reference_code == 0,
            citation_failure_count,
            **citation_metrics,
        ),
        "determinism": {"status": "NOT_CHECKED", "failures": 0},
    }

    determinism_failures = 0
    first_projection_hashes = _projection_hashes(args.run_dir)
    first_anchor_hashes = _source_anchor_hashes(gold_report)
    if not args.skip_determinism and all(gate["status"] == "PASS" for name, gate in gates.items() if name != "determinism"):
        second_structure = _run_py(*chunk_args)
        second_gold = _run_py(
            "tests/validation/validate_citation_gold_v3.py",
            "--gold", str(args.gold),
            "--run-dir", str(args.run_dir),
        )
        second_gold_report = _load_json(args.run_dir / "citation-gold-v3-validation.json")
        second_projection_hashes = _projection_hashes(args.run_dir)
        second_anchor_hashes = _source_anchor_hashes(second_gold_report)
        deterministic = (
            second_structure == 0
            and second_gold == 0
            and first_projection_hashes == second_projection_hashes
            and first_anchor_hashes == second_anchor_hashes
        )
        determinism_failures = 0 if deterministic else 1
        gates["determinism"] = _gate(
            deterministic,
            determinism_failures,
            projection_hashes=second_projection_hashes,
            source_anchor_hashes=second_anchor_hashes,
        )

    overall_pass = all(gate["status"] == "PASS" for gate in gates.values())
    summary = {
        "overall_status": "PASS" if overall_pass else "FAIL",
        "git_commit": _git_commit(),
        "gold_version": "citation-gold-v3",
        "gold_hash": _sha256(args.gold),
        "pdf_count": chunk_report.get("pdf_count", 0),
        "overlap_tokens": chunk_report.get("overlap_tokens", 0),
        "canonical_source_loss": survival_report.get("canonical_source_loss", 0),
        "gold_canonical_source_loss": survival_report.get("gold_canonical_source_loss", 0),
        "chunk_source_holes": holes,
        "chunk_source_overlaps": overlaps,
        "wrong_regions": wrong_regions,
        "wrong_namespaces": wrong_namespaces,
        "hard_max_violations": hard_max_violations,
        "empty_chunks": empty_chunks,
        **citation_metrics,
        "determinism_failures": determinism_failures,
        "gates": gates,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    acceptance_path = args.run_dir / "acceptance-summary.json"
    acceptance_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    ledger_path = args.run_dir / "failure-ledger.json"
    ledger = _load_json(ledger_path)
    if not isinstance(ledger, list):
        ledger = []
    ledger.append(
        {
            "iteration": args.iteration if args.iteration is not None else len(ledger) + 1,
            "timestamp": summary["timestamp"],
            "git_commit": summary["git_commit"],
            "overall_status": summary["overall_status"],
            "gates": {name: gate["status"] for name, gate in gates.items()},
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
        "gates": gates,
        "metrics": summary,
        "reports": {
            "source": ["canonical-source-survival.json", "chunk-source-coverage.json"],
            "structure": ["chunk-corpus-review.json", "chunk-regions-scopes.json"],
            "citation_gold_v3": ["citation-gold-v3-validation.json", "reference-usage.json"],
        },
    }
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    package = _write_result_package(run_dir=args.run_dir, gold=args.gold, summary=summary)

    print(json.dumps(summary, indent=2))
    print(f"acceptance-summary: {acceptance_path}")
    print(f"contract-report: {contract_path}")
    print(f"result-package: {package}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
