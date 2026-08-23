#!/usr/bin/env python3
"""Validate production citations are not known negative false positives."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.validation.gold_audit_candidate_builder import is_negative_bracket


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


def validate_paper(*, chunks_json: dict) -> dict:
    failures: list[dict] = []
    for mention in chunks_json.get("citation_mentions", []):
        surface = mention.get("surface_text") or ""
        if not surface.startswith("[") or not surface.endswith("]"):
            continue
        inner = surface[1:-1]
        domain = (mention.get("metadata") or {}).get("source_domain", "text")
        if is_negative_bracket(inner, source_domain=domain):
            failures.append(
                {
                    "failure_type": "negative_false_positive",
                    "surface": surface,
                    "element_id": mention.get("element_id"),
                    "page": mention.get("page"),
                }
            )
    return {
        "negative_false_positives": len(failures),
        "failures": failures,
        "pass": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("data/validation/runs/chunk"))
    args = parser.parse_args()

    papers = []
    total = 0
    for path in sorted(args.run_dir.glob("*.chunks.json")):
        chunks_json = json.loads(path.read_text(encoding="utf-8"))
        result = validate_paper(chunks_json=chunks_json)
        total += result["negative_false_positives"]
        papers.append({"result_file": str(path), **result})

    report = {
        "git_commit": _git_commit(),
        "negative_false_positives": total,
        "papers": papers,
        "pass": total == 0,
    }
    output = args.run_dir / "negative-citations.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "report": str(output)}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
