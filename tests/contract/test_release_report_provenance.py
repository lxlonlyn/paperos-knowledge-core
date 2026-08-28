"""Direct contract for release-report validation provenance and gate semantics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from tests.validation.release_provenance import (
    ENGINEERING_GATE_NAMES,
    SEARCH_QUALITY_PENDING,
    VALIDATION_ORIGIN_CURRENT,
    VALIDATION_ORIGIN_REUSED,
    _drop_legacy_gate_fields,
    _engineering_decision,
    _gate_record,
    _legacy_engineering_evidence,
    _merge_query_reviews,
)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _legacy_field_contract() -> dict[str, object]:
    normalized = _drop_legacy_gate_fields(
        {
            "head": "1" * 40,
            "gates": {"reranker_blocker": True},
            "reranker_blocker": {"status": "PASS"},
        }
    )
    _require("gates" not in normalized, "Legacy blocking gates remain public")
    _require(
        "reranker_blocker" not in normalized,
        "Provisional reranker quality remains a blocking gate",
    )
    return normalized


def _query_provenance_contract() -> list[dict[str, object]]:
    previous_head = "1" * 40
    current_head = "2" * 40
    merged = _merge_query_reviews(
        ["old_query", "current_query"],
        [
            {"id": "old_query", "status": "PASS"},
            {"id": "current_query", "status": "FAIL"},
        ],
        [{"id": "current_query", "status": "PASS"}],
        previous_head=previous_head,
        current_head=current_head,
    )
    old, current = merged
    _require(
        old["status"] == "PASS"
        and old["validation_origin"] == VALIDATION_ORIGIN_REUSED
        and old["validated_head"] == previous_head
        and old["executed_this_run"] is False,
        "Historical query was presented as a current-HEAD execution",
    )
    _require(
        current["status"] == "PASS"
        and current["validation_origin"] == VALIDATION_ORIGIN_CURRENT
        and current["validated_head"] == current_head
        and current["executed_this_run"] is True,
        "Current query execution provenance is incorrect",
    )
    return merged


def _gate_provenance_contract() -> dict[str, dict[str, object]]:
    previous_head = "1" * 40
    current_head = "2" * 40
    legacy_report = {
        "gates": {
            "active_pointer": True,
            "contracts": True,
            "hard_max": True,
            "figure": True,
            "source_projection": True,
            "evidence_replay": True,
            "lifecycle": True,
            "gpu_restricted": True,
            "full_pipeline": True,
        },
        "clean_room": {
            "pipeline_completed_pdf_to_llm": True,
            "explicit_filter": {"status": "PASS"},
            "citation_provenance": {"status": "PASS"},
        },
    }
    gates = _legacy_engineering_evidence(
        legacy_report,
        legacy_head=previous_head,
    )
    for name in (
        "active_revision",
        "hard_filters",
        "hard_max",
        "figure_provenance",
        "source_provenance",
        "citation_provenance",
        "evidence_replay",
        "lifecycle",
        "gpu_restriction",
        "clean_room_pipeline",
    ):
        gate = gates[name]
        _require(
            gate["status"] == "PASS"
            and gate["validation_origin"] == VALIDATION_ORIGIN_REUSED
            and gate["validated_head"] == previous_head
            and gate["executed_this_run"] is False,
            f"Historical gate provenance is incorrect: {name}",
        )
    for name in ("contracts", "ci", "compile", "ruff", "mypy", "node_build"):
        gates[name] = _gate_record(
            True,
            origin=VALIDATION_ORIGIN_CURRENT,
            validated_head=current_head,
            executed_this_run=True,
        )
    _require(
        set(gates) == set(ENGINEERING_GATE_NAMES),
        "Formal engineering gate set is incomplete",
    )
    _require(
        _engineering_decision(gates) == "GO",
        "Pending semantic quality incorrectly blocks engineering release",
    )
    _require(
        SEARCH_QUALITY_PENDING not in gates,
        "Search quality was accidentally made an engineering gate",
    )
    for name, gate in gates.items():
        _require(
            {"validation_origin", "validated_head", "executed_this_run"}
            <= set(gate),
            f"Gate lacks validation provenance: {name}",
        )
    return gates


def _ci_contract() -> dict[str, object]:
    workflow = (
        REPOSITORY_ROOT / ".github/workflows/cross-platform.yml"
    ).read_text(encoding="utf-8")
    required = (
        "ubuntu-latest",
        "windows-latest",
        "npm run build",
        "python -m compileall",
        "ruff check",
        "mypy paperos_core",
        "test_portable_data_paths.py",
        "test_runtime_query_contracts.py",
        "external-boundaries:",
        "test_active_canonical_revision.py",
        "test_query_filter_contracts.py",
    )
    missing = [item for item in required if item not in workflow]
    _require(not missing, f"Cross-platform CI is missing required gates: {missing}")
    _require(
        "runs-on: [self-hosted, linux, x64, paperos-external]" in workflow,
        "Real Cognee/vector contracts are not isolated to the Linux external job",
    )
    return {"status": "passed", "required_entries": list(required)}


def main() -> None:
    report = {
        "status": "passed",
        "legacy_fields": _legacy_field_contract(),
        "queries": _query_provenance_contract(),
        "gates": _gate_provenance_contract(),
        "ci": _ci_contract(),
        "search_quality_status": SEARCH_QUALITY_PENDING,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
