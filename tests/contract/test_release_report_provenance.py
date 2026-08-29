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
    VALIDATION_HEAD_ATTRIBUTED,
    VALIDATION_HEAD_LEGACY_UNATTRIBUTED,
    VALIDATION_HEAD_MIXED,
    VALIDATION_ORIGIN_CURRENT,
    VALIDATION_ORIGIN_MIXED_REUSED,
    VALIDATION_ORIGIN_REUSED,
    _annotate_validation,
    _composite_reused_validation,
    _drop_legacy_gate_fields,
    _engineering_decision,
    _gate_record,
    _legacy_engineering_evidence,
    _merge_query_reviews,
    _reused_validation,
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
    explicit = _reused_validation(
        {"id": "explicit_query", "status": "PASS", "validated_head": previous_head}
    )
    _require(
        explicit["validated_head"] == previous_head
        and explicit["validation_head_status"] == VALIDATION_HEAD_ATTRIBUTED,
        "Explicit historical HEAD was not preserved",
    )
    merged = _merge_query_reviews(
        ["old_query", "current_query"],
        [
            {"id": "old_query", "status": "PASS"},
            {"id": "current_query", "status": "FAIL"},
        ],
        [{"id": "current_query", "status": "PASS"}],
        current_head=current_head,
        previous_provenance_trusted=False,
    )
    old, current = merged
    _require(
        old["status"] == "PASS"
        and old["validation_origin"] == VALIDATION_ORIGIN_REUSED
        and old["validated_head"] is None
        and old["validation_head_status"]
        == VALIDATION_HEAD_LEGACY_UNATTRIBUTED
        and old["executed_this_run"] is False,
        "Unattributed historical query received a guessed HEAD",
    )
    _require(
        current["status"] == "PASS"
        and current["validation_origin"] == VALIDATION_ORIGIN_CURRENT
        and current["validated_head"] == current_head
        and current["executed_this_run"] is True,
        "Current query execution provenance is incorrect",
    )
    diagnostic = _reused_validation(
        {
            "id": "adadiv_self_limitation_default",
            "status": "PASS",
            "validated_head": previous_head,
            "window_count": 3,
            "winning_window_index": 2,
            "rerank_rank": 12,
        },
        force_unattributed=True,
    )
    _require(
        diagnostic["status"] == "PASS"
        and diagnostic["validated_head"] is None
        and diagnostic["validation_head_status"]
        == VALIDATION_HEAD_LEGACY_UNATTRIBUTED,
        "Legacy reranker diagnostic retained a guessed clean-room HEAD",
    )
    return merged


def _composite_provenance_contract() -> dict[str, object]:
    head_a = "a" * 40
    head_b = "b" * 40
    child_a = _annotate_validation(
        {"status": "PASS"},
        origin=VALIDATION_ORIGIN_REUSED,
        validated_head=head_a,
        executed_this_run=False,
    )
    child_b = _annotate_validation(
        {"status": "PASS"},
        origin=VALIDATION_ORIGIN_REUSED,
        validated_head=head_b,
        executed_this_run=False,
    )
    composite = _composite_reused_validation(
        {"status": "PASS", "validated_head": head_a},
        children=[child_a, child_b],
    )
    _require(
        composite["validation_origin"] == VALIDATION_ORIGIN_MIXED_REUSED
        and composite["validated_head"] is None
        and composite["validation_head_status"] == VALIDATION_HEAD_MIXED,
        "Mixed composite was assigned one child's HEAD",
    )
    return composite


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
        legacy_structural_head=previous_head,
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
    for name in (
        "contracts",
        "ci_local_equivalent",
        "ci_workflow_contract",
        "compile",
        "ruff",
        "mypy",
        "node_build",
    ):
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
        "mixed_composite": _composite_provenance_contract(),
        "gates": _gate_provenance_contract(),
        "ci": _ci_contract(),
        "search_quality_status": SEARCH_QUALITY_PENDING,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
