"""Pure release-report provenance helpers with no external-service imports."""

from __future__ import annotations

from typing import Any

PROVENANCE_SCHEMA_VERSION = 2
VALIDATION_ORIGIN_CURRENT = "current_head"
VALIDATION_ORIGIN_REUSED = "reused_previous_run"
VALIDATION_ORIGIN_MIXED_REUSED = "mixed_reused"
VALIDATION_HEAD_ATTRIBUTED = "attributed"
VALIDATION_HEAD_LEGACY_UNATTRIBUTED = "legacy_unattributed"
VALIDATION_HEAD_MIXED = "mixed_children"
SEARCH_QUALITY_PENDING = "PENDING_RERANK_OPTIMIZATION"
RERANK_PROVISIONAL_NOTICE = (
    "Rerank quality is provisional and will be revalidated after the dedicated "
    "rerank optimization task."
)
ENGINEERING_GATE_NAMES = (
    "contracts",
    "ci_local_equivalent",
    "ci_workflow_contract",
    "compile",
    "ruff",
    "mypy",
    "node_build",
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
)


def _drop_legacy_gate_fields(report: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(report)
    normalized.pop("gates", None)
    normalized.pop("reranker_blocker", None)
    return normalized


def _validation_fields(
    *,
    origin: str,
    validated_head: str | None,
    executed_this_run: bool,
    validation_head_status: str | None = None,
) -> dict[str, object]:
    return {
        "validation_origin": origin,
        "validated_head": validated_head,
        "executed_this_run": executed_this_run,
        "validation_head_status": validation_head_status
        or (
            VALIDATION_HEAD_ATTRIBUTED
            if validated_head is not None
            else VALIDATION_HEAD_LEGACY_UNATTRIBUTED
        ),
    }


def _annotate_validation(
    payload: dict[str, Any],
    *,
    origin: str,
    validated_head: str | None,
    executed_this_run: bool,
    validation_head_status: str | None = None,
) -> dict[str, Any]:
    return {
        **payload,
        **_validation_fields(
            origin=origin,
            validated_head=validated_head,
            executed_this_run=executed_this_run,
            validation_head_status=validation_head_status,
        ),
    }


def _reused_validation(
    payload: dict[str, Any], *, force_unattributed: bool = False
) -> dict[str, Any]:
    value = payload.get("validated_head")
    validated_head = (
        str(value)
        if not force_unattributed and isinstance(value, str) and value
        else None
    )
    return _annotate_validation(
        payload,
        origin=VALIDATION_ORIGIN_REUSED,
        validated_head=validated_head,
        executed_this_run=False,
        validation_head_status=(
            VALIDATION_HEAD_ATTRIBUTED
            if validated_head is not None
            else VALIDATION_HEAD_LEGACY_UNATTRIBUTED
        ),
    )


def _composite_reused_validation(
    payload: dict[str, Any],
    *,
    children: list[dict[str, Any]],
) -> dict[str, Any]:
    heads = {
        str(child["validated_head"])
        for child in children
        if isinstance(child.get("validated_head"), str)
        and child["validated_head"]
    }
    all_attributed = bool(children) and all(
        isinstance(child.get("validated_head"), str)
        and child["validated_head"]
        and child.get("validation_head_status") == VALIDATION_HEAD_ATTRIBUTED
        for child in children
    )
    if all_attributed and len(heads) == 1:
        return _annotate_validation(
            payload,
            origin=VALIDATION_ORIGIN_REUSED,
            validated_head=next(iter(heads)),
            executed_this_run=False,
        )
    return _annotate_validation(
        payload,
        origin=VALIDATION_ORIGIN_MIXED_REUSED,
        validated_head=None,
        executed_this_run=False,
        validation_head_status=VALIDATION_HEAD_MIXED,
    )


def _merge_query_reviews(
    case_ids: list[str],
    previous_reviews: list[dict[str, Any]],
    current_reviews: list[dict[str, Any]],
    *,
    current_head: str,
    previous_provenance_trusted: bool = True,
) -> list[dict[str, Any]]:
    previous_by_id = {item["id"]: item for item in previous_reviews}
    current_by_id = {item["id"]: item for item in current_reviews}
    return [
        (
            _annotate_validation(
                current_by_id[case_id],
                origin=VALIDATION_ORIGIN_CURRENT,
                validated_head=current_head,
                executed_this_run=True,
            )
            if case_id in current_by_id
            else _reused_validation(
                previous_by_id[case_id],
                force_unattributed=not previous_provenance_trusted,
            )
        )
        for case_id in case_ids
    ]


def _gate_record(
    passed: bool,
    *,
    origin: str,
    validated_head: str | None,
    executed_this_run: bool,
    validation_head_status: str | None = None,
) -> dict[str, object]:
    return {
        "status": "PASS" if passed else "FAIL",
        **_validation_fields(
            origin=origin,
            validated_head=validated_head,
            executed_this_run=executed_this_run,
            validation_head_status=validation_head_status,
        ),
    }


def _gate_passed(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return isinstance(value, dict) and value.get("status") == "PASS"


def _engineering_decision(gates: dict[str, dict[str, object]]) -> str:
    return (
        "GO"
        if set(gates) == set(ENGINEERING_GATE_NAMES)
        and all(_gate_passed(gates[name]) for name in ENGINEERING_GATE_NAMES)
        else "NO-GO"
    )


def _legacy_engineering_evidence(
    report: dict[str, Any], *, legacy_structural_head: str | None = None
) -> dict[str, dict[str, object]]:
    existing = report.get("engineering_gates")
    if isinstance(existing, dict):
        records: dict[str, dict[str, object]] = {}
        for name in ENGINEERING_GATE_NAMES:
            source_name = name
            if name in {"ci_local_equivalent", "ci_workflow_contract"}:
                source_name = name if name in existing else "ci"
            value = existing.get(source_name)
            if isinstance(value, dict):
                records[name] = _reused_validation(value)
        if set(records) == set(ENGINEERING_GATE_NAMES):
            return records

    legacy = report.get("gates", {})
    clean_room = report.get("clean_room", {})
    if not legacy_structural_head:
        raise RuntimeError(
            "Legacy structural gates require an explicitly verified execution HEAD."
        )
    structural = {
        "active_revision": bool(legacy.get("active_pointer")),
        "hard_filters": bool(legacy.get("contracts"))
        and clean_room.get("explicit_filter", {}).get("status") == "PASS",
        "hard_max": bool(legacy.get("hard_max")),
        "figure_provenance": bool(legacy.get("figure")),
        "source_provenance": bool(legacy.get("source_projection")),
        "citation_provenance": (
            clean_room.get("citation_provenance", {}).get("status") == "PASS"
        ),
        "evidence_replay": bool(legacy.get("evidence_replay")),
        "lifecycle": bool(legacy.get("lifecycle")),
        "gpu_restriction": bool(legacy.get("gpu_restricted")),
        "clean_room_pipeline": bool(
            clean_room.get("pipeline_completed_pdf_to_llm")
            and legacy.get("full_pipeline")
        ),
    }
    records = {
        name: _gate_record(
            passed,
            origin=VALIDATION_ORIGIN_REUSED,
            validated_head=legacy_structural_head,
            executed_this_run=False,
        )
        for name, passed in structural.items()
    }
    for name in (
        "contracts",
        "ci_local_equivalent",
        "ci_workflow_contract",
        "compile",
        "ruff",
        "mypy",
        "node_build",
    ):
        records[name] = _gate_record(
            False,
            origin=VALIDATION_ORIGIN_REUSED,
            validated_head=None,
            executed_this_run=False,
            validation_head_status=VALIDATION_HEAD_LEGACY_UNATTRIBUTED,
        )
    return records
