"""Permanent Cognee retrieval boundary contract.

Run directly; this project intentionally does not use pytest.

Static:
    python tests/contract/test_cognee_retrieval_boundary.py

Live retained dataset:
    python tests/contract/test_cognee_retrieval_boundary.py
        --live-data-dir data/validation/runs/<latest>
        --dataset <dataset-name>
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

FORBIDDEN_RETRIEVAL_NAMES = {
    "get_vector_engine",
    "get_graph_engine",
    "get_embedding_engine",
    "search_datapoint_vectors",
    "open_table",
}
FORBIDDEN_RETRIEVAL_TEXT = {
    "lancedb",
    "_DataPoint_text",
    "ChunkDataPoint_text",
}


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def static_contract() -> dict[str, object]:
    failures: list[str] = []
    retrieval_root = REPOSITORY_ROOT / "paperos_core" / "retrieval"
    for path in sorted(retrieval_root.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Name, ast.Attribute)):
                name = node.id if isinstance(node, ast.Name) else node.attr
                if name in FORBIDDEN_RETRIEVAL_NAMES:
                    failures.append(f"{path.name}: forbidden retrieval symbol {name}")
        for token in FORBIDDEN_RETRIEVAL_TEXT:
            if token.casefold() in source.casefold():
                failures.append(f"{path.name}: forbidden retrieval text {token}")

    compat = REPOSITORY_ROOT / "paperos_core" / "adapters" / "cognee" / "compat.py"
    compat_source = compat.read_text(encoding="utf-8")
    _require(
        "public" in compat_source
        and "ChunkDataPoint" in compat_source
        and "search_datapoint_vectors" in compat_source,
        "The private vector fallback lacks its required public-API limitation note.",
    )

    adapter_root = REPOSITORY_ROOT / "paperos_core" / "adapters" / "cognee"
    public_module_owners = {
        adapter_root / "configurator.py",
        adapter_root / "pipeline.py",
        adapter_root / "search.py",
    }
    llm_gateway_owner = adapter_root / "llm.py"
    for path in sorted((REPOSITORY_ROOT / "paperos_core").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "cognee" and path not in public_module_owners:
                        failures.append(f"{path}: public cognee import is not an exact exception")
                    elif alias.name.startswith("cognee.") and path != compat:
                        failures.append(f"{path}: private Cognee import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "cognee.infrastructure.llm":
                    names = {alias.name for alias in node.names}
                    if path != llm_gateway_owner or names != {"LLMGateway"}:
                        failures.append(f"{path}: LLMGateway exception is not exact")
                elif module == "cognee" or module.startswith("cognee."):
                    if path != compat:
                        failures.append(f"{path}: private Cognee import {module}")

    _require(not failures, "\n".join(failures))
    return {
        "status": "passed",
        "retrieval_module_count": len(list(retrieval_root.glob("*.py"))),
        "public_cognee_import_owners": sorted(
            owner.relative_to(REPOSITORY_ROOT).as_posix()
            for owner in public_module_owners
        ),
        "llm_gateway_owner": "paperos_core/adapters/cognee/llm.py",
        "private_fallback_owner": "paperos_core/adapters/cognee/compat.py",
    }


def _dataset_from_manifests(data_root: Path) -> str:
    names: set[str] = set()
    for path in (data_root / "cognee" / "manifests").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = payload.get("dataset", {})
        if isinstance(dataset, dict) and dataset.get("name"):
            names.add(str(dataset["name"]))
    _require(len(names) == 1, f"Expected one retained Cognee dataset, found: {sorted(names)}")
    return next(iter(names))


def _allowed_canonical_ids(data_root: Path, dataset: str) -> set[str]:
    result: set[str] = set()
    for path in (data_root / "cognee" / "manifests").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset_payload = payload.get("dataset")
        if isinstance(dataset_payload, dict):
            manifest_dataset = dataset_payload.get("name")
            if manifest_dataset and manifest_dataset != dataset:
                continue
        mapping = payload.get("canonical_to_cognee_id", {})
        if isinstance(mapping, dict):
            result.update(str(canonical_id) for canonical_id in mapping)
    return result


async def live_contract(
    data_root: Path,
    *,
    dataset: str | None,
    query: str,
) -> dict[str, Any]:
    resolved = data_root.expanduser().resolve(strict=False)
    from paperos_core.application import create_application
    from paperos_core.config import load_settings

    selected_dataset = dataset or _dataset_from_manifests(resolved)
    configured = load_settings()
    settings = configured.model_copy(
        update={
            "data": configured.data.model_copy(
                update={"directory": resolved, "dataset": selected_dataset}
            )
        }
    )
    application = create_application(settings)
    from paperos_core.adapters.cognee.search import CogneeSearchAdapter

    adapter = CogneeSearchAdapter(application.paths, application.knowledge_pipeline.compat)
    allowed = _allowed_canonical_ids(resolved, selected_dataset)
    _require(allowed, "No canonical IDs are registered for the selected dataset.")
    try:
        await application.start()
        searched = await adapter.graph_search(
            query,
            dataset=selected_dataset,
            top_k=8,
            search_type="GRAPH_COMPLETION",
        )
        recalled = await adapter.recall_context(
            query,
            dataset=selected_dataset,
            top_k=8,
            search_type="GRAPH_COMPLETION",
        )
    finally:
        await application.aclose()

    canonical_ids_present = all(
        hit.canonical_id in allowed for hit in (*searched, *recalled)
    )
    provenance_present = all(
        hit.source_chunk_ids or hit.references for hit in (*searched, *recalled)
    )
    public_api_sufficient = bool(searched and recalled) and (
        canonical_ids_present and provenance_present
    )
    return {
        "status": "passed" if public_api_sufficient else "unsupported_by_cognee_1_4_0",
        "dataset": selected_dataset,
        "query": query,
        "search_hit_count": len(searched),
        "recall_hit_count": len(recalled),
        "public_api_sufficient": public_api_sufficient,
        "canonical_ids_present": canonical_ids_present,
        "chunk_or_graph_provenance_present": provenance_present,
        "fallback_required": not public_api_sufficient,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-data-dir", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--query", default="weak coupling topology preservation")
    args = parser.parse_args()
    report: dict[str, object] = {"static": static_contract()}
    if args.live_data_dir is not None:
        report["live"] = asyncio.run(
            live_contract(
                args.live_data_dir,
                dataset=args.dataset,
                query=args.query,
            )
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
