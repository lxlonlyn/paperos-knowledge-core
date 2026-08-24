"""Permanent Cognee retrieval boundary contract.

Run directly; this project intentionally does not use pytest.

Static:
    python tests/contract/test_cognee_retrieval_boundary.py

Live retained dataset:
    python tests/contract/test_cognee_retrieval_boundary.py
        --live-data-dir data/validation/scholarly_work_reference/output
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


async def live_contract(
    data_root: Path,
    *,
    dataset: str | None,
    query: str | None,
) -> dict[str, Any]:
    resolved = data_root.expanduser().resolve(strict=False)
    from paperos_core.application import create_application
    from paperos_core.config import load_settings
    from paperos_core.retrieval.candidates import QueryRequest
    from paperos_core.retrieval.corpus import CorpusView

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
    try:
        await application.start()
        question = query or "What methods and limitations are described?"
        response = await application.services.retrieval.query(
            QueryRequest(query=question, dataset=selected_dataset)
        )
        corpus = CorpusView.load(
            application.paths,
            application.canonical_repository,
            application.registry,
            application.scholarly_registry,
        )
        grounded = all(
            item.chunk_id in corpus.chunks
            and item.text == corpus.chunks[item.chunk_id].text
            and item.document_id == corpus.chunks[item.chunk_id].document_id
            for item in response.evidence
        )
        _require(response.evidence, "Chunk vector boundary returned no evidence.")
        _require(grounded, "Retrieval returned non-canonical evidence.")
        _require(
            set(response.channels_used) <= {"lexical", "vector"},
            f"Unexpected first-stage channels: {response.channels_used}",
        )
        report = {
            "status": "passed",
            "dataset": selected_dataset,
            "chunk_ids": [item.chunk_id for item in response.evidence],
            "channels": response.channels_used,
            "source_grounded": grounded,
        }
    finally:
        await application.aclose()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-data-dir", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument(
        "--query",
        help="Optional shared query override; default queries come from real DataPoints.",
    )
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
