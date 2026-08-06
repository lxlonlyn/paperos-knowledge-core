"""Contract spike: does Cognee's public pipeline auto-establish provenance?

The spike feeds a custom DataPoint through ``cognee.run_custom_pipeline`` and
``add_data_points`` in two modes:

1. with a plain ``PipelineItem`` (no relational Data registration);
2. with a ``PipelineItem`` whose id points at a registered Cognee Data row.

If mode 1 produced provenance, PaperOS could delete its private
``ensure_dataset``/``register_data_item`` ORM compatibility code. The spike
asserts the opposite: Cognee's public pipeline auto-creates the dataset and
logs pipeline runs, but stable data provenance requires the minimal private
registration that stays centralized in ``compat.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from uuid import UUID

import pytest
from cognee.infrastructure.engine import DataPoint

from paperos_core.adapters.cognee.compat import (
    CogneeCompatibilityAdapter,
    PipelineItem,
    task,
)
from paperos_core.domain.datapoints import cognee_uuid
from paperos_core.domain.documents import SourceFile
from paperos_core.paths import build_data_paths

_TRACKED_ENV = (
    "SYSTEM_ROOT_DIRECTORY",
    "DATA_ROOT_DIRECTORY",
    "CACHE_ROOT_DIRECTORY",
    "COGNEE_LOGS_DIR",
    "DB_PROVIDER",
    "DB_PATH",
    "DB_NAME",
    "VECTOR_DB_PROVIDER",
    "VECTOR_DB_URL",
    "GRAPH_DATABASE_PROVIDER",
    "GRAPH_DATASET_DATABASE_HANDLER",
    "GRAPH_FILE_PATH",
    "COGNEE_SKIP_CONNECTION_TEST",
    "ENABLE_BACKEND_ACCESS_CONTROL",
    "REQUIRE_AUTHENTICATION",
    "TELEMETRY_DISABLED",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_ENDPOINT",
    "EMBEDDING_MODEL",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_MAX_COMPLETION_TOKENS",
    "EMBEDDING_BATCH_SIZE",
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_ENDPOINT",
)


class _SpikeNode(DataPoint):  # type: ignore[misc]
    text: str
    metadata: dict = {"index_fields": []}  # noqa: RUF012


async def _store_spike_node(
    data: list[object],
    ctx: object = None,
    *,
    compat: CogneeCompatibilityAdapter,
) -> list[object]:
    node = _SpikeNode(id=cognee_uuid("spike_node"), text="spike node")
    await compat.add_data_points(
        [node],
        custom_edges=[],
        embed_triplets=False,
        ctx=ctx,
    )
    return data


def _source(root: Path) -> SourceFile:
    return SourceFile(
        id="src_spike",
        sha256="0" * 64,
        original_filename="spike.pdf",
        size_bytes=10,
        storage_path=root / "spike.pdf",
    )


def _spike_env(root: Path) -> dict[str, str]:
    return {
        "SYSTEM_ROOT_DIRECTORY": str(root / "system"),
        "DATA_ROOT_DIRECTORY": str(root / "data"),
        "CACHE_ROOT_DIRECTORY": str(root / "cache"),
        "COGNEE_LOGS_DIR": str(root / "logs"),
        "DB_PROVIDER": "sqlite",
        "DB_PATH": str(root / "cognee.db"),
        "DB_NAME": "cognee_db",
        "VECTOR_DB_PROVIDER": "lancedb",
        "VECTOR_DB_URL": str(root / "lancedb"),
        "GRAPH_DATABASE_PROVIDER": "kuzu",
        "GRAPH_DATASET_DATABASE_HANDLER": "kuzu",
        "GRAPH_FILE_PATH": str(root / "kuzu"),
        "COGNEE_SKIP_CONNECTION_TEST": "true",
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "REQUIRE_AUTHENTICATION": "false",
        "TELEMETRY_DISABLED": "true",
        "EMBEDDING_PROVIDER": "openai_compatible",
        "EMBEDDING_ENDPOINT": "http://127.0.0.1:9/v1",
        "EMBEDDING_MODEL": "default",
        "EMBEDDING_DIMENSIONS": "768",
        "EMBEDDING_MAX_COMPLETION_TOKENS": "2048",
        "EMBEDDING_BATCH_SIZE": "5",
        "LLM_PROVIDER": "custom",
        "LLM_MODEL": "openai/spike",
        "LLM_ENDPOINT": "http://127.0.0.1:9/v1",
    }


async def _run_spike(
    root: Path,
    *,
    registered: bool,
) -> CogneeCompatibilityAdapter:
    import cognee

    compat = CogneeCompatibilityAdapter(build_data_paths(root))
    dataset = await compat.ensure_dataset("spike")
    data_id = None
    if registered:
        data_id = await compat.register_data_item(
            dataset=dataset,
            source=_source(root),
            snapshot_id="snapshot_spike",
            document_id="doc_spike",
            title="Spike",
        )
    item = PipelineItem(id=data_id, data_id=data_id, bundle=None, source=None)
    run_infos = await cognee.run_custom_pipeline(
        tasks=[task(_store_spike_node, batch_size=1, compat=compat).task],
        data=item,
        dataset="spike",
        pipeline_name="spike_pipeline",
    )
    run_info = next(iter(run_infos.values()))
    counts = await compat.provenance_counts(
        dataset_id=dataset.id,
        data_id=data_id,
        pipeline_run_id=UUID(str(run_info.pipeline_run_id)),
    )
    return counts


@pytest.mark.asyncio
async def test_public_pipeline_requires_registered_data_for_provenance(
    gate1_run_dir: Path,
) -> None:
    root = gate1_run_dir / "provenance-spike"
    saved = {name: os.environ.get(name) for name in _TRACKED_ENV}
    try:
        os.environ.update(_spike_env(root))
        CogneeCompatibilityAdapter.reset_configuration_caches()
        unregistered = await _run_spike(root, registered=False)
        assert unregistered.provenance_backend == "none"
        assert unregistered.provenance_node_count == 0
        assert unregistered.provenance_edge_count == 0
        registered = await _run_spike(root, registered=True)
        assert registered.provenance_backend in {"relational", "graph"}
        assert registered.provenance_node_count > 0
        assert registered.provenance_edge_count >= 0
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        CogneeCompatibilityAdapter.reset_configuration_caches()
