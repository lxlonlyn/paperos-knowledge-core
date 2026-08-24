"""Direct contract for current-Snapshot selection and enrichment reuse.

This project intentionally does not use pytest. Run:

    python tests/contract/test_rebuild_enrichment_lifecycle.py \
      --live-data-dir data/validation/scholarly_work_reference/output
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.adapters.cognee.pipeline_tasks import semantic_enrichment_task
from paperos_core.domain.canonical import ChunkProjection
from paperos_core.domain.knowledge import SemanticEnrichment
from paperos_core.errors import CogneeStorageError
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.paths import build_data_paths


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _empty_enrichment() -> SemanticEnrichment:
    return SemanticEnrichment(
        entities=[],
        claims=[],
        relations=[],
        summaries=[],
        model="contract-model",
        provider="contract-provider",
        model_version="contract-model",
        prompt_name="semantic_enrichment",
        prompt_version="contract",
        prompt_sha256="0" * 64,
        covered_chunk_ids=[],
        uncovered_chunk_ids=[],
        coverage_ratio=0.0,
    )


class _LLMProbe:
    def __init__(self, enrichment: SemanticEnrichment) -> None:
        self.enrichment = enrichment
        self.calls = 0

    async def enrich(
        self, bundle: Any, chunks: list[Any], scholarly: Any = None
    ) -> SemanticEnrichment:
        self.calls += 1
        return self.enrichment


def _identity_bound(snapshot_id: str) -> Any:
    return SimpleNamespace(
        bundle=SimpleNamespace(snapshot=SimpleNamespace(id=snapshot_id)),
        projection=ChunkProjection(snapshot_id=snapshot_id, chunks=[]),
        scholarly=None,
    )


def current_snapshot_contract(live_data_dir: Path) -> dict[str, object]:
    repository = CanonicalRepository(build_data_paths(live_data_dir))
    all_ids = repository.list_snapshot_ids()
    latest_ids = repository.list_latest_snapshot_ids()
    document_ids = {
        repository.get_snapshot(snapshot_id).document_id for snapshot_id in all_ids
    }
    _require(len(all_ids) > len(latest_ids), "Historical snapshots were not retained.")
    _require(
        len(latest_ids) == len(document_ids) == 4,
        "Expected four current Documents and four latest Snapshots.",
    )
    return {
        "status": "passed",
        "all_snapshot_count": len(all_ids),
        "current_document_count": len(document_ids),
        "latest_snapshot_count": len(latest_ids),
        "latest_snapshot_ids": latest_ids,
    }


async def enrichment_reuse_contract() -> dict[str, object]:
    enrichment = _empty_enrichment()
    with tempfile.TemporaryDirectory(prefix="paperos-enrichment-contract-") as directory:
        root = Path(directory)

        existing_id = "snapshot_existing"
        (root / f"{existing_id}.json").write_text(
            enrichment.model_dump_json(indent=2),
            encoding="utf-8",
        )
        existing_probe = _LLMProbe(enrichment)
        reused = await semantic_enrichment_task(
            [_identity_bound(existing_id)],
            llm=existing_probe,
            enrichment_root=root,
            reuse_existing=True,
            generate_if_missing=False,
        )
        _require(len(reused) == 1, "Existing enrichment was not returned.")
        _require(existing_probe.calls == 0, "Reuse unexpectedly called the LLM.")

        missing_probe = _LLMProbe(enrichment)
        try:
            await semantic_enrichment_task(
                [_identity_bound("snapshot_missing")],
                llm=missing_probe,
                enrichment_root=root,
                reuse_existing=True,
                generate_if_missing=False,
            )
        except CogneeStorageError:
            pass
        else:
            raise RuntimeError("Missing enrichment did not fail immediately.")
        _require(missing_probe.calls == 0, "Missing-artifact failure called the LLM.")

        generated_id = "snapshot_generated"
        generation_probe = _LLMProbe(enrichment)
        await semantic_enrichment_task(
            [_identity_bound(generated_id)],
            llm=generation_probe,
            enrichment_root=root,
            reuse_existing=True,
            generate_if_missing=True,
        )
        _require(generation_probe.calls == 1, "Explicit generation did not call the LLM.")
        _require(
            (root / f"{generated_id}.json").is_file(),
            "Generated enrichment was not persisted.",
        )

    return {
        "status": "passed",
        "existing_artifact_llm_calls": existing_probe.calls,
        "missing_artifact_llm_calls": missing_probe.calls,
        "explicit_generation_llm_calls": generation_probe.calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-data-dir", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "current_snapshots": current_snapshot_contract(args.live_data_dir),
        "enrichment_reuse": asyncio.run(enrichment_reuse_contract()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
