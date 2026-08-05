"""Debugger entry for real, retained stages of the PaperOS pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.application import create_application
from paperos_core.config import load_settings


async def run(args: argparse.Namespace) -> dict[str, object]:
    application = create_application(load_settings())
    application.storage.initialize()
    started = False
    try:
        if args.mode in {"cognee", "full"}:
            await application.start()
            started = True
        if args.mode == "mineru":
            result = await application.services.ingestion.ingest_pdf_to_parser(args.pdf)
            return result.model_dump(mode="json")
        if args.mode == "canonical":
            parse_run = application.parser_artifacts.get_parse_run(args.parse_run_id)
            source = application.registry.get_source(parse_run.source_file_id)
            job = next(
                item
                for item in application.registry.list_jobs(limit=10_000)
                if item.source_file_id == source.id
            )
            bundle = application.canonical_mapper.build_canonical_snapshot(
                source=source,
                parse_run=parse_run,
                artifacts=application.parser_artifacts.list_artifacts(parse_run.id),
                manifest_path=application.canonical_repository.snapshot_manifest_path(
                    source.id, parse_run.id
                ),
                dataset_id=job.dataset_id,
            )
            persisted = application.canonical_repository.save_snapshot(bundle)
            return persisted.snapshot.model_dump(mode="json")
        if args.mode == "enrich":
            bundle = application.canonical_repository.get_bundle(args.snapshot_id)
            report, path = await application.knowledge_pipeline.ingest_bundle(bundle)
            return {
                "path": str(path),
                "report": report.model_dump(mode="json"),
            }
        if args.mode == "cognee":
            bundle = application.canonical_repository.get_bundle(args.snapshot_id)
            report, _ = await application.knowledge_pipeline.ingest_bundle(bundle)
            return report.model_dump(mode="json")
        result = await application.services.ingestion.ingest_pdf_to_knowledge(args.pdf)
        return result.public_dict()
    finally:
        if started:
            await application.aclose()
        else:
            await application.local_inference_client.aclose()
            await application.mineru.aclose()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("mineru", "canonical", "enrich", "cognee", "full")
    )
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--parse-run-id")
    parser.add_argument("--snapshot-id")
    args = parser.parse_args()
    required = {
        "mineru": args.pdf,
        "canonical": args.parse_run_id,
        "enrich": args.snapshot_id,
        "cognee": args.snapshot_id,
        "full": args.pdf,
    }
    if required[args.mode] is None:
        parser.error(f"mode {args.mode} is missing its required input argument")
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
