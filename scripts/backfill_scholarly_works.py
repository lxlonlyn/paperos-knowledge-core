"""Backfill stable ScholarlyWork identities from retained canonical snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paperos_core.config import load_settings
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry
from paperos_core.paths import build_data_paths
from paperos_core.storage.initializer import StorageInitializer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill ScholarlyWork identities in the configured PaperOS registry."
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="PaperOS TOML path; defaults to config/paperos.toml.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Override the configured PaperOS data root.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    data_dir = args.data_dir if args.data_dir is not None else settings.data_dir
    paths = build_data_paths(
        data_dir,
        registry_filename=settings.storage.registry_filename,
        lexical_filename=settings.storage.lexical_filename,
    )
    storage = StorageInitializer(paths)
    storage.initialize()
    repository = CanonicalRepository(paths)
    registry = ScholarlyRegistry(paths)
    contexts = registry.backfill(repository)
    snapshot = registry.identity_snapshot()
    print(
        json.dumps(
            {
                "status": "completed",
                "processed_documents": len(contexts),
                "active_works": len(snapshot["works"]),
                "document_links": len(snapshot["document_links"]),
                "reference_links": len(snapshot["reference_links"]),
                "redirects": len(snapshot["redirects"]),
                "work_ids": [work["id"] for work in snapshot["works"]],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
