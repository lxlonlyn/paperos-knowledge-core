"""Permanent portable-data pytest contract."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile
from contextlib import closing
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.config import load_settings
from paperos_core.errors import ConfigurationError
from paperos_core.indexes.lexical_store import LexicalStore
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.parser_artifacts import ParserArtifactRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.locations import PROJECT_ROOT
from paperos_core.paths import build_data_paths
from paperos_core.storage.path_refs import DataPathCodec

PATH_COLUMNS = (
    ("source_files", "storage_path"),
    ("parse_runs", "artifact_manifest_path"),
    ("parser_artifacts", "storage_path"),
    ("canonical_snapshots", "manifest_path"),
)
PATH_KEYS = {
    "storage_path",
    "storage_relpath",
    "artifact_manifest_path",
    "artifact_manifest_relpath",
    "manifest_path",
    "manifest_relpath",
    "asset_path",
    "lexical_database",
    "cognee_manifest_path",
    "enrichment_path",
    "run_root",
}


def _require(condition: object, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _portable_reference(value: str, *, allow_dot: bool = False) -> bool:
    if allow_dot and value == ".":
        return True
    portable = PurePosixPath(value)
    windows = PureWindowsPath(value)
    return (
        bool(value)
        and "\\" not in value
        and not portable.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and ".." not in portable.parts
    )


def codec_contract() -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="paperos-codec-") as directory:
        root = Path(directory)
        codec = DataPathCodec(root)
        target = root / "parsed" / "source" / "run" / "manifest.json"
        encoded = codec.encode(target)
        _require(
            encoded == "parsed/source/run/manifest.json",
            f"Unexpected portable encoding: {encoded}",
        )
        _require(codec.decode(encoded) == target.resolve(), "Portable decode mismatch.")
        rejected = 0
        for unsafe in (
            "/tmp/outside",
            "../outside",
            "parsed/../outside",
            r"C:\paperos\data\raw\source.pdf",
            r"parsed\source\manifest.json",
        ):
            try:
                codec.decode(unsafe)
            except ConfigurationError:
                rejected += 1
        _require(rejected == 5, "DataPathCodec accepted an unsafe path.")
    return {
        "status": "passed",
        "posix_separator": True,
        "absolute_and_escape_rejected": True,
    }


def cross_platform_smoke_contract() -> dict[str, object]:
    """Load the shipped config and public entry modules from an unrelated cwd."""

    modules = (
        "paperos_core.api.app",
        "paperos_core.documents",
        "paperos_core.jobs.worker",
        "paperos_core.retrieval.service",
        "server",
    )
    original_cwd = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="paperos-portable-smoke-") as directory:
        try:
            os.chdir(directory)
            settings = load_settings(PROJECT_ROOT / "config/paperos.example.toml")
            for module in modules:
                importlib.import_module(module)
        finally:
            os.chdir(original_cwd)
    _require(settings.config_path == PROJECT_ROOT / "config/paperos.example.toml", "Example config path changed.")
    _require(settings.data_dir == PROJECT_ROOT / "data", "Example data path is cwd-dependent.")
    _require(
        settings.local_inference.embedding_model_path
        == PROJECT_ROOT / "data/models/embedding/embeddinggemma-300M-Q8_0.gguf",
        "Example embedding model path is cwd-dependent.",
    )
    _require(settings.local_inference.enabled, "Example local endpoint is not runnable.")
    return {
        "status": "passed",
        "example_config": True,
        "cwd_independent": True,
        "imported_modules": list(modules),
    }


def sqlite_contract(data_root: Path) -> dict[str, object]:
    paths = build_data_paths(data_root)
    codec = DataPathCodec(paths.root)
    checked = 0
    with closing(sqlite3.connect(paths.registry_db)) as connection, connection:
        connection.row_factory = sqlite3.Row
        for table, column in PATH_COLUMNS:
            rows = connection.execute(f"SELECT {column} FROM {table}").fetchall()
            for row in rows:
                value = str(row[column])
                _require(
                    _portable_reference(value),
                    f"SQLite contains a non-portable path: {table}.{column}={value}",
                )
                codec.decode(value)
                checked += 1
        for row in connection.execute(
            "SELECT payload, result FROM operational_jobs"
        ).fetchall():
            for column in ("payload", "result"):
                if row[column]:
                    value = str(row[column])
                    _require(
                        str(paths.root) not in value,
                        f"Operational job {column} contains the data root.",
                    )
    return {"status": "passed", "path_record_count": checked}


def _check_json_value(
    value: Any,
    *,
    key: str | None,
    source: Path,
    old_root: str,
) -> int:
    checked = 0
    if isinstance(value, dict):
        for child_key, child in value.items():
            checked += _check_json_value(
                child,
                key=str(child_key),
                source=source,
                old_root=old_root,
            )
    elif isinstance(value, list):
        for child in value:
            checked += _check_json_value(
                child,
                key=key,
                source=source,
                old_root=old_root,
            )
    elif isinstance(value, str):
        _require(old_root not in value, f"Persistent JSON retains old data root: {source}")
        parsed = urlsplit(value)
        if key in PATH_KEYS and not (parsed.scheme and "://" in value):
            _require(
                _portable_reference(value, allow_dot=key == "run_root"),
                f"Persistent JSON path is not portable: {source} / {key}={value}",
            )
            checked += 1
    return checked


def json_contract(data_root: Path, *, old_root: str | None = None) -> dict[str, object]:
    root_text = old_root or str(data_root.resolve())
    checked = 0
    files = 0
    persistent_roots = (
        data_root / "raw",
        data_root / "parsed",
        data_root / "canonical",
        data_root / "cognee",
        data_root / "indexes",
        data_root / "jobs",
        data_root / "logs",
        data_root / "exports",
        data_root / "cache",
    )
    candidates = {
        path for root in persistent_roots if root.exists() for path in root.rglob("*")
    }
    for path in sorted(candidates):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        if path.name.endswith("-process.json"):
            continue
        files += 1
        if path.suffix.lower() == ".json":
            payloads = [json.loads(path.read_text(encoding="utf-8"))]
        else:
            payloads = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        for payload in payloads:
            checked += _check_json_value(
                payload,
                key=None,
                source=path,
                old_root=root_text,
            )
    return {
        "status": "passed",
        "json_file_count": files,
        "path_value_count": checked,
    }


def relocation_contract(data_root: Path) -> dict[str, object]:
    source_paths = build_data_paths(data_root)
    with tempfile.TemporaryDirectory(prefix="paperos-relocation-") as directory:
        relocated_root = Path(directory) / "data"
        for name in ("raw", "parsed", "canonical", "indexes"):
            source = source_paths.root / name
            if source.exists():
                shutil.copytree(source, relocated_root / name)
        registry_target = relocated_root / "jobs" / "registry.sqlite3"
        registry_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_paths.registry_db, registry_target)
        chunks = source_paths.cognee / "chunks"
        if chunks.exists():
            shutil.copytree(chunks, relocated_root / "cognee" / "chunks")

        relocated = build_data_paths(relocated_root)
        registry = SourceRegistry(relocated)
        parser = ParserArtifactRepository(relocated)
        canonical = CanonicalRepository(relocated)
        snapshot_ids = canonical.list_snapshot_ids()
        _require(snapshot_ids, "Relocated data contains no canonical snapshots.")
        for snapshot_id in snapshot_ids:
            bundle = canonical.get_bundle(snapshot_id)
            registered_source = registry.get_source(bundle.snapshot.source_file_id)
            _require(registered_source.storage_path.is_file(), "Relocated raw PDF is missing.")
            parser.verify_artifact_checksums(bundle.snapshot.parse_run_id)
            canonical.verify_snapshot(snapshot_id)

        lexical = LexicalStore(relocated.indexes / "lexical.sqlite3")
        lexical_status = lexical.status()
        _require(
            cast(int, lexical_status["record_count"]) > 0,
            "Relocated FTS projection cannot continue serving queries.",
        )
        return {
            "status": "passed",
            "snapshot_count": len(snapshot_ids),
            "raw_parsed_canonical_checksums": True,
            "fts_query_projection_available": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--old-data-root")
    parser.add_argument("--relocate", action="store_true")
    args = parser.parse_args()
    report: dict[str, object] = {
        "codec": codec_contract(),
        "cross_platform_smoke": cross_platform_smoke_contract(),
    }
    if args.data_dir is not None:
        root = args.data_dir.expanduser().resolve(strict=False)
        report["sqlite"] = sqlite_contract(root)
        report["json"] = json_contract(root, old_root=args.old_data_root)
        if args.relocate:
            report["relocation"] = relocation_contract(root)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def test_portable_data_paths_contract() -> None:
    codec_contract()
    cross_platform_smoke_contract()


if __name__ == "__main__":
    main()
