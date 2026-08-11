"""Migrate legacy absolute PaperOS path records to portable data references."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.storage.path_refs import DataPathCodec


@dataclass(frozen=True, slots=True)
class PathColumn:
    table: str
    id_column: str
    path_column: str
    checksum_column: str | None = None


@dataclass(frozen=True, slots=True)
class DatabaseUpdate:
    table: str
    id_column: str
    row_id: str
    path_column: str
    value: str


PATH_COLUMNS = (
    PathColumn("source_files", "id", "storage_path", "sha256"),
    PathColumn("parse_runs", "id", "artifact_manifest_path"),
    PathColumn("parser_artifacts", "id", "storage_path", "sha256"),
    PathColumn("canonical_snapshots", "id", "manifest_path"),
)


class MigrationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _is_windows_absolute(value: str) -> bool:
    return PureWindowsPath(value).is_absolute()


def _is_absolute(value: str) -> bool:
    return Path(value).is_absolute() or _is_windows_absolute(value)


def _relative_parts(value: str, old_root: str) -> tuple[str, ...]:
    if _is_windows_absolute(value):
        candidate = PureWindowsPath(value)
        root = PureWindowsPath(old_root)
        try:
            return candidate.relative_to(root).parts
        except ValueError as exc:
            raise MigrationError(
                f"Legacy Windows path is not below --old-data-root: {value}"
            ) from exc
    posix_candidate = Path(value).expanduser().resolve(strict=False)
    posix_root = Path(old_root).expanduser().resolve(strict=False)
    try:
        return posix_candidate.relative_to(posix_root).parts
    except ValueError as exc:
        raise MigrationError(
            f"Legacy path is not below the selected old data root: {value}"
        ) from exc


def _convert_path(
    value: str,
    *,
    data_root: Path,
    codec: DataPathCodec,
    old_data_root: str | None,
) -> tuple[str, Path]:
    if not _is_absolute(value):
        return value, codec.decode(value)
    selected_old_root = old_data_root or str(data_root)
    relative = _relative_parts(value, selected_old_root)
    target = data_root.joinpath(*relative).resolve(strict=False)
    return codec.encode(target), target


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def _database_updates(
    connection: sqlite3.Connection,
    *,
    data_root: Path,
    codec: DataPathCodec,
    old_data_root: str | None,
) -> list[DatabaseUpdate]:
    updates: list[DatabaseUpdate] = []
    for spec in PATH_COLUMNS:
        if not _table_exists(connection, spec.table):
            continue
        selected = [spec.id_column, spec.path_column]
        if spec.checksum_column is not None:
            selected.append(spec.checksum_column)
        rows = connection.execute(
            f"SELECT {', '.join(selected)} FROM {spec.table}"
        ).fetchall()
        for row in rows:
            current = str(row[spec.path_column])
            converted, target = _convert_path(
                current,
                data_root=data_root,
                codec=codec,
                old_data_root=old_data_root,
            )
            if not target.is_file():
                raise MigrationError(
                    f"Referenced file does not exist for {spec.table}.{spec.path_column}: "
                    f"{target}"
                )
            if spec.checksum_column is not None:
                expected = str(row[spec.checksum_column]).lower()
                actual = _sha256(target)
                if actual != expected:
                    raise MigrationError(
                        f"Checksum mismatch for {spec.table} row {row[spec.id_column]}"
                    )
            if converted != current:
                updates.append(
                    DatabaseUpdate(
                        table=spec.table,
                        id_column=spec.id_column,
                        row_id=str(row[spec.id_column]),
                        path_column=spec.path_column,
                        value=converted,
                    )
                )
    return updates


def _portable_string(
    value: str,
    *,
    data_root: Path,
    codec: DataPathCodec,
    old_data_root: str | None,
) -> str:
    parsed = urlsplit(value)
    if parsed.scheme and "://" in value:
        return value
    if not _is_absolute(value):
        return value
    try:
        converted, _target = _convert_path(
            value,
            data_root=data_root,
            codec=codec,
            old_data_root=old_data_root,
        )
    except MigrationError:
        return value
    return converted


def _portable_value(
    value: Any,
    *,
    data_root: Path,
    codec: DataPathCodec,
    old_data_root: str | None,
) -> Any:
    if isinstance(value, str):
        return _portable_string(
            value,
            data_root=data_root,
            codec=codec,
            old_data_root=old_data_root,
        )
    if isinstance(value, list):
        return [
            _portable_value(
                item,
                data_root=data_root,
                codec=codec,
                old_data_root=old_data_root,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            str(key): _portable_value(
                item,
                data_root=data_root,
                codec=codec,
                old_data_root=old_data_root,
            )
            for key, item in value.items()
        }
    return value


def _json_file_updates(
    data_root: Path,
    *,
    codec: DataPathCodec,
    old_data_root: str | None,
) -> dict[Path, bytes]:
    updates: dict[Path, bytes] = {}
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
        try:
            text = path.read_text(encoding="utf-8")
            if path.suffix.lower() == ".json":
                payload = json.loads(text)
                converted = _portable_value(
                    payload,
                    data_root=data_root,
                    codec=codec,
                    old_data_root=old_data_root,
                )
                if converted == payload:
                    continue
                rendered = json.dumps(
                    converted, ensure_ascii=False, sort_keys=True, indent=2
                ) + "\n"
            else:
                changed = False
                rendered_lines = []
                for line in text.splitlines():
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    converted = _portable_value(
                        payload,
                        data_root=data_root,
                        codec=codec,
                        old_data_root=old_data_root,
                    )
                    if converted != payload:
                        changed = True
                    rendered_lines.append(
                        json.dumps(converted, ensure_ascii=False, sort_keys=True)
                    )
                rendered = "\n".join(rendered_lines) + (
                    "\n" if rendered_lines else ""
                )
                if not changed:
                    continue
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MigrationError(f"Unable to inspect persistent JSON {path}: {exc}") from exc
        encoded = rendered.encode()
        if encoded != text.encode():
            updates[path] = encoded

    for manifest_path in data_root.glob("canonical/*/*/manifest.json"):
        effective = updates.get(manifest_path, manifest_path.read_bytes())
        manifest = json.loads(effective.decode())
        changed = False
        for entry in manifest.get("files", []):
            artifact_path = manifest_path.parent / str(entry["path"])
            artifact = updates.get(artifact_path, artifact_path.read_bytes())
            digest = hashlib.sha256(artifact).hexdigest()
            size = len(artifact)
            if entry.get("sha256") != digest or entry.get("size_bytes") != size:
                entry["sha256"] = digest
                entry["size_bytes"] = size
                changed = True
        if changed:
            updates[manifest_path] = (
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)
                + "\n"
            ).encode()
    return updates


def _apply(
    connection: sqlite3.Connection,
    *,
    database_updates: list[DatabaseUpdate],
    file_updates: dict[Path, bytes],
) -> None:
    backup_root = Path(tempfile.mkdtemp(prefix="paperos-path-migration-"))
    backups: dict[Path, Path] = {}
    staged: dict[Path, Path] = {}
    try:
        for index, (path, content) in enumerate(file_updates.items()):
            backup = backup_root / "backup" / str(index)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
            backups[path] = backup
            temporary = path.parent / f".{path.name}.portable.tmp"
            temporary.write_bytes(content)
            staged[path] = temporary

        connection.execute("BEGIN IMMEDIATE")
        for update in database_updates:
            connection.execute(
                f"UPDATE {update.table} SET {update.path_column}=? "
                f"WHERE {update.id_column}=?",
                (update.value, update.row_id),
            )
        for path, temporary in staged.items():
            os.replace(temporary, path)
        connection.commit()
    except BaseException:
        connection.rollback()
        for path, backup in backups.items():
            if backup.is_file():
                shutil.copy2(backup, path)
        raise
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        shutil.rmtree(backup_root, ignore_errors=True)


def migrate(
    data_root: Path,
    *,
    old_data_root: str | None,
    dry_run: bool,
) -> dict[str, object]:
    resolved = data_root.expanduser().resolve(strict=False)
    registry = resolved / "jobs" / "registry.sqlite3"
    if not registry.is_file():
        raise MigrationError(f"PaperOS registry does not exist: {registry}")
    codec = DataPathCodec(resolved)
    with sqlite3.connect(registry, timeout=30) as connection:
        connection.row_factory = sqlite3.Row
        database_updates = _database_updates(
            connection,
            data_root=resolved,
            codec=codec,
            old_data_root=old_data_root,
        )
        file_updates = _json_file_updates(
            resolved,
            codec=codec,
            old_data_root=old_data_root,
        )
        if not dry_run:
            _apply(
                connection,
                database_updates=database_updates,
                file_updates=file_updates,
            )
    return {
        "status": "dry-run" if dry_run else "migrated",
        "data_root": str(resolved),
        "database_update_count": len(database_updates),
        "json_update_count": len(file_updates),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate PaperOS absolute data paths to portable references."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--old-data-root",
        help="Original data root when the directory has already been copied or moved.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            migrate(
                args.data_dir,
                old_data_root=args.old_data_root,
                dry_run=args.dry_run,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
