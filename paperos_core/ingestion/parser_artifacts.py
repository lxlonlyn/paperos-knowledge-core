"""Immutable ParseRun and parser-artifact persistence."""

from __future__ import annotations

import io
import json
import mimetypes
import sqlite3
import stat
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from paperos_core.adapters.mineru.schemas import MinerUParseResult
from paperos_core.domain.documents import SourceFile, utc_now
from paperos_core.domain.enums import ParserArtifactType, ParseRunStatus
from paperos_core.domain.ids import parse_run_id, parser_artifact_id
from paperos_core.domain.parsing import ParserArtifact, ParseRun
from paperos_core.errors import ParserArtifactValidationError, SourceRegistryError
from paperos_core.ingestion.validation import calculate_sha256
from paperos_core.paths import DataPaths
from paperos_core.storage.immutable import ImmutableConflictError, write_immutable_bytes
from paperos_core.storage.path_refs import DataPathCodec

_MAX_UNCOMPRESSED_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
_WINDOWS_ILLEGAL_FILENAME_CHARACTERS = frozenset('<>:"|?*')
_WINDOWS_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{number}" for number in range(1, 10)}
    | {f"lpt{number}" for number in range(1, 10)}
)
_RECOVERABLE_PARSE_RUN_STATUSES = (
    ParseRunStatus.PENDING,
    ParseRunStatus.SUBMITTED,
    ParseRunStatus.RUNNING,
)
OPERATION_ID_REQUEST_OPTION = "_paperos_operational_job_id"


class ParserArtifactRepository:
    def __init__(self, paths: DataPaths) -> None:
        self.paths = paths
        self.path_codec = DataPathCodec(paths.root)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.paths.registry_db, timeout=30)
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _json(value: Any) -> str | None:
        return json.dumps(value, ensure_ascii=False, sort_keys=True) if value is not None else None

    @staticmethod
    def _parse_json(value: str | None) -> Any:
        return json.loads(value) if value is not None else None

    def _run_from_row(self, row: sqlite3.Row) -> ParseRun:
        return ParseRun(
            id=row["id"],
            source_file_id=row["source_file_id"],
            provider=row["provider"],
            backend=row["backend"],
            status=ParseRunStatus(row["status"]),
            request_options=self._parse_json(row["request_options"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            completed_at=(
                datetime.fromisoformat(row["completed_at"]) if row["completed_at"] else None
            ),
            artifact_manifest_path=self.path_codec.decode(str(row["artifact_manifest_path"])),
            schema_version=row["schema_version"],
            pipeline_version=row["pipeline_version"],
            provider_task_id=row["provider_task_id"],
            provider_version=row["provider_version"],
            provider_model=row["provider_model"],
            error_code=row["error_code"],
            error_message=row["error_message"],
            raw_metadata=self._parse_json(row["raw_metadata"]),
        )

    def _artifact_from_row(self, row: sqlite3.Row) -> ParserArtifact:
        return ParserArtifact(
            id=row["id"],
            parse_run_id=row["parse_run_id"],
            artifact_type=ParserArtifactType(row["artifact_type"]),
            storage_path=self.path_codec.decode(str(row["storage_path"])),
            sha256=row["sha256"],
            size_bytes=row["size_bytes"],
            created_at=datetime.fromisoformat(row["created_at"]),
            media_type=row["media_type"],
            page=row["page"],
            provider_name=row["provider_name"],
            provider_metadata=self._parse_json(row["provider_metadata"]),
            id_version=row["id_version"],
        )

    def create_parse_run(
        self,
        source: SourceFile,
        *,
        provider: str,
        backend: str,
        request_options: dict[str, Any],
    ) -> ParseRun:
        run_id = parse_run_id()
        root = (self.paths.parsed / source.id / run_id).resolve(strict=False)
        self.paths.assert_within_root(root)
        root.mkdir(parents=True, exist_ok=False)
        run = ParseRun(
            id=run_id,
            source_file_id=source.id,
            provider=provider,
            backend=backend,
            status=ParseRunStatus.PENDING,
            request_options=request_options,
            artifact_manifest_path=root / "manifest.json",
        )
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO parse_runs (
                        id, source_file_id, provider, backend, status, request_options,
                        created_at, completed_at, artifact_manifest_path, schema_version,
                        pipeline_version, provider_task_id, provider_version, provider_model,
                        error_code, error_message, raw_metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run.id,
                        run.source_file_id,
                        run.provider,
                        run.backend,
                        run.status.value,
                        self._json(run.request_options),
                        run.created_at.isoformat(),
                        None,
                        self.path_codec.encode(run.artifact_manifest_path),
                        run.schema_version,
                        run.pipeline_version,
                        None,
                        None,
                        None,
                        None,
                        None,
                        None,
                    ),
                )
        except sqlite3.Error as exc:
            raise SourceRegistryError(
                f"Unable to create ParseRun: {exc}", affected=self.paths.registry_db
            ) from exc
        return run

    def get_parse_run(self, run_id: str) -> ParseRun:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM parse_runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            raise SourceRegistryError(f"ParseRun '{run_id}' does not exist.", affected=run_id)
        return self._run_from_row(row)

    def find_replay_run(
        self,
        source_file_id: str,
        *,
        operation_id: str,
    ) -> ParseRun | None:
        """Find the durable parser checkpoint for one logical operation."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM parse_runs WHERE source_file_id = ? "
                "ORDER BY created_at DESC, id DESC",
                (source_file_id,),
            ).fetchall()
        for row in rows:
            run = self._run_from_row(row)
            if run.request_options.get(OPERATION_ID_REQUEST_OPTION) != operation_id:
                continue
            if run.status == ParseRunStatus.FAILED:
                return None
            if run.provider_task_id is not None or run.artifact_manifest_path.is_file():
                return run
        return None

    def durable_result_artifacts(
        self,
        parse_run: ParseRun,
    ) -> list[ParserArtifact] | None:
        """Return a fully persisted provider result, or None before its commit marker."""

        manifest_path = parse_run.artifact_manifest_path
        if not manifest_path.is_file():
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ParserArtifactValidationError(
                "Parser artifact manifest cannot be read.",
                affected=manifest_path,
            ) from exc
        if (
            not isinstance(manifest, dict)
            or manifest.get("parse_run_id") != parse_run.id
            or manifest.get("source_file_id") != parse_run.source_file_id
            or manifest.get("provider_task_id") != parse_run.provider_task_id
        ):
            raise ParserArtifactValidationError(
                "Parser artifact manifest does not match its ParseRun.",
                affected=manifest_path,
            )
        artifacts = self.list_artifacts(parse_run.id)
        if not artifacts:
            raise ParserArtifactValidationError(
                "Parser artifact manifest has no registered artifacts.",
                affected=parse_run.id,
            )
        self.verify_artifact_checksums(parse_run.id)
        return artifacts

    def recover_interrupted_runs(self) -> int:
        """Mark non-terminal parse attempts left by a prior process as interrupted."""

        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE parse_runs SET status = ?, completed_at = ? "
                "WHERE status IN (?, ?, ?)",
                (
                    ParseRunStatus.INTERRUPTED.value,
                    utc_now().isoformat(),
                    *(status.value for status in _RECOVERABLE_PARSE_RUN_STATUSES),
                ),
            )
            updated_count = cursor.rowcount
        return updated_count

    def update_parse_run(
        self,
        run_id: str,
        *,
        status: ParseRunStatus,
        provider: str | None = None,
        backend: str | None = None,
        provider_task_id: str | None = None,
        provider_model: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        raw_metadata: dict[str, Any] | None = None,
    ) -> ParseRun:
        current = self.get_parse_run(run_id)
        completed_at = (
            utc_now()
            if status
            in {ParseRunStatus.COMPLETED, ParseRunStatus.FAILED, ParseRunStatus.INTERRUPTED}
            else None
        )
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE parse_runs
                SET provider = ?, backend = ?, status = ?, completed_at = ?,
                    provider_task_id = ?,
                    provider_model = ?, error_code = ?, error_message = ?, raw_metadata = ?
                WHERE id = ?
                """,
                (
                    provider or current.provider,
                    backend or current.backend,
                    status.value,
                    completed_at.isoformat() if completed_at else None,
                    provider_task_id or current.provider_task_id,
                    provider_model or current.provider_model,
                    error_code,
                    error_message,
                    self._json(raw_metadata)
                    if raw_metadata is not None
                    else self._json(current.raw_metadata),
                    run_id,
                ),
            )
        return self.get_parse_run(run_id)

    def _write_immutable(self, path: Path, content: bytes) -> None:
        self.paths.assert_within_root(path)
        try:
            write_immutable_bytes(path, content)
        except ImmutableConflictError as exc:
            raise ParserArtifactValidationError(
                "Immutable parser artifact already exists with different content.",
                affected=path,
            ) from exc
        except OSError as exc:
            raise ParserArtifactValidationError(
                f"Unable to persist immutable parser artifact: {exc}", affected=path
            ) from exc

    @staticmethod
    def _artifact_type(relative_path: str) -> ParserArtifactType:
        name = PurePosixPath(relative_path).name.lower()
        suffix = PurePosixPath(relative_path).suffix.lower()
        parts = {part.lower() for part in PurePosixPath(relative_path).parts}
        if suffix in {".md", ".markdown"}:
            return ParserArtifactType.MARKDOWN
        if "content_list" in name and suffix == ".json":
            return ParserArtifactType.CONTENT_LIST
        if suffix == ".json" and ("model" in name or "middle" in name or "layout" in name):
            return ParserArtifactType.MODEL_OUTPUT
        if (
            "images" in parts
            or "assets" in parts
            or suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".svg"}
        ):
            return ParserArtifactType.ASSET
        return ParserArtifactType.OTHER

    def _insert_artifact(
        self,
        parse_run: ParseRun,
        path: Path,
        artifact_type: ParserArtifactType,
        *,
        relative_path: str,
        provider_metadata: dict[str, Any] | None = None,
    ) -> ParserArtifact:
        digest = calculate_sha256(path)
        artifact = ParserArtifact(
            id=parser_artifact_id(parse_run.id, artifact_type.value, relative_path, digest),
            parse_run_id=parse_run.id,
            artifact_type=artifact_type,
            storage_path=path,
            sha256=digest,
            size_bytes=path.stat().st_size,
            media_type=mimetypes.guess_type(path.name)[0],
            provider_name=parse_run.provider,
            provider_metadata=provider_metadata,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO parser_artifacts (
                    id, parse_run_id, artifact_type, storage_path, sha256, size_bytes,
                    created_at, media_type, page, provider_name, provider_metadata,
                    id_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.id,
                    artifact.parse_run_id,
                    artifact.artifact_type.value,
                    self.path_codec.encode(artifact.storage_path),
                    artifact.sha256,
                    artifact.size_bytes,
                    artifact.created_at.isoformat(),
                    artifact.media_type,
                    artifact.page,
                    artifact.provider_name,
                    self._json(artifact.provider_metadata),
                    artifact.id_version,
                ),
            )
        return artifact

    @staticmethod
    def _safe_member(info: zipfile.ZipInfo) -> PurePosixPath:
        # ZipInfo normalizes OS-native separators in ``filename`` on Windows;
        # ``orig_filename`` retains the archive's portable namespace spelling.
        filename = info.orig_filename
        raw_parts = filename.split("/")
        relative = PurePosixPath(filename)
        mode = info.external_attr >> 16
        if (
            "\\" in filename
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in raw_parts)
            or stat.S_ISLNK(mode)
            or not filename
        ):
            raise ParserArtifactValidationError(
                "MinerU archive contains an unsafe path.", affected=filename
            )
        for part in raw_parts:
            if (
                any(character in _WINDOWS_ILLEGAL_FILENAME_CHARACTERS for character in part)
                or any(ord(character) < 32 or ord(character) == 127 for character in part)
                or part.endswith((".", " "))
                or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES
            ):
                raise ParserArtifactValidationError(
                    "MinerU archive path is not in the portable Windows/Linux namespace.",
                    affected=filename,
                )
        return relative

    @classmethod
    def _safe_members(
        cls, infos: list[zipfile.ZipInfo]
    ) -> list[tuple[zipfile.ZipInfo, PurePosixPath]]:
        members = [(info, cls._safe_member(info)) for info in infos]
        folded_paths: dict[str, str] = {}
        for _info, relative in members:
            portable_path = relative.as_posix()
            folded = portable_path.casefold()
            previous = folded_paths.get(folded)
            if previous is not None:
                raise ParserArtifactValidationError(
                    "MinerU archive contains a case-fold path collision.",
                    affected=f"{previous} / {portable_path}",
                )
            folded_paths[folded] = portable_path
        return members

    def persist_result(
        self, parse_run: ParseRun, result: MinerUParseResult
    ) -> list[ParserArtifact]:
        root = parse_run.artifact_manifest_path.parent
        artifacts: list[ParserArtifact] = []

        generated = (
            (
                "provider_result.zip",
                result.archive_bytes,
                ParserArtifactType.ARCHIVE,
            ),
            (
                "provider_response.json",
                json.dumps(
                    result.final_metadata, ensure_ascii=False, sort_keys=True, indent=2
                ).encode(),
                ParserArtifactType.PROVIDER_RESPONSE,
            ),
            (
                "task_metadata.json",
                json.dumps(
                    {"poll_history": result.poll_history, "warnings": result.warnings},
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                ).encode(),
                ParserArtifactType.TASK_METADATA,
            ),
        )
        for name, content, artifact_type in generated:
            path = root / name
            self._write_immutable(path, content)
            artifacts.append(
                self._insert_artifact(parse_run, path, artifact_type, relative_path=name)
            )

        try:
            archive = zipfile.ZipFile(io.BytesIO(result.archive_bytes))
        except zipfile.BadZipFile as exc:
            raise ParserArtifactValidationError(
                "MinerU result archive cannot be opened as ZIP.",
                affected=parse_run.id,
            ) from exc
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if not infos:
            raise ParserArtifactValidationError(
                "MinerU result archive contains no files.", affected=parse_run.id
            )
        if sum(info.file_size for info in infos) > _MAX_UNCOMPRESSED_ARCHIVE_BYTES:
            raise ParserArtifactValidationError(
                "MinerU result archive exceeds the safe uncompressed size limit.",
                affected=parse_run.id,
            )
        members = self._safe_members(infos)
        with archive:
            for info, relative in members:
                content = archive.read(info)
                target = root / "artifacts" / Path(*relative.parts)
                self._write_immutable(target, content)
                artifact_type = self._artifact_type(relative.as_posix())
                artifacts.append(
                    self._insert_artifact(
                        parse_run,
                        target,
                        artifact_type,
                        relative_path=f"artifacts/{relative.as_posix()}",
                        provider_metadata={
                            "archive_path": relative.as_posix(),
                            "compressed_size": info.compress_size,
                        },
                    )
                )

        required = {artifact.artifact_type for artifact in artifacts}
        missing = {
            ParserArtifactType.MARKDOWN,
            ParserArtifactType.CONTENT_LIST,
            ParserArtifactType.MODEL_OUTPUT,
        } - required
        if missing:
            raise ParserArtifactValidationError(
                "MinerU result is missing required artifacts: "
                + ", ".join(sorted(item.value for item in missing)),
                affected=parse_run.id,
            )

        manifest = {
            "schema_version": "1.0",
            "parse_run_id": parse_run.id,
            "source_file_id": parse_run.source_file_id,
            "provider": result.provider,
            "provider_task_id": result.provider_task_id,
            "backend": result.backend,
            "artifacts": [
                {
                    "id": artifact.id,
                    "type": artifact.artifact_type.value,
                    "path": str(artifact.storage_path.relative_to(root)),
                    "sha256": artifact.sha256,
                    "size_bytes": artifact.size_bytes,
                }
                for artifact in artifacts
            ],
        }
        self._write_immutable(
            parse_run.artifact_manifest_path,
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode(),
        )
        return artifacts

    def list_artifacts(self, run_id: str) -> list[ParserArtifact]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM parser_artifacts WHERE parse_run_id = ? ORDER BY storage_path",
                (run_id,),
            ).fetchall()
        return [self._artifact_from_row(row) for row in rows]

    def verify_artifact_checksums(self, run_id: str) -> None:
        for artifact in self.list_artifacts(run_id):
            if (
                not artifact.storage_path.is_file()
                or artifact.storage_path.stat().st_size != artifact.size_bytes
                or calculate_sha256(artifact.storage_path) != artifact.sha256
            ):
                raise ParserArtifactValidationError(
                    "Parser artifact checksum verification failed.",
                    affected=artifact.storage_path,
                )
