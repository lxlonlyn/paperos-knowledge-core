"""Central, idempotent owner of all PaperOS local schemas."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from paperos_core.errors import ConfigurationError
from paperos_core.paths import DataPaths

REGISTRY_TABLES = frozenset(
    {
        "source_files",
        "ingestion_jobs",
        "parse_runs",
        "parser_artifacts",
        "canonical_snapshots",
        "operational_jobs",
        "feedback",
        "corrections",
        "improvements",
        "document_tombstones",
    }
)


@dataclass(frozen=True, slots=True)
class StorageStatus:
    valid: bool
    registry_database: Path
    lexical_database: Path
    missing_tables: tuple[str, ...]
    fts5_available: bool


class StorageInitializer:
    def __init__(self, paths: DataPaths) -> None:
        self.paths = paths
        self.lexical_database = paths.indexes / "lexical.sqlite3"

    def initialize(self) -> None:
        self.paths.initialize()
        try:
            with sqlite3.connect(self.paths.registry_db, timeout=30) as connection:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.executescript(_REGISTRY_SCHEMA)
                _migrate_canonical_snapshot_projection_split(connection)
            self.initialize_lexical()
        except sqlite3.Error as exc:
            raise ConfigurationError(
                f"Unable to initialize PaperOS local schema: {exc}",
                affected=self.paths.root,
            ) from exc

    def initialize_lexical(self) -> None:
        self.lexical_database.parent.mkdir(parents=True, exist_ok=True)
        try:
            with sqlite3.connect(self.lexical_database, timeout=30) as connection:
                connection.executescript(_LEXICAL_SCHEMA)
        except sqlite3.Error as exc:
            raise ConfigurationError(
                f"Unable to initialize PaperOS FTS schema: {exc}",
                affected=self.lexical_database,
            ) from exc

    def validate(self) -> StorageStatus:
        missing = set(REGISTRY_TABLES)
        fts5 = False
        if self.paths.registry_db.is_file():
            with sqlite3.connect(self.paths.registry_db, timeout=30) as connection:
                present = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
            missing -= present
        if self.lexical_database.is_file():
            with sqlite3.connect(self.lexical_database, timeout=30) as connection:
                fts5 = bool(
                    connection.execute(
                        "SELECT sqlite_compileoption_used('ENABLE_FTS5')"
                    ).fetchone()[0]
                )
                lexical = {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
            if "lexical_records" not in lexical or "lexical_fts" not in lexical:
                missing.add("lexical_records/lexical_fts")
        else:
            missing.add("lexical_records/lexical_fts")
        return StorageStatus(
            valid=not missing and fts5,
            registry_database=self.paths.registry_db,
            lexical_database=self.lexical_database,
            missing_tables=tuple(sorted(missing)),
            fts5_available=fts5,
        )


_REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS source_files (
    id TEXT PRIMARY KEY, sha256 TEXT NOT NULL UNIQUE,
    original_filename TEXT NOT NULL, stored_filename TEXT NOT NULL,
    media_type TEXT NOT NULL, size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
    storage_path TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
    schema_version TEXT NOT NULL, id_version TEXT NOT NULL, source_url TEXT,
    user_metadata TEXT, dataset_id TEXT
);
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    id TEXT PRIMARY KEY, source_file_id TEXT NOT NULL, dataset_id TEXT NOT NULL,
    status TEXT NOT NULL, current_operation TEXT NOT NULL, created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL, attempt_count INTEGER NOT NULL DEFAULT 0,
    error_code TEXT, error_message TEXT, completed_at TEXT, requested_options TEXT,
    schema_version TEXT NOT NULL, id_version TEXT NOT NULL,
    FOREIGN KEY (source_file_id) REFERENCES source_files(id)
);
CREATE INDEX IF NOT EXISTS ingestion_jobs_source_idx ON ingestion_jobs(source_file_id);
CREATE INDEX IF NOT EXISTS ingestion_jobs_status_idx ON ingestion_jobs(status);
CREATE TABLE IF NOT EXISTS parse_runs (
    id TEXT PRIMARY KEY, source_file_id TEXT NOT NULL, provider TEXT NOT NULL,
    backend TEXT NOT NULL, status TEXT NOT NULL, request_options TEXT NOT NULL,
    created_at TEXT NOT NULL, completed_at TEXT, artifact_manifest_path TEXT NOT NULL UNIQUE,
    schema_version TEXT NOT NULL, pipeline_version TEXT NOT NULL, provider_task_id TEXT,
    provider_version TEXT, provider_model TEXT, error_code TEXT, error_message TEXT,
    raw_metadata TEXT, FOREIGN KEY (source_file_id) REFERENCES source_files(id)
);
CREATE TABLE IF NOT EXISTS parser_artifacts (
    id TEXT PRIMARY KEY, parse_run_id TEXT NOT NULL, artifact_type TEXT NOT NULL,
    storage_path TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL, media_type TEXT, page INTEGER, provider_name TEXT,
    provider_metadata TEXT, id_version TEXT NOT NULL,
    FOREIGN KEY (parse_run_id) REFERENCES parse_runs(id)
);
CREATE INDEX IF NOT EXISTS parser_artifacts_run_idx ON parser_artifacts(parse_run_id);
CREATE TABLE IF NOT EXISTS canonical_snapshots (
    id TEXT PRIMARY KEY, source_file_id TEXT NOT NULL, parse_run_id TEXT NOT NULL,
    document_id TEXT NOT NULL, manifest_path TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL,
    schema_version TEXT NOT NULL, id_version TEXT NOT NULL, pipeline_version TEXT NOT NULL,
    cleaning_version TEXT NOT NULL, classification_version TEXT NOT NULL,
    reference_processing_version TEXT NOT NULL,
    UNIQUE(parse_run_id, schema_version, pipeline_version),
    FOREIGN KEY (source_file_id) REFERENCES source_files(id),
    FOREIGN KEY (parse_run_id) REFERENCES parse_runs(id)
);
CREATE INDEX IF NOT EXISTS canonical_snapshot_source_idx ON canonical_snapshots(source_file_id);
CREATE TABLE IF NOT EXISTS operational_jobs (
    id TEXT PRIMARY KEY, job_type TEXT NOT NULL, payload TEXT NOT NULL,
    status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    error TEXT, result TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY, feedback_type TEXT NOT NULL, target_id TEXT NOT NULL,
    query_id TEXT, answer_id TEXT, evidence_ids TEXT NOT NULL, comment TEXT,
    replacement_text TEXT, created_by TEXT, created_at TEXT NOT NULL,
    schema_version TEXT NOT NULL, id_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY, target_id TEXT NOT NULL, replacement_or_correction TEXT NOT NULL,
    status TEXT NOT NULL, created_at TEXT NOT NULL, schema_version TEXT NOT NULL,
    id_version TEXT NOT NULL, derived_from_feedback_id TEXT NOT NULL UNIQUE,
    source_chunk_ids TEXT NOT NULL, supersedes_object_id TEXT, version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS improvements (
    id TEXT PRIMARY KEY, feedback_id TEXT NOT NULL UNIQUE, target_id TEXT NOT NULL,
    improvement_type TEXT NOT NULL, text TEXT, status TEXT NOT NULL,
    evidence_ids TEXT NOT NULL, source_chunk_ids TEXT NOT NULL,
    derived_from_ids TEXT NOT NULL, correction_id TEXT, version INTEGER NOT NULL,
    created_at TEXT NOT NULL, schema_version TEXT NOT NULL, id_version TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS document_tombstones (
    document_id TEXT PRIMARY KEY, deleted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

_LEXICAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS lexical_records (
    object_id TEXT PRIMARY KEY, object_type TEXT NOT NULL, document_id TEXT NOT NULL,
    canonical_snapshot_id TEXT NOT NULL, schema_version TEXT NOT NULL,
    index_version TEXT NOT NULL, field_name TEXT NOT NULL, section_id TEXT,
    section_path TEXT, text TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS lexical_fts USING fts5(
    object_id UNINDEXED, text, content='lexical_records',
    content_rowid='rowid', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS lexical_records_ai AFTER INSERT ON lexical_records BEGIN
    INSERT INTO lexical_fts(rowid, object_id, text) VALUES (new.rowid, new.object_id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS lexical_records_ad AFTER DELETE ON lexical_records BEGIN
    INSERT INTO lexical_fts(lexical_fts, rowid, object_id, text)
    VALUES ('delete', old.rowid, old.object_id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS lexical_records_au AFTER UPDATE ON lexical_records BEGIN
    INSERT INTO lexical_fts(lexical_fts, rowid, object_id, text)
    VALUES ('delete', old.rowid, old.object_id, old.text);
    INSERT INTO lexical_fts(rowid, object_id, text) VALUES (new.rowid, new.object_id, new.text);
END;
CREATE INDEX IF NOT EXISTS lexical_document_idx ON lexical_records(document_id);
"""


def _migrate_canonical_snapshot_projection_split(
    connection: sqlite3.Connection,
) -> None:
    """Remove the legacy chunking column without touching canonical rows."""
    columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(canonical_snapshots)")
    }
    if "chunking_version" in columns:
        connection.execute(
            "ALTER TABLE canonical_snapshots DROP COLUMN chunking_version"
        )
