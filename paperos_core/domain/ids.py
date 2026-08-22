"""Versioned deterministic identity helpers."""

from __future__ import annotations

import hashlib
import re
import uuid

SOURCE_FILE_SCHEMA_VERSION = "1.0"
SOURCE_FILE_ID_VERSION = "1"
INGESTION_JOB_SCHEMA_VERSION = "1.0"
INGESTION_JOB_ID_VERSION = "1"
PARSE_RUN_SCHEMA_VERSION = "1.0"
PARSER_ARTIFACT_ID_VERSION = "1"
CANONICAL_SCHEMA_VERSION = "1.0"
CANONICAL_ID_VERSION = "1"
CANONICAL_PIPELINE_VERSION = "gate3.2"
CLEANING_VERSION = "1"
CLASSIFICATION_VERSION = "1"
CHUNKING_VERSION = "4"
REFERENCE_PROCESSING_VERSION = "1"
KNOWLEDGE_TRIPLET_ID_VERSION = "1"
SCHOLARLY_WORK_SCHEMA_VERSION = "1.0"
SCHOLARLY_WORK_ID_VERSION = "1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def normalize_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if not _SHA256_PATTERN.fullmatch(normalized):
        raise ValueError("SHA-256 must contain exactly 64 hexadecimal characters")
    return normalized


def stable_id(prefix: str, *parts: str, id_version: str) -> str:
    if not prefix or not id_version or not parts:
        raise ValueError("Stable IDs require a prefix, ID version, and identity parts")
    canonical = "\x1f".join(("paperos", prefix, id_version, *parts))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:32]}"


def source_file_id(sha256: str, *, id_version: str = SOURCE_FILE_ID_VERSION) -> str:
    return stable_id("src", normalize_sha256(sha256), id_version=id_version)


def ingestion_job_id() -> str:
    """Jobs represent requests, so each request receives a distinct identity."""
    return f"job_{uuid.uuid4().hex}"


def parse_run_id() -> str:
    return f"parse_{uuid.uuid4().hex}"


def scholarly_work_id() -> str:
    """Allocate a permanent identity once; never derive it from mutable metadata."""
    return f"work_{uuid.uuid4().hex}"


def parser_artifact_id(
    parse_id: str,
    artifact_type: str,
    relative_path: str,
    sha256: str,
    *,
    id_version: str = PARSER_ARTIFACT_ID_VERSION,
) -> str:
    return stable_id(
        "artifact",
        parse_id,
        artifact_type,
        relative_path,
        normalize_sha256(sha256),
        id_version=id_version,
    )


def canonical_snapshot_id(
    parse_id: str,
    *,
    schema_version: str = CANONICAL_SCHEMA_VERSION,
    pipeline_version: str = CANONICAL_PIPELINE_VERSION,
    id_version: str = CANONICAL_ID_VERSION,
) -> str:
    return stable_id(
        "snapshot",
        parse_id,
        schema_version,
        pipeline_version,
        id_version=id_version,
    )


def document_id(
    source_id: str,
    *,
    schema_version: str = CANONICAL_SCHEMA_VERSION,
    id_version: str = CANONICAL_ID_VERSION,
) -> str:
    return stable_id("doc", source_id, schema_version, id_version=id_version)


def person_id(
    document_id_value: str,
    display_name: str,
    author_order: int,
    *,
    id_version: str = CANONICAL_ID_VERSION,
) -> str:
    return stable_id(
        "person",
        document_id_value,
        str(author_order),
        display_name.casefold(),
        id_version=id_version,
    )


def section_id(
    document_id_value: str,
    order: int,
    path: str,
    *,
    id_version: str = CANONICAL_ID_VERSION,
) -> str:
    return stable_id("section", document_id_value, str(order), path, id_version=id_version)


def element_id(
    document_id_value: str,
    order: int,
    artifact_id: str,
    artifact_item_index: int,
    content_digest: str,
    *,
    id_version: str = CANONICAL_ID_VERSION,
) -> str:
    return stable_id(
        "element",
        document_id_value,
        str(order),
        artifact_id,
        str(artifact_item_index),
        content_digest,
        id_version=id_version,
    )


def chunk_id(
    document_id_value: str,
    order: int,
    element_span_ids: list[str],
    *,
    chunking_version: str = CHUNKING_VERSION,
    id_version: str = CANONICAL_ID_VERSION,
) -> str:
    """Identify one chunk from the exact element-internal spans it covers."""
    return stable_id(
        "chunk",
        document_id_value,
        str(order),
        chunking_version,
        *element_span_ids,
        id_version=id_version,
    )


def reference_entry_id(
    document_id_value: str,
    order: int,
    raw_text: str,
    *,
    id_version: str = CANONICAL_ID_VERSION,
) -> str:
    return stable_id(
        "reference",
        document_id_value,
        str(order),
        raw_text,
        id_version=id_version,
    )


def citation_span_id(
    document_id_value: str,
    element_id: str,
    character_start: int,
    character_end: int,
    *,
    id_version: str = CANONICAL_ID_VERSION,
) -> str:
    return stable_id(
        "cite_span",
        document_id_value,
        element_id,
        str(character_start),
        str(character_end),
        id_version=id_version,
    )


def citation_mention_id(
    document_id_value: str,
    element_id: str,
    citation_span_id_value: str,
    atomic_key: str,
    group_index: int,
    *,
    id_version: str = CANONICAL_ID_VERSION,
) -> str:
    return stable_id(
        "cite_mention",
        document_id_value,
        element_id,
        citation_span_id_value,
        atomic_key,
        str(group_index),
        id_version=id_version,
    )


def semantic_object_id(
    prefix: str,
    canonical_snapshot_id_value: str,
    content: str,
    source_ids: list[str],
    *,
    id_version: str = CANONICAL_ID_VERSION,
) -> str:
    return stable_id(
        prefix,
        canonical_snapshot_id_value,
        content,
        *sorted(source_ids),
        id_version=id_version,
    )


def knowledge_triplet_id(
    canonical_snapshot_id_value: str,
    source_id: str,
    relation_type: str,
    target_id: str,
    source_chunk_ids: list[str],
    *,
    id_version: str = KNOWLEDGE_TRIPLET_ID_VERSION,
) -> str:
    """Identify one versioned, searchable projection of a typed graph edge."""
    return stable_id(
        "triplet",
        canonical_snapshot_id_value,
        source_id,
        relation_type,
        target_id,
        *sorted(source_chunk_ids),
        id_version=id_version,
    )
