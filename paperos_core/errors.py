"""Typed PaperOS errors with stable, user-facing error codes."""

from __future__ import annotations

import re
from pathlib import Path, PureWindowsPath
from typing import Any

_DEFAULT_PUBLIC_MESSAGE = "The request could not be completed."
_PUBLIC_MESSAGES = {
    "paperos_error": _DEFAULT_PUBLIC_MESSAGE,
    "configuration_error": "PaperOS configuration is invalid.",
    "invalid_dataset": "The requested dataset is invalid.",
    "missing_source_file": "The source file is unavailable.",
    "invalid_pdf": "The source PDF is invalid.",
    "pdf_too_large": "The source PDF exceeds the configured size limit.",
    "source_changed_during_ingestion": "The source changed during ingestion.",
    "storage_integrity_error": "Stored data failed an integrity check.",
    "source_registry_error": "The source registry operation failed.",
    "source_not_found": "The requested source was not found.",
    "ingestion_job_not_found": "The requested ingestion job was not found.",
    "invalid_ingestion_job_transition": "The ingestion job transition is invalid.",
    "mineru_configuration_error": "The document parser configuration is invalid.",
    "mineru_authentication_error": "Document parser authentication failed.",
    "mineru_quota_error": "The document parser quota is unavailable.",
    "mineru_timeout": "The document parser timed out.",
    "mineru_provider_error": "The document parser is unavailable.",
    "mineru_parse_failure": "The document could not be parsed.",
    "parser_artifact_validation_error": "A parser artifact failed validation.",
    "canonical_mapping_error": "The parsed document could not be mapped.",
    "canonical_validation_error": "Canonical document validation failed.",
    "canonical_storage_error": "Canonical document storage failed.",
    "local_inference_configuration_error": "Local inference configuration is invalid.",
    "local_inference_unavailable": "Local inference is unavailable.",
    "local_inference_response_error": "Local inference returned an invalid response.",
    "cognee_configuration_error": "Knowledge engine configuration is invalid.",
    "cognee_storage_error": "The knowledge engine is unavailable.",
    "semantic_enrichment_error": "Semantic enrichment failed.",
    "index_storage_error": "The retrieval index is unavailable.",
    "feedback_validation_error": "Feedback validation failed.",
    "feedback_storage_error": "Feedback storage failed.",
    "document_not_found": "The requested document was not found.",
    "job_queue_error": "The job queue operation failed.",
    "mineru_unavailable": "The document parser is unavailable.",
    "llm_unavailable": "The language model is unavailable.",
    "local_models_unavailable": "Local inference is unavailable.",
    "vector_unavailable": "The vector index is unavailable.",
    "cognee_graph_unavailable": "The knowledge graph is unavailable.",
    "worker_unavailable": "The operational worker is unavailable.",
    "operational_job_failed": "The operation could not be completed.",
}


def public_diagnostic(code: str) -> dict[str, str]:
    """Return a stable public diagnostic without internal exception text."""

    return {
        "code": code,
        "message": _PUBLIC_MESSAGES.get(code, _DEFAULT_PUBLIC_MESSAGE),
    }


class PaperOSError(Exception):
    """Base error returned by application and CLI boundaries."""

    code = "paperos_error"
    retryable = False

    def __init__(
        self,
        message: str,
        *,
        affected: str | Path | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.affected = str(affected) if affected is not None else None
        self.retryable = self.retryable if retryable is None else retryable
        self.details = details or {}

    def as_dict(self) -> dict[str, Any]:
        error: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.affected is not None:
            error["affected"] = self.affected
        if self.details:
            error["details"] = self.details
        return {"error": error}

    def as_api_dict(self) -> dict[str, Any]:
        """Serialize client-safe fields without machine-local path disclosure."""

        error: dict[str, Any] = {
            **public_diagnostic(self.code),
            "retryable": self.retryable,
        }
        details = _safe_api_value(self.details)
        if isinstance(details, dict) and details:
            error["details"] = details
        return {"error": error}


_PATH_DETAIL_KEYS = {"affected", "directory", "file", "filename", "path", "root"}
_DANGEROUS_DETAIL_KEYS = {
    "artifact_errors",
    "attempts",
    "exception",
    "failures",
    "last_error",
    "message",
    "stack",
    "traceback",
}
_SAFE_STRING_KEY_SUFFIXES = (
    "_code",
    "_id",
    "_ids",
    "_key",
    "_keys",
    "_role",
    "_roles",
    "_sha256",
    "_status",
    "_type",
    "_types",
)
_SAFE_STRING_KEYS = {"actual", "expected", "missing", "reason", "status"}
_POSIX_PATH_IN_TEXT = re.compile(r"(?:^|[\s\[({'\"=:])/(?!/)[^\s\])}'\",;]+")
_WINDOWS_PATH_IN_TEXT = re.compile(
    r"(?:^|[\s\[({'\"=:])(?:[A-Za-z]:[\\/]|\\\\[^\\/\s]+[\\/])"
)
_REDACTED = object()


def _is_path_detail_key(key: str) -> bool:
    normalized = key.casefold()
    return normalized in _PATH_DETAIL_KEYS or any(
        normalized.endswith(f"_{suffix}") for suffix in _PATH_DETAIL_KEYS
    )


def _contains_local_reference(value: str) -> bool:
    selected = value.strip()
    return (
        "file://" in selected.casefold()
        or Path(selected).is_absolute()
        or PureWindowsPath(selected).is_absolute()
        or _POSIX_PATH_IN_TEXT.search(selected) is not None
        or _WINDOWS_PATH_IN_TEXT.search(selected) is not None
    )


def _is_safe_string_key(key: str | None) -> bool:
    if key is None:
        return False
    normalized = key.casefold()
    return normalized in _SAFE_STRING_KEYS or normalized.endswith(
        _SAFE_STRING_KEY_SUFFIXES
    )


def _safe_api_value(value: Any, *, key: str | None = None) -> Any:
    normalized_key = key.casefold() if key is not None else None
    if key is not None and (
        _is_path_detail_key(key) or normalized_key in _DANGEROUS_DETAIL_KEYS
    ):
        return _REDACTED
    if isinstance(value, Path):
        return _REDACTED
    if isinstance(value, str):
        if _contains_local_reference(value) or not _is_safe_string_key(key):
            return _REDACTED
        return value
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        safe_items_dict: dict[str, Any] = {}
        for item_key, item_value in value.items():
            rendered_key = str(item_key)
            if _contains_local_reference(rendered_key):
                continue
            safe = _safe_api_value(item_value, key=rendered_key)
            if safe is not _REDACTED:
                safe_items_dict[rendered_key] = safe
        return safe_items_dict
    if isinstance(value, (list, tuple)):
        safe_items_list = []
        for item in value:
            safe = _safe_api_value(item, key=key)
            if safe is not _REDACTED:
                safe_items_list.append(safe)
        return safe_items_list
    return _REDACTED


class ConfigurationError(PaperOSError):
    code = "configuration_error"


class InvalidDatasetError(PaperOSError):
    code = "invalid_dataset"


class MissingSourceFileError(PaperOSError):
    code = "missing_source_file"


class InvalidPDFError(PaperOSError):
    code = "invalid_pdf"


class FileTooLargeError(PaperOSError):
    code = "pdf_too_large"


class SourceChangedError(PaperOSError):
    code = "source_changed_during_ingestion"
    retryable = True


class StorageIntegrityError(PaperOSError):
    code = "storage_integrity_error"


class SourceRegistryError(PaperOSError):
    code = "source_registry_error"


class SourceNotFoundError(PaperOSError):
    code = "source_not_found"


class JobNotFoundError(PaperOSError):
    code = "ingestion_job_not_found"


class InvalidJobTransitionError(PaperOSError):
    code = "invalid_ingestion_job_transition"


class MinerUConfigurationError(PaperOSError):
    code = "mineru_configuration_error"


class MinerUAuthenticationError(PaperOSError):
    code = "mineru_authentication_error"


class MinerUQuotaError(PaperOSError):
    code = "mineru_quota_error"
    retryable = True


class MinerUTimeoutError(PaperOSError):
    code = "mineru_timeout"
    retryable = True


class MinerUProviderError(PaperOSError):
    code = "mineru_provider_error"
    retryable = True


class MinerUParseError(PaperOSError):
    code = "mineru_parse_failure"


class ParserArtifactValidationError(PaperOSError):
    code = "parser_artifact_validation_error"


class CanonicalMappingError(PaperOSError):
    code = "canonical_mapping_error"


class CanonicalValidationError(PaperOSError):
    code = "canonical_validation_error"


class CanonicalStorageError(PaperOSError):
    code = "canonical_storage_error"


class LocalInferenceConfigurationError(PaperOSError):
    code = "local_inference_configuration_error"


class LocalInferenceUnavailableError(PaperOSError):
    code = "local_inference_unavailable"
    retryable = True


class LocalInferenceResponseError(PaperOSError):
    code = "local_inference_response_error"
    retryable = True


class CogneeConfigurationError(PaperOSError):
    code = "cognee_configuration_error"


class CogneeStorageError(PaperOSError):
    code = "cognee_storage_error"
    retryable = True


class SemanticEnrichmentError(PaperOSError):
    code = "semantic_enrichment_error"
    retryable = True


class IndexStorageError(PaperOSError):
    code = "index_storage_error"


class FeedbackValidationError(PaperOSError):
    code = "feedback_validation_error"


class FeedbackStorageError(PaperOSError):
    code = "feedback_storage_error"


class DocumentNotFoundError(PaperOSError):
    code = "document_not_found"


class JobQueueError(PaperOSError):
    code = "job_queue_error"
