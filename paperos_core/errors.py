"""Typed PaperOS errors with stable, user-facing error codes."""

from __future__ import annotations

from pathlib import Path
from typing import Any


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
