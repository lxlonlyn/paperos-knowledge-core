"""Centrally declared domain enums."""

from enum import StrEnum


class IngestionJobStatus(StrEnum):
    PENDING = "pending"
    VALIDATING = "validating"
    PARSING = "parsing"
    NORMALIZING = "normalizing"
    WRITING = "writing"
    INDEXING = "indexing"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ParseRunStatus(StrEnum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ParserArtifactType(StrEnum):
    MARKDOWN = "markdown"
    CONTENT_LIST = "content_list"
    MODEL_OUTPUT = "model_output"
    ASSET = "asset"
    TASK_METADATA = "task_metadata"
    PROVIDER_RESPONSE = "provider_response"
    ARCHIVE = "archive"
    OTHER = "other"


class ElementType(StrEnum):
    TITLE = "title"
    PARAGRAPH = "paragraph"
    FORMULA = "formula"
    FIGURE = "figure"
    TABLE = "table"
    CAPTION = "caption"
    FOOTNOTE = "footnote"
    CODE = "code"
    LIST = "list"
    LIST_ITEM = "list_item"
    REFERENCE = "reference"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_NUMBER = "page_number"
    OTHER = "other"


class ReferenceResolutionStatus(StrEnum):
    UNRESOLVED = "unresolved"
    CANDIDATE = "candidate"
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    REJECTED = "rejected"
