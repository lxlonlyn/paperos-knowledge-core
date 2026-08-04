"""Public API schemas reusing the application domain models."""

from paperos_core.documents import (
    DocumentDeletionReport,
    DocumentDetail,
    DocumentSummary,
)
from paperos_core.feedback.models import (
    FeedbackRecord,
    FeedbackRequest,
    ImprovementReport,
)
from paperos_core.retrieval.candidates import QueryRequest, QueryResponse

__all__ = [
    "DocumentDeletionReport",
    "DocumentDetail",
    "DocumentSummary",
    "FeedbackRecord",
    "FeedbackRequest",
    "ImprovementReport",
    "QueryRequest",
    "QueryResponse",
]
