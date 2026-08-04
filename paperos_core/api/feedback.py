"""Feedback API domain exports."""

from paperos_core.feedback.models import (
    FeedbackRecord,
    FeedbackRequest,
    ImprovementReport,
)
from paperos_core.feedback.service import FeedbackService

__all__ = [
    "FeedbackRecord",
    "FeedbackRequest",
    "FeedbackService",
    "ImprovementReport",
]
