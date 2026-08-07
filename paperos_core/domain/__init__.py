"""Shared PaperOS domain models."""

from paperos_core.domain.canonical import (
    CanonicalBundle,
    CanonicalIngestionResult,
    CanonicalKnowledgeBundle,
    CanonicalSnapshot,
    Chunk,
    Document,
    Element,
    Person,
    ReferenceEntry,
    Section,
    SourceSpan,
)
from paperos_core.domain.documents import IngestionJob, IngestionResult, SourceFile
from paperos_core.domain.enums import ElementType, IngestionJobStatus
from paperos_core.domain.parsing import ParsedIngestionResult, ParserArtifact, ParseRun

__all__ = [
    "CanonicalBundle",
    "CanonicalIngestionResult",
    "CanonicalKnowledgeBundle",
    "CanonicalSnapshot",
    "Chunk",
    "Document",
    "Element",
    "ElementType",
    "IngestionJob",
    "IngestionJobStatus",
    "IngestionResult",
    "ParseRun",
    "ParsedIngestionResult",
    "ParserArtifact",
    "Person",
    "ReferenceEntry",
    "Section",
    "SourceFile",
    "SourceSpan",
]
