"""Read-only corpus view over retained canonical source artifacts."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Literal

from paperos_core.domain.canonical import CanonicalBundle, Chunk
from paperos_core.ingestion.canonical_repository import CanonicalRepository
from paperos_core.ingestion.registry import SourceRegistry
from paperos_core.ingestion.scholarly_registry import ScholarlyRegistry
from paperos_core.paths import DataPaths
from paperos_core.retrieval.candidates import Candidate


@dataclass(slots=True)
class CorpusView:
    paths: DataPaths
    bundles: dict[str, CanonicalBundle]
    chunks: dict[str, Chunk]
    chunk_bundles: dict[str, CanonicalBundle]
    source_filenames: dict[str, str]
    work_id_by_document: dict[str, str] = field(default_factory=dict)
    document_ids_by_work: dict[str, set[str]] = field(default_factory=dict)
    work_titles: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(
        cls,
        paths: DataPaths,
        canonical_repository: CanonicalRepository,
        registry: SourceRegistry,
        scholarly_registry: ScholarlyRegistry | None = None,
    ) -> CorpusView:
        retained_bundles = canonical_repository.list_bundles()
        bundles = {bundle.document.id: bundle for bundle in retained_bundles}
        with sqlite3.connect(paths.registry_db) as connection:
            exists = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='document_tombstones'"
            ).fetchone()
            deleted = (
                {
                    str(row[0])
                    for row in connection.execute(
                        "SELECT document_id FROM document_tombstones"
                    ).fetchall()
                }
                if exists
                else set()
            )
        bundles = {
            document_id: bundle
            for document_id, bundle in bundles.items()
            if document_id not in deleted
        }
        retained_bundles = [
            bundle
            for bundle in retained_bundles
            if bundle.document.id in bundles
        ]
        chunks = {
            chunk.id: chunk
            for bundle in retained_bundles
            for chunk in canonical_repository.get_chunk_projection(
                bundle.snapshot.id
            ).chunks
        }
        chunk_bundles = {
            chunk.id: bundle
            for bundle in retained_bundles
            for chunk in canonical_repository.get_chunk_projection(
                bundle.snapshot.id
            ).chunks
        }
        source_filenames = {
            bundle.document.source_file_id: registry.get_source(
                bundle.document.source_file_id
            ).original_filename
            for bundle in bundles.values()
        }
        work_id_by_document: dict[str, str] = {}
        document_ids_by_work: dict[str, set[str]] = {}
        work_titles: dict[str, str] = {}
        if scholarly_registry is not None:
            for work in scholarly_registry.list_works():
                work_titles[work.id] = work.title
            for document_id in bundles:
                work = scholarly_registry.work_for_document(document_id)
                if work is None:
                    continue
                work_id_by_document[document_id] = work.id
                document_ids_by_work.setdefault(work.id, set()).add(document_id)
                work_titles[work.id] = work.title
        return cls(
            paths=paths,
            bundles=bundles,
            chunks=chunks,
            chunk_bundles=chunk_bundles,
            source_filenames=source_filenames,
            work_id_by_document=work_id_by_document,
            document_ids_by_work=document_ids_by_work,
            work_titles=work_titles,
        )

    def candidate_for_chunk(
        self,
        chunk_id: str,
        *,
        channel: str,
        score: float,
        object_id: str | None = None,
        object_type: str = "chunk",
        knowledge_kind: Literal[
            "source_fact",
            "structured_relation",
            "system_inference",
            "user_confirmed",
        ] = "source_fact",
        derived_from_ids: list[str] | None = None,
        text: str | None = None,
        source_work_id: str | None = None,
        subject_work_ids: list[str] | None = None,
        candidate_id: str | None = None,
    ) -> Candidate:
        chunk = self.chunks[chunk_id]
        bundle = self.chunk_bundles[chunk_id]
        resolved_source_work = source_work_id or self.work_id_by_document.get(
            chunk.document_id
        )
        return Candidate(
            id=candidate_id or chunk.id,
            object_id=object_id or chunk.id,
            object_type=object_type,
            document_id=chunk.document_id,
            source_file_id=bundle.document.source_file_id,
            source_filename=self.source_filenames[bundle.document.source_file_id],
            canonical_snapshot_id=bundle.snapshot.id,
            chunk_id=chunk.id,
            section_id=chunk.section_id,
            section_path=chunk.section_path,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            text=text if text is not None else chunk.text,
            channels=[channel],
            channel_scores={channel: score},
            knowledge_kind=knowledge_kind,
            derived_from_ids=derived_from_ids or [],
            source_work_id=resolved_source_work,
            subject_work_ids=list(subject_work_ids or []),
        )

    def filtered_document_ids(
        self,
        requested_document_ids: list[str] | None,
        dataset_name: str,
    ) -> set[str]:
        dataset_documents = {
            document_id
            for document_id, bundle in self.bundles.items()
            if bundle.snapshot.dataset_id == dataset_name
        }
        if requested_document_ids is None:
            return dataset_documents
        return dataset_documents.intersection(requested_document_ids)

    def document_ids_for_works(self, work_ids: list[str] | set[str]) -> set[str]:
        selected: set[str] = set()
        for work_id in work_ids:
            selected.update(self.document_ids_by_work.get(work_id, set()))
        return selected

    def explicitly_mentioned_document_ids(self, query: str) -> set[str]:
        """Resolve unambiguous title/identifier mentions without an LLM planner."""
        normalized_query = _normalized_title_text(query)
        title_tokens = {
            document_id: _normalized_title_text(bundle.document.title).split()
            for document_id, bundle in self.bundles.items()
        }
        prefixes = {
            document_id: {
                " ".join(tokens[:length])
                for length in range(2, len(tokens) + 1)
                if len(" ".join(tokens[:length])) >= 7
            }
            for document_id, tokens in title_tokens.items()
        }
        matched: set[str] = set()
        for document_id, bundle in self.bundles.items():
            title = " ".join(title_tokens[document_id])
            identifiers = {
                _normalized_title_text(token)
                for token in re.findall(
                    r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", bundle.document.title
                )
                if len(token) >= 5
            }
            unique_prefixes = {
                prefix
                for prefix in prefixes[document_id]
                if not any(
                    prefix in other_prefixes
                    for other_id, other_prefixes in prefixes.items()
                    if other_id != document_id
                )
            }
            mentions = {title, *identifiers, *unique_prefixes}
            if any(
                _contains_title_phrase(normalized_query, mention)
                for mention in mentions
            ):
                matched.add(document_id)
        return matched


def _normalized_title_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _contains_title_phrase(normalized_query: str, phrase: str) -> bool:
    return f" {phrase} " in f" {normalized_query} "
