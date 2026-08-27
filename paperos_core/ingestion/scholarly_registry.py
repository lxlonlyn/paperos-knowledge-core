"""Persistent resolver for stable ScholarlyWork identities."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any

from paperos_core.domain.canonical import (
    CanonicalBundle,
    Chunk,
    Document,
    ReferenceEntry,
)
from paperos_core.domain.documents import utc_now
from paperos_core.domain.ids import (
    SCHOLARLY_WORK_ID_VERSION,
    SCHOLARLY_WORK_SCHEMA_VERSION,
    scholarly_work_id,
)
from paperos_core.domain.scholarly import (
    ReferenceWorkResolution,
    ScholarlyContext,
    ScholarlyWork,
    WorkIdentifierKind,
    WorkIdentityStatus,
)
from paperos_core.paths import DataPaths

if TYPE_CHECKING:
    from paperos_core.ingestion.canonical_repository import CanonicalRepository

_DOI_PREFIX = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
_ARXIV_PREFIX = re.compile(
    r"^(?:https?://arxiv\.org/(?:abs|pdf)/|arxiv:\s*)", re.IGNORECASE
)
_ARXIV_VERSION = re.compile(r"v\d+$", re.IGNORECASE)
_ALPHA_REFERENCE = re.compile(
    r"^\[[A-Za-z][^]]*]\s*(?P<authors>[^:]{2,300}):\s*(?P<body>.+)$"
)
_REFERENCE_PREFIX = re.compile(r"^\[[^]]+]\s*")
_YEAR_FIRST_REFERENCE = re.compile(
    r"^(?P<authors>.{3,500}?)\.\s+(?P<year>(?:19|20)\d{2})\.\s+"
    r"(?P<title>[^.]{4,500})(?:\.\s|$)"
)
_TITLE_AFTER_AUTHORS_REFERENCE = re.compile(
    r"^(?P<authors>.{3,500}?)\.\s+(?P<title>[^.]{4,500})\.\s+(?P<body>.+)$"
)
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_STATUS_PRIORITY = {
    WorkIdentityStatus.PROVISIONAL: 0,
    WorkIdentityStatus.IDENTIFIED: 1,
    WorkIdentityStatus.INGESTED: 2,
}


class ScholarlyRegistry:
    """Own Work identities in registry.db; Cognee only receives projections."""

    _STATE_TABLES = (
        "scholarly_works",
        "work_identifiers",
        "document_work_links",
        "reference_work_links",
        "work_redirects",
    )

    def __init__(self, paths: DataPaths, *, database_path: Path | None = None) -> None:
        self.paths = paths
        self.database_path = database_path or paths.registry_db

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def candidate_database_path(self, snapshot_id: str) -> Path:
        path = (
            self.paths.cognee / "scholarly_staging" / f"{snapshot_id}.sqlite3"
        ).resolve(strict=False)
        self.paths.assert_within_root(path)
        return path

    def candidate_manifest_path(self, snapshot_id: str) -> Path:
        return self.candidate_database_path(snapshot_id).with_suffix(".json")

    def resolve_candidate_bundle(
        self,
        bundle: CanonicalBundle,
        chunks: list[Chunk],
    ) -> ScholarlyContext:
        """Resolve a candidate against an isolated copy of active registry state."""

        snapshot_id = bundle.snapshot.id
        database_path = self.candidate_database_path(snapshot_id)
        manifest_path = self.candidate_manifest_path(snapshot_id)
        self.discard_candidate(snapshot_id)
        database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._connect() as source:
                source.execute("BEGIN")
                active_pointers = self._active_pointers(source)
                base_digest = self._state_digest(source)
                with sqlite3.connect(database_path, timeout=30) as candidate:
                    source.backup(candidate)
            _write_json(
                manifest_path,
                {
                    "snapshot_id": snapshot_id,
                    "active_pointers": active_pointers,
                    "base_state_digest": base_digest,
                },
            )
            staged = ScholarlyRegistry(self.paths, database_path=database_path)
            return staged.resolve_bundle(bundle, chunks)
        except Exception:
            self.discard_candidate(snapshot_id)
            raise

    def publish_candidate(
        self,
        snapshot_id: str,
        repository: CanonicalRepository,
        *,
        expected_previous_snapshot_id: str | None = None,
    ) -> str | None:
        """Atomically publish candidate scholarly state and its active pointer."""

        database_path = self.candidate_database_path(snapshot_id)
        manifest_path = self.candidate_manifest_path(snapshot_id)
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"Scholarly candidate metadata is unavailable for {snapshot_id}"
            ) from exc
        if metadata.get("snapshot_id") != snapshot_id or not database_path.is_file():
            raise RuntimeError(f"Scholarly candidate is incomplete for {snapshot_id}")

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if self._active_pointers(connection) != metadata.get("active_pointers"):
                raise RuntimeError(
                    "Active canonical state changed while scholarly candidate was building"
                )
            if self._state_digest(connection) != metadata.get("base_state_digest"):
                raise RuntimeError(
                    "Scholarly registry changed while candidate was building"
                )
            if expected_previous_snapshot_id is not None:
                current = connection.execute(
                    "SELECT active.snapshot_id "
                    "FROM canonical_snapshots AS candidate "
                    "LEFT JOIN active_canonical_snapshots AS active "
                    "ON active.document_id = candidate.document_id "
                    "WHERE candidate.id = ?",
                    (snapshot_id,),
                ).fetchone()
                current_snapshot_id = (
                    str(current["snapshot_id"])
                    if current is not None and current["snapshot_id"] is not None
                    else None
                )
                if current_snapshot_id != expected_previous_snapshot_id:
                    raise RuntimeError(
                        "Active canonical revision changed before candidate publication"
                    )
            connection.execute(
                "ATTACH DATABASE ? AS scholarly_candidate",
                (str(database_path),),
            )
            connection.execute("PRAGMA defer_foreign_keys = ON")
            for table in reversed(self._STATE_TABLES):
                connection.execute(f"DELETE FROM {table}")
            for table in self._STATE_TABLES:
                connection.execute(
                    f"INSERT INTO {table} SELECT * FROM scholarly_candidate.{table}"
                )
            previous = repository.activate_snapshot(
                snapshot_id,
                connection=connection,
            )
        self.discard_candidate(snapshot_id)
        return previous

    def discard_candidate(self, snapshot_id: str) -> None:
        database_path = self.candidate_database_path(snapshot_id)
        for path in (
            database_path,
            Path(f"{database_path}-wal"),
            Path(f"{database_path}-shm"),
            self.candidate_manifest_path(snapshot_id),
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                # Published state is atomic; staging cleanup remains retryable.
                continue

    @staticmethod
    def _active_pointers(connection: sqlite3.Connection) -> dict[str, str]:
        return {
            str(row[0]): str(row[1])
            for row in connection.execute(
                "SELECT document_id, snapshot_id FROM active_canonical_snapshots "
                "ORDER BY document_id"
            ).fetchall()
        }

    @classmethod
    def _state_digest(cls, connection: sqlite3.Connection) -> str:
        digest = hashlib.sha256()
        for table in cls._STATE_TABLES:
            digest.update(table.encode("utf-8"))
            rows = connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for row in rows:
                digest.update(
                    json.dumps(
                        list(row),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
        return digest.hexdigest()

    @staticmethod
    def normalize_doi(value: str | None) -> str | None:
        if not value or not value.strip():
            return None
        normalized = _DOI_PREFIX.sub("", value.strip()).strip().casefold()
        return normalized.rstrip(".,;)") or None

    @staticmethod
    def normalize_arxiv(value: str | None) -> str | None:
        if not value or not value.strip():
            return None
        normalized = _ARXIV_PREFIX.sub("", value.strip()).strip()
        normalized = normalized.removesuffix(".pdf")
        normalized = _ARXIV_VERSION.sub("", normalized).casefold()
        return normalized.rstrip(".,;)") or None

    @staticmethod
    def normalize_text(value: str | None) -> str:
        if not value:
            return ""
        folded = unicodedata.normalize("NFKC", value).casefold()
        return " ".join(_NON_WORD.sub(" ", folded).split())

    @classmethod
    def normalize_author(cls, value: str | None) -> str | None:
        normalized = cls.normalize_text(value)
        return normalized or None

    @staticmethod
    def _split_reference_authors(value: str) -> list[str]:
        normalized = re.sub(r"\s+(?:and|&)\s+", ", ", value, flags=re.IGNORECASE)
        return [
            item.strip().strip(",")
            for item in normalized.split(",")
            if item.strip().strip(",")
        ]

    @classmethod
    def infer_reference_identity(
        cls, raw_text: str
    ) -> tuple[str | None, list[str]]:
        """Conservatively recover identity from common bibliography styles."""
        raw = raw_text.strip()
        alpha = _ALPHA_REFERENCE.match(raw)
        if alpha is not None:
            authors = cls._split_reference_authors(alpha.group("authors"))
            title = alpha.group("body").split(". ", 1)[0].strip().rstrip(".")
            return (title, authors) if authors and title else (None, [])

        unkeyed = _REFERENCE_PREFIX.sub("", raw)
        comma_identity = cls._infer_comma_separated_reference(unkeyed)
        if comma_identity is not None:
            return comma_identity

        match = _YEAR_FIRST_REFERENCE.match(unkeyed)
        if match is None:
            match = _TITLE_AFTER_AUTHORS_REFERENCE.match(unkeyed)
            if match is None or re.search(r"(?:19|20)\d{2}", match.group("body")) is None:
                return None, []
        authors_text = match.group("authors").strip()
        if "," not in authors_text and " and " not in authors_text.casefold():
            return None, []
        authors = cls._split_reference_authors(authors_text)
        title = match.group("title").strip().rstrip(".")
        if not authors or len(title.split()) < 2:
            return None, []
        return title, authors

    @classmethod
    def _infer_comma_separated_reference(
        cls, raw_text: str
    ) -> tuple[str, list[str]] | None:
        """Parse author lists without mistaking initials for sentence boundaries."""

        authors: list[str] = []
        for part in (item.strip() for item in raw_text.split(",")):
            if not part:
                continue
            split = cls._split_author_and_title(part)
            if split is not None:
                author, title = split
                authors.append(author)
                return title, authors
            if cls._looks_like_author_name(part):
                authors.append(re.sub(r"^(?:and|&)\s+", "", part).strip())
                continue
            title = part.strip().rstrip(".")
            if authors and len(title.split()) >= 2:
                return title, authors
            return None
        return None

    @classmethod
    def _split_author_and_title(cls, value: str) -> tuple[str, str] | None:
        if any(character.isdigit() for character in value):
            return None
        boundaries = [match.start() for match in re.finditer(r"\.\s+", value)]
        for boundary in reversed(boundaries):
            author = value[: boundary + 1].strip().rstrip(".")
            title_body = value[boundary + 1 :].strip()
            title = title_body.split(". ", 1)[0].strip().rstrip(".")
            if cls._looks_like_author_name(author) and len(title.split()) >= 2:
                return re.sub(r"^(?:and|&)\s+", "", author).strip(), title
        return None

    @staticmethod
    def _looks_like_author_name(value: str) -> bool:
        candidate = re.sub(r"^(?:and|&)\s+", "", value.strip()).strip()
        if not candidate or any(character.isdigit() for character in candidate):
            return False
        tokens = candidate.split()
        if not 2 <= len(tokens) <= 4:
            return False
        surname = tokens[-1].strip(".,;:()[]{}")
        first = tokens[0].strip(".,;:()[]{}")
        return bool(
            surname
            and first
            and surname[0].isupper()
            and first[0].isupper()
            and ":" not in candidate
        )

    @classmethod
    def _authors_compatible(cls, first: str | None, second: str | None) -> bool:
        if not first or not second:
            return False
        if first == second:
            return True
        first_tokens = first.split()
        second_tokens = second.split()
        boundary_tokens = {
            token
            for token in (
                first_tokens[0],
                first_tokens[-1],
                second_tokens[0],
                second_tokens[-1],
            )
            if len(token) >= 2
        }
        return bool(boundary_tokens & set(first_tokens) & set(second_tokens))

    @classmethod
    def _can_merge_ingested_duplicate(
        cls,
        first: ScholarlyWork,
        second: ScholarlyWork,
    ) -> bool:
        """Conservatively reconcile one ingested Work with an identity duplicate."""
        statuses = {first.identity_status, second.identity_status}
        if WorkIdentityStatus.INGESTED not in statuses:
            return False
        if (
            first.identity_status is WorkIdentityStatus.INGESTED
            and second.identity_status is WorkIdentityStatus.INGESTED
        ):
            return False
        if (
            not first.normalized_title
            or first.normalized_title != second.normalized_title
            or first.year is None
            or second.year is None
            or first.year != second.year
            or not cls._authors_compatible(
                first.normalized_first_author,
                second.normalized_first_author,
            )
        ):
            return False
        first_doi = cls.normalize_doi(first.doi)
        second_doi = cls.normalize_doi(second.doi)
        if first_doi and second_doi and first_doi != second_doi:
            return False
        first_arxiv = cls.normalize_arxiv(first.arxiv_id)
        second_arxiv = cls.normalize_arxiv(second.arxiv_id)
        return not (
            first_arxiv and second_arxiv and first_arxiv != second_arxiv
        )

    @staticmethod
    def _title_similarity(first: str, second: str) -> float:
        if first == second:
            return 1.0
        return SequenceMatcher(None, first, second, autojunk=False).ratio()

    @classmethod
    def identity_attributes_match(
        cls,
        work: ScholarlyWork,
        *,
        title: str,
        year: int | None,
        first_author: str | None,
        doi: str | None = None,
        arxiv_id: str | None = None,
    ) -> bool:
        """Apply the registry's conservative read-only identity predicate."""
        incoming_doi = cls.normalize_doi(doi)
        candidate_doi = cls.normalize_doi(work.doi)
        if incoming_doi and candidate_doi and incoming_doi != candidate_doi:
            return False
        incoming_arxiv = cls.normalize_arxiv(arxiv_id)
        candidate_arxiv = cls.normalize_arxiv(work.arxiv_id)
        if incoming_arxiv and candidate_arxiv and incoming_arxiv != candidate_arxiv:
            return False
        normalized_title = cls.normalize_text(title)
        normalized_author = cls.normalize_author(first_author)
        author_available = bool(
            normalized_author and work.normalized_first_author
        )
        if author_available and not cls._authors_compatible(
            normalized_author,
            work.normalized_first_author,
        ):
            return False
        threshold = 0.86 if author_available else 0.90
        return bool(
            normalized_title
            and year is not None
            and work.year == year
            and cls._title_similarity(normalized_title, work.normalized_title)
            >= threshold
        )

    def canonicalize_work_id(
        self, work_id: str, connection: sqlite3.Connection | None = None
    ) -> str:
        owns_connection = connection is None
        db = connection or self._connect()
        try:
            current = work_id
            seen: set[str] = set()
            while current not in seen:
                seen.add(current)
                row = db.execute(
                    "SELECT survivor_work_id FROM work_redirects "
                    "WHERE loser_work_id = ?",
                    (current,),
                ).fetchone()
                if row is None:
                    return current
                current = str(row["survivor_work_id"])
            raise RuntimeError(f"Work redirect cycle detected at {current}")
        finally:
            if owns_connection:
                db.close()

    def get_work(self, work_id: str) -> ScholarlyWork:
        with self._connect() as connection:
            canonical_id = self.canonicalize_work_id(work_id, connection)
            return self._get_work(connection, canonical_id)

    def list_works(self, *, include_redirected: bool = False) -> list[ScholarlyWork]:
        query = "SELECT * FROM scholarly_works"
        if not include_redirected:
            query += (
                " WHERE id NOT IN "
                "(SELECT loser_work_id FROM work_redirects)"
            )
        query += " ORDER BY created_at, id"
        with self._connect() as connection:
            return [
                self._row_to_work(row)
                for row in connection.execute(query).fetchall()
            ]

    def list_redirects(self) -> dict[str, str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT loser_work_id, survivor_work_id FROM work_redirects "
                "ORDER BY loser_work_id"
            ).fetchall()
            return {
                str(row["loser_work_id"]): self.canonicalize_work_id(
                    str(row["survivor_work_id"]), connection
                )
                for row in rows
            }

    def work_for_document(self, document_id: str) -> ScholarlyWork | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT work_id FROM document_work_links WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            if row is None:
                return None
            work_id = self.canonicalize_work_id(str(row["work_id"]), connection)
            return self._get_work(connection, work_id)

    def work_for_reference(self, reference_id: str) -> ScholarlyWork | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT work_id FROM reference_work_links WHERE reference_id = ?",
                (reference_id,),
            ).fetchone()
            if row is None or row["work_id"] is None:
                return None
            work_id = self.canonicalize_work_id(str(row["work_id"]), connection)
            return self._get_work(connection, work_id)

    def resolve_document(self, document: Document) -> ScholarlyWork:
        authors = [person.display_name for person in document.authors]
        first_author = self.normalize_author(authors[0] if authors else None)
        doi = self.normalize_doi(document.doi)
        arxiv = self.normalize_arxiv(document.arxiv_id)
        title = document.title.strip() or document.id
        normalized_title = self.normalize_text(title)
        now = utc_now().isoformat()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            linked = connection.execute(
                "SELECT work_id FROM document_work_links WHERE document_id = ?",
                (document.id,),
            ).fetchone()
            candidates: list[str] = []
            if linked is not None:
                candidates.append(
                    self.canonicalize_work_id(str(linked["work_id"]), connection)
                )
            for kind, value in (
                (WorkIdentifierKind.DOI, doi),
                (WorkIdentifierKind.ARXIV, arxiv),
            ):
                matched = self._find_identifier(connection, kind, value)
                if matched is not None:
                    candidates.append(matched)

            if candidates:
                work_id = candidates[0]
                for candidate in candidates[1:]:
                    work_id = self._merge(connection, work_id, candidate)
                for candidate in self._find_title_candidates(
                    connection,
                    normalized_title,
                    document.year,
                    first_author,
                    incoming_doi=doi,
                    incoming_arxiv=arxiv,
                ):
                    candidate = self.canonicalize_work_id(candidate, connection)
                    work_id = self.canonicalize_work_id(work_id, connection)
                    if candidate == work_id:
                        continue
                    if self._can_merge_ingested_duplicate(
                        self._get_work(connection, work_id),
                        self._get_work(connection, candidate),
                    ):
                        work_id = self._merge(connection, work_id, candidate)
            else:
                title_candidates = self._find_title_candidates(
                    connection,
                    normalized_title,
                    document.year,
                    first_author,
                    incoming_doi=doi,
                    incoming_arxiv=arxiv,
                )
                work_id = (
                    title_candidates[0]
                    if len(title_candidates) == 1
                    else self._create_work(
                        connection,
                        title=title,
                        normalized_title=normalized_title,
                        doi=document.doi,
                        arxiv_id=document.arxiv_id,
                        year=document.year,
                        authors=authors,
                        first_author=first_author,
                        status=WorkIdentityStatus.INGESTED,
                        confidence=1.0,
                    )
                )

            work_id = self._promote(
                connection,
                work_id,
                title=title,
                normalized_title=normalized_title,
                doi=document.doi,
                arxiv_id=document.arxiv_id,
                year=document.year,
                authors=authors,
                first_author=first_author,
                status=WorkIdentityStatus.INGESTED,
                confidence=1.0,
            )
            work_id = self._attach_identifiers(
                connection,
                work_id,
                doi=document.doi,
                arxiv_id=document.arxiv_id,
                title=title,
            )
            connection.execute(
                """
                INSERT INTO document_work_links(
                    document_id, work_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    work_id = excluded.work_id,
                    updated_at = excluded.updated_at
                """,
                (document.id, work_id, now, now),
            )
            self._reconcile_unresolved(connection)
            return self._get_work(connection, work_id)

    def resolve_reference(
        self,
        reference: ReferenceEntry,
        *,
        source_chunk_ids: list[str] | None = None,
    ) -> ReferenceWorkResolution:
        authors = list(reference.authors)
        title = (reference.title or "").strip()
        if not title or not authors:
            inferred_title, inferred_authors = self.infer_reference_identity(
                reference.raw_text
            )
            title = title or inferred_title or ""
            authors = authors or inferred_authors
        first_author = self.normalize_author(authors[0] if authors else None)
        doi = self.normalize_doi(reference.doi)
        arxiv = self.normalize_arxiv(reference.arxiv_id)
        normalized_title = self.normalize_text(title)
        now = utc_now().isoformat()

        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            work_id: str | None = None
            status = "unresolved"
            confidence = 0.0

            exact_ids: list[str] = []
            for kind, value in (
                (WorkIdentifierKind.DOI, doi),
                (WorkIdentifierKind.ARXIV, arxiv),
            ):
                matched = self._find_identifier(connection, kind, value)
                if matched is not None:
                    exact_ids.append(matched)
            if exact_ids:
                work_id = exact_ids[0]
                for candidate in exact_ids[1:]:
                    work_id = self._merge(connection, work_id, candidate)
                status = "resolved"
                confidence = 1.0 if doi else 0.99
            elif normalized_title and reference.year is not None and first_author:
                candidates = self._find_title_candidates(
                    connection,
                    normalized_title,
                    reference.year,
                    first_author,
                    incoming_doi=doi,
                    incoming_arxiv=arxiv,
                )
                if len(candidates) == 1:
                    work_id = candidates[0]
                    status = "resolved"
                    confidence = 0.92
                elif len(candidates) > 1:
                    status = "ambiguous"
                else:
                    work_id = self._create_work(
                        connection,
                        title=title,
                        normalized_title=normalized_title,
                        doi=reference.doi,
                        arxiv_id=reference.arxiv_id,
                        year=reference.year,
                        authors=authors,
                        first_author=first_author,
                        status=(
                            WorkIdentityStatus.IDENTIFIED
                            if doi or arxiv
                            else WorkIdentityStatus.PROVISIONAL
                        ),
                        confidence=0.95 if doi or arxiv else 0.82,
                    )
                    work_id = self._attach_identifiers(
                        connection,
                        work_id,
                        doi=reference.doi,
                        arxiv_id=reference.arxiv_id,
                        title=title,
                    )
                    status = "resolved"
                    confidence = 0.95 if doi or arxiv else 0.82
            elif doi or arxiv:
                work_id = self._create_work(
                    connection,
                    title=title or reference.raw_text.strip()[:500] or reference.id,
                    normalized_title=(
                        normalized_title
                        or self.normalize_text(reference.raw_text[:500])
                        or reference.id
                    ),
                    doi=reference.doi,
                    arxiv_id=reference.arxiv_id,
                    year=reference.year,
                    authors=authors,
                    first_author=first_author,
                    status=WorkIdentityStatus.IDENTIFIED,
                    confidence=0.95,
                )
                work_id = self._attach_identifiers(
                    connection,
                    work_id,
                    doi=reference.doi,
                    arxiv_id=reference.arxiv_id,
                    title=title,
                )
                status = "resolved"
                confidence = 0.95

            if work_id is not None:
                work_id = self.canonicalize_work_id(work_id, connection)
            connection.execute(
                """
                INSERT INTO reference_work_links(
                    reference_id, source_document_id, work_id,
                    resolution_status, confidence, normalized_doi,
                    normalized_arxiv, normalized_title, year,
                    normalized_first_author, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(reference_id) DO UPDATE SET
                    source_document_id = excluded.source_document_id,
                    work_id = excluded.work_id,
                    resolution_status = excluded.resolution_status,
                    confidence = excluded.confidence,
                    normalized_doi = excluded.normalized_doi,
                    normalized_arxiv = excluded.normalized_arxiv,
                    normalized_title = excluded.normalized_title,
                    year = excluded.year,
                    normalized_first_author = excluded.normalized_first_author,
                    updated_at = excluded.updated_at
                """,
                (
                    reference.id,
                    reference.document_id,
                    work_id,
                    status,
                    confidence,
                    doi,
                    arxiv,
                    normalized_title or None,
                    reference.year,
                    first_author,
                    now,
                    now,
                ),
            )
            return ReferenceWorkResolution(
                reference_id=reference.id,
                source_document_id=reference.document_id,
                work_id=work_id,
                resolution_status=status,
                confidence=confidence,
                source_chunk_ids=source_chunk_ids or [],
            )

    def resolve_bundle(
        self, bundle: CanonicalBundle, chunks: list[Chunk]
    ) -> ScholarlyContext:
        document_work = self.resolve_document(bundle.document)
        citing_chunks_by_reference: dict[str, list[str]] = {}
        for chunk in chunks:
            for reference_id in chunk.citation_reference_entry_ids:
                citing_chunks_by_reference.setdefault(reference_id, []).append(
                    chunk.id
                )

        resolutions = [
            self.resolve_reference(
                reference,
                source_chunk_ids=citing_chunks_by_reference.get(reference.id, []),
            )
            for reference in bundle.references
        ]
        works_by_id = {document_work.id: document_work}
        for resolution in resolutions:
            if resolution.work_id is not None:
                work = self.get_work(resolution.work_id)
                resolution.work_id = work.id
                works_by_id[work.id] = work
        document_work = self.get_work(document_work.id)
        works_by_id[document_work.id] = document_work
        return ScholarlyContext(
            document_work=document_work,
            works=sorted(works_by_id.values(), key=lambda work: work.id),
            reference_resolutions=resolutions,
        )

    def backfill(
        self, repository: CanonicalRepository
    ) -> list[ScholarlyContext]:
        contexts: list[ScholarlyContext] = []
        for bundle in sorted(
            repository.list_active_bundles(),
            key=lambda item: (item.snapshot.created_at, item.snapshot.id),
        ):
            projection = repository.get_chunk_projection(bundle.snapshot.id)
            contexts.append(self.resolve_bundle(bundle, projection.chunks))
        return contexts

    def merge(self, first_work_id: str, second_work_id: str) -> ScholarlyWork:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            survivor = self._merge(connection, first_work_id, second_work_id)
            self._reconcile_unresolved(connection)
            return self._get_work(connection, survivor)

    def identity_snapshot(self) -> dict[str, Any]:
        with self._connect() as connection:
            works = [
                self._row_to_work(row).model_dump(mode="json")
                for row in connection.execute(
                    "SELECT * FROM scholarly_works "
                    "WHERE id NOT IN (SELECT loser_work_id FROM work_redirects) "
                    "ORDER BY id"
                ).fetchall()
            ]
            document_links = [
                {
                    "document_id": str(row["document_id"]),
                    "work_id": self.canonicalize_work_id(
                        str(row["work_id"]), connection
                    ),
                }
                for row in connection.execute(
                    "SELECT document_id, work_id FROM document_work_links "
                    "ORDER BY document_id"
                ).fetchall()
            ]
            reference_links = [
                {
                    "reference_id": str(row["reference_id"]),
                    "work_id": (
                        self.canonicalize_work_id(
                            str(row["work_id"]), connection
                        )
                        if row["work_id"] is not None
                        else None
                    ),
                    "resolution_status": str(row["resolution_status"]),
                }
                for row in connection.execute(
                    "SELECT reference_id, work_id, resolution_status "
                    "FROM reference_work_links ORDER BY reference_id"
                ).fetchall()
            ]
        return {
            "works": works,
            "document_links": document_links,
            "reference_links": reference_links,
            "redirects": self.list_redirects(),
        }

    def _get_work(
        self, connection: sqlite3.Connection, work_id: str
    ) -> ScholarlyWork:
        row = connection.execute(
            "SELECT * FROM scholarly_works WHERE id = ?", (work_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Unknown ScholarlyWork: {work_id}")
        return self._row_to_work(row)

    @staticmethod
    def _row_to_work(row: sqlite3.Row) -> ScholarlyWork:
        return ScholarlyWork(
            id=str(row["id"]),
            title=str(row["title"]),
            normalized_title=str(row["normalized_title"]),
            doi=str(row["doi"]) if row["doi"] is not None else None,
            arxiv_id=(
                str(row["arxiv_id"]) if row["arxiv_id"] is not None else None
            ),
            year=int(row["year"]) if row["year"] is not None else None,
            authors=list(json.loads(str(row["authors"]))),
            normalized_first_author=(
                str(row["normalized_first_author"])
                if row["normalized_first_author"] is not None
                else None
            ),
            identity_status=WorkIdentityStatus(str(row["identity_status"])),
            identity_confidence=float(row["identity_confidence"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
            updated_at=datetime.fromisoformat(str(row["updated_at"])),
            schema_version=str(row["schema_version"]),
            id_version=str(row["id_version"]),
        )

    def _find_identifier(
        self,
        connection: sqlite3.Connection,
        kind: WorkIdentifierKind,
        normalized_value: str | None,
    ) -> str | None:
        if normalized_value is None:
            return None
        row = connection.execute(
            "SELECT work_id FROM work_identifiers "
            "WHERE kind = ? AND normalized_value = ? "
            "ORDER BY created_at, work_id LIMIT 1",
            (kind.value, normalized_value),
        ).fetchone()
        if row is None:
            return None
        return self.canonicalize_work_id(str(row["work_id"]), connection)

    def _find_title_candidates(
        self,
        connection: sqlite3.Connection,
        normalized_title: str,
        year: int | None,
        first_author: str | None,
        *,
        incoming_doi: str | None = None,
        incoming_arxiv: str | None = None,
    ) -> list[str]:
        if not normalized_title:
            return []
        if year is None:
            rows = connection.execute(
                """
                SELECT id, doi, arxiv_id, normalized_title,
                       normalized_first_author, year
                FROM scholarly_works
                WHERE id NOT IN (SELECT loser_work_id FROM work_redirects)
                ORDER BY created_at, id
                """
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id, doi, arxiv_id, normalized_title,
                       normalized_first_author, year
                FROM scholarly_works
                WHERE (year = ? OR year IS NULL)
                  AND id NOT IN (SELECT loser_work_id FROM work_redirects)
                ORDER BY created_at, id
                """,
                (year,),
            ).fetchall()
        compatible: list[str] = []
        for row in rows:
            candidate_doi = self.normalize_doi(row["doi"])
            candidate_arxiv = self.normalize_arxiv(row["arxiv_id"])
            if incoming_doi and candidate_doi and incoming_doi != candidate_doi:
                continue
            if (
                incoming_arxiv
                and candidate_arxiv
                and incoming_arxiv != candidate_arxiv
            ):
                continue
            candidate_title = str(row["normalized_title"])
            candidate_author = (
                str(row["normalized_first_author"])
                if row["normalized_first_author"] is not None
                else None
            )
            candidate_year = (
                int(row["year"]) if row["year"] is not None else None
            )
            if year is not None and candidate_year is not None and year != candidate_year:
                continue
            author_available = bool(first_author and candidate_author)
            if author_available and not self._authors_compatible(
                first_author,
                candidate_author,
            ):
                continue
            similarity = self._title_similarity(
                normalized_title,
                candidate_title,
            )
            if similarity < 1.0:
                same_year = year is not None and year == candidate_year
                if not same_year:
                    continue
                threshold = 0.86 if author_available else 0.90
                if similarity < threshold:
                    continue
            compatible.append(str(row["id"]))
        return compatible

    def _create_work(
        self,
        connection: sqlite3.Connection,
        *,
        title: str,
        normalized_title: str,
        doi: str | None,
        arxiv_id: str | None,
        year: int | None,
        authors: list[str],
        first_author: str | None,
        status: WorkIdentityStatus,
        confidence: float,
    ) -> str:
        work_id = scholarly_work_id()
        now = utc_now().isoformat()
        connection.execute(
            """
            INSERT INTO scholarly_works(
                id, title, normalized_title, doi, arxiv_id, year, authors,
                normalized_first_author, identity_status, identity_confidence,
                created_at, updated_at, schema_version, id_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                work_id,
                title,
                normalized_title,
                doi,
                arxiv_id,
                year,
                json.dumps(authors, ensure_ascii=False),
                first_author,
                status.value,
                confidence,
                now,
                now,
                SCHOLARLY_WORK_SCHEMA_VERSION,
                SCHOLARLY_WORK_ID_VERSION,
            ),
        )
        return work_id

    def _attach_identifiers(
        self,
        connection: sqlite3.Connection,
        work_id: str,
        *,
        doi: str | None,
        arxiv_id: str | None,
        title: str,
    ) -> str:
        for kind, normalized, raw in (
            (WorkIdentifierKind.DOI, self.normalize_doi(doi), doi),
            (WorkIdentifierKind.ARXIV, self.normalize_arxiv(arxiv_id), arxiv_id),
            (
                WorkIdentifierKind.TITLE,
                self.normalize_text(title) or None,
                title or None,
            ),
        ):
            if normalized is None or raw is None:
                continue
            if kind is not WorkIdentifierKind.TITLE:
                existing = self._find_identifier(connection, kind, normalized)
                if existing is not None and existing != work_id:
                    work_id = self._merge(connection, work_id, existing)
            connection.execute(
                """
                INSERT OR IGNORE INTO work_identifiers(
                    work_id, kind, normalized_value, raw_value, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    work_id,
                    kind.value,
                    normalized,
                    raw,
                    utc_now().isoformat(),
                ),
            )
        return self.canonicalize_work_id(work_id, connection)

    def _promote(
        self,
        connection: sqlite3.Connection,
        work_id: str,
        *,
        title: str,
        normalized_title: str,
        doi: str | None,
        arxiv_id: str | None,
        year: int | None,
        authors: list[str],
        first_author: str | None,
        status: WorkIdentityStatus,
        confidence: float,
    ) -> str:
        work_id = self.canonicalize_work_id(work_id, connection)
        current = self._get_work(connection, work_id)
        promoted_status = (
            status
            if _STATUS_PRIORITY[status] >= _STATUS_PRIORITY[current.identity_status]
            else current.identity_status
        )
        use_new = status is WorkIdentityStatus.INGESTED
        connection.execute(
            """
            UPDATE scholarly_works SET
                title = ?, normalized_title = ?, doi = ?, arxiv_id = ?,
                year = ?, authors = ?, normalized_first_author = ?,
                identity_status = ?, identity_confidence = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                title if use_new or not current.title else current.title,
                (
                    normalized_title
                    if use_new or not current.normalized_title
                    else current.normalized_title
                ),
                doi or current.doi,
                arxiv_id or current.arxiv_id,
                year if year is not None else current.year,
                json.dumps(
                    authors if use_new or not current.authors else current.authors,
                    ensure_ascii=False,
                ),
                first_author or current.normalized_first_author,
                promoted_status.value,
                max(confidence, current.identity_confidence),
                utc_now().isoformat(),
                work_id,
            ),
        )
        return work_id

    def _merge(
        self,
        connection: sqlite3.Connection,
        first_work_id: str,
        second_work_id: str,
    ) -> str:
        first_id = self.canonicalize_work_id(first_work_id, connection)
        second_id = self.canonicalize_work_id(second_work_id, connection)
        if first_id == second_id:
            return first_id
        first = self._get_work(connection, first_id)
        second = self._get_work(connection, second_id)
        first_key = (
            _STATUS_PRIORITY[first.identity_status],
            -first.created_at.timestamp(),
            first.id,
        )
        second_key = (
            _STATUS_PRIORITY[second.identity_status],
            -second.created_at.timestamp(),
            second.id,
        )
        survivor, loser = (
            (first, second) if first_key >= second_key else (second, first)
        )

        identifiers = connection.execute(
            "SELECT kind, normalized_value, raw_value, created_at "
            "FROM work_identifiers WHERE work_id = ?",
            (loser.id,),
        ).fetchall()
        connection.execute(
            "DELETE FROM work_identifiers WHERE work_id = ?", (loser.id,)
        )
        for row in identifiers:
            connection.execute(
                """
                INSERT OR IGNORE INTO work_identifiers(
                    work_id, kind, normalized_value, raw_value, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    survivor.id,
                    row["kind"],
                    row["normalized_value"],
                    row["raw_value"],
                    row["created_at"],
                ),
            )
        connection.execute(
            "UPDATE document_work_links SET work_id = ?, updated_at = ? "
            "WHERE work_id = ?",
            (survivor.id, utc_now().isoformat(), loser.id),
        )
        connection.execute(
            "UPDATE reference_work_links SET work_id = ?, updated_at = ? "
            "WHERE work_id = ?",
            (survivor.id, utc_now().isoformat(), loser.id),
        )
        connection.execute(
            "UPDATE work_redirects SET survivor_work_id = ? "
            "WHERE survivor_work_id = ?",
            (survivor.id, loser.id),
        )
        connection.execute(
            "INSERT OR REPLACE INTO work_redirects("
            "loser_work_id, survivor_work_id, created_at"
            ") VALUES (?, ?, ?)",
            (loser.id, survivor.id, utc_now().isoformat()),
        )
        self._promote(
            connection,
            survivor.id,
            title=survivor.title,
            normalized_title=survivor.normalized_title,
            doi=survivor.doi or loser.doi,
            arxiv_id=survivor.arxiv_id or loser.arxiv_id,
            year=survivor.year if survivor.year is not None else loser.year,
            authors=survivor.authors or loser.authors,
            first_author=(
                survivor.normalized_first_author
                or loser.normalized_first_author
            ),
            status=survivor.identity_status,
            confidence=max(
                survivor.identity_confidence, loser.identity_confidence
            ),
        )
        return survivor.id

    def _reconcile_unresolved(self, connection: sqlite3.Connection) -> None:
        rows = connection.execute(
            """
            SELECT * FROM reference_work_links
            WHERE work_id IS NULL
              AND resolution_status IN ('unresolved', 'ambiguous')
            ORDER BY reference_id
            """
        ).fetchall()
        now = utc_now().isoformat()
        for row in rows:
            work_id = self._find_identifier(
                connection, WorkIdentifierKind.DOI, row["normalized_doi"]
            )
            confidence = 1.0
            if work_id is None:
                work_id = self._find_identifier(
                    connection,
                    WorkIdentifierKind.ARXIV,
                    row["normalized_arxiv"],
                )
                confidence = 0.99
            if work_id is None:
                candidates = self._find_title_candidates(
                    connection,
                    str(row["normalized_title"] or ""),
                    int(row["year"]) if row["year"] is not None else None,
                    (
                        str(row["normalized_first_author"])
                        if row["normalized_first_author"] is not None
                        else None
                    ),
                    incoming_doi=(
                        str(row["normalized_doi"])
                        if row["normalized_doi"] is not None
                        else None
                    ),
                    incoming_arxiv=(
                        str(row["normalized_arxiv"])
                        if row["normalized_arxiv"] is not None
                        else None
                    ),
                )
                if len(candidates) == 1:
                    work_id = candidates[0]
                    confidence = 0.92
                elif len(candidates) > 1:
                    connection.execute(
                        "UPDATE reference_work_links SET "
                        "resolution_status = 'ambiguous', updated_at = ? "
                        "WHERE reference_id = ?",
                        (now, row["reference_id"]),
                    )
                    continue
            if work_id is not None:
                connection.execute(
                    "UPDATE reference_work_links SET work_id = ?, "
                    "resolution_status = 'resolved', confidence = ?, "
                    "updated_at = ? WHERE reference_id = ?",
                    (work_id, confidence, now, row["reference_id"]),
                )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
