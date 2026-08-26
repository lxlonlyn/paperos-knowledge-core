"""Citation mention extraction with bibliography-first label and author-year resolution."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TypeVar

from paperos_core.domain.canonical import Chunk, CitationMention, Element, ReferenceEntry, Section
from paperos_core.domain.ids import citation_mention_id, citation_span_id
from paperos_core.ingestion.bibliography_scope import (
    FAILURE_NAMESPACE_NOT_ASSIGNED,
    REGION_MAIN,
    BibliographyScope,
    ScopedBibliography,
    assign_bibliography_scopes,
    repair_numeric_label_sequence,
)
from paperos_core.ingestion.citation_candidates import detect_citation_candidates
from paperos_core.ingestion.normalization import plain_text

REF_LABEL_RE = re.compile(
    r"""
    ^\s*
    (?:
        \[(?P<bracket>[^\[\]\s]{1,40})\]
        |
        \((?P<paren>[^()\s]{1,40})\)
        |
        (?P<number>\d{1,4})[.)]
    )
    \s*
    """,
    re.VERBOSE,
)
BRACKET_RE = re.compile(r"\[(?P<body>[^\[\]\n]{1,400})\]")
NUMERIC_RANGE_RE = re.compile(r"^\s*(\d+)\s*[-–−—]\s*(\d+)\s*$")
_NUMERIC_LABEL = re.compile(r"^\d+$")
_YEAR_RE = re.compile(r"(?<!\d)([12]\d{3})([a-d]?)(?!\d)")
_MAX_NUMERIC_RANGE = 20

_AUTHOR_YEAR_PAREN_RE = re.compile(
    r"""
    \(
    (?P<author>[A-ZÀ-ÖØ-Þ][\w''\-]+(?:\s+(?:et\s+al\.?|and|&)\s*[\w''\-]+)*)
    \s*,\s*
    (?P<year>[12]\d{3}[a-d]?)
    \)
    """,
    re.VERBOSE,
)
_AUTHOR_YEAR_INLINE_RE = re.compile(
    r"""
    (?<![A-Za-z])
    (?P<author>[A-ZÀ-ÖØ-Þ][\w''\-]+(?:\s+et\s+al\.?)?(?:\s+and\s+[A-ZÀ-ÖØ-Þ][\w''\-]+)?)
    \s+
    \(
    (?P<year>[12]\d{3}[a-d]?)
    \)
    """,
    re.VERBOSE,
)
_BRACKET_AUTHOR_YEAR_RE = re.compile(
    r"^(?P<author>.+?)\s+(?P<year>[12]\d{3}[a-d]?)\s*$"
)
_SECTION_REF_RE = re.compile(
    r"\b(?:sec(?:tion)?|cor(?:ollary)?|thm|theorem|fig|figure|eq(?:uation)?)\b",
    re.IGNORECASE,
)
_YEAR_ONLY_BRACKET_RE = re.compile(r"^[12]\d{3}[a-d]?$")
_LEFT_AUTHOR_RE = re.compile(
    r"(?P<author>[A-ZÀ-ÖØ-Þ][\w''\-]+(?:\s+(?:et\s+al\.?|and|&)\s*[\w''\-]+)*)\s*$"
)
_VENUE_FALSE_POSITIVE_RE = re.compile(
    r"\b(?:processing|symposium|conference|transactions|journal|review|press)\s*\(",
    re.IGNORECASE,
)

FAILURE_MISSING_REFERENCE_ENTRY = "MISSING_REFERENCE_ENTRY"
FAILURE_AMBIGUOUS_LABEL = "AMBIGUOUS_LABEL"
FAILURE_AMBIGUOUS_AUTHOR_YEAR = "AMBIGUOUS_AUTHOR_YEAR"
FAILURE_SCOPE_NOT_FOUND = "SCOPE_NOT_FOUND"
FAILURE_UNPARSEABLE = "UNPARSEABLE"

_KeyT = TypeVar("_KeyT")

@dataclass
class ReferenceIndexes:
    label_map: dict[str, list[ReferenceEntry]] = field(default_factory=dict)
    author_year_map: dict[tuple[str, str], list[ReferenceEntry]] = field(
        default_factory=dict
    )
    references: list[ReferenceEntry] = field(default_factory=list)
    scope_id: str | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class AtomicResolution:
    atomic_key: str
    reference: ReferenceEntry | None
    resolution_status: str
    failure_reason: str | None
    resolution_kind: str | None
    bibliography_scope_id: str | None = None


def normalize_label(label: str) -> str:
    value = label.strip()
    if len(value) >= 2 and (
        (value[0] == "[" and value[-1] == "]")
        or (value[0] == "(" and value[-1] == ")")
    ):
        value = value[1:-1].strip()
    return value


def normalize_citation_label(label: str) -> str:
    value = normalize_label(label)
    value = value.replace("∗", "*").replace("⋆", "*").replace("\\*", "*")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"\s+", "", value)
    return value.casefold()


def extract_reference_label(raw_text: str) -> tuple[str | None, str | None]:
    """Return (label, label_kind) from bibliography leading marker."""
    normalized = plain_text(raw_text)
    match = REF_LABEL_RE.match(normalized)
    if not match:
        return None, None
    label = normalize_label(
        match.group("bracket") or match.group("paren") or match.group("number") or ""
    )
    if not label:
        return None, None
    if _NUMERIC_LABEL.fullmatch(label):
        return label, "numeric"
    return label, "symbolic"


def extract_citation_label(raw_text: str) -> tuple[str | None, str | None]:
    """Backward-compatible alias for bibliography label extraction."""
    return extract_reference_label(raw_text)


def build_reference_indexes(references: list[ReferenceEntry]) -> ReferenceIndexes:
    scoped = build_scoped_reference_indexes(
        references=references,
        elements=[],
        sections=[],
    )
    if "default" in scoped.scope_indexes:
        return scoped.scope_indexes["default"]
    return next(iter(scoped.scope_indexes.values()))


def build_scoped_reference_indexes(
    *,
    references: list[ReferenceEntry],
    elements: list[Element],
    sections: list[Section],
) -> ScopedBibliography[ReferenceIndexes]:
    reference_scope, assigned_scopes = assign_bibliography_scopes(
        references=references,
        elements=elements,
        sections=sections,
    )
    scope_ids = sorted(set(reference_scope.values()))
    if not scope_ids:
        scope_ids = ["default"]
    repaired = list(references)
    for scope_id in scope_ids:
        repaired = repair_numeric_label_sequence(
            repaired,
            scope_id=scope_id,
            reference_scope=reference_scope,
        )
    references_by_id = {reference.id: reference for reference in repaired}
    for reference in repaired:
        scope_id = reference_scope.get(reference.id, "default")
        reference = reference.model_copy(
            update={
                "bibliography_scope_id": scope_id,
                "citation_namespace_id": scope_id,
            }
        )
        references_by_id[reference.id] = reference
    repaired = [references_by_id[reference.id] for reference in repaired]

    scope_indexes: dict[str, ReferenceIndexes] = {}
    scopes: dict[str, BibliographyScope] = {}
    grouped: dict[str, list[ReferenceEntry]] = {scope_id: [] for scope_id in scope_ids}
    for reference in repaired:
        scope_id = reference.bibliography_scope_id or reference_scope.get(
            reference.id, "default"
        )
        grouped.setdefault(scope_id, []).append(reference)

    for scope_id, scoped_refs in grouped.items():
        scope_indexes[scope_id] = _indexes_for_references(scoped_refs, scope_id=scope_id)
        existing = assigned_scopes.get(scope_id)
        scopes[scope_id] = BibliographyScope(
            namespace_id=scope_id,
            bibliography_region_id=(
                existing.bibliography_region_id if existing else None
            ),
            parent_region=existing.parent_region if existing else REGION_MAIN,
            owner_body_region_id=existing.owner_body_region_id if existing else None,
            reference_ids=[reference.id for reference in scoped_refs],
        )

    if "default" not in scope_indexes:
        scope_indexes["default"] = _indexes_for_references(repaired, scope_id="default")

    return ScopedBibliography(
        scopes=scopes,
        reference_scope=reference_scope,
        scope_indexes=scope_indexes,
    )


def build_reference_label_index(
    references: list[ReferenceEntry],
) -> dict[str, ReferenceEntry]:
    """Legacy single-reference label index (first wins on duplicates)."""
    index: dict[str, ReferenceEntry] = {}
    for key, entries in build_reference_indexes(references).label_map.items():
        if key not in index:
            index[key] = entries[0]
    return index


def expand_atom(atom: str) -> list[str] | None:
    value = normalize_label(atom.strip())
    if not value:
        return None
    match = NUMERIC_RANGE_RE.match(value)
    if not match:
        return [value]
    start = int(match.group(1))
    end = int(match.group(2))
    if end <= start or end - start > _MAX_NUMERIC_RANGE:
        return None
    return [str(item) for item in range(start, end + 1)]


def parse_bracket_atoms(inner: str) -> list[str] | None:
    inner = _normalize_bracket_inner(inner).strip()
    if not inner:
        return None
    if _looks_like_author_year_bracket(inner):
        return None
    raw_parts = re.split(r"\s*[,;]\s*", inner)
    if len(raw_parts) > 1:
        keys: list[str] = []
        for part in raw_parts:
            expanded = expand_atom(part.strip())
            if not expanded:
                return None
            keys.extend(expanded)
        return keys or None
    expanded_single = expand_atom(inner)
    if expanded_single:
        return expanded_single
    return [inner]


def _looks_like_author_year_bracket(inner: str) -> bool:
    if not _YEAR_RE.search(inner):
        return False
    compact = re.sub(r"\s+", "", inner)
    return not re.fullmatch(r"[\d,;\-–−—]+", compact)


def resolve_bracket(
    inner: str,
    indexes: ReferenceIndexes,
    *,
    left_context: str | None = None,
) -> list[AtomicResolution] | None:
    inner = _normalize_bracket_inner(inner).strip()
    if not inner:
        return None
    if _looks_like_author_year_bracket(inner):
        author_year = _resolve_bracket_author_year(inner, indexes, left_context=left_context)
        return author_year or None
    atoms = parse_bracket_atoms(inner)
    if atoms is None:
        ocr_atoms = _split_ocred_symbolic_group(inner)
        if ocr_atoms:
            return _resolve_atomic_keys(ocr_atoms, indexes)
        return None
    if len(atoms) == 1 and atoms[0] not in indexes.label_map:
        ocr_atoms = _split_ocred_symbolic_group(atoms[0])
        if ocr_atoms:
            atoms = ocr_atoms
    if len(atoms) == 1 and normalize_citation_label(atoms[0]) in indexes.label_map:
        return _resolve_atomic_keys(atoms, indexes)
    if len(atoms) == 1:
        author_year = _resolve_bracket_author_year(inner, indexes, left_context=left_context)
        if author_year:
            return author_year
    return _resolve_atomic_keys(atoms, indexes)


def extract_citation_mentions_from_text(
    *,
    document_id: str,
    snapshot_id: str,
    element_id: str,
    text: str,
    reference_index: (
        dict[str, ReferenceEntry]
        | ReferenceIndexes
        | ScopedBibliography[ReferenceIndexes]
    ),
    document_region: str | None = None,
    bibliography_scope_ids: list[str] | None = None,
    bibliography_scope_id: str | None = None,
    citation_namespace_id: str | None = None,
    region_instance_id: str | None = None,
) -> list[CitationMention]:
    scoped, indexes = _resolve_reference_context(
        reference_index,
        document_region=document_region,
        bibliography_scope_ids=bibliography_scope_ids,
        bibliography_scope_id=citation_namespace_id or bibliography_scope_id,
    )
    mentions: list[CitationMention] = []
    for candidate in detect_citation_candidates(text):
        if candidate.kind == "bracket":
            inner = candidate.metadata.get("inner", "")
            bracket_start = candidate.bracket_start or candidate.start
            if _is_negative_bracket_domain(inner, text, bracket_start):
                continue
            if "author" in candidate.metadata:
                inner = f"{candidate.metadata['author']} {candidate.metadata['year']}"
            _emit_bracket_mentions(
                inner=inner,
                surface=candidate.surface,
                indexes=indexes,
                document_id=document_id,
                snapshot_id=snapshot_id,
                element_id=element_id,
                start=candidate.start,
                end=candidate.end,
                document_region=document_region,
                region_instance_id=region_instance_id,
                left_context=text[max(0, candidate.start - 240) : candidate.start],
                mentions=mentions,
            )
            continue
        if _is_venue_false_positive(text, candidate.start, candidate.surface):
            continue
        mentions.extend(
            _author_year_mentions(
                author=candidate.metadata["author"],
                year=candidate.metadata["year"],
                indexes=indexes,
                document_id=document_id,
                snapshot_id=snapshot_id,
                element_id=element_id,
                surface=candidate.surface,
                start=candidate.start,
                end=candidate.end,
                match_kind=candidate.kind,
                document_region=document_region,
            )
        )

    _ = scoped
    return mentions


def _emit_bracket_mentions(
    *,
    inner: str,
    surface: str,
    indexes: ReferenceIndexes,
    document_id: str,
    snapshot_id: str,
    element_id: str,
    start: int,
    end: int,
    document_region: str | None,
    region_instance_id: str | None,
    left_context: str | None = None,
    mentions: list[CitationMention],
) -> bool:
    if (
        not _looks_like_author_year_bracket(inner)
        and not _looks_like_citation_bracket(inner)
        and _split_ocred_symbolic_group(inner) is None
    ):
        return False
    resolved = resolve_bracket(inner, indexes, left_context=left_context)
    if resolved:
        if not _should_emit_bracket_citation(inner, resolved, indexes):
            return False
        mentions.extend(
            _mentions_from_atomic_resolutions(
                resolutions=resolved,
                document_id=document_id,
                snapshot_id=snapshot_id,
                element_id=element_id,
                surface=surface,
                start=start,
                end=end,
                match_kind="bracket",
                document_region=document_region,
            )
        )
        return True
    if _looks_like_author_year_bracket(inner):
        author_resolved = _resolve_bracket_author_year(inner, indexes, left_context=left_context)
        if author_resolved:
            mentions.extend(
                _mentions_from_atomic_resolutions(
                    resolutions=author_resolved,
                    document_id=document_id,
                    snapshot_id=snapshot_id,
                    element_id=element_id,
                    surface=surface,
                    start=start,
                    end=end,
                    match_kind="bracket",
                    document_region=document_region,
                )
            )
            return True
        mentions.extend(
            _unresolved_author_year_bracket_mentions(
                inner=inner,
                document_id=document_id,
                snapshot_id=snapshot_id,
                element_id=element_id,
                surface=surface,
                start=start,
                end=end,
                document_region=document_region,
            )
        )
        return True
    if not _looks_like_citation_bracket(inner):
        return False
    atoms = parse_bracket_atoms(inner) or _split_ocred_symbolic_group(inner)
    if not atoms:
        atoms = [inner]
    resolutions = _resolve_atomic_keys(atoms, indexes)
    mentions.extend(
        _mentions_from_atomic_resolutions(
            resolutions=resolutions,
            document_id=document_id,
            snapshot_id=snapshot_id,
            element_id=element_id,
            surface=surface,
            start=start,
            end=end,
            match_kind="bracket",
            document_region=document_region,
        )
    )
    return True


def _should_emit_bracket_citation(
    inner: str,
    resolutions: list[AtomicResolution],
    indexes: ReferenceIndexes,
) -> bool:
    if _looks_like_author_year_bracket(inner) or _looks_like_citation_bracket(inner):
        return True
    if any(item.resolution_status == "resolved" for item in resolutions):
        return True
    return bool(any(normalize_citation_label(item.atomic_key) in indexes.label_map for item in resolutions))


def _author_year_mentions(
    *,
    author: str,
    year: str,
    indexes: ReferenceIndexes,
    document_id: str,
    snapshot_id: str,
    element_id: str,
    surface: str,
    start: int,
    end: int,
    match_kind: str,
    document_region: str | None,
) -> list[CitationMention]:
    resolved = _resolve_author_year_text(author, year, indexes)
    return _mentions_from_atomic_resolutions(
        resolutions=resolved,
        document_id=document_id,
        snapshot_id=snapshot_id,
        element_id=element_id,
        surface=surface,
        start=start,
        end=end,
        match_kind=match_kind,
        document_region=document_region,
    )


def attach_mentions_to_chunks(
    mentions: list[CitationMention],
    *,
    chunks: list[Chunk],
) -> list[CitationMention]:
    attached: list[CitationMention] = []
    for mention in mentions:
        chunk_id, diagnostic = _owning_chunk_for_mention(mention, chunks)
        metadata = dict(mention.metadata)
        if diagnostic:
            metadata["chunk_attachment_diagnostic"] = diagnostic
        attached.append(
            mention.model_copy(update={"chunk_id": chunk_id, "metadata": metadata})
        )
    return attached


def _resolve_reference_context(
    reference_index: (
        dict[str, ReferenceEntry]
        | ReferenceIndexes
        | ScopedBibliography[ReferenceIndexes]
    ),
    *,
    document_region: str | None,
    bibliography_scope_ids: list[str] | None,
    bibliography_scope_id: str | None = None,
) -> tuple[
    ScopedBibliography[ReferenceIndexes] | None,
    ReferenceIndexes,
]:
    if isinstance(reference_index, ScopedBibliography):
        scoped = reference_index
        scope_id = bibliography_scope_id
        if scope_id and scope_id in scoped.scope_indexes:
            return scoped, scoped.scope_indexes[scope_id]
        _ = (document_region, bibliography_scope_ids)
        return scoped, ReferenceIndexes(
            scope_id=scope_id,
            failure_reason=FAILURE_NAMESPACE_NOT_ASSIGNED,
        )
    if isinstance(reference_index, ReferenceIndexes):
        return None, reference_index
    return None, _legacy_index_to_indexes(reference_index)


def _owning_chunk_for_mention(
    mention: CitationMention,
    chunks: list[Chunk],
) -> tuple[str | None, str | None]:
    candidates: list[tuple[str, int]] = []
    for chunk in chunks:
        for span in chunk.spans:
            if span.element_id != mention.element_id:
                continue
            if (
                span.character_start_in_element <= mention.character_start
                and span.character_end_in_element >= mention.character_end
            ):
                span_size = (
                    span.character_end_in_element - span.character_start_in_element
                )
                candidates.append((chunk.id, span_size))
    if not candidates:
        return None, "no_containing_chunk_span"
    if len(candidates) == 1:
        return candidates[0][0], None
    candidates.sort(key=lambda item: item[1])
    return candidates[0][0], "multiple_containing_chunks_chose_smallest_span"


def _indexes_for_references(
    references: list[ReferenceEntry],
    *,
    scope_id: str,
) -> ReferenceIndexes:
    label_map: dict[str, list[ReferenceEntry]] = {}
    author_year_map: dict[tuple[str, str], list[ReferenceEntry]] = {}
    for reference in references:
        label = reference.citation_label or extract_reference_label(reference.raw_text)[0]
        if label:
            key = normalize_citation_label(label)
            _append_unique(label_map, key, reference)
        ref_num = reference.parsed_fields.get("reference_number")
        if ref_num is not None:
            num_key = normalize_citation_label(str(ref_num))
            if num_key != normalize_citation_label(label or ""):
                _append_unique(label_map, num_key, reference)
        for author_key, year_key in _bibliography_author_year_keys(reference):
            _append_unique(author_year_map, (author_key, year_key), reference)
    return ReferenceIndexes(
        label_map=label_map,
        author_year_map=author_year_map,
        references=list(references),
        scope_id=scope_id,
    )


def _normalize_bracket_inner(inner: str) -> str:
    collapsed = re.sub(r"(?<=\d)\s+(?=\d)", "", inner)
    return collapsed


def _resolve_atomic_keys(
    keys: list[str],
    indexes: ReferenceIndexes,
) -> list[AtomicResolution]:
    resolved: list[AtomicResolution] = []
    for key in keys:
        lookup_key = normalize_citation_label(key)
        entries = indexes.label_map.get(lookup_key)
        reference, failure = _resolve_unique(entries or [], key)
        if reference is None:
            author_match = _BRACKET_AUTHOR_YEAR_RE.match(key.strip())
            if author_match:
                author_resolved = _resolve_author_year_text(
                    author_match.group("author"),
                    author_match.group("year"),
                    indexes,
                )
                if author_resolved:
                    resolved.extend(author_resolved)
                    continue
            resolved.append(
                AtomicResolution(
                    atomic_key=key,
                    reference=None,
                    resolution_status="unresolved",
                    failure_reason=(
                        indexes.failure_reason
                        or failure
                        or FAILURE_MISSING_REFERENCE_ENTRY
                    ),
                    resolution_kind=None,
                    bibliography_scope_id=indexes.scope_id,
                )
            )
            continue
        resolved.append(
            AtomicResolution(
                atomic_key=key,
                reference=reference,
                resolution_status="resolved",
                failure_reason=None,
                resolution_kind="label",
                bibliography_scope_id=indexes.scope_id,
            )
        )
    return resolved


def _resolve_bracket_author_year(
    inner: str,
    indexes: ReferenceIndexes,
    *,
    left_context: str | None = None,
) -> list[AtomicResolution] | None:
    parts = [part.strip() for part in inner.split(";") if part.strip()]
    if not parts:
        return None
    resolved: list[AtomicResolution] = []
    for part in parts:
        for atom in _expand_author_year_part(part):
            match = _BRACKET_AUTHOR_YEAR_RE.match(atom.strip())
            if not match:
                resolved.append(
                    AtomicResolution(
                        atomic_key=atom,
                        reference=None,
                        resolution_status="unresolved",
                        failure_reason=FAILURE_UNPARSEABLE,
                        resolution_kind=None,
                        bibliography_scope_id=indexes.scope_id,
                    )
                )
                continue
            atom_resolved = _resolve_author_year_text(
                match.group("author"),
                match.group("year"),
                indexes,
                left_context=left_context,
            )
            resolved.extend(atom_resolved)
    return resolved or None


_BRACKET_AUTHOR_YEAR_PART_RE = re.compile(
    r"^(?P<author>.+?)\s+(?P<year>[12]\d{3}[a-d]?)(?:\s*,\s*(?P<tail>.+))?\s*$",
    re.IGNORECASE,
)


def _expand_author_year_part(part: str) -> list[str]:
    part = part.strip()
    match = _BRACKET_AUTHOR_YEAR_PART_RE.match(part)
    if not match:
        return [part]
    author = match.group("author").strip()
    year = match.group("year")
    tail = (match.group("tail") or "").strip()
    if not tail:
        return [part]
    base_year = re.sub(r"[a-z]$", "", year, flags=re.IGNORECASE)
    if _YEAR_RE.fullmatch(tail):
        return [f"{author} {year}", f"{author} {tail}"]
    if re.fullmatch(r"[a-z]", tail, flags=re.IGNORECASE):
        return [f"{author} {year}", f"{author} {base_year}{tail}"]
    return [part]


def _author_key_candidates(author: str) -> list[str]:
    base = _normalize_author_key(author)
    candidates = [base]
    particle = re.match(r"^(van|von|de)\s+(\S+)(.*)$", base)
    if particle:
        shortened = f"{particle.group(2)}{particle.group(3)}".strip()
        if shortened:
            candidates.append(shortened)
    return list(dict.fromkeys(candidates))


def _lookup_author_year_entries(
    author: str,
    year: str,
    indexes: ReferenceIndexes,
) -> list[ReferenceEntry]:
    year_key = year.casefold()
    for candidate in _author_key_candidates(author):
        entries = indexes.author_year_map.get((candidate, year_key))
        if entries:
            return entries
    normalized = _normalize_author_key(author)
    first_author, second_author = _citation_author_constraints(author)
    matched: list[ReferenceEntry] = []
    seen: set[str] = set()
    for (key, map_year), entries in indexes.author_year_map.items():
        if map_year != year_key:
            continue
        if second_author is not None:
            pair_key = f"{first_author} and {second_author}"
            if not (
                _fuzzy_surname_match(first_author, key)
                or _fuzzy_surname_match(pair_key, key)
                or _fuzzy_surname_match(normalized, key)
            ):
                continue
        elif not (
            _fuzzy_surname_match(normalized, key)
            or key in normalized
            or normalized in key
        ):
            continue
        for entry in entries:
            if entry.id not in seen:
                seen.add(entry.id)
                matched.append(entry)
    return matched


def _disambiguate_by_context(
    entries: list[ReferenceEntry],
    left_context: str | None,
) -> ReferenceEntry | None:
    if len(entries) <= 1:
        return entries[0] if entries else None
    if not left_context:
        return None
    context = _normalize_author_key(left_context)
    context_tokens = {token for token in re.split(r"\W+", context) if len(token) >= 4}
    best_score = 0
    best_entry: ReferenceEntry | None = None
    for entry in entries:
        raw = plain_text(entry.raw_text)
        year_match = _YEAR_RE.search(raw)
        title_text = raw[year_match.end() :].strip() if year_match else raw
        title_tokens = {
            token
            for token in re.split(r"\W+", _normalize_author_key(title_text))
            if len(token) >= 4
        }
        score = len(context_tokens & title_tokens)
        if score > best_score:
            best_score = score
            best_entry = entry
    return best_entry if best_score > 0 else None


def _resolve_author_year_text(
    author: str,
    year: str,
    indexes: ReferenceIndexes,
    *,
    left_context: str | None = None,
) -> list[AtomicResolution]:
    author_key = _normalize_author_key(author)
    year_key = year.casefold()
    normalized = f"{author_key}:{year_key}"
    entries = _lookup_author_year_entries(author, year, indexes)
    reference, failure = _resolve_unique(entries or [], normalized)
    if reference is None and failure == FAILURE_AMBIGUOUS_LABEL:
        reference = _disambiguate_by_context(entries, left_context)
        if reference is not None:
            failure = None
    if reference is None:
        reference = _fuzzy_author_year_match(author, year, indexes.references)
        if reference is not None:
            failure = None
    if reference is None:
        return [
            AtomicResolution(
                atomic_key=normalized,
                reference=None,
                resolution_status="unresolved",
                failure_reason=(
                    indexes.failure_reason
                    or failure
                    or FAILURE_AMBIGUOUS_AUTHOR_YEAR
                ),
                resolution_kind=None,
                bibliography_scope_id=indexes.scope_id,
            )
        ]
    return [
        AtomicResolution(
            atomic_key=normalized,
            reference=reference,
            resolution_status="resolved",
            failure_reason=None,
            resolution_kind="author_year",
            bibliography_scope_id=indexes.scope_id,
        )
    ]


def _bibliography_author_year_keys(
    reference: ReferenceEntry,
) -> list[tuple[str, str]]:
    raw = plain_text(reference.raw_text)
    year_match = _YEAR_RE.search(raw)
    if year_match is None:
        return []
    year_token = f"{year_match.group(1)}{year_match.group(2)}".casefold()
    authors_text = raw[: year_match.start()].strip().rstrip(".,;")
    if not authors_text:
        return []
    surnames = _all_author_surnames(authors_text)
    if not surnames:
        return []
    keys: set[tuple[str, str]] = set()
    first = _normalize_author_key(surnames[0])
    keys.add((first, year_token))
    keys.add((_normalize_author_key(f"{surnames[0]} et al"), year_token))
    if len(surnames) >= 2:
        keys.add(
            (
                _normalize_author_key(f"{surnames[0]} and {surnames[1]}"),
                year_token,
            )
        )
        keys.add(
            (
                _normalize_author_key(f"{surnames[0]} & {surnames[1]}"),
                year_token,
            )
        )
    return sorted(keys)


def _all_author_surnames(authors_text: str) -> list[str]:
    segments = re.split(r",|\s+and\s+|\s*&\s*", authors_text)
    surnames: list[str] = []
    for segment in segments:
        cleaned = segment.strip()
        if not cleaned:
            continue
        lowered = cleaned.casefold()
        if lowered in {"et al", "et al.", "eds", "ed", "ed."}:
            continue
        if lowered.endswith(("et al.", "et al")):
            cleaned = re.sub(r"\s+et\s+al\.?$", "", cleaned, flags=re.IGNORECASE)
        surname = _surname_from_author(cleaned)
        if surname and surname.casefold() not in {"et", "al"}:
            surnames.append(surname)
    return surnames


def _citation_author_constraints(author: str) -> tuple[str, str | None]:
    normalized = _normalize_author_key(author)
    if " et al" in normalized:
        return normalized.split(" et al", 1)[0].strip(), None
    if " and " in normalized:
        first, second = normalized.split(" and ", 1)
        return first.strip(), second.strip()
    return normalized, None


def _fuzzy_author_year_match(
    author: str,
    year: str,
    references: list[ReferenceEntry],
) -> ReferenceEntry | None:
    first_author, second_author = _citation_author_constraints(author)
    if not first_author:
        return None
    year_base = re.sub(r"[a-z]$", "", year, flags=re.IGNORECASE)
    scored: list[tuple[int, ReferenceEntry]] = []
    for reference in references:
        if not _reference_matches_year(reference, year, year_base):
            continue
        bib_authors = _all_author_surnames(
            plain_text(reference.raw_text).split(str(reference.year or year_base))[0]
        )
        bib_first = _first_author_surname_from_reference(reference, year)
        if not bib_authors and not bib_first:
            continue
        first_surname = bib_first or (
            _normalize_author_key(bib_authors[0]) if bib_authors else ""
        )
        score = 0
        if first_surname == first_author:
            score = 4
        elif _fuzzy_surname_match(first_author, first_surname):
            score = 3
        elif " et al" in author.casefold():
            first_given = _first_author_given_name(reference, year)
            if first_given == first_author:
                # Some source papers cite the first author's given name (for
                # example, ``Elena et al.``).  This is still constrained to
                # the first-author role and loses to a surname match.
                score = 1
        if score == 0:
            continue
        if second_author is not None:
            if len(bib_authors) < 2:
                continue
            second = _normalize_author_key(bib_authors[1])
            if second != second_author and not _fuzzy_surname_match(
                second_author, bib_authors[1]
            ):
                continue
        scored.append((score, reference))
    if not scored:
        return None
    best = max(score for score, _ in scored)
    candidates = [reference for score, reference in scored if score == best]
    resolved, _ = _resolve_unique(candidates, "first_author_fuzzy")
    return resolved


def _first_author_given_name(reference: ReferenceEntry, year: str) -> str:
    raw = plain_text(reference.raw_text)
    year_base = re.sub(r"[a-z]$", "", year, flags=re.IGNORECASE)
    authors_text = raw
    for token in (year, year_base):
        if token and token in authors_text:
            authors_text = authors_text.split(token, 1)[0]
            break
    first_chunk = re.split(r",|\sand\s|\s&\s", authors_text.strip(), maxsplit=1)[0]
    tokens = [
        token
        for token in re.split(r"\s+", first_chunk)
        if token and not re.fullmatch(r"[A-Z]\.?", token)
    ]
    return _normalize_author_key(tokens[0]) if len(tokens) >= 2 else ""


def _fuzzy_surname_match(citation_token: str, bibliography_token: str) -> bool:
    left = _normalize_author_key(citation_token).replace(" ", "")
    right = _normalize_author_key(bibliography_token).replace(" ", "")
    if left == right:
        return True
    if len(left) < 4 or len(right) < 4:
        if left == right:
            return True
        if len(left) == 3 and (right.startswith(left) or right.endswith(left)):
            return True
        return bool(len(right) == 3 and (left.startswith(right) or left.endswith(right)))
    if SequenceMatcher(None, left, right).ratio() >= 0.84:
        return True
    return bool(left[:3] == right[:3] and abs(len(left) - len(right)) <= 2)


def _reference_matches_year(
    reference: ReferenceEntry,
    year: str,
    year_base: str,
) -> bool:
    raw = plain_text(reference.raw_text).casefold()
    year_cf = year.casefold()
    if year_cf in raw:
        return True
    if re.search(r"[a-z]$", year, flags=re.IGNORECASE):
        return False
    if reference.year is not None and str(reference.year) == year_base:
        return not re.search(rf"{year_base}[a-d]\b", raw)
    return False


def _surname_from_author(author: str) -> str:
    tokens = [token for token in re.split(r"\s+", author.strip()) if token]
    if not tokens:
        return ""
    token = tokens[-1]
    token = unicodedata.normalize("NFKD", token)
    token = "".join(char for char in token if char.isalpha() or char in "-'")
    return token


def _normalize_author_key(author: str) -> str:
    value = plain_text(author).casefold()
    value = value.replace("\u2019", "'")
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\bet\s+al\.?\b", "et al", value)
    value = value.replace(".", "")
    return value.strip()


def _first_author_surname_from_reference(reference: ReferenceEntry, year: str) -> str:
    raw = plain_text(reference.raw_text)
    year_base = re.sub(r"[a-z]$", "", year, flags=re.IGNORECASE)
    authors_text = raw
    for token in (year, year_base):
        if token and token in authors_text:
            authors_text = authors_text.split(token, 1)[0]
            break
    authors_text = authors_text.strip().rstrip(".,;")
    first_chunk = re.split(r",|\sand\s|\s&\s", authors_text, maxsplit=1)[0]
    tokens = [
        token
        for token in re.split(r"\s+", first_chunk.strip())
        if token and not re.fullmatch(r"[A-Z]\.", token) and token != "."
    ]
    tokens = [token for token in tokens if re.search(r"\w", token, flags=re.UNICODE)]
    if len(tokens) >= 2 and len(tokens[-1]) <= 3:
        return _normalize_author_key(tokens[-2] + tokens[-1])
    if tokens:
        return _normalize_author_key(tokens[-1])
    return ""


def _author_surname_tokens_from_reference(
    reference: ReferenceEntry,
    year: str,
) -> list[str]:
    raw = plain_text(reference.raw_text)
    year_base = re.sub(r"[a-z]$", "", year, flags=re.IGNORECASE)
    authors_text = raw
    for token in (year, year_base):
        if token and token in authors_text:
            authors_text = authors_text.split(token, 1)[0]
            break
    authors_text = authors_text.strip().rstrip(".,;")
    first_chunk = re.split(r",|\sand\s|\s&\s", authors_text, maxsplit=1)[0]
    tokens = [
        token
        for token in re.split(r"\s+", first_chunk.strip())
        if token and not re.fullmatch(r"[A-Z]\.", token) and token != "."
    ]
    normalized = [_normalize_author_key(token) for token in tokens if re.search(r"\w", token, flags=re.UNICODE)]
    merged: list[str] = []
    for index, token in enumerate(normalized):
        if len(token) <= 3 and merged:
            merged[-1] = merged[-1] + token
        else:
            merged.append(token)
    tokens = list(merged)
    if tokens:
        trailing = tokens[-1]
        if len(trailing) > 4:
            suffix = trailing[-3:]
            if suffix and suffix not in tokens:
                tokens.append(suffix)
    return tokens


def _work_identity(reference: ReferenceEntry) -> str:
    if reference.doi:
        return f"doi:{reference.doi.casefold()}"
    if reference.arxiv_id:
        return f"arxiv:{reference.arxiv_id.casefold()}"
    raw = plain_text(reference.raw_text)
    year_match = _YEAR_RE.search(raw)
    title_text = plain_text(
        reference.title or str(reference.parsed_fields.get("title") or "")
    )
    if not title_text and year_match is not None:
        after_year = raw[year_match.end() :].strip().lstrip(".,; ")
        if after_year:
            title_text = after_year
    if not title_text:
        title_text = raw
    title = re.sub(r"\s+", " ", title_text).strip().casefold()
    year = reference.year if reference.year is not None else ""
    if year_match is not None and not year:
        year = year_match.group(1)
    return f"title:{title}:{year}"


def _resolve_unique(
    entries: list[ReferenceEntry],
    key: str,
) -> tuple[ReferenceEntry | None, str | None]:
    if not entries:
        return None, FAILURE_MISSING_REFERENCE_ENTRY
    if len(entries) == 1:
        return entries[0], None
    identities = {_work_identity(entry) for entry in entries}
    if len(identities) == 1:
        return min(entries, key=lambda item: item.order), None
    return None, FAILURE_AMBIGUOUS_LABEL


def _append_unique(
    mapping: dict[_KeyT, list[ReferenceEntry]],
    key: _KeyT,
    reference: ReferenceEntry,
) -> None:
    bucket = mapping.setdefault(key, [])
    if reference not in bucket:
        bucket.append(reference)


def _extend_unique(
    mapping: dict[_KeyT, list[ReferenceEntry]],
    key: _KeyT,
    entries: list[ReferenceEntry],
) -> None:
    for reference in entries:
        _append_unique(mapping, key, reference)


def _span_resolution_status(resolutions: list[AtomicResolution]) -> str:
    resolved_count = sum(
        1 for item in resolutions if item.resolution_status == "resolved"
    )
    if resolved_count == 0:
        return "unresolved"
    if resolved_count == len(resolutions):
        return "resolved"
    return "partially_resolved"


def _mentions_from_atomic_resolutions(
    *,
    resolutions: list[AtomicResolution],
    document_id: str,
    snapshot_id: str,
    element_id: str,
    surface: str,
    start: int,
    end: int,
    match_kind: str,
    document_region: str | None,
) -> list[CitationMention]:
    span_id = citation_span_id(document_id, element_id, start, end)
    span_status = _span_resolution_status(resolutions)
    mentions: list[CitationMention] = []
    for index, item in enumerate(resolutions):
        mention_id = citation_mention_id(
            document_id,
            element_id,
            span_id,
            item.atomic_key,
            index,
        )
        mentions.append(
            CitationMention(
                id=mention_id,
                document_id=document_id,
                canonical_snapshot_id=snapshot_id,
                citation_span_id=span_id,
                surface_text=surface,
                atomic_key=item.atomic_key,
                normalized_keys=[item.atomic_key],
                element_id=element_id,
                character_start=start,
                character_end=end,
                group_index=index,
                group_size=len(resolutions),
                bibliography_scope_id=item.bibliography_scope_id,
                citation_namespace_id=item.bibliography_scope_id,
                document_region=document_region,
                reference_entry_id=item.reference.id if item.reference else None,
                resolution_status=item.resolution_status,
                span_resolution_status=span_status,
                failure_reason=item.failure_reason,
                metadata={
                    "match_kind": match_kind,
                    "resolution_kind": item.resolution_kind,
                },
            )
        )
    return mentions


def _unresolved_author_year_bracket_mentions(
    *,
    inner: str,
    document_id: str,
    snapshot_id: str,
    element_id: str,
    surface: str,
    start: int,
    end: int,
    document_region: str | None,
) -> list[CitationMention]:
    atoms: list[str] = []
    for part in re.split(r"\s*;\s*", inner):
        atoms.extend(_expand_author_year_part(part.strip()))
    if not atoms:
        atoms = [inner]
    resolutions = [
        AtomicResolution(
            atomic_key=atom,
            reference=None,
            resolution_status="unresolved",
            failure_reason=FAILURE_AMBIGUOUS_AUTHOR_YEAR,
            resolution_kind=None,
        )
        for atom in atoms
    ]
    return _mentions_from_atomic_resolutions(
        resolutions=resolutions,
        document_id=document_id,
        snapshot_id=snapshot_id,
        element_id=element_id,
        surface=surface,
        start=start,
        end=end,
        match_kind="bracket",
        document_region=document_region,
    )


def _unresolved_span_mention(
    *,
    document_id: str,
    snapshot_id: str,
    element_id: str,
    surface: str,
    inner: str,
    start: int,
    end: int,
    match_kind: str,
    failure_reason: str,
    document_region: str | None,
) -> CitationMention:
    span_id = citation_span_id(document_id, element_id, start, end)
    mention_id = citation_mention_id(document_id, element_id, span_id, inner, 0)
    return CitationMention(
        id=mention_id,
        document_id=document_id,
        canonical_snapshot_id=snapshot_id,
        citation_span_id=span_id,
        surface_text=surface,
        atomic_key=inner,
        normalized_keys=[inner],
        element_id=element_id,
        character_start=start,
        character_end=end,
        group_index=0,
        group_size=1,
        document_region=document_region,
        reference_entry_id=None,
        resolution_status="unresolved",
        span_resolution_status="unresolved",
        failure_reason=failure_reason,
        metadata={"match_kind": match_kind},
    )


def _legacy_index_to_indexes(
    reference_index: dict[str, ReferenceEntry],
) -> ReferenceIndexes:
    label_map: dict[str, list[ReferenceEntry]] = {}
    for key, reference in reference_index.items():
        label_map.setdefault(key, []).append(reference)
        label_map.setdefault(key.casefold(), []).append(reference)
    return ReferenceIndexes(label_map=label_map, references=list(reference_index.values()))


def _mask_span(text: str, start: int, end: int) -> str:
    return text[:start] + (" " * (end - start)) + text[end:]


def _is_negative_bracket_domain(inner: str, text: str, start: int) -> bool:
    compact = re.sub(r"\s+", "", inner)
    if compact in {";", "", ","} or inner.strip() in {";", ""}:
        return True
    if _SECTION_REF_RE.search(inner):
        return True
    if "," in inner and not re.search(r"\d\s*[-–−—]\s*\d", inner):
        parts = [part.strip() for part in inner.split(",")]
        if (
            len(parts) > 1
            and all(part.isdigit() for part in parts)
            and len(set(parts)) == 1
        ):
            # Repeated tensor/image dimensions are not citation labels.
            return True
        if len(parts) == 2 and parts[0].isdigit():
            if parts[0] == "0" and parts[1].casefold() in {"1", "t"}:
                return True
            if parts[1].isdigit():
                return bool(int(parts[0]) == 0 and int(parts[1]) == 1)
            if not _is_numeric_citation_part(parts[1]):
                return True
    if _YEAR_ONLY_BRACKET_RE.fullmatch(inner.strip()):
        return _left_author_phrase(text, start) is None
    return False


def _is_numeric_citation_part(part: str) -> bool:
    value = part.strip()
    if not value:
        return False
    if value.isdigit():
        return True
    if re.fullmatch(r"\d+[a-d]?", value, flags=re.IGNORECASE):
        return True
    if NUMERIC_RANGE_RE.match(value):
        return True
    return bool(re.fullmatch(r"\d+\s*[-–−—]\s*\d+", value))


def _left_author_match(text: str, bracket_start: int) -> tuple[str, int] | None:
    prefix = text[max(0, bracket_start - 120) : bracket_start]
    patterns = (
        r"(?P<author>[A-ZÀ-ÖØ-Þ][\w''\u00C0-\u024F\-]+(?:\s+et\s+al\.?)?(?:\s+and\s+[A-ZÀ-ÖØ-Þ][\w''\u00C0-\u024F\-]+)?)\s*$",
        r"(?P<author>[A-ZÀ-ÖØ-Þ][\w''\u00C0-\u024F\-]+)\s*$",
    )
    stripped = prefix.rstrip()
    for pattern in patterns:
        match = re.search(pattern, stripped)
        if match is not None:
            author = match.group("author").strip()
            absolute_start = max(0, bracket_start - 120) + match.start("author")
            return author, absolute_start
    return None


def _left_author_phrase(text: str, bracket_start: int) -> str | None:
    match = _left_author_match(text, bracket_start)
    return match[0] if match is not None else None


def _split_ocred_symbolic_group(inner: str) -> list[str] | None:
    if re.search(r"\\mathrm\s*\{", inner, re.IGNORECASE):
        tokens = re.findall(
            r"\\mathrm\s*\{\s*([A-Za-z*]+)\s*\}\s*(?:\^\s*\{\s*\\?\s*ast\s*\})?",
            inner,
            flags=re.IGNORECASE,
        )
        years = re.findall(r"(?<!\d)([12]\d{3})(?!\d)", inner)
        if tokens and years:
            return [f"{token}{years[0]}" for token in tokens]
    if ". " not in inner and "." not in inner:
        return None
    parts = re.split(r"\.\s*", inner.strip())
    normalized: list[str] = []
    for part in parts:
        token = re.sub(r"\s+", "", part.strip())
        if token:
            normalized.append(token)
    if len(normalized) < 2:
        return None
    return normalized


def _is_venue_false_positive(text: str, start: int, surface: str) -> bool:
    window = text[max(0, start - 80) : start + len(surface)]
    if _VENUE_FALSE_POSITIVE_RE.search(window):
        return True
    author = surface.strip().strip("()")
    return bool(author.casefold().startswith("processing "))


def _looks_like_citation_bracket(inner: str) -> bool:
    if not inner or len(inner) > 240:
        return False
    if re.search(r"[\\{}^$]", inner):
        return False
    if re.search(r"\\(?:mathrm|mathcal|widehat|right)", inner):
        return False
    compact = re.sub(r"\s+", "", inner).casefold()
    if compact in {"0,1", "0,t", "-1,1", "1,1", "lg", "cs"}:
        return False
    if re.fullmatch(r"[a-z]{1,3}", compact):
        return False
    if re.fullmatch(r"[A-Z]{1,4}", inner.strip()):
        return False
    if re.fullmatch(r"[12]\d{3}[a-d]?", inner.strip(), flags=re.IGNORECASE):
        return False
    if re.fullmatch(r"[a-z]{2,12}\.[A-Za-z]{2,12}", inner.strip()):
        return False
    if re.search(r"\bsec(?:tion)?\b|\bcor(?:ollary)?\b|\beq(?:uation)?\b", inner, re.IGNORECASE):
        return False
    return not re.fullmatch(r";\s*", inner.strip())
