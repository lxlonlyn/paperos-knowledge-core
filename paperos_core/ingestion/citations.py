"""Citation mention extraction with bibliography-first label and author-year resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from paperos_core.domain.canonical import Chunk, CitationMention, ReferenceEntry, Section
from paperos_core.domain.ids import citation_mention_id, citation_span_id
from paperos_core.ingestion.bibliography_scope import (
    BibliographyScope,
    REGION_ABSTRACT,
    REGION_MAIN,
    REGION_SUPPLEMENT,
    ScopedBibliography,
    assign_bibliography_scopes,
    repair_numeric_label_sequence,
    resolve_element_region,
    scopes_for_region,
)
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

FAILURE_MISSING_REFERENCE_ENTRY = "MISSING_REFERENCE_ENTRY"
FAILURE_AMBIGUOUS_LABEL = "AMBIGUOUS_LABEL"
FAILURE_AMBIGUOUS_AUTHOR_YEAR = "AMBIGUOUS_AUTHOR_YEAR"
FAILURE_SCOPE_NOT_FOUND = "SCOPE_NOT_FOUND"
FAILURE_UNPARSEABLE = "UNPARSEABLE"


@dataclass
class ReferenceIndexes:
    label_map: dict[str, list[ReferenceEntry]] = field(default_factory=dict)
    author_year_map: dict[tuple[str, str], list[ReferenceEntry]] = field(
        default_factory=dict
    )
    references: list[ReferenceEntry] = field(default_factory=list)
    scope_id: str | None = None


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
    elements: list,
    sections: list[Section],
) -> ScopedBibliography:
    reference_scope, parent_regions = assign_bibliography_scopes(
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
        reference = reference.model_copy(update={"bibliography_scope_id": scope_id})
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
        scopes[scope_id] = BibliographyScope(
            scope_id=scope_id,
            parent_region=parent_regions.get(scope_id, REGION_MAIN),
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
    if re.fullmatch(r"[\d,;\-–−—]+", compact):
        return False
    return True


def resolve_bracket(
    inner: str,
    indexes: ReferenceIndexes,
) -> list[AtomicResolution] | None:
    inner = _normalize_bracket_inner(inner).strip()
    if not inner:
        return None
    if _looks_like_author_year_bracket(inner):
        author_year = _resolve_bracket_author_year(inner, indexes)
        return author_year or None
    atoms = parse_bracket_atoms(inner)
    if atoms is None:
        return None
    if len(atoms) == 1 and atoms[0] in indexes.label_map:
        return _resolve_atomic_keys(atoms, indexes)
    if len(atoms) == 1:
        author_year = _resolve_bracket_author_year(inner, indexes)
        if author_year:
            return author_year
    return _resolve_atomic_keys(atoms, indexes)


def extract_citation_mentions_from_text(
    *,
    document_id: str,
    snapshot_id: str,
    element_id: str,
    text: str,
    reference_index: dict[str, ReferenceEntry] | ReferenceIndexes | ScopedBibliography,
    document_region: str | None = None,
    bibliography_scope_ids: list[str] | None = None,
) -> list[CitationMention]:
    scoped, indexes = _resolve_reference_context(
        reference_index,
        document_region=document_region,
        bibliography_scope_ids=bibliography_scope_ids,
    )
    mentions: list[CitationMention] = []
    working = text

    for match in BRACKET_RE.finditer(text):
        inner = match.group("body").strip()
        if not _looks_like_citation_bracket(inner):
            continue
        resolved = resolve_bracket(inner, indexes)
        if resolved:
            mentions.extend(
                _mentions_from_atomic_resolutions(
                    resolutions=resolved,
                    document_id=document_id,
                    snapshot_id=snapshot_id,
                    element_id=element_id,
                    surface=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    match_kind="bracket",
                    document_region=document_region,
                )
            )
            working = _mask_span(working, match.start(), match.end())
            continue
        if _looks_like_author_year_bracket(inner):
            author_resolved = _resolve_bracket_author_year(inner, indexes)
            if author_resolved:
                mentions.extend(
                    _mentions_from_atomic_resolutions(
                        resolutions=author_resolved,
                        document_id=document_id,
                        snapshot_id=snapshot_id,
                        element_id=element_id,
                        surface=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        match_kind="bracket",
                        document_region=document_region,
                    )
                )
                working = _mask_span(working, match.start(), match.end())
                continue
            mentions.extend(
                _unresolved_author_year_bracket_mentions(
                    inner=inner,
                    document_id=document_id,
                    snapshot_id=snapshot_id,
                    element_id=element_id,
                    surface=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    document_region=document_region,
                )
            )
            working = _mask_span(working, match.start(), match.end())
            continue
        atoms = parse_bracket_atoms(inner)
        if atoms:
            unresolved = _resolve_atomic_keys(atoms, indexes)
            mentions.extend(
                _mentions_from_atomic_resolutions(
                    resolutions=unresolved,
                    document_id=document_id,
                    snapshot_id=snapshot_id,
                    element_id=element_id,
                    surface=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    match_kind="bracket",
                    document_region=document_region,
                )
            )
        else:
            mentions.append(
                _unresolved_span_mention(
                    document_id=document_id,
                    snapshot_id=snapshot_id,
                    element_id=element_id,
                    surface=match.group(0),
                    inner=inner,
                    start=match.start(),
                    end=match.end(),
                    match_kind="bracket",
                    failure_reason=FAILURE_UNPARSEABLE,
                    document_region=document_region,
                )
            )
        working = _mask_span(working, match.start(), match.end())

    for match in _AUTHOR_YEAR_PAREN_RE.finditer(working):
        resolved = _resolve_author_year_text(
            match.group("author"), match.group("year"), indexes
        )
        if resolved:
            mentions.extend(
                _mentions_from_atomic_resolutions(
                    resolutions=resolved,
                    document_id=document_id,
                    snapshot_id=snapshot_id,
                    element_id=element_id,
                    surface=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    match_kind="author_year_paren",
                    document_region=document_region,
                )
            )
        else:
            mentions.append(
                _unresolved_span_mention(
                    document_id=document_id,
                    snapshot_id=snapshot_id,
                    element_id=element_id,
                    surface=match.group(0),
                    inner=f"{match.group('author')} {match.group('year')}",
                    start=match.start(),
                    end=match.end(),
                    match_kind="author_year_paren",
                    failure_reason=FAILURE_AMBIGUOUS_AUTHOR_YEAR,
                    document_region=document_region,
                )
            )
        working = _mask_span(working, match.start(), match.end())

    for match in _AUTHOR_YEAR_INLINE_RE.finditer(working):
        resolved = _resolve_author_year_text(
            match.group("author"), match.group("year"), indexes
        )
        if resolved:
            mentions.extend(
                _mentions_from_atomic_resolutions(
                    resolutions=resolved,
                    document_id=document_id,
                    snapshot_id=snapshot_id,
                    element_id=element_id,
                    surface=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    match_kind="author_year_inline",
                    document_region=document_region,
                )
            )
        else:
            mentions.append(
                _unresolved_span_mention(
                    document_id=document_id,
                    snapshot_id=snapshot_id,
                    element_id=element_id,
                    surface=match.group(0),
                    inner=f"{match.group('author')} {match.group('year')}",
                    start=match.start(),
                    end=match.end(),
                    match_kind="author_year_inline",
                    failure_reason=FAILURE_AMBIGUOUS_AUTHOR_YEAR,
                    document_region=document_region,
                )
            )

    _ = scoped
    return mentions


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
    reference_index: dict[str, ReferenceEntry] | ReferenceIndexes | ScopedBibliography,
    *,
    document_region: str | None,
    bibliography_scope_ids: list[str] | None,
) -> tuple[ScopedBibliography | None, ReferenceIndexes]:
    if isinstance(reference_index, ScopedBibliography):
        scoped = reference_index
        scope_ids = bibliography_scope_ids
        if scope_ids is None and document_region is not None:
            scope_ids = scopes_for_region(document_region, scoped)
        if scope_ids:
            merged = ReferenceIndexes(scope_id=scope_ids[0])
            for scope_id in scope_ids:
                indexes = scoped.scope_indexes.get(scope_id)
                if indexes is None:
                    continue
                for key, entries in indexes.label_map.items():
                    _extend_unique(merged.label_map, key, entries)
                for key, entries in indexes.author_year_map.items():
                    _extend_unique(merged.author_year_map, key, entries)
                merged.references.extend(indexes.references)
            return scoped, merged
        default = scoped.scope_indexes.get("default") or next(
            iter(scoped.scope_indexes.values())
        )
        return scoped, default
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
            key = normalize_label(label)
            _append_unique(label_map, key, reference)
        ref_num = reference.parsed_fields.get("reference_number")
        if ref_num is not None:
            num_key = str(ref_num)
            if num_key != normalize_label(label or ""):
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
        entries = indexes.label_map.get(key)
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
                    failure_reason=failure or FAILURE_MISSING_REFERENCE_ENTRY,
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
) -> list[AtomicResolution] | None:
    parts = [part.strip() for part in inner.split(";") if part.strip()]
    if not parts:
        return None
    resolved: list[AtomicResolution] = []
    for part in parts:
        for atom in _expand_author_year_part(part):
            match = _BRACKET_AUTHOR_YEAR_RE.match(atom.strip())
            if not match:
                continue
            atom_resolved = _resolve_author_year_text(
                match.group("author"), match.group("year"), indexes
            )
            if atom_resolved:
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


def _resolve_author_year_text(
    author: str,
    year: str,
    indexes: ReferenceIndexes,
) -> list[AtomicResolution] | None:
    author_key = _normalize_author_key(author)
    year_key = year.casefold()
    entries = indexes.author_year_map.get((author_key, year_key))
    reference, failure = _resolve_unique(entries or [], f"{author_key}:{year_key}")
    if reference is None and failure is None:
        reference = _fuzzy_author_year_match(author, year, indexes.references)
        failure = None if reference else FAILURE_AMBIGUOUS_AUTHOR_YEAR
    if reference is None:
        return None
    normalized = f"{author_key}:{year_key}"
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
        if lowered.endswith("et al.") or lowered.endswith("et al"):
            cleaned = re.sub(r"\s+et\s+al\.?$", "", cleaned, flags=re.IGNORECASE)
        surname = _surname_from_author(cleaned)
        if surname and surname.casefold() not in {"et", "al"}:
            surnames.append(surname)
    return surnames


def _citation_author_constraints(author: str) -> tuple[str | None, str | None]:
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
    candidates: list[ReferenceEntry] = []
    for reference in references:
        if not _reference_matches_year(reference, year, year_base):
            continue
        bib_authors = _all_author_surnames(
            plain_text(reference.raw_text).split(str(reference.year or year_base))[0]
        )
        if not bib_authors:
            continue
        if _normalize_author_key(bib_authors[0]) != first_author:
            continue
        if second_author is not None:
            if len(bib_authors) < 2:
                continue
            if _normalize_author_key(bib_authors[1]) != second_author:
                continue
        if not _fuzzy_surname_match(first_author, bib_authors[0]):
            continue
        candidates.append(reference)
    resolved, _ = _resolve_unique(candidates, "fuzzy")
    return resolved


def _fuzzy_surname_match(citation_token: str, bibliography_token: str) -> bool:
    left = citation_token.casefold().replace(" ", "")
    right = bibliography_token.casefold().replace(" ", "")
    if left == right:
        return True
    if len(left) < 4 or len(right) < 4:
        return False
    return SequenceMatcher(None, left, right).ratio() >= 0.84


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
    return re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ\-']", "", token)


def _normalize_author_key(author: str) -> str:
    value = plain_text(author).casefold()
    value = value.replace("’", "'")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\bet\s+al\.?\b", "et al", value)
    value = value.replace(".", "")
    return value.strip()


def _work_identity(reference: ReferenceEntry) -> str:
    if reference.doi:
        return f"doi:{reference.doi.casefold()}"
    if reference.arxiv_id:
        return f"arxiv:{reference.arxiv_id.casefold()}"
    title = plain_text(reference.title or reference.raw_text)[:120].casefold()
    year = reference.year if reference.year is not None else ""
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
        return sorted(entries, key=lambda item: item.order)[0], None
    return None, FAILURE_AMBIGUOUS_LABEL


def _append_unique(
    mapping: dict,
    key,
    reference: ReferenceEntry,
) -> None:
    bucket = mapping.setdefault(key, [])
    if reference not in bucket:
        bucket.append(reference)


def _extend_unique(mapping: dict, key, entries: list[ReferenceEntry]) -> None:
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


def _looks_like_citation_bracket(inner: str) -> bool:
    if not inner or len(inner) > 240:
        return False
    if re.search(r"[\\{}^$]", inner):
        return False
    if re.search(r"\\(?:mathrm|mathcal|widehat|right)", inner):
        return False
    compact = re.sub(r"\s+", "", inner).casefold()
    if compact in {"0,1", "0,t", "-1,1", "1,1"}:
        return False
    if re.fullmatch(r"[12]\d{3}[a-d]?", inner.strip(), flags=re.IGNORECASE):
        return False
    if re.fullmatch(r"[a-z]{2,12}\.[A-Za-z]{2,12}", inner.strip()):
        return False
    if re.search(r"\bsec(?:tion)?\b|\bcor(?:ollary)?\b|\beq(?:uation)?\b", inner, re.I):
        return False
    if re.fullmatch(r";\s*", inner.strip()):
        return False
    return True
