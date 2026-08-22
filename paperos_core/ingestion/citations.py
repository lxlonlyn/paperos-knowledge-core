"""Citation mention extraction with bibliography-first label and author-year resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from paperos_core.domain.canonical import CitationMention, ReferenceEntry
from paperos_core.domain.ids import citation_mention_id
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


@dataclass
class ReferenceIndexes:
    label_map: dict[str, list[ReferenceEntry]] = field(default_factory=dict)
    author_year_map: dict[tuple[str, str], list[ReferenceEntry]] = field(
        default_factory=dict
    )
    references: list[ReferenceEntry] = field(default_factory=list)


@dataclass(frozen=True)
class ResolvedCitation:
    reference: ReferenceEntry
    normalized_key: str
    resolution_kind: str


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


def _normalize_bracket_inner(inner: str) -> str:
    collapsed = re.sub(r"(?<=\d)\s+(?=\d)", "", inner)
    return collapsed


def resolve_bracket(
    inner: str,
    indexes: ReferenceIndexes,
) -> list[ResolvedCitation] | None:
    inner = _normalize_bracket_inner(inner).strip()
    if not inner:
        return None

    if inner in indexes.label_map:
        return _resolve_label_keys([inner], indexes)

    expanded_single = expand_atom(inner)
    if expanded_single and all(key in indexes.label_map for key in expanded_single):
        return _resolve_label_keys(expanded_single, indexes)

    raw_parts = re.split(r"\s*[,;]\s*", inner)
    if len(raw_parts) > 1:
        keys: list[str] = []
        for part in raw_parts:
            expanded = expand_atom(part)
            if not expanded:
                break
            keys.extend(expanded)
        else:
            if keys and all(key in indexes.label_map for key in keys):
                return _resolve_label_keys(keys, indexes)

    author_year = _resolve_bracket_author_year(inner, indexes)
    if author_year:
        return author_year

    return None


def extract_citation_mentions_from_text(
    *,
    document_id: str,
    snapshot_id: str,
    element_id: str,
    text: str,
    reference_index: dict[str, ReferenceEntry] | ReferenceIndexes,
) -> list[CitationMention]:
    indexes = (
        reference_index
        if isinstance(reference_index, ReferenceIndexes)
        else _legacy_index_to_indexes(reference_index)
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
                _mention_from_resolved(
                    resolved=resolved,
                    document_id=document_id,
                    snapshot_id=snapshot_id,
                    element_id=element_id,
                    surface=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    match_kind="bracket",
                )
            )
            working = _mask_span(working, match.start(), match.end())
            continue
        mentions.append(
            CitationMention(
                id=citation_mention_id(
                    document_id,
                    element_id,
                    match.group(0),
                    match.start(),
                ),
                document_id=document_id,
                canonical_snapshot_id=snapshot_id,
                surface_text=match.group(0),
                normalized_keys=[inner],
                element_id=element_id,
                character_start=match.start(),
                character_end=match.end(),
                reference_entry_id=None,
                resolution_status="unresolved",
                metadata={"match_kind": "bracket"},
            )
        )

    for match in _AUTHOR_YEAR_PAREN_RE.finditer(working):
        resolved = _resolve_author_year_text(
            match.group("author"), match.group("year"), indexes
        )
        if not resolved:
            continue
        mentions.extend(
            _mention_from_resolved(
                resolved=resolved,
                document_id=document_id,
                snapshot_id=snapshot_id,
                element_id=element_id,
                surface=match.group(0),
                start=match.start(),
                end=match.end(),
                match_kind="author_year_paren",
            )
        )
        working = _mask_span(working, match.start(), match.end())

    for match in _AUTHOR_YEAR_INLINE_RE.finditer(working):
        resolved = _resolve_author_year_text(
            match.group("author"), match.group("year"), indexes
        )
        if not resolved:
            continue
        mentions.extend(
            _mention_from_resolved(
                resolved=resolved,
                document_id=document_id,
                snapshot_id=snapshot_id,
                element_id=element_id,
                surface=match.group(0),
                start=match.start(),
                end=match.end(),
                match_kind="author_year_inline",
            )
        )

    return mentions


def attach_mentions_to_chunks(
    mentions: list[CitationMention],
    *,
    element_to_chunk: dict[str, str],
) -> list[CitationMention]:
    attached: list[CitationMention] = []
    for mention in mentions:
        chunk_id = element_to_chunk.get(mention.element_id)
        attached.append(mention.model_copy(update={"chunk_id": chunk_id}))
    return attached


def _legacy_index_to_indexes(
    reference_index: dict[str, ReferenceEntry],
) -> ReferenceIndexes:
    label_map: dict[str, list[ReferenceEntry]] = {}
    for key, reference in reference_index.items():
        label_map.setdefault(key, []).append(reference)
        label_map.setdefault(key.casefold(), []).append(reference)
    return ReferenceIndexes(label_map=label_map)


def _resolve_label_keys(
    keys: list[str],
    indexes: ReferenceIndexes,
) -> list[ResolvedCitation] | None:
    resolved: list[ResolvedCitation] = []
    for key in keys:
        entries = indexes.label_map.get(key)
        if not entries:
            return None
        reference = _resolve_unique(entries, key)
        if reference is None:
            return None
        resolved.append(
            ResolvedCitation(
                reference=reference,
                normalized_key=key,
                resolution_kind="label",
            )
        )
    return resolved


def _resolve_bracket_author_year(
    inner: str,
    indexes: ReferenceIndexes,
) -> list[ResolvedCitation] | None:
    parts = [part.strip() for part in inner.split(";") if part.strip()]
    if not parts:
        return None
    resolved: list[ResolvedCitation] = []
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
) -> list[ResolvedCitation] | None:
    author_key = _normalize_author_key(author)
    year_key = year.casefold()
    entries = indexes.author_year_map.get((author_key, year_key))
    reference = _resolve_unique(entries or [], f"{author_key}:{year_key}")
    if reference is None:
        reference = _fuzzy_author_year_match(author, year, indexes.references)
    if reference is None:
        return None
    normalized = f"{author_key}:{year_key}"
    return [
        ResolvedCitation(
            reference=reference,
            normalized_key=normalized,
            resolution_kind="author_year",
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
        second = _normalize_author_key(surnames[1])
        keys.add((second, year_token))
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


def _citation_surname_tokens(author: str) -> list[str]:
    normalized = _normalize_author_key(author)
    normalized = re.sub(r"\bet al\b", "", normalized).strip()
    if " and " in normalized:
        return [token.strip() for token in normalized.split(" and ") if token.strip()]
    return [normalized] if normalized else []


def _fuzzy_author_year_match(
    author: str,
    year: str,
    references: list[ReferenceEntry],
) -> ReferenceEntry | None:
    tokens = _citation_surname_tokens(author)
    if not tokens:
        return None
    year_base = re.sub(r"[a-z]$", "", year, flags=re.IGNORECASE)
    candidates: list[ReferenceEntry] = []
    for reference in references:
        if not _reference_matches_year(reference, year, year_base):
            continue
        compact = plain_text(reference.raw_text).casefold().replace(" ", "")
        if all(_token_matches_compact(token, compact) for token in tokens):
            candidates.append(reference)
    return _resolve_unique(candidates, "fuzzy")


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


def _token_matches_compact(token: str, compact_raw: str) -> bool:
    needle = token.casefold().replace(" ", "")
    if not needle:
        return False
    if needle in compact_raw:
        return True
    if len(needle) < 4:
        return False
    window = len(needle)
    for index in range(max(1, len(compact_raw) - window)):
        fragment = compact_raw[index : index + window]
        if SequenceMatcher(None, needle, fragment).ratio() >= 0.84:
            return True
    return False


def _extract_surnames(authors_text: str) -> list[str]:
    return _all_author_surnames(authors_text)[:2]


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


def _resolve_unique(
    entries: list[ReferenceEntry],
    key: str,
) -> ReferenceEntry | None:
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]
    return None


def _append_unique(
    mapping: dict,
    key,
    reference: ReferenceEntry,
) -> None:
    bucket = mapping.setdefault(key, [])
    if reference not in bucket:
        bucket.append(reference)


def _mention_from_resolved(
    *,
    resolved: list[ResolvedCitation],
    document_id: str,
    snapshot_id: str,
    element_id: str,
    surface: str,
    start: int,
    end: int,
    match_kind: str,
) -> list[CitationMention]:
    mentions: list[CitationMention] = []
    for index, item in enumerate(resolved):
        mention_id = citation_mention_id(
            document_id,
            element_id,
            f"{surface}#{item.normalized_key}#{index}",
            start,
        )
        mentions.append(
            CitationMention(
                id=mention_id,
                document_id=document_id,
                canonical_snapshot_id=snapshot_id,
                surface_text=surface,
                normalized_keys=[item.normalized_key],
                element_id=element_id,
                character_start=start,
                character_end=end,
                reference_entry_id=item.reference.id,
                resolution_status="resolved",
                metadata={
                    "match_kind": match_kind,
                    "resolution_kind": item.resolution_kind,
                    "group_index": index,
                    "group_size": len(resolved),
                },
            )
        )
    return mentions


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
