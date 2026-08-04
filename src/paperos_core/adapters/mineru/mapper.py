"""The sole MinerU artifact-to-canonical mapping boundary."""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from paperos_core.config import IngestionConfig
from paperos_core.domain.canonical import (
    CanonicalBundle,
    CanonicalSnapshot,
    Document,
    Element,
    Person,
    Section,
    SourceSpan,
)
from paperos_core.domain.documents import SourceFile
from paperos_core.domain.enums import ElementType, ParserArtifactType, ParseRunStatus
from paperos_core.domain.ids import (
    CANONICAL_PIPELINE_VERSION,
    CANONICAL_SCHEMA_VERSION,
    canonical_snapshot_id,
    document_id,
    element_id,
    person_id,
    section_id,
)
from paperos_core.domain.parsing import ParserArtifact, ParseRun
from paperos_core.errors import CanonicalMappingError
from paperos_core.ingestion.chunking import build_chunks
from paperos_core.ingestion.classification import classify_element
from paperos_core.ingestion.cleaning import (
    MarginText,
    adjacent_duplicate_indexes,
    repeated_margin_indexes,
)
from paperos_core.ingestion.normalization import (
    normalize_doi,
    normalize_text,
    plain_text,
    strip_heading_number,
)
from paperos_core.ingestion.references import parse_reference_entry

_DOI = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
_YEAR = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_HEADING_NUMBER = re.compile(r"^\s*(\d+(?:\.\d+)*)\.?\s+")


class MinerUCanonicalMapper:
    """Interpret persisted MinerU structures exactly once."""

    def __init__(self, config: IngestionConfig) -> None:
        self.config = config

    def build_canonical_snapshot(
        self,
        *,
        source: SourceFile,
        parse_run: ParseRun,
        artifacts: list[ParserArtifact],
        manifest_path: Path,
        dataset_id: str | None = None,
    ) -> CanonicalBundle:
        if parse_run.status != ParseRunStatus.COMPLETED:
            raise CanonicalMappingError(
                "Canonical mapping requires a completed ParseRun.",
                affected=parse_run.id,
            )
        if parse_run.source_file_id != source.id:
            raise CanonicalMappingError(
                "ParseRun does not belong to the supplied SourceFile.",
                affected=parse_run.id,
            )
        content_artifact, content = self._load_content_list(artifacts)
        markdown = self._load_markdown(artifacts)
        snapshot_identifier = canonical_snapshot_id(parse_run.id)
        document_identifier = document_id(source.id)
        timestamp = parse_run.completed_at or parse_run.created_at
        snapshot = CanonicalSnapshot(
            id=snapshot_identifier,
            source_file_id=source.id,
            parse_run_id=parse_run.id,
            document_id=document_identifier,
            manifest_path=manifest_path,
            dataset_id=(dataset_id or source.dataset_id or "papers"),
            created_at=timestamp,
            schema_version=CANONICAL_SCHEMA_VERSION,
            pipeline_version=CANONICAL_PIPELINE_VERSION,
        )

        drop_indexes = self._cleanup_indexes(content)
        title = self._extract_title(content)
        frontmatter = self._frontmatter(content)
        authors = self._extract_authors(content, document_identifier, title=title)
        abstract = self._extract_named_block(content, "abstract")
        if abstract is None:
            abstract = self._extract_unheaded_abstract(content)
        keywords = self._extract_keywords(content)
        doi = self._extract_doi(content, markdown)
        year = self._extract_year(content, frontmatter)
        document = Document(
            id=document_identifier,
            source_file_id=source.id,
            parse_run_id=parse_run.id,
            canonical_snapshot_id=snapshot_identifier,
            title=title,
            language=self._detect_language(content),
            document_type="research_paper",
            abstract=abstract,
            year=year,
            doi=doi,
            keywords=keywords or None,
            authors=authors,
            affiliations=self._extract_affiliations(content),
            created_at=timestamp,
        )
        sections, section_for_item = self._sections(
            content=content,
            artifact_id=content_artifact.id,
            document_id_value=document_identifier,
            snapshot_id=snapshot_identifier,
        )
        elements = self._elements(
            content=content,
            artifact=content_artifact,
            document_id_value=document_identifier,
            snapshot_id=snapshot_identifier,
            section_for_item=section_for_item,
            drop_indexes=drop_indexes,
        )
        references = [
            parse_reference_entry(
                document_id=document_identifier,
                snapshot_id=snapshot_identifier,
                order=order,
                raw_text=element.raw_text or element.text or "",
                source_element_id=element.id,
            )
            for order, element in enumerate(
                item for item in elements if item.element_type == ElementType.REFERENCE
            )
        ]
        chunks = build_chunks(
            document_id=document_identifier,
            snapshot_id=snapshot_identifier,
            sections=sections,
            elements=elements,
            target_tokens=self.config.chunk_target_tokens,
            overlap_tokens=self.config.chunk_overlap_tokens,
        )
        warnings: list[str] = []
        if not authors:
            warnings.append("No author names could be mapped from parser artifacts.")
        if doi is None:
            warnings.append("No DOI was present in parser artifacts.")
        if not references:
            warnings.append("No reference entries were present in parser artifacts.")
        if not any(item.element_type == ElementType.TABLE for item in elements):
            warnings.append("MinerU returned no table element for this document.")
        return CanonicalBundle(
            snapshot=snapshot,
            document=document,
            sections=sections,
            elements=elements,
            chunks=chunks,
            references=references,
            warnings=warnings,
        )

    @staticmethod
    def _load_content_list(
        artifacts: list[ParserArtifact],
    ) -> tuple[ParserArtifact, list[dict[str, Any]]]:
        errors: list[str] = []
        for artifact in artifacts:
            if artifact.artifact_type != ParserArtifactType.CONTENT_LIST:
                continue
            try:
                payload = json.loads(artifact.storage_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                errors.append(f"{artifact.storage_path.name}: {exc}")
                continue
            if (
                isinstance(payload, list)
                and payload
                and all(isinstance(item, dict) for item in payload)
                and any("page_idx" in item for item in payload)
            ):
                return artifact, payload
        raise CanonicalMappingError(
            "No supported MinerU content-list artifact was found.",
            details={"artifact_errors": errors},
        )

    @staticmethod
    def _load_markdown(artifacts: list[ParserArtifact]) -> str:
        for artifact in artifacts:
            if artifact.artifact_type == ParserArtifactType.MARKDOWN:
                try:
                    return artifact.storage_path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError) as exc:
                    raise CanonicalMappingError(
                        f"Unable to read MinerU Markdown artifact: {exc}",
                        affected=artifact.storage_path,
                    ) from exc
        raise CanonicalMappingError("MinerU Markdown artifact is missing.")

    @staticmethod
    def _text(item: dict[str, Any]) -> str:
        value = item.get("text")
        return value if isinstance(value, str) else ""

    def _cleanup_indexes(self, content: list[dict[str, Any]]) -> set[int]:
        margins: list[MarginText] = []
        text_rows: list[tuple[int, str]] = []
        for index, item in enumerate(content):
            kind = item.get("type")
            text = self._text(item)
            page_idx = item.get("page_idx")
            page = page_idx + 1 if isinstance(page_idx, int) else None
            if kind in {"header", "footer"}:
                margins.append(MarginText(index, str(kind), plain_text(text), page))
            if kind in {"text", "ref_text"}:
                text_rows.append((index, text))
        return repeated_margin_indexes(margins) | adjacent_duplicate_indexes(text_rows)

    def _extract_title(self, content: list[dict[str, Any]]) -> str:
        raw_title = ""
        for item in content:
            if item.get("text_level") == 1 and self._text(item).strip():
                raw_title = plain_text(self._text(item))
                break
        if not raw_title:
            for item in content:
                if item.get("type") == "text" and self._text(item).strip():
                    raw_title = plain_text(self._text(item))
                    break
        if not raw_title:
            raise CanonicalMappingError("MinerU content list contains no document title.")
        for item in content[:40]:
            text = plain_text(self._text(item))
            match = re.search(
                r"(?:19|20)\d{2}\.\s+(.+?)\.\s+"
                r"(?:ACM|IEEE|Computer|Journal|Proceedings)\b",
                text,
                flags=re.IGNORECASE,
            )
            if match:
                candidate = match.group(1).strip()
                if (
                    SequenceMatcher(None, raw_title.casefold(), candidate.casefold()).ratio()
                    >= 0.85
                ):
                    return candidate
        return raw_title

    def _frontmatter(self, content: list[dict[str, Any]]) -> str:
        values: list[str] = []
        for item in content:
            text = self._text(item)
            if item.get("text_level") is not None and _HEADING_NUMBER.match(text):
                break
            if text:
                values.append(plain_text(text))
        return "\n".join(values)

    def _extract_authors(
        self,
        content: list[dict[str, Any]],
        document_id_value: str,
        *,
        title: str,
    ) -> list[Person]:
        citation_names = self._citation_authors(content, title)
        if citation_names:
            return [
                Person(
                    id=person_id(document_id_value, name, order),
                    display_name=name,
                    name_parts=name.split(),
                    raw_name=name,
                )
                for order, name in enumerate(citation_names)
            ]
        title_seen = False
        candidate = ""
        for item in content:
            raw = self._text(item)
            text = plain_text(re.sub(r"<sup>.*?</sup>", "", raw, flags=re.IGNORECASE))
            if not text:
                continue
            if not title_seen:
                title_seen = text == title
                continue
            lowered = text.casefold()
            if item.get("text_level") is not None or lowered.startswith(
                ("abstract", "highlights", "received", "keywords")
            ):
                break
            if len(text) <= 350 and (
                "," in text or re.search(r"\s+(?:and|&)\s+", text, re.IGNORECASE)
            ):
                candidate = text
                break
        if not candidate:
            return []
        names = re.split(r"\s*,\s*|\s+(?:and|&)\s+", candidate)
        cleaned: list[str] = []
        for name in names:
            value = name.strip(" ,;*†‡")
            if 1 < len(value.split()) <= 7 and not any(char.isdigit() for char in value):
                cleaned.append(value)
        return [
            Person(
                id=person_id(document_id_value, name, order),
                display_name=name,
                name_parts=name.split(),
                raw_name=name,
            )
            for order, name in enumerate(cleaned)
        ]

    def _citation_authors(self, content: list[dict[str, Any]], title: str) -> list[str]:
        for item in content[:50]:
            text = plain_text(self._text(item))
            if not text or title.casefold() not in text.casefold():
                continue
            match = re.match(r"^(.+?)\.\s+(?:19|20)\d{2}\.\s+", text)
            if not match:
                continue
            names = [
                name.strip(" ,.")
                for name in re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", match.group(1))
                if name.strip(" ,.")
            ]
            if 1 <= len(names) <= 20 and all(1 < len(name.split()) <= 8 for name in names):
                return names
        return []

    def _extract_affiliations(self, content: list[dict[str, Any]]) -> list[str]:
        results: list[str] = []
        for item in content[:20]:
            raw = self._text(item)
            if re.match(r"^<sup>[a-z0-9]+</sup>", raw, flags=re.IGNORECASE) and len(raw) > 40:
                results.append(plain_text(raw))
        return results

    def _extract_named_block(self, content: list[dict[str, Any]], heading: str) -> str | None:
        collecting = False
        values: list[str] = []
        for item in content:
            text = plain_text(self._text(item))
            if item.get("text_level") is not None:
                if collecting:
                    break
                collecting = strip_heading_number(text).casefold() == heading.casefold()
                continue
            if collecting and text:
                if text.casefold().startswith("keywords:"):
                    break
                values.append(text)
        return "\n".join(values) or None

    def _extract_unheaded_abstract(self, content: list[dict[str, Any]]) -> str | None:
        for item in content:
            text = plain_text(self._text(item))
            if item.get("text_level") is not None and _HEADING_NUMBER.match(text):
                break
            if (
                item.get("type") == "text"
                and len(text) >= 500
                and not text.casefold().startswith(("fig.", "figure", "table"))
            ):
                return text
        return None

    def _extract_keywords(self, content: list[dict[str, Any]]) -> list[str]:
        for item in content:
            text = plain_text(self._text(item))
            lowered = text.casefold()
            if lowered.startswith("keywords:"):
                return [
                    value.strip()
                    for value in re.split(r"[;,]", text.split(":", 1)[1])
                    if value.strip()
                ]
            if lowered.startswith("additional key words and phrases:"):
                return [
                    value.strip() for value in text.split(":", 1)[1].split(",") if value.strip()
                ]
        return []

    def _extract_doi(self, content: list[dict[str, Any]], markdown: str) -> str | None:
        searchable = "\n".join(self._text(item) for item in content) + "\n" + markdown
        match = _DOI.search(searchable)
        return normalize_doi(match.group(0)) if match else None

    @staticmethod
    def _extract_year(content: list[dict[str, Any]], frontmatter: str) -> int | None:
        first_page = "\n".join(
            plain_text(str(item.get("text", ""))) for item in content if item.get("page_idx") == 0
        )
        for pattern in (
            r"publication date:[^\n]{0,60}?((?:19|20)\d{2})",
            r"©\s*((?:19|20)\d{2})",
            r"(?:accepted|available online)[^\n]{0,60}?((?:19|20)\d{2})",
        ):
            priority = re.search(pattern, first_page, flags=re.IGNORECASE)
            if priority:
                return int(priority.group(1))
        citation = re.search(r"\.\s*((?:19|20)\d{2})\.\s+", first_page)
        if citation:
            return int(citation.group(1))
        years = [int(value) for value in _YEAR.findall(frontmatter)]
        if not years:
            return None
        return max(set(years), key=lambda year: (years.count(year), year))

    @staticmethod
    def _detect_language(content: list[dict[str, Any]]) -> str:
        sample = " ".join(
            str(item.get("text", "")) for item in content[:80] if item.get("type") == "text"
        )
        ascii_letters = sum(character.isascii() and character.isalpha() for character in sample)
        letters = sum(character.isalpha() for character in sample)
        return "en" if letters and ascii_letters / letters > 0.85 else "und"

    def _sections(
        self,
        *,
        content: list[dict[str, Any]],
        artifact_id: str,
        document_id_value: str,
        snapshot_id: str,
    ) -> tuple[list[Section], dict[int, str | None]]:
        sections: list[Section] = []
        current: str | None = None
        item_sections: dict[int, str | None] = {}
        stack: list[Section] = []
        for item_index, item in enumerate(content):
            raw_title = self._text(item)
            is_document_title = item.get("text_level") == 1
            if item.get("text_level") is not None and raw_title and not is_document_title:
                match = _HEADING_NUMBER.match(plain_text(raw_title))
                level = len(match.group(1).split(".")) if match else 1
                title = strip_heading_number(raw_title)
                while stack and stack[-1].level >= level:
                    stack.pop()
                parent = stack[-1] if stack else None
                path = " / ".join([*(section.title for section in stack), title])
                page = item["page_idx"] + 1 if isinstance(item.get("page_idx"), int) else None
                span = self._source_span(artifact_id, item_index, item)
                section = Section(
                    id=section_id(document_id_value, len(sections), path),
                    document_id=document_id_value,
                    canonical_snapshot_id=snapshot_id,
                    title=title,
                    raw_title=plain_text(raw_title),
                    level=level,
                    order=len(sections),
                    path=path,
                    parent_section_id=parent.id if parent else None,
                    page_start=page,
                    page_end=page,
                    source_span=span,
                    section_type=self._section_type(title),
                )
                sections.append(section)
                stack.append(section)
                current = section.id
            item_sections[item_index] = current

        last_page = max(
            (item["page_idx"] + 1 for item in content if isinstance(item.get("page_idx"), int)),
            default=1,
        )
        finalized: list[Section] = []
        for index, section in enumerate(sections):
            next_pages = [
                other.page_start
                for other in sections[index + 1 :]
                if other.level <= section.level and other.page_start is not None
            ]
            page_end = max(
                section.page_start or 1,
                (next_pages[0] if next_pages else last_page),
            )
            finalized.append(section.model_copy(update={"page_end": page_end}))
        return finalized, item_sections

    @staticmethod
    def _section_type(title: str) -> str | None:
        normalized = title.casefold()
        for kind in (
            "abstract",
            "introduction",
            "conclusion",
            "conclusions",
            "references",
            "acknowledgment",
            "acknowledgements",
        ):
            if kind in normalized:
                return "conclusion" if kind == "conclusions" else kind
        return None

    def _elements(
        self,
        *,
        content: list[dict[str, Any]],
        artifact: ParserArtifact,
        document_id_value: str,
        snapshot_id: str,
        section_for_item: dict[int, str | None],
        drop_indexes: set[int],
    ) -> list[Element]:
        elements: list[Element] = []
        for item_index, item in enumerate(content):
            if item_index in drop_indexes or item.get("type") == "page_number":
                continue
            provider_type = str(item.get("type") or "other")
            raw_text = self._text(item)
            is_heading = item.get("text_level") is not None
            asset = item.get("img_path")
            asset_path = self._safe_asset_path(artifact, asset)
            captions = self._string_list(
                item.get("image_caption") or item.get("chart_caption") or item.get("table_caption")
            )
            footnotes = self._string_list(
                item.get("image_footnote")
                or item.get("chart_footnote")
                or item.get("table_footnote")
            )
            classification_text = captions[0] if captions else raw_text
            canonical_type = classify_element(
                provider_type,
                text=classification_text,
                is_heading=is_heading,
                has_asset=asset_path is not None,
            )
            text = plain_text(raw_text) if raw_text else None
            if canonical_type == ElementType.FORMULA:
                latex = self._latex(raw_text)
                text = latex
            else:
                latex = None
            html = item.get("table_body") if isinstance(item.get("table_body"), str) else None
            if not any((text, html, asset_path, captions, footnotes)):
                continue
            element = self._new_element(
                document_id_value=document_id_value,
                snapshot_id=snapshot_id,
                artifact_id=artifact.id,
                item_index=item_index,
                order=len(elements),
                element_type=canonical_type,
                section_id=section_for_item.get(item_index),
                raw_text=raw_text or None,
                text=text,
                latex=latex,
                html=html,
                asset_path=asset_path,
                item=item,
            )
            elements.append(element)
            caption_ids: list[str] = []
            footnote_ids: list[str] = []
            for caption_index, caption in enumerate(captions):
                caption_element = self._new_element(
                    document_id_value=document_id_value,
                    snapshot_id=snapshot_id,
                    artifact_id=artifact.id,
                    item_index=item_index,
                    order=len(elements),
                    element_type=ElementType.CAPTION,
                    section_id=section_for_item.get(item_index),
                    raw_text=caption,
                    text=plain_text(caption),
                    item=item,
                    parent_element_id=element.id,
                    metadata={"caption_index": caption_index},
                )
                elements.append(caption_element)
                caption_ids.append(caption_element.id)
            for footnote_index, footnote in enumerate(footnotes):
                footnote_element = self._new_element(
                    document_id_value=document_id_value,
                    snapshot_id=snapshot_id,
                    artifact_id=artifact.id,
                    item_index=item_index,
                    order=len(elements),
                    element_type=ElementType.FOOTNOTE,
                    section_id=section_for_item.get(item_index),
                    raw_text=footnote,
                    text=plain_text(footnote),
                    item=item,
                    parent_element_id=element.id,
                    metadata={"footnote_index": footnote_index},
                )
                elements.append(footnote_element)
                footnote_ids.append(footnote_element.id)
            if caption_ids or footnote_ids:
                elements[-(1 + len(caption_ids) + len(footnote_ids))] = element.model_copy(
                    update={
                        "caption_element_ids": caption_ids,
                        "footnote_element_ids": footnote_ids,
                    }
                )
        return elements

    def _new_element(
        self,
        *,
        document_id_value: str,
        snapshot_id: str,
        artifact_id: str,
        item_index: int,
        order: int,
        element_type: ElementType,
        section_id: str | None,
        raw_text: str | None,
        text: str | None,
        item: dict[str, Any],
        latex: str | None = None,
        html: str | None = None,
        asset_path: Path | None = None,
        parent_element_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Element:
        digest_source = "\x1f".join(
            (
                element_type.value,
                text or "",
                latex or "",
                html or "",
                str(asset_path or ""),
                str(order),
            )
        )
        digest = hashlib.sha256(digest_source.encode()).hexdigest()
        page = item.get("page_idx")
        page_number = page + 1 if isinstance(page, int) else None
        bbox = self._bbox(item.get("bbox"))
        return Element(
            id=element_id(document_id_value, order, artifact_id, item_index, digest),
            document_id=document_id_value,
            canonical_snapshot_id=snapshot_id,
            element_type=element_type,
            order=order,
            section_id=section_id,
            parent_element_id=parent_element_id,
            text=text,
            raw_text=normalize_text(raw_text) if raw_text else None,
            latex=latex,
            html=html,
            asset_path=asset_path,
            page=page_number,
            bounding_box=bbox,
            source_span=SourceSpan(
                artifact_id=artifact_id,
                item_index=item_index,
                page=page_number,
                bounding_box=bbox,
            ),
            metadata=metadata or {},
        )

    @staticmethod
    def _safe_asset_path(content_artifact: ParserArtifact, value: object) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        artifact_root = content_artifact.storage_path.parent.resolve()
        candidate = (artifact_root / value).resolve()
        try:
            candidate.relative_to(artifact_root)
        except ValueError as exc:
            raise CanonicalMappingError(
                "MinerU asset path escapes the parser artifact directory.",
                affected=value,
            ) from exc
        return candidate if candidate.is_file() else None

    @staticmethod
    def _string_list(value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [item for item in value if isinstance(item, str) and item.strip()]

    @staticmethod
    def _latex(value: str) -> str:
        text = normalize_text(value)
        text = re.sub(r"^\$\$\s*", "", text)
        return re.sub(r"\s*\$\$$", "", text).strip()

    @staticmethod
    def _bbox(value: object) -> tuple[float, float, float, float] | None:
        if (
            isinstance(value, list)
            and len(value) == 4
            and all(isinstance(item, int | float) for item in value)
        ):
            return tuple(float(item) for item in value)  # type: ignore[return-value]
        return None

    def _source_span(self, artifact_id: str, item_index: int, item: dict[str, Any]) -> SourceSpan:
        page_idx = item.get("page_idx")
        return SourceSpan(
            artifact_id=artifact_id,
            item_index=item_index,
            page=page_idx + 1 if isinstance(page_idx, int) else None,
            bounding_box=self._bbox(item.get("bbox")),
        )
