"""Academic chunker behavior: sections, spans, tables, formulas, overlap."""

from __future__ import annotations

import re

from paperos_core.domain.canonical import Element, Section, SourceSpan
from paperos_core.domain.enums import ElementType
from paperos_core.domain.ids import chunk_id
from paperos_core.ingestion.chunking import build_chunks, resolve_cognee_tokenizer


class _CountingTokenizer:
    """Deterministic tokenizer: Latin words plus CJK characters each count once."""

    def count_tokens(self, text: str) -> int:
        words = len(text.split())
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
        return max(1, words + cjk)


TOKENIZER = _CountingTokenizer()


def _element(
    element_id: str,
    element_type: ElementType,
    order: int,
    *,
    text: str | None = None,
    section_id: str | None = None,
    latex: str | None = None,
    markdown: str | None = None,
    page: int = 1,
    caption_element_ids: list[str] | None = None,
) -> Element:
    return Element(
        id=element_id,
        document_id="doc_test",
        canonical_snapshot_id="snapshot_test",
        element_type=element_type,
        order=order,
        section_id=section_id,
        text=text,
        latex=latex,
        markdown=markdown,
        page=page,
        bounding_box=(0.0, 0.0, 100.0, 50.0),
        caption_element_ids=caption_element_ids or [],
        source_span=SourceSpan(artifact_id="artifact", item_index=order, page=page),
    )


def _section(section_id: str, order: int, title: str) -> Section:
    return Section(
        id=section_id,
        document_id="doc_test",
        canonical_snapshot_id="snapshot_test",
        title=title,
        level=1,
        order=order,
        path=f"/{title}",
    )


def test_sections_are_never_crossed() -> None:
    intro = _section("section_intro", 0, "Introduction")
    methods = _section("section_methods", 1, "Methods")
    elements = [
        _element(
            "element_a",
            ElementType.PARAGRAPH,
            0,
            text="alpha " * 60,
            section_id=intro.id,
        ),
        _element(
            "element_b",
            ElementType.PARAGRAPH,
            1,
            text="beta " * 60,
            section_id=methods.id,
        ),
    ]
    chunks = build_chunks(
        document_id="doc_test",
        snapshot_id="snapshot_test",
        sections=[intro, methods],
        elements=elements,
        target_tokens=20,
        overlap_tokens=0,
        tokenizer=TOKENIZER,
    )
    assert len(chunks) > 2
    for chunk in chunks:
        assert chunk.section_id in {intro.id, methods.id}
        span_sections = {
            element.section_id
            for element in elements
            if element.id in chunk.element_ids
        }
        assert span_sections <= {chunk.section_id}


def test_two_short_sections_are_never_merged() -> None:
    first = _section("section_first", 0, "First")
    second = _section("section_second", 1, "Second")
    elements = [
        _element(
            f"first_{index}",
            ElementType.PARAGRAPH,
            index,
            text="short ",
            section_id=first.id,
        )
        for index in range(2)
    ] + [
        _element(
            f"second_{index}",
            ElementType.PARAGRAPH,
            index,
            text="tiny ",
            section_id=second.id,
        )
        for index in range(2)
    ]
    chunks = build_chunks(
        document_id="doc_test",
        snapshot_id="snapshot_test",
        sections=[first, second],
        elements=elements,
        target_tokens=100,
        overlap_tokens=0,
        tokenizer=TOKENIZER,
    )
    assert len(chunks) == 2
    assert {chunk.section_id for chunk in chunks} == {first.id, second.id}
    assert not any(
        first.id in chunk.element_ids and second.id in chunk.element_ids
        for chunk in chunks
    )


def test_overlap_never_exceeds_target_token_limit() -> None:
    section = _section("section_intro", 0, "Introduction")
    elements = [
        _element(
            f"element_{index}",
            ElementType.PARAGRAPH,
            index,
            text="word " * 4,
            section_id=section.id,
        )
        for index in range(14)
    ]
    chunks = build_chunks(
        document_id="doc_test",
        snapshot_id="snapshot_test",
        sections=[section],
        elements=elements,
        target_tokens=16,
        overlap_tokens=6,
        tokenizer=TOKENIZER,
    )
    assert len(chunks) > 2
    assert all(chunk.token_count and chunk.token_count <= 16 for chunk in chunks)
    for chunk in chunks:
        assert len(chunk.element_span_ids) == len(set(chunk.element_span_ids))
        for span_id in chunk.overlap_element_span_ids:
            assert span_id in chunk.element_span_ids


def test_oversized_element_is_split_into_span_identified_chunks() -> None:
    section = _section("section_intro", 0, "Introduction")
    element = _element(
        "element_long",
        ElementType.PARAGRAPH,
        0,
        text=("sentence " * 400).strip(),
        section_id=section.id,
    )
    chunks = build_chunks(
        document_id="doc_test",
        snapshot_id="snapshot_test",
        sections=[section],
        elements=[element],
        target_tokens=30,
        overlap_tokens=0,
        tokenizer=TOKENIZER,
    )
    assert len(chunks) > 1
    assert all(chunk.token_count and chunk.token_count <= 30 for chunk in chunks)
    all_span_ids = [
        span_id for chunk in chunks for span_id in chunk.element_span_ids
    ]
    assert len(all_span_ids) == len(set(all_span_ids))
    assert all(span_id.startswith("element_long:") for span_id in all_span_ids)
    assert all(chunk.element_ids == ["element_long"] for chunk in chunks)
    assert all(chunk.page_start == 1 and chunk.page_end == 1 for chunk in chunks)
    assert all(chunk.bounding_box == (0.0, 0.0, 100.0, 50.0) for chunk in chunks)


def test_table_produces_searchable_text() -> None:
    section = _section("section_results", 0, "Results")
    element = _element(
        "element_table",
        ElementType.TABLE,
        0,
        markdown="| col | value |\n| --- | --- |\n| a | 1 |",
        section_id=section.id,
    )
    chunks = build_chunks(
        document_id="doc_test",
        snapshot_id="snapshot_test",
        sections=[section],
        elements=[element],
        target_tokens=200,
        overlap_tokens=0,
        tokenizer=TOKENIZER,
    )
    assert chunks and chunks[0].text.startswith("[Table]")
    assert "| col | value |" in chunks[0].text


def test_formula_includes_caption_and_section_context() -> None:
    section = _section("section_math", 0, "Math")
    caption = _element(
        "element_caption",
        ElementType.CAPTION,
        0,
        text="Equation 1: the loss function.",
        section_id=section.id,
    )
    formula = _element(
        "element_formula",
        ElementType.FORMULA,
        1,
        latex="L = \\sum_i (y_i - \\hat{y}_i)^2",
        section_id=section.id,
        caption_element_ids=[caption.id],
    )
    chunks = build_chunks(
        document_id="doc_test",
        snapshot_id="snapshot_test",
        sections=[section],
        elements=[caption, formula],
        target_tokens=200,
        overlap_tokens=0,
        tokenizer=TOKENIZER,
    )
    texts = " ".join(chunk.text for chunk in chunks)
    assert "[Formula]" in texts
    assert "L = \\sum_i" in texts
    assert "Caption: Equation 1" in texts
    assert "Section: Math" in texts


def test_table_formula_and_oversized_paragraph_respect_token_limits() -> None:
    section = _section("section_mixed", 0, "Mixed")
    elements = [
        _element(
            "element_table",
            ElementType.TABLE,
            0,
            markdown="| a | b |\n| --- | --- |\n| 1 | 2 |",
            section_id=section.id,
        ),
        _element(
            "element_formula",
            ElementType.FORMULA,
            1,
            latex="E = mc^2",
            section_id=section.id,
        ),
        _element(
            "element_long",
            ElementType.PARAGRAPH,
            2,
            text=("long " * 300).strip(),
            section_id=section.id,
        ),
    ]
    chunks = build_chunks(
        document_id="doc_test",
        snapshot_id="snapshot_test",
        sections=[section],
        elements=elements,
        target_tokens=20,
        overlap_tokens=4,
        tokenizer=TOKENIZER,
    )
    assert chunks
    assert all(chunk.token_count and chunk.token_count <= 20 for chunk in chunks)
    all_text = " ".join(chunk.text for chunk in chunks)
    assert "[Table]" in all_text
    assert "[Formula]" in all_text
    assert "E = mc^2" in all_text


def test_overlap_records_exact_source_spans() -> None:
    section = _section("section_intro", 0, "Introduction")
    elements = [
        _element(
            f"element_{index}",
            ElementType.PARAGRAPH,
            index,
            text=f"paragraph {index} " + ("word " * 40),
            section_id=section.id,
        )
        for index in range(6)
    ]
    chunks = build_chunks(
        document_id="doc_test",
        snapshot_id="snapshot_test",
        sections=[section],
        elements=elements,
        target_tokens=25,
        overlap_tokens=8,
        tokenizer=TOKENIZER,
    )
    overlapped = [chunk for chunk in chunks if chunk.overlap_source_chunk_ids]
    assert overlapped
    for chunk in overlapped:
        previous_id = chunk.overlap_source_chunk_ids[0]
        assert previous_id in {item.id for item in chunks}
        assert chunk.overlap_element_span_ids
        for span_id in chunk.overlap_element_span_ids:
            previous = next(item for item in chunks if item.id == previous_id)
            assert span_id in previous.element_span_ids


def test_chunk_id_embeds_element_span() -> None:
    assert chunk_id("doc", 0, ["element_a:0"]) != chunk_id("doc", 0, ["element_a:1"])
    assert chunk_id("doc", 0, ["element_a:0", "element_b:2"]) != chunk_id(
        "doc", 0, ["element_b:2", "element_a:0"]
    )


def test_chunks_are_deterministic_across_runs() -> None:
    section = _section("section_intro", 0, "Introduction")
    elements = [
        _element(
            f"element_{index}",
            ElementType.PARAGRAPH,
            index,
            text=("word " * 35),
            section_id=section.id,
        )
        for index in range(4)
    ]
    first = build_chunks(
        document_id="doc_test",
        snapshot_id="snapshot_test",
        sections=[section],
        elements=elements,
        target_tokens=20,
        overlap_tokens=4,
        tokenizer=TOKENIZER,
    )
    second = build_chunks(
        document_id="doc_test",
        snapshot_id="snapshot_test",
        sections=[section],
        elements=elements,
        target_tokens=20,
        overlap_tokens=4,
        tokenizer=TOKENIZER,
    )
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert [chunk.text for chunk in first] == [chunk.text for chunk in second]


def test_cognee_tokenizer_resolves_without_raising() -> None:
    tokenizer = resolve_cognee_tokenizer()
    assert callable(tokenizer.count_tokens)
    assert isinstance(tokenizer.count_tokens("hello world"), int)
    assert tokenizer.count_tokens("hello world") > 0
