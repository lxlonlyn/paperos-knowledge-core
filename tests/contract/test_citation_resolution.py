"""Deterministic citation resolution contract tests."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from paperos_core.domain.canonical import ReferenceEntry
from paperos_core.ingestion.citations import (
    build_reference_indexes,
    expand_atom,
    extract_citation_mentions_from_text,
    extract_reference_label,
    resolve_bracket,
)


def _require(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _reference(order: int, raw_text: str) -> ReferenceEntry:
    label, label_kind = extract_reference_label(raw_text)
    ref_num = int(label) if label and label.isdigit() else None
    return ReferenceEntry(
        id=f"reference_{order}",
        document_id="doc_test",
        canonical_snapshot_id="snapshot_test",
        raw_text=raw_text,
        order=order,
        year=2020 if "2020" in raw_text else 2007 if "2007" in raw_text else None,
        citation_label=label,
        parsed_fields={
            "citation_label": label,
            "label_kind": label_kind,
            "reference_number": ref_num,
        },
    )


def test_numeric_label_group_and_range() -> None:
    references = [
        _reference(0, "[1] First paper."),
        _reference(1, "[2] Second paper."),
        _reference(2, "[3] Third paper."),
        _reference(3, "[4] Fourth paper."),
        _reference(4, "[5] Fifth paper."),
    ]
    indexes = build_reference_indexes(references)
    _require(resolve_bracket("1", indexes) is not None, "single numeric label")
    group = resolve_bracket("1, 2", indexes)
    _require(group is not None and len(group) == 2, "[1,2] group")
    ranged = resolve_bracket("1-3", indexes)
    _require(ranged is not None and len(ranged) == 3, "[1-3] hyphen range")
    en_dash = resolve_bracket("2–4", indexes)
    _require(en_dash is not None and len(en_dash) == 3, "[2–4] en-dash range")
    _require(expand_atom("3-5") == ["3", "4", "5"], "expand_atom range")


def test_symbolic_labels() -> None:
    references = [
        _reference(0, "[DC98] Desbrun et al."),
        _reference(1, "[Hec02] Heckbert."),
        _reference(2, "[ABC+21] Foo."),
    ]
    indexes = build_reference_indexes(references)
    _require(resolve_bracket("DC98", indexes) is not None, "symbolic single")
    group = resolve_bracket("DC98, Hec02", indexes)
    _require(group is not None and len(group) == 2, "symbolic group")


def test_author_year_in_brackets() -> None:
    references = [
        ReferenceEntry(
            id="reference_ay",
            document_id="doc_test",
            canonical_snapshot_id="snapshot_test",
            raw_text="Olga Sorkine and Daniel Alexa. 2007. As-rigid-as-possible surface modeling.",
            order=0,
            year=2007,
            citation_label=None,
            parsed_fields={},
        ),
        ReferenceEntry(
            id="reference_dinh",
            document_id="doc_test",
            canonical_snapshot_id="snapshot_test",
            raw_text="Laurent Dinh, David Krueger, and Yoshua Bengio. 2015. NICE.",
            order=1,
            year=2015,
            citation_label=None,
            parsed_fields={},
        ),
    ]
    indexes = build_reference_indexes(references)
    mentions = extract_citation_mentions_from_text(
        document_id="doc_test",
        snapshot_id="snapshot_test",
        element_id="element_test",
        text="Following [Sorkine and Alexa 2007], we extend [Dinh et al. 2015].",
        reference_index=indexes,
    )
    _require(len(mentions) == 2, "two author-year bracket mentions")
    _require(all(item.reference_entry_id for item in mentions), "all resolved")


def test_ocred_symbolic_citation_inside_latex_math() -> None:
    reference = _reference(0, "[LWJ∗22] Liu et al. Learning smooth neural functions.")
    indexes = build_reference_indexes([reference])
    surface = r"$\mathrm { [ L W J ^ { * } } 2 2 ]$"
    text = f"Liu et al. {surface} propose Lipschitz regularization."
    mentions = extract_citation_mentions_from_text(
        document_id="doc_test",
        snapshot_id="snapshot_test",
        element_id="element_test",
        text=text,
        reference_index=indexes,
    )
    _require(len(mentions) == 1, "one OCR-spaced LaTeX citation mention")
    mention = mentions[0]
    _require(mention.reference_entry_id == reference.id, "symbolic label resolved")
    _require(mention.resolution_status == "resolved", "citation resolution status")
    _require(mention.surface_text == surface, "source surface is preserved")
    _require(
        text[mention.character_start : mention.character_end] == surface,
        "source coordinates are exact",
    )

    formula_mentions = extract_citation_mentions_from_text(
        document_id="doc_test",
        snapshot_id="snapshot_test",
        element_id="element_formula",
        text=r"The tensor is $\mathrm{[X]} = A^2$.",
        reference_index=indexes,
    )
    _require(not formula_mentions, "ordinary inline math is not a citation")


def main() -> None:
    test_numeric_label_group_and_range()
    test_symbolic_labels()
    test_author_year_in_brackets()
    test_ocred_symbolic_citation_inside_latex_math()
    print("PASS citation resolution contracts")


if __name__ == "__main__":
    main()
