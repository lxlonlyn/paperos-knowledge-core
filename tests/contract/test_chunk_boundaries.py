"""Permanent chunk-boundary and table provenance contracts."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

from paperos_core.ingestion.chunk_dp import partition_units
from paperos_core.ingestion.sentence_units import SentenceUnit
from tests.validation.chunk import (
    boundaries__synthetic_multi_part_table_contract,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))


def _unit(text: str, *, kind: str, tokens: int = 20) -> SentenceUnit:
    return SentenceUnit(
        text=text,
        tokens=tokens,
        element_id=f"element_{kind}_{text}",
        span_key="0:1",
        character_start_in_element=0,
        character_end_in_element=1,
        token_start=0,
        token_end=tokens,
        section_id="section",
        section_path="Methods",
        page=1,
        bounding_box=None,
        paragraph_end=True,
        subsection_end=False,
        unit_kind=kind,
    )


def test_formula_lead_and_continuation_stay_together_when_they_fit() -> None:
    units = [
        _unit("The update is defined as:", kind="sentence"),
        _unit("x_{t+1}=F(x_t)", kind="formula"),
        _unit("where F is the learned flow.", kind="sentence"),
        replace(_unit("Independent discussion.", kind="sentence"), subsection_end=True),
    ]
    ranges = partition_units(
        units,
        target_tokens=50,
        hard_max_tokens=75,
        count=lambda text: len(text.split()),
    )
    assert any(start == 0 and end >= 3 for start, end in ranges)


def test_multi_part_table_provenance_contract() -> None:
    result = boundaries__synthetic_multi_part_table_contract()
    assert result["table_part_count"] >= 3
    assert result["multi_part_table_provenance_errors"] == 0, result["failures"]
