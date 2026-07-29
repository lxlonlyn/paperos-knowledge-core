"""Central canonical element classification rules."""

from __future__ import annotations

import re

from paperos_core.domain.enums import ElementType
from paperos_core.ingestion.normalization import normalized_match_text

CLASSIFICATION_VERSION = "1"


def classify_element(
    provider_type: str,
    *,
    text: str = "",
    is_heading: bool = False,
    has_asset: bool = False,
) -> ElementType:
    """Map a provider label to the stable canonical vocabulary.

    This function deliberately receives only neutral scalar features; provider field
    inspection stays in the adapter mapper.
    """
    if is_heading:
        return ElementType.TITLE
    normalized_type = provider_type.casefold()
    if normalized_type == "table":
        return ElementType.TABLE
    if normalized_type in {"equation", "formula", "equation_interline"}:
        return ElementType.FORMULA
    if normalized_type in {"image", "figure", "chart"} or has_asset:
        if re.match(r"^\s*table\b", normalized_match_text(text)):
            return ElementType.TABLE
        return ElementType.FIGURE
    if normalized_type in {"ref_text", "reference"}:
        return ElementType.REFERENCE
    if normalized_type in {"page_footnote", "footnote"}:
        return ElementType.FOOTNOTE
    if normalized_type == "header":
        return ElementType.HEADER
    if normalized_type == "footer":
        return ElementType.FOOTER
    if normalized_type == "page_number":
        return ElementType.PAGE_NUMBER
    if normalized_type == "code":
        return ElementType.CODE
    if normalized_type in {"list", "list_item"}:
        return ElementType(normalized_type)
    if normalized_type in {"text", "paragraph", "aside_text"}:
        if re.match(r"^\s*(?:[-•]|\d+[.)])\s+", text):
            return ElementType.LIST_ITEM
        return ElementType.PARAGRAPH
    return ElementType.OTHER
