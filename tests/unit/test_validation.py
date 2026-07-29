import pytest

from paperos_core.errors import FileTooLargeError, InvalidPDFError
from paperos_core.ingestion.validation import validate_pdf, validate_pdf_markers


def test_real_pdf_validation_and_checksum(real_pdf_case) -> None:
    pdf_path, case = real_pdf_case
    validated = validate_pdf(pdf_path, max_file_mb=200)
    assert validated.sha256 == case["sha256"]
    assert validated.size_bytes == case["bytes"]
    assert validated.media_type == "application/pdf"


def test_pdf_marker_validation_errors_are_explicit() -> None:
    with pytest.raises(InvalidPDFError, match="%PDF-"):
        validate_pdf_markers(b"not a PDF", b"%%EOF")
    with pytest.raises(InvalidPDFError, match="%%EOF"):
        validate_pdf_markers(b"%PDF-1.7\n", b"missing marker")


def test_real_pdf_size_limit_is_enforced(real_pdf_case) -> None:
    pdf_path, _ = real_pdf_case
    with pytest.raises(FileTooLargeError) as raised:
        validate_pdf(pdf_path, max_file_mb=1)
    assert raised.value.code == "pdf_too_large"
    assert raised.value.details["max_bytes"] == 1024 * 1024
