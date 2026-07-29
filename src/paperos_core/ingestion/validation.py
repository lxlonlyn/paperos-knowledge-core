"""Local PDF validation and checksum calculation for Gate 1."""

from __future__ import annotations

import hashlib
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from paperos_core.errors import (
    FileTooLargeError,
    InvalidPDFError,
    MissingSourceFileError,
    SourceChangedError,
)

PDF_MEDIA_TYPE = "application/pdf"
_HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ValidatedPDF:
    path: Path
    original_filename: str
    media_type: str
    size_bytes: int
    sha256: str


def validate_pdf_markers(header: bytes, tail: bytes) -> None:
    if not header.startswith(b"%PDF-"):
        raise InvalidPDFError("File does not start with a valid %PDF- header.")
    version = header[5:8]
    supported_versions = {*(f"1.{minor}".encode() for minor in range(8)), b"2.0"}
    if version not in supported_versions:
        raise InvalidPDFError("PDF header contains an unsupported or malformed version.")
    if b"%%EOF" not in tail:
        raise InvalidPDFError("PDF is missing the required %%EOF end marker.")


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
                digest.update(block)
    except OSError as exc:
        raise MissingSourceFileError(f"Unable to read source PDF: {exc}", affected=path) from exc
    return digest.hexdigest()


def validate_pdf(path: Path, *, max_file_mb: int) -> ValidatedPDF:
    supplied_path = path.expanduser()
    try:
        resolved = supplied_path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise MissingSourceFileError(
            "Source PDF does not exist or cannot be resolved.", affected=supplied_path
        ) from exc
    if not resolved.is_file():
        raise MissingSourceFileError("Source PDF path is not a regular file.", affected=resolved)
    if resolved.suffix.lower() != ".pdf":
        raise InvalidPDFError("Source file must have a .pdf extension.", affected=resolved)
    guessed_type, _ = mimetypes.guess_type(resolved.name)
    if guessed_type != PDF_MEDIA_TYPE:
        raise InvalidPDFError(
            f"Source file MIME type is '{guessed_type or 'unknown'}', expected application/pdf.",
            affected=resolved,
        )

    before = resolved.stat()
    if before.st_size <= 0:
        raise InvalidPDFError("Source PDF is empty.", affected=resolved)
    max_bytes = max_file_mb * 1024 * 1024
    if before.st_size > max_bytes:
        raise FileTooLargeError(
            f"Source PDF is {before.st_size} bytes; configured limit is {max_bytes} bytes.",
            affected=resolved,
            details={"size_bytes": before.st_size, "max_bytes": max_bytes},
        )

    try:
        with resolved.open("rb") as stream:
            header = stream.read(16)
            stream.seek(max(0, before.st_size - 8192))
            tail = stream.read()
    except OSError as exc:
        raise MissingSourceFileError(
            f"Unable to inspect source PDF: {exc}", affected=resolved
        ) from exc
    try:
        validate_pdf_markers(header, tail)
    except InvalidPDFError as exc:
        exc.affected = str(resolved)
        raise

    sha256 = calculate_sha256(resolved)
    after = resolved.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
    ):
        raise SourceChangedError(
            "Source PDF changed while it was being validated; retry with a stable file.",
            affected=resolved,
        )
    return ValidatedPDF(
        path=resolved,
        original_filename=resolved.name,
        media_type=PDF_MEDIA_TYPE,
        size_bytes=before.st_size,
        sha256=sha256,
    )
