"""Portable creation of checksum-validated immutable filesystem artifacts."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

_COPY_CHUNK_SIZE = 1024 * 1024


class ImmutableConflictError(Exception):
    """The target name is already bound to different or invalid content."""

    def __init__(self, target: Path) -> None:
        super().__init__(f"Immutable target already exists with different content: {target}")
        self.target = target


class ImmutableSourceChangedError(Exception):
    """A source changed between validation and immutable publication."""

    def __init__(self, source: Path) -> None:
        super().__init__(f"Source changed while being copied: {source}")
        self.source = source


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_COPY_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_target(target: Path, *, expected_size: int, expected_sha256: str) -> None:
    if (
        target.is_symlink()
        or not target.is_file()
        or target.stat().st_size != expected_size
        or _sha256(target) != expected_sha256
    ):
        raise ImmutableConflictError(target)


def _publish(
    temporary_name: str,
    target: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    try:
        os.link(temporary_name, target)
    except FileExistsError:
        _validate_target(
            target,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )


def write_immutable_bytes(target: Path, content: bytes) -> None:
    """Create ``target`` once, or validate an identical existing artifact."""

    expected_sha256 = hashlib.sha256(content).hexdigest()
    expected_size = len(content)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        _validate_target(
            target,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        return

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".immutable-",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        _publish(
            temporary_name,
            target,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def copy_immutable_file(
    source: Path,
    target: Path,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    """Copy a validated source without overwriting an immutable target."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        _validate_target(
            target,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
        return

    temporary_name: str | None = None
    try:
        digest = hashlib.sha256()
        copied = 0
        with (
            source.open("rb") as source_stream,
            tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".immutable-",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as temporary,
        ):
            temporary_name = temporary.name
            for block in iter(lambda: source_stream.read(_COPY_CHUNK_SIZE), b""):
                temporary.write(block)
                digest.update(block)
                copied += len(block)
            temporary.flush()
            os.fsync(temporary.fileno())
        if copied != expected_size or digest.hexdigest() != expected_sha256:
            raise ImmutableSourceChangedError(source)
        _publish(
            temporary_name,
            target,
            expected_size=expected_size,
            expected_sha256=expected_sha256,
        )
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


__all__ = [
    "ImmutableConflictError",
    "ImmutableSourceChangedError",
    "copy_immutable_file",
    "write_immutable_bytes",
]
