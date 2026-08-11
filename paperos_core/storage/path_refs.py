"""Portable references for files owned by one PaperOS data directory."""

from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath

from paperos_core.errors import ConfigurationError


class DataPathCodec:
    """Translate runtime absolute paths to portable data-root references."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.expanduser().resolve(strict=False)

    def encode(self, path: Path) -> str:
        """Return a POSIX reference for an absolute path inside ``data_root``."""

        candidate = path.expanduser().resolve(strict=False)
        try:
            relative = candidate.relative_to(self.data_root)
        except ValueError as exc:
            raise ConfigurationError(
                "Persistent path is outside the configured PaperOS data directory.",
                affected=path,
                details={"data_root": str(self.data_root)},
            ) from exc
        value = PurePosixPath(*relative.parts).as_posix()
        if value in {"", "."}:
            raise ConfigurationError(
                "The data root itself cannot be stored as a file reference.",
                affected=path,
            )
        return value

    def decode(self, value: str) -> Path:
        """Resolve a persisted reference and reject absolute or escaping input."""

        if not value or "\x00" in value or "\\" in value:
            raise ConfigurationError(
                "Persistent data path must be a non-empty POSIX reference.",
                affected=value,
            )
        portable = PurePosixPath(value)
        windows = PureWindowsPath(value)
        if portable.is_absolute() or windows.is_absolute() or windows.drive:
            raise ConfigurationError(
                "Persistent data path must not be absolute.", affected=value
            )
        if any(part in {"", ".", ".."} for part in portable.parts):
            raise ConfigurationError(
                "Persistent data path contains an unsafe segment.", affected=value
            )
        resolved = (self.data_root / Path(*portable.parts)).resolve(strict=False)
        try:
            resolved.relative_to(self.data_root)
        except ValueError as exc:
            raise ConfigurationError(
                "Persistent data path escapes the configured data directory.",
                affected=value,
            ) from exc
        return resolved


__all__ = ["DataPathCodec"]
