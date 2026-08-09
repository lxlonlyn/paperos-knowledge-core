"""Versioned prompt loading from the repository's sole prompt source."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from paperos_core.errors import ConfigurationError
from paperos_core.locations import PROMPTS_ROOT

_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
_VERSION = re.compile(r"<!--\s*prompt-version:\s*([^\s]+)\s*-->")


@dataclass(frozen=True, slots=True)
class PromptDescriptor:
    name: str
    version: str
    sha256: str
    text: str


class PromptRepository:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (
            root
            if root is not None
            else PROMPTS_ROOT
        ).resolve(strict=False)

    def load(self, name: str) -> str:
        return self.describe(name).text

    def describe(self, name: str) -> PromptDescriptor:
        if not _NAME.fullmatch(name):
            raise ConfigurationError("Invalid prompt name.", affected=name)
        path = (self.root / f"{name}.md").resolve(strict=False)
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ConfigurationError("Prompt path escapes prompt root.", affected=path) from exc
        if not path.is_file():
            raise ConfigurationError("Required prompt file is missing.", affected=path)
        content = path.read_text(encoding="utf-8")
        match = _VERSION.search(content)
        if match is None:
            raise ConfigurationError(
                "Prompt file is missing its prompt-version marker.", affected=path
            )
        text = _VERSION.sub("", content, count=1).strip()
        if not text:
            raise ConfigurationError("Prompt text is empty.", affected=path)
        return PromptDescriptor(
            name=name,
            version=match.group(1),
            sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            text=text,
        )
