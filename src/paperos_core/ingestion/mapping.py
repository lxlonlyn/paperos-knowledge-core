"""Stable canonical mapping interface.

The concrete MinerU field interpretation lives only in
``paperos_core.adapters.mineru.mapper``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from paperos_core.domain.canonical import CanonicalBundle
from paperos_core.domain.documents import SourceFile
from paperos_core.domain.parsing import ParserArtifact, ParseRun


class CanonicalMapper(Protocol):
    def build_canonical_snapshot(
        self,
        *,
        source: SourceFile,
        parse_run: ParseRun,
        artifacts: list[ParserArtifact],
        manifest_path: Path,
        dataset_id: str | None = None,
    ) -> CanonicalBundle: ...
