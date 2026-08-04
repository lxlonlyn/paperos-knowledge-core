"""Central runtime path resolver used by every PaperOS component."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from paperos_core.errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class DataPaths:
    root: Path
    raw: Path
    parsed: Path
    canonical: Path
    cognee: Path
    indexes: Path
    models: Path
    jobs: Path
    cache: Path
    exports: Path
    logs: Path
    test_corpus: Path
    test_runs: Path
    tmp: Path

    @property
    def registry_db(self) -> Path:
        return self.jobs / "registry.sqlite3"

    def runtime_directories(self) -> tuple[Path, ...]:
        return tuple(getattr(self, field.name) for field in fields(self) if field.name != "root")

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for path in self.runtime_directories():
            path.mkdir(parents=True, exist_ok=True)

    def assert_within_root(self, path: Path) -> None:
        try:
            path.resolve(strict=False).relative_to(self.root)
        except ValueError as exc:
            raise ConfigurationError(
                "Runtime path escapes the configured PaperOS data directory.",
                affected=path,
                details={"data_dir": str(self.root)},
            ) from exc


def build_data_paths(root: Path) -> DataPaths:
    resolved = root.expanduser().resolve(strict=False)
    paths = DataPaths(
        root=resolved,
        raw=resolved / "raw",
        parsed=resolved / "parsed",
        canonical=resolved / "canonical",
        cognee=resolved / "cognee",
        indexes=resolved / "indexes",
        models=resolved / "models",
        jobs=resolved / "jobs",
        cache=resolved / "cache",
        exports=resolved / "exports",
        logs=resolved / "logs",
        test_corpus=resolved / "test-corpus",
        test_runs=resolved / "test-runs",
        tmp=resolved / "tmp",
    )
    for child in paths.runtime_directories():
        paths.assert_within_root(child)
    return paths


def initialize_data_paths(paths: DataPaths) -> DataPaths:
    paths.initialize()
    return paths
