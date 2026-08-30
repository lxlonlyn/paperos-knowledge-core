"""Central runtime path resolver used by every PaperOS component."""

from __future__ import annotations

from dataclasses import dataclass
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
    tmp: Path
    registry_filename: str = "registry.sqlite3"
    lexical_filename: str = "lexical.sqlite3"

    @property
    def registry_db(self) -> Path:
        return self.jobs / self.registry_filename

    @property
    def lexical_db(self) -> Path:
        return self.indexes / self.lexical_filename

    def runtime_directories(self) -> tuple[Path, ...]:
        return (
            self.raw,
            self.parsed,
            self.canonical,
            self.cognee,
            self.indexes,
            self.models,
            self.jobs,
            self.cache,
            self.exports,
            self.logs,
            self.tmp,
        )

    def initialize(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for path in self.runtime_directories():
            path.mkdir(parents=True, exist_ok=True)
        for name in ("system", "data", "vector", "graph"):
            (self.cognee / name).mkdir(parents=True, exist_ok=True)

    def assert_within_root(self, path: Path) -> None:
        try:
            path.resolve(strict=False).relative_to(self.root)
        except ValueError as exc:
            raise ConfigurationError(
                "Runtime path escapes the configured PaperOS data directory.",
                affected=path,
                details={"data_dir": str(self.root)},
            ) from exc


def build_data_paths(
    root: Path,
    *,
    registry_filename: str = "registry.sqlite3",
    lexical_filename: str = "lexical.sqlite3",
) -> DataPaths:
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
        tmp=resolved / "tmp",
        registry_filename=registry_filename,
        lexical_filename=lexical_filename,
    )
    for child in paths.runtime_directories():
        paths.assert_within_root(child)
    for database, directory in (
        (paths.registry_db, paths.jobs),
        (paths.lexical_db, paths.indexes),
    ):
        if database.parent != directory:
            raise ConfigurationError(
                "Configured database filename must stay within its storage directory.",
                affected=database,
            )
    return paths


def initialize_data_paths(paths: DataPaths) -> DataPaths:
    paths.initialize()
    return paths
