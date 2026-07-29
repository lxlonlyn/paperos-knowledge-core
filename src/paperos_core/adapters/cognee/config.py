"""Single Cognee/DeepSeek configuration boundary backed by the project `.env`."""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, SecretStr

from paperos_core.errors import CogneeConfigurationError
from paperos_core.paths import DataPaths


class CogneeRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    llm_provider: str
    llm_model: str
    llm_endpoint: str
    llm_api_key: SecretStr
    embedding_provider: str
    embedding_model: str
    embedding_endpoint: str
    embedding_dimensions: int
    db_provider: str
    vector_db_provider: str
    graph_database_provider: str
    env_path: Path


def _read_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise CogneeConfigurationError("Cognee .env file does not exist.", affected=path)
    values: dict[str, str] = {}
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    except OSError as exc:
        raise CogneeConfigurationError(f"Unable to read Cognee .env: {exc}", affected=path) from exc
    return values


def configure_cognee_environment(paths: DataPaths, *, env_path: Path) -> CogneeRuntimeConfig:
    values = _read_env(env_path)
    required = (
        "DB_PROVIDER",
        "VECTOR_DB_PROVIDER",
        "GRAPH_DATABASE_PROVIDER",
        "LLM_PROVIDER",
        "LLM_MODEL",
        "LLM_ENDPOINT",
        "LLM_API_KEY",
        "EMBEDDING_PROVIDER",
        "EMBEDDING_MODEL",
        "EMBEDDING_ENDPOINT",
        "EMBEDDING_DIMENSIONS",
    )
    missing = [key for key in required if not (os.getenv(key) or values.get(key))]
    if missing:
        raise CogneeConfigurationError(
            "Cognee configuration is missing required values: " + ", ".join(missing),
            affected=env_path,
        )
    for key, value in values.items():
        if not os.getenv(key):
            os.environ[key] = value
    _apply_runtime_overrides(paths)
    paths.cognee.mkdir(parents=True, exist_ok=True)
    return CogneeRuntimeConfig(
        llm_provider=os.environ["LLM_PROVIDER"],
        llm_model=os.environ["LLM_MODEL"],
        llm_endpoint=os.environ["LLM_ENDPOINT"].rstrip("/"),
        llm_api_key=SecretStr(os.environ["LLM_API_KEY"]),
        embedding_provider=os.environ["EMBEDDING_PROVIDER"],
        embedding_model=os.environ["EMBEDDING_MODEL"],
        embedding_endpoint=os.environ["EMBEDDING_ENDPOINT"].rstrip("/"),
        embedding_dimensions=int(os.environ["EMBEDDING_DIMENSIONS"]),
        db_provider=os.environ["DB_PROVIDER"],
        vector_db_provider=os.environ["VECTOR_DB_PROVIDER"],
        graph_database_provider=os.environ["GRAPH_DATABASE_PROVIDER"],
        env_path=env_path,
    )


def reassert_cognee_runtime(paths: DataPaths) -> None:
    """Undo Cognee's import-time `.env` override and clear cached configurations."""
    _apply_runtime_overrides(paths)
    from cognee.base_config import get_base_config  # type: ignore[import-untyped]
    from cognee.infrastructure.databases.graph.config import (  # type: ignore[import-untyped]
        get_graph_config,
    )
    from cognee.infrastructure.databases.relational.config import (  # type: ignore[import-untyped]
        get_relational_config,
    )
    from cognee.infrastructure.databases.vector.config import (  # type: ignore[import-untyped]
        get_vectordb_config,
    )
    from cognee.infrastructure.databases.vector.embeddings.config import (  # type: ignore[import-untyped]
        get_embedding_config,
    )

    get_base_config.cache_clear()
    get_graph_config.cache_clear()
    get_relational_config.cache_clear()
    get_vectordb_config.cache_clear()
    get_embedding_config.cache_clear()


def _apply_runtime_overrides(paths: DataPaths) -> None:
    os.environ.update(
        {
            "SYSTEM_ROOT_DIRECTORY": str(paths.cognee / "system"),
            "DATA_ROOT_DIRECTORY": str(paths.cognee / "data"),
            "CACHE_ROOT_DIRECTORY": str(paths.cognee / "cache"),
            "COGNEE_LOGS_DIR": str(paths.logs / "cognee"),
            "VECTOR_DB_URL": str(paths.cognee / "vector"),
            "ENABLE_BACKEND_ACCESS_CONTROL": "false",
            "REQUIRE_AUTHENTICATION": "false",
            "TELEMETRY_DISABLED": "true",
            "NO_PROXY": _no_proxy(os.getenv("NO_PROXY", "")),
            "no_proxy": _no_proxy(os.getenv("no_proxy", "")),
        }
    )


def _no_proxy(existing: str) -> str:
    entries = [item.strip() for item in existing.split(",") if item.strip()]
    for value in ("127.0.0.1", "localhost"):
        if value not in entries:
            entries.append(value)
    return ",".join(entries)
