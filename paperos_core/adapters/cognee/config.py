"""Narrow translation from PaperOS settings to Cognee's process environment."""

from __future__ import annotations

import os

from paperos_core.config import CogneeSettings


def configure_cognee(settings: CogneeSettings) -> None:
    """Apply all and only the Cognee environment translation in one place."""

    cognee_root = settings.system_database.parent.parent
    local = settings.local_inference
    llm = settings.llm
    os.environ.update(
        {
            "SYSTEM_ROOT_DIRECTORY": str(settings.system_database.parent),
            "DATA_ROOT_DIRECTORY": str(cognee_root / "data"),
            "CACHE_ROOT_DIRECTORY": str(cognee_root / "cache"),
            "COGNEE_LOGS_DIR": str(cognee_root / "logs"),
            "DB_PROVIDER": settings.db_provider,
            "DB_PATH": str(settings.system_database),
            "DB_NAME": "cognee_db",
            "VECTOR_DB_PROVIDER": settings.vector_provider,
            "VECTOR_DB_URL": str(settings.vector_database),
            "GRAPH_DATABASE_PROVIDER": settings.graph_provider,
            "GRAPH_DATASET_DATABASE_HANDLER": settings.graph_provider,
            "GRAPH_FILE_PATH": str(settings.graph_database),
            "LLM_PROVIDER": llm.provider,
            "LLM_MODEL": llm.model,
            "LLM_ENDPOINT": llm.endpoint.rstrip("/"),
            "LLM_API_KEY": llm.api_key_value() or "",
            "EMBEDDING_PROVIDER": "openai_compatible",
            "EMBEDDING_MODEL": local.embedding.model,
            "EMBEDDING_ENDPOINT": (
                f"http://{local.host}:{local.port}/v1"
            ),
            "EMBEDDING_API_KEY": "paperos-internal",
            "EMBEDDING_DIMENSIONS": str(local.embedding.dimensions),
            "EMBEDDING_MAX_COMPLETION_TOKENS": str(local.embedding.max_tokens),
            "EMBEDDING_BATCH_SIZE": "5",
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
