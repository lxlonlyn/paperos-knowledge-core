"""Apply PaperOS TOML settings to Cognee's public runtime configuration API."""

from __future__ import annotations

import os
from dataclasses import dataclass
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

from paperos_core.config import RuntimeSettings
from paperos_core.paths import DataPaths


@dataclass(frozen=True, slots=True)
class CogneeStoragePaths:
    root: Path
    system: Path
    data: Path
    vector: Path
    graph: Path


class CogneeConfigurator:
    """Configure Cognee before any engine, gateway, or pipeline is created."""

    def apply(
        self,
        settings: RuntimeSettings,
        paths: DataPaths,
    ) -> CogneeStoragePaths:
        root = paths.cognee.resolve(strict=False)
        resolved = CogneeStoragePaths(
            root=root,
            system=root / "system",
            data=root / "data",
            vector=root / "vector",
            graph=root / "graph",
        )

        # Cognee initializes logging during import, before its public path setters
        # are callable. Derive that import-time location from PaperOS data rather
        # than accepting Cognee's machine-specific home-directory default.
        os.environ["COGNEE_LOGS_DIR"] = str((paths.logs / "cognee").resolve(strict=False))

        # Cognee's OpenAI-compatible embedding client inherits HTTP proxy settings.
        # A local endpoint must never be sent through a system proxy: besides being
        # unnecessary, proxies commonly return 502 without the request ever reaching
        # the PaperOS child process. Preserve the user's existing bypasses and add
        # only loopback hosts before Cognee constructs its client.
        self._ensure_loopback_proxy_bypass(settings.cognee.embedding.endpoint)

        import cognee  # type: ignore[import-untyped]

        cognee.config.system_root_directory(str(resolved.system))
        cognee.config.data_root_directory(str(resolved.data))

        llm = settings.cognee.llm
        cognee.config.set_llm_config(
            {
                "llm_provider": llm.provider,
                "llm_model": llm.model,
                "llm_endpoint": llm.endpoint,
                "llm_api_key": llm.api_key_value(),
                "llm_max_completion_tokens": llm.max_completion_tokens,
                "llm_temperature": llm.temperature,
            }
        )

        embedding = settings.cognee.embedding
        cognee.config.set_embedding_config(
            {
                "embedding_provider": embedding.provider,
                "embedding_model": embedding.model,
                "embedding_endpoint": embedding.endpoint,
                "embedding_api_key": embedding.api_key_value(),
                "embedding_dimensions": embedding.dimensions,
                "embedding_max_completion_tokens": embedding.max_tokens,
                "embedding_batch_size": embedding.batch_size,
            }
        )

        storage = settings.cognee.storage
        relational_root = resolved.system / "databases"
        cognee.config.set_relational_db_config(
            {
                "db_provider": storage.relational_provider,
                "db_path": str(relational_root),
                "db_name": storage.database_name,
            }
        )
        cognee.config.set_vector_db_config(
            {
                "vector_db_provider": storage.vector_provider,
                "vector_dataset_database_handler": storage.vector_provider,
                "vector_db_url": str(resolved.vector),
                "vector_db_subprocess_enabled": storage.vector_subprocess_enabled,
            }
        )
        graph_filename = f"cognee_graph_{storage.graph_provider}"
        cognee.config.set_graph_db_config(
            {
                "graph_database_provider": storage.graph_provider,
                "graph_dataset_database_handler": storage.graph_provider,
                "graph_filename": graph_filename,
                "graph_file_path": str(resolved.graph / graph_filename),
                "graph_database_subprocess_enabled": storage.graph_subprocess_enabled,
            }
        )
        return resolved

    @staticmethod
    def _ensure_loopback_proxy_bypass(endpoint: str) -> None:
        hostname = urlsplit(endpoint).hostname
        if hostname is None:
            return
        try:
            is_loopback = ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = hostname.casefold() == "localhost"
        if not is_loopback:
            return

        existing = [
            item.strip()
            for key in ("NO_PROXY", "no_proxy")
            for item in os.environ.get(key, "").split(",")
            if item.strip()
        ]
        required = ("localhost", "127.0.0.1", "::1", hostname)
        merged = list(dict.fromkeys([*existing, *required]))
        value = ",".join(merged)
        os.environ["NO_PROXY"] = value
        os.environ["no_proxy"] = value


__all__ = ["CogneeConfigurator", "CogneeStoragePaths"]
