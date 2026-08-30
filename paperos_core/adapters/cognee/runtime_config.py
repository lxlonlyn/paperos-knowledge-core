"""Read-only view of configuration resolved by Cognee itself."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True, slots=True)
class CogneeRuntimeConfig:
    llm_provider: str
    llm_model: str
    llm_endpoint: str
    embedding_provider: str
    embedding_model: str
    embedding_endpoint: str
    embedding_dimensions: int
    embedding_max_tokens: int
    db_provider: str
    db_path: str
    db_name: str
    vector_db_provider: str
    vector_db_url: str
    graph_database_provider: str
    graph_file_path: str

    def embedding_targets(self, host: str, port: int) -> bool:
        parsed = urlparse(self.embedding_endpoint)
        endpoint_host = (parsed.hostname or "").casefold()
        configured_host = host.casefold()
        loopback_aliases = {"127.0.0.1", "localhost"}
        same_host = endpoint_host == configured_host or {
            endpoint_host,
            configured_host,
        } <= loopback_aliases
        return same_host and parsed.port == port


class CogneeRuntimeConfigReader:
    """Expose a credential-free snapshot of Cognee's current settings."""

    def read(self) -> CogneeRuntimeConfig:
        from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter

        values = CogneeCompatibilityAdapter.runtime_config_snapshot()
        return CogneeRuntimeConfig(
            llm_provider=str(values["llm_provider"] or ""),
            llm_model=str(values["llm_model"] or ""),
            llm_endpoint=str(values["llm_endpoint"] or ""),
            embedding_provider=str(values["embedding_provider"] or ""),
            embedding_model=str(values["embedding_model"] or ""),
            embedding_endpoint=str(values["embedding_endpoint"] or ""),
            embedding_dimensions=int(values["embedding_dimensions"] or 0),
            embedding_max_tokens=int(values["embedding_max_tokens"] or 0),
            db_provider=str(values["db_provider"] or ""),
            db_path=str(values["db_path"] or ""),
            db_name=str(values["db_name"] or ""),
            vector_db_provider=str(values["vector_db_provider"] or ""),
            vector_db_url=str(values["vector_db_url"] or ""),
            graph_database_provider=str(values["graph_database_provider"] or ""),
            graph_file_path=str(values["graph_file_path"] or ""),
        )

    async def test_llm_connection(self) -> None:
        from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter

        await CogneeCompatibilityAdapter.test_llm_connection()
