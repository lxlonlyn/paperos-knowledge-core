"""PaperOS-owned structured configuration.

The git-ignored TOML is the only persistent configuration. Three optional
secret environment variables may override TOML keys without becoming a second system.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path, PureWindowsPath
from typing import Literal
from urllib.parse import urlparse

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

from paperos_core.errors import ConfigurationError
from paperos_core.locations import CONFIG_ROOT, PROJECT_ROOT

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CONFIG_PATH = CONFIG_ROOT / "paperos.toml"


class StrictSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataSettings(StrictSettings):
    directory: Path = DEFAULT_DATA_DIR
    dataset: str = "papers"

    @field_validator("dataset")
    @classmethod
    def dataset_must_not_be_blank(cls, value: str) -> str:
        selected = value.strip()
        if not selected:
            raise ValueError("data.dataset must not be blank")
        return selected


def _safe_storage_name(value: str) -> str:
    selected = value.strip()
    if (
        not selected
        or ".." in selected
        or "/" in selected
        or "\\" in selected
        or Path(selected).is_absolute()
        or PureWindowsPath(selected).is_absolute()
    ):
        raise ValueError("storage names must be safe relative names without path separators or '..'")
    return selected


class StorageSettings(StrictSettings):
    registry_filename: str = "registry.sqlite3"
    lexical_filename: str = "lexical.sqlite3"

    @field_validator("registry_filename", "lexical_filename")
    @classmethod
    def filenames_must_be_safe(cls, value: str) -> str:
        return _safe_storage_name(value)


class APISettings(StrictSettings):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class MinerUSettings(StrictSettings):
    provider: str = "cloud"
    endpoint: str = ""
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    use_async_tasks: bool = True
    poll_interval_seconds: float = Field(default=2.0, gt=0)
    timeout_seconds: int = Field(default=1800, gt=0)
    preferred_backend: str = "auto"

    def api_key_value(self) -> str | None:
        return self.api_key.get_secret_value() if self.api_key is not None else None


class CogneeLLMSettings(StrictSettings):
    provider: str = "openai"
    model: str = "openai/gpt-5-mini"
    endpoint: str = ""
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    max_completion_tokens: int = Field(default=16384, gt=0)
    temperature: float = Field(default=0.0, ge=0)

    @field_validator("provider", "model")
    @classmethod
    def required_values_must_not_be_blank(cls, value: str) -> str:
        selected = value.strip()
        if not selected:
            raise ValueError("Cognee LLM provider and model must not be blank")
        return selected

    def api_key_value(self) -> str | None:
        return self.api_key.get_secret_value() if self.api_key is not None else None


class CogneeEmbeddingSettings(StrictSettings):
    provider: str = "custom"
    model: str = "default"
    endpoint: str = "http://127.0.0.1:8081/v1"
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    dimensions: int = Field(default=768, gt=0)
    max_tokens: int = Field(default=2048, gt=0)
    batch_size: int = Field(default=5, gt=0)

    @field_validator("provider", "model")
    @classmethod
    def required_values_must_not_be_blank(cls, value: str) -> str:
        selected = value.strip()
        if not selected:
            raise ValueError("Cognee embedding provider and model must not be blank")
        return selected

    def api_key_value(self) -> str | None:
        return self.api_key.get_secret_value() if self.api_key is not None else None


class CogneeStorageSettings(StrictSettings):
    relational_provider: str = "sqlite"
    database_name: str = "cognee_db"
    vector_provider: str = "lancedb"
    graph_provider: str = "kuzu"
    vector_subprocess_enabled: bool = True
    graph_subprocess_enabled: bool = True

    @field_validator("relational_provider", "vector_provider", "graph_provider")
    @classmethod
    def providers_must_not_be_blank(cls, value: str) -> str:
        selected = value.strip().lower()
        if not selected:
            raise ValueError("Cognee storage providers must not be blank")
        return selected

    @field_validator("database_name")
    @classmethod
    def database_name_must_be_safe(cls, value: str) -> str:
        return _safe_storage_name(value)


class CogneeSettings(StrictSettings):
    llm: CogneeLLMSettings = Field(default_factory=CogneeLLMSettings)
    embedding: CogneeEmbeddingSettings = Field(default_factory=CogneeEmbeddingSettings)
    storage: CogneeStorageSettings = Field(default_factory=CogneeStorageSettings)


class IngestionSettings(StrictSettings):
    max_file_mb: int = Field(default=200, gt=0)
    chunk_target_tokens: int = Field(default=900, gt=0)
    chunk_hard_max_tokens: int = Field(default=1200, gt=0)
    chunk_overlap_tokens: int = Field(default=0, ge=0)
    semantic_enrichment_enabled: bool = False
    claim_enrichment_enabled: bool = False


class LocalInferenceSettings(StrictSettings):
    enabled: bool = True
    host: Literal["127.0.0.1", "localhost"] = "127.0.0.1"
    port: int = Field(default=8081, ge=1, le=65535)
    embedding_model_path: Path = Path("../data/models/embedding/embeddinggemma-300M-Q8_0.gguf")
    reranker_model_path: Path = Path("../data/models/reranker/qwen3-reranker-0.6b-q8_0.gguf")
    cuda_devices: list[int] = Field(default_factory=list)
    startup_timeout: int = Field(default=180, gt=0)
    request_timeout: int = Field(default=120, gt=0)

    @field_validator("cuda_devices")
    @classmethod
    def cuda_devices_must_be_unique(cls, value: list[int]) -> list[int]:
        if any(device < 0 for device in value):
            raise ValueError("local_inference.cuda_devices must be non-negative")
        if len(set(value)) != len(value):
            raise ValueError("local_inference.cuda_devices must not contain duplicates")
        return value


class RetrievalSettings(StrictSettings):
    top_k: int = Field(default=12, gt=0)
    candidate_pool_size: int = Field(default=40, gt=0)
    rerank_enabled: bool = True
    synthesis_max_input_tokens: int = Field(default=48_000, gt=0)


class RuntimeSettings(StrictSettings):
    data: DataSettings = Field(default_factory=DataSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    cognee: CogneeSettings = Field(default_factory=CogneeSettings)
    mineru: MinerUSettings = Field(default_factory=MinerUSettings)
    local_inference: LocalInferenceSettings = Field(default_factory=LocalInferenceSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    api: APISettings = Field(default_factory=APISettings)
    config_path: Path | None = Field(default=None, exclude=True)

    @model_validator(mode="after")
    def local_inference_dependencies_must_be_enabled(self) -> RuntimeSettings:
        local = self.local_inference
        embedding = self.cognee.embedding
        if _endpoint_targets_local_inference(embedding.endpoint, local.host, local.port) and not (
            local.enabled
        ):
            raise ValueError(
                "cognee.embedding.endpoint targets [local_inference] host/port, so "
                "local_inference.enabled must be true"
            )
        if self.retrieval.rerank_enabled and not local.enabled:
            raise ValueError(
                "retrieval.rerank_enabled=true requires local_inference.enabled=true; "
                "no remote reranker is configured"
            )
        if (
            self.ingestion.claim_enrichment_enabled
            and not self.ingestion.semantic_enrichment_enabled
        ):
            raise ValueError(
                "ingestion.claim_enrichment_enabled=true requires "
                "ingestion.semantic_enrichment_enabled=true"
            )
        return self

    @property
    def data_dir(self) -> Path:
        return self.data.directory

    @property
    def dataset(self) -> str:
        return self.data.dataset


def _endpoint_targets_local_inference(endpoint: str, host: str, port: int) -> bool:
    """Match the configured endpoint using the runtime's loopback alias rules."""

    try:
        parsed = urlparse(endpoint)
        endpoint_port = parsed.port
    except ValueError:
        return False
    endpoint_host = (parsed.hostname or "").casefold()
    configured_host = host.casefold()
    loopback_aliases = {"127.0.0.1", "localhost"}
    same_host = endpoint_host == configured_host or {
        endpoint_host,
        configured_host,
    } <= loopback_aliases
    return same_host and endpoint_port == port


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    expanded = Path(value).expanduser()
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return expanded.resolve(strict=False)


def resolve_local_model_path(settings: RuntimeSettings, configured: Path) -> Path:
    """Resolve a user-owned model path relative to the single TOML file."""

    base_dir = settings.config_path.parent if settings.config_path else CONFIG_ROOT
    return _resolve_path(configured, base_dir=base_dir)


def load_settings(
    path: Path | None = None,
) -> RuntimeSettings:
    """Load the single PaperOS TOML plus the three supported secret overrides."""
    explicit_config = path is not None
    config_path = (path or DEFAULT_CONFIG_PATH).expanduser().resolve(strict=False)
    raw: dict[str, object] = {}
    if config_path.exists():
        if not config_path.is_file():
            raise ConfigurationError(
                "PaperOS config path is not a regular file.", affected=config_path
            )
        try:
            with config_path.open("rb") as stream:
                raw = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError(
                f"Unable to read PaperOS TOML configuration: {exc}",
                affected=config_path,
            ) from exc
    elif explicit_config:
        raise ConfigurationError(
            "The requested PaperOS config file does not exist.", affected=config_path
        )

    data_raw = raw.setdefault("data", {})
    if not isinstance(data_raw, dict):
        raise ConfigurationError("The [data] section must be a TOML table.")
    config_root = config_path.parent
    data_root = _resolve_path(data_raw.get("directory", DEFAULT_DATA_DIR), base_dir=config_root)
    secret_overrides = {
        ("mineru", "api_key"): os.getenv("PAPEROS_MINERU_API_KEY"),
        ("cognee", "llm", "api_key"): os.getenv("PAPEROS_LLM_API_KEY"),
        ("cognee", "embedding", "api_key"): os.getenv("PAPEROS_EMBEDDING_API_KEY"),
    }
    for keys, value in secret_overrides.items():
        if value is None:
            continue
        target = raw
        for key in keys[:-1]:
            nested = target.setdefault(key, {})
            if not isinstance(nested, dict):
                raise ConfigurationError(
                    f"The [{'.'.join(keys[:-1])}] section must be a TOML table."
                )
            target = nested
        target[keys[-1]] = value

    data_raw["directory"] = data_root

    try:
        settings = RuntimeSettings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(
            f"Invalid PaperOS configuration: {exc}", affected=config_path
        ) from exc
    local = settings.local_inference
    local = local.model_copy(
        update={
            "embedding_model_path": _resolve_path(local.embedding_model_path, base_dir=config_root),
            "reranker_model_path": _resolve_path(local.reranker_model_path, base_dir=config_root),
        }
    )
    return settings.model_copy(
        update={
            "local_inference": local,
            "config_path": config_path,
        }
    )
