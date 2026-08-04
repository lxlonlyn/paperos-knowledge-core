"""The single structured PaperOS configuration and environment secret boundary."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from paperos_core.errors import ConfigurationError

DEFAULT_DATA_DIR = Path("~/paperos-knowledge-core/data")
DEFAULT_CONFIG_PATH = Path("config/paperos.toml")


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


class DeepSeekSettings(StrictSettings):
    endpoint: str = ""
    model: str = ""
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    timeout_seconds: int = Field(default=180, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)

    def api_key_value(self) -> str | None:
        return self.api_key.get_secret_value() if self.api_key is not None else None


class IngestionSettings(StrictSettings):
    max_file_mb: int = Field(default=200, gt=0)
    chunk_target_tokens: int = Field(default=900, gt=0)
    chunk_overlap_tokens: int = Field(default=135, ge=0)


class EmbeddingSettings(StrictSettings):
    model_path: Path = Path("models/embedding/embeddinggemma-300M-Q8_0.gguf")
    model: str = "default"
    dimensions: int = Field(default=768, gt=0)
    max_tokens: int = Field(default=2048, gt=0)


class RerankerSettings(StrictSettings):
    model_path: Path = Path("models/reranker/qwen3-reranker-0.6b-q8_0.gguf")
    candidate_limit: int = Field(default=40, gt=0)


class QueryExpansionSettings(StrictSettings):
    model_path: Path = Path("models/query-expansion/qmd-query-expansion-1.7B-q4_k_m.gguf")
    max_output_tokens: int = Field(default=512, gt=0)


class LocalInferenceSettings(StrictSettings):
    host: Literal["127.0.0.1", "localhost"] = "127.0.0.1"
    port: int = Field(default=8081, ge=1, le=65535)
    request_timeout_seconds: int = Field(default=120, gt=0)
    startup_timeout_seconds: int = Field(default=180, gt=0)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    query_expansion: QueryExpansionSettings = Field(default_factory=QueryExpansionSettings)


class CogneeSettings(StrictSettings):
    system_database: Path = Path("cognee/system/databases")
    vector_database: Path = Path("cognee/vector")
    graph_database: Path = Path("cognee/graph")
    db_provider: str = "sqlite"
    vector_provider: str = "lancedb"
    graph_provider: str = "kuzu"
    deepseek: DeepSeekSettings = Field(default_factory=DeepSeekSettings, exclude=True)
    local_inference: LocalInferenceSettings = Field(
        default_factory=LocalInferenceSettings, exclude=True
    )


class RetrievalProfileSettings(StrictSettings):
    lexical_weight: float = Field(default=1.0, ge=0)
    semantic_weight: float = Field(default=1.0, ge=0)
    graph_weight: float = Field(default=1.0, ge=0)
    global_context_weight: float = Field(default=0.6, ge=0)
    confirmed_knowledge_weight: float = Field(default=1.2, ge=0)


class RetrievalProfilesSettings(StrictSettings):
    comprehensive: RetrievalProfileSettings = Field(default_factory=RetrievalProfileSettings)
    truth: RetrievalProfileSettings = Field(default_factory=RetrievalProfileSettings)
    associative: RetrievalProfileSettings = Field(default_factory=RetrievalProfileSettings)


class RetrievalSettings(StrictSettings):
    default_profile: Literal["comprehensive", "truth", "associative"] = "comprehensive"
    top_k: int = Field(default=12, gt=0)
    candidate_pool_size: int = Field(default=40, gt=0)
    graph_depth: int = Field(default=2, ge=0)
    max_chunks_per_document: int = Field(default=3, gt=0)
    max_chunks_per_section: int = Field(default=2, gt=0)
    profiles: RetrievalProfilesSettings = Field(default_factory=RetrievalProfilesSettings)


class RuntimeSettings(StrictSettings):
    data: DataSettings = Field(default_factory=DataSettings)
    mineru: MinerUSettings = Field(default_factory=MinerUSettings)
    deepseek: DeepSeekSettings = Field(default_factory=DeepSeekSettings)
    local_inference: LocalInferenceSettings = Field(default_factory=LocalInferenceSettings)
    cognee: CogneeSettings = Field(default_factory=CogneeSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    api: APISettings = Field(default_factory=APISettings)
    config_path: Path | None = Field(default=None, exclude=True)

    @property
    def data_dir(self) -> Path:
        return self.data.directory

    @property
    def dataset(self) -> str:
        return self.data.dataset


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    expanded = Path(value).expanduser()
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return expanded.resolve(strict=False)


def load_settings(
    path: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> RuntimeSettings:
    """Load TOML, then apply the documented environment-only overrides."""

    env = os.environ if environ is None else environ
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
    selected_data = env.get("PAPEROS_DATA_DIR", data_raw.get("directory", DEFAULT_DATA_DIR))
    data_root = _resolve_path(selected_data, base_dir=Path.cwd())
    data_raw["directory"] = data_root

    mineru_raw = raw.setdefault("mineru", {})
    deepseek_raw = raw.setdefault("deepseek", {})
    if not isinstance(mineru_raw, dict) or not isinstance(deepseek_raw, dict):
        raise ConfigurationError("The [mineru] and [deepseek] sections must be TOML tables.")
    if "api_key" in mineru_raw or "api_key" in deepseek_raw:
        raise ConfigurationError(
            "API keys are forbidden in paperos.toml; use MINERU_API_KEY and "
            "DEEPSEEK_API_KEY environment variables."
        )
    mineru_key = env.get("MINERU_API_KEY", "").strip()
    deepseek_key = env.get("DEEPSEEK_API_KEY", "").strip()
    if mineru_key:
        mineru_raw["api_key"] = mineru_key
    if deepseek_key:
        deepseek_raw["api_key"] = deepseek_key

    cognee_raw = raw.setdefault("cognee", {})
    if not isinstance(cognee_raw, dict):
        raise ConfigurationError("The [cognee] section must be a TOML table.")
    for key, default in (
        ("system_database", "cognee/system/databases"),
        ("vector_database", "cognee/vector"),
        ("graph_database", "cognee/graph"),
    ):
        cognee_raw[key] = _resolve_path(cognee_raw.get(key, default), base_dir=data_root)

    try:
        settings = RuntimeSettings.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(
            f"Invalid PaperOS configuration: {exc}", affected=config_path
        ) from exc
    cognee = settings.cognee.model_copy(
        update={
            "deepseek": settings.deepseek,
            "local_inference": settings.local_inference,
        }
    )
    return settings.model_copy(
        update={
            "cognee": cognee,
            "config_path": config_path if config_path.exists() else None,
        }
    )
