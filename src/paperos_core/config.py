"""Validated, centralized PaperOS TOML configuration loading."""

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


class StrictConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictConfigModel):
    data_dir: Path = DEFAULT_DATA_DIR
    dataset: str = "papers"


class APIConfig(StrictConfigModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)


class WorkerConfig(StrictConfigModel):
    concurrency: int = Field(default=1, ge=1)
    poll_interval_seconds: float = Field(default=1.0, gt=0)


class MinerUConfig(StrictConfigModel):
    provider: str = "mineru_cloud"
    endpoint: str = ""
    api_key_env: str = "MINERU_API_KEY"
    api_key: SecretStr | None = Field(default=None, repr=False)
    use_async_tasks: bool = True
    poll_interval_seconds: float = Field(default=2.0, gt=0)
    timeout_seconds: int = Field(default=1800, gt=0)
    preferred_backend: str = "auto"

    @field_validator("api_key", mode="before")
    @classmethod
    def empty_api_key_is_unset(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    def api_key_value(self) -> str | None:
        return self.api_key.get_secret_value() if self.api_key is not None else None


class IngestionConfig(StrictConfigModel):
    max_file_mb: int = Field(default=200, gt=0)
    retain_parser_debug_files: bool = True
    chunk_target_tokens: int = Field(default=900, gt=0)
    chunk_overlap_tokens: int = Field(default=135, ge=0)
    chunk_boundary_window_tokens: int = Field(default=200, ge=0)
    resolve_references: bool = True
    semantic_enrichment: bool = True


class EmbeddingModelConfig(StrictConfigModel):
    enabled: bool = True
    model_path: Path = Path("models/embedding/embeddinggemma-300M-Q8_0.gguf")
    dimensions: int = Field(default=768, gt=0)
    max_tokens: int = Field(default=2048, gt=0)


class RerankerModelConfig(StrictConfigModel):
    enabled: bool = True
    model_path: Path = Path("models/reranker/qwen3-reranker-0.6b-q8_0.gguf")
    candidate_limit: int = Field(default=40, gt=0)


class QueryExpansionModelConfig(StrictConfigModel):
    enabled: bool = True
    model_path: Path = Path("models/query-expansion/qmd-query-expansion-1.7B-q4_k_m.gguf")
    max_output_tokens: int = Field(default=512, gt=0)


class ModelsConfig(StrictConfigModel):
    gateway_endpoint: str = "http://127.0.0.1:8081"
    request_timeout_seconds: int = Field(default=120, gt=0)
    embedding: EmbeddingModelConfig = Field(default_factory=EmbeddingModelConfig)
    reranker: RerankerModelConfig = Field(default_factory=RerankerModelConfig)
    query_expansion: QueryExpansionModelConfig = Field(default_factory=QueryExpansionModelConfig)


class LexicalIndexConfig(StrictConfigModel):
    path: Path = Path("indexes/lexical.sqlite")
    fts_table: str = "chunk_fts"


class IndexesConfig(StrictConfigModel):
    lexical: LexicalIndexConfig = Field(default_factory=LexicalIndexConfig)


class RetrievalProfileConfig(StrictConfigModel):
    lexical_weight: float = Field(default=1.0, ge=0)
    semantic_weight: float = Field(default=1.0, ge=0)
    graph_weight: float = Field(default=1.0, ge=0)
    global_context_weight: float = Field(default=0.6, ge=0)
    confirmed_knowledge_weight: float = Field(default=1.2, ge=0)


class RetrievalProfilesConfig(StrictConfigModel):
    comprehensive: RetrievalProfileConfig = Field(default_factory=RetrievalProfileConfig)
    truth: RetrievalProfileConfig = Field(default_factory=RetrievalProfileConfig)
    associative: RetrievalProfileConfig = Field(default_factory=RetrievalProfileConfig)


class RetrievalConfig(StrictConfigModel):
    default_profile: Literal["comprehensive", "truth", "associative"] = "comprehensive"
    top_k: int = Field(default=12, gt=0)
    candidate_pool_size: int = Field(default=40, gt=0)
    graph_depth: int = Field(default=2, ge=0)
    max_chunks_per_document: int = Field(default=3, gt=0)
    max_chunks_per_section: int = Field(default=2, gt=0)
    profiles: RetrievalProfilesConfig = Field(default_factory=RetrievalProfilesConfig)


class TestingConfig(StrictConfigModel):
    corpus_dir: Path = Path("test-corpus/pdfs")
    expected_dir: Path = Path("test-corpus/expected")
    queries_dir: Path = Path("test-corpus/queries")
    run_dir: Path = Path("test-runs")
    require_real_pdf: bool = True
    require_live_mineru: bool = True
    require_live_cognee: bool = True
    require_live_local_models: bool = True
    require_live_deepseek: bool = True
    fail_on_missing_dependency: bool = True
    minimum_pdf_count: int = Field(default=1, ge=1)


class PaperOSConfig(StrictConfigModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    mineru_ocr: MinerUConfig = Field(default_factory=MinerUConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    indexes: IndexesConfig = Field(default_factory=IndexesConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    testing: TestingConfig = Field(default_factory=TestingConfig)
    config_path: Path | None = Field(default=None, exclude=True)
    configured_data_dir: Path | None = Field(default=None, exclude=True)

    @property
    def data_dir(self) -> Path:
        return self.project.data_dir

    @property
    def dataset(self) -> str:
        return self.project.dataset


def _resolve_path(value: str | Path, *, base_dir: Path) -> Path:
    expanded = Path(value).expanduser()
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return expanded.resolve(strict=False)


def load_config(
    path: Path | None = None,
    *,
    data_dir: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> PaperOSConfig:
    """Load TOML with environment overrides for paths and configured secrets."""

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

    project_raw = raw.setdefault("project", {})
    if not isinstance(project_raw, dict):
        raise ConfigurationError(
            "The [project] configuration section must be a TOML table.",
            affected=config_path,
        )

    toml_data_dir = project_raw.get("data_dir", DEFAULT_DATA_DIR)
    configured_data_dir = _resolve_path(toml_data_dir, base_dir=Path.cwd())
    env_data_dir = env.get("PAPEROS_DATA_DIR")
    selected_data_dir = data_dir or (Path(env_data_dir) if env_data_dir else toml_data_dir)
    project_raw["data_dir"] = _resolve_path(selected_data_dir, base_dir=Path.cwd())

    mineru_raw = raw.setdefault("mineru_ocr", {})
    if not isinstance(mineru_raw, dict):
        raise ConfigurationError(
            "The [mineru_ocr] configuration section must be a TOML table.",
            affected=config_path,
        )
    api_key_env = mineru_raw.get("api_key_env", "MINERU_API_KEY")
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        raise ConfigurationError(
            "mineru_ocr.api_key_env must name a non-empty environment variable.",
            affected=config_path,
        )
    environment_api_key = env.get(api_key_env)
    if environment_api_key:
        mineru_raw["api_key"] = environment_api_key

    try:
        config = PaperOSConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigurationError(
            f"Invalid PaperOS configuration: {exc}", affected=config_path
        ) from exc
    return config.model_copy(
        update={
            "config_path": config_path if config_path.exists() else None,
            "configured_data_dir": configured_data_dir,
        }
    )
