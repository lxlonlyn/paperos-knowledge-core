"""Provider switching must be configuration-only for LLM/embedding/vector/graph."""

from __future__ import annotations

import os
from pathlib import Path

from paperos_core.adapters.cognee.compat import CogneeCompatibilityAdapter
from paperos_core.adapters.cognee.config import configure_cognee
from paperos_core.config import CogneeEmbeddingSettings, EmbeddingSettings, load_settings
from paperos_core.paths import build_data_paths
from paperos_core.runtime.local_inference.client import LocalInferenceClient
from paperos_core.runtime.local_inference.runtime import LocalInferenceRuntime

_TRACKED_ENV = (
    "LLM_PROVIDER",
    "LLM_MODEL",
    "LLM_ENDPOINT",
    "LLM_API_KEY",
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_ENDPOINT",
    "EMBEDDING_API_KEY",
    "EMBEDDING_DIMENSIONS",
    "VECTOR_DB_PROVIDER",
    "GRAPH_DATABASE_PROVIDER",
)


def _load(
    gate1_run_dir: Path,
    toml: str,
    *,
    environ: dict[str, str] | None = None,
):
    path = gate1_run_dir / "provider-switch" / "paperos.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml, encoding="utf-8")
    return load_settings(path, environ=environ or {})


def test_remote_embedding_and_openai_llm_switch_by_config_only(
    gate1_run_dir: Path,
) -> None:
    settings = _load(
        gate1_run_dir,
        f'''[data]
directory = "{gate1_run_dir / 'provider-data'}"
dataset = "papers"
[llm]
provider = "openai"
model = "openai/gpt-5-mini"
endpoint = ""
[cognee]
vector_provider = "qdrant"
graph_provider = "networkx"
[cognee.embedding]
provider = "openai"
model = "text-embedding-3-small"
endpoint = "https://api.openai.com/v1"
dimensions = 1536
local_runtime = false
''',
        environ={
            "LLM_API_KEY": "llm-key",
            "EMBEDDING_API_KEY": "embedding-key",
        },
    )
    saved = {name: os.environ.get(name) for name in _TRACKED_ENV}
    try:
        configure_cognee(settings)
        CogneeCompatibilityAdapter.reset_configuration_caches()
        assert os.environ["LLM_PROVIDER"] == "openai"
        assert os.environ["LLM_MODEL"] == "openai/gpt-5-mini"
        assert os.environ["LLM_API_KEY"] == "llm-key"
        assert os.environ["EMBEDDING_PROVIDER"] == "openai"
        assert os.environ["EMBEDDING_ENDPOINT"] == "https://api.openai.com/v1"
        assert os.environ["EMBEDDING_API_KEY"] == "embedding-key"
        assert os.environ["EMBEDDING_DIMENSIONS"] == "1536"
        assert os.environ["VECTOR_DB_PROVIDER"] == "qdrant"
        assert os.environ["GRAPH_DATABASE_PROVIDER"] == "networkx"
        assert settings.cognee.embedding.local_runtime is False
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_local_runtime_injects_loopback_endpoint(gate1_run_dir: Path) -> None:
    settings = _load(
        gate1_run_dir,
        f'''[data]
directory = "{gate1_run_dir / 'provider-data-local'}"
dataset = "papers"
[llm]
provider = "custom"
model = "example-model"
endpoint = "https://api.example.com/v1"
[cognee.embedding]
provider = "openai_compatible"
model = "default"
endpoint = ""
dimensions = 768
local_runtime = true
''',
        environ={"EMBEDDING_API_KEY": "paperos-internal"},
    )
    saved = {name: os.environ.get(name) for name in _TRACKED_ENV}
    try:
        configure_cognee(settings)
        CogneeCompatibilityAdapter.reset_configuration_caches()
        assert (
            os.environ["EMBEDDING_ENDPOINT"]
            == f"http://{settings.local_inference.host}:{settings.local_inference.port}/v1"
        )
        assert os.environ["EMBEDDING_API_KEY"] == "paperos-internal"
        assert os.environ["EMBEDDING_PROVIDER"] == "openai_compatible"
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_local_runtime_starts_when_embedding_or_reranker_is_local(
    gate1_run_dir: Path,
) -> None:
    paths = build_data_paths(gate1_run_dir / "local-runtime-condition")
    client = LocalInferenceClient("http://127.0.0.1:1", 1)
    remote = _load(
        gate1_run_dir,
        f'''[data]
directory = "{gate1_run_dir / 'provider-data-remote'}"
dataset = "papers"
[llm]
provider = "custom"
model = "example-model"
endpoint = "https://api.example.com/v1"
[cognee.embedding]
provider = "openai"
model = "text-embedding-3-small"
endpoint = "https://api.openai.com/v1"
local_runtime = false
''',
    )
    assert LocalInferenceRuntime(remote, paths, client).required is False
    rerank_local = remote.model_copy(
        update={
            "retrieval": remote.retrieval.model_copy(update={"rerank_enabled": True})
        }
    )
    assert LocalInferenceRuntime(rerank_local, paths, client).required is True
    embedding_local = _load(
        gate1_run_dir,
        f'''[data]
directory = "{gate1_run_dir / 'provider-data-embedding-local'}"
dataset = "papers"
[llm]
provider = "custom"
model = "example-model"
endpoint = "https://api.example.com/v1"
[cognee.embedding]
provider = "openai_compatible"
model = "default"
endpoint = ""
local_runtime = true
''',
    )
    assert LocalInferenceRuntime(embedding_local, paths, client).required is True


def test_embedding_parameters_have_a_single_owner() -> None:
    local_fields = set(EmbeddingSettings.model_fields)
    assert {"model_path", "sha256"} <= local_fields
    assert "model" not in local_fields
    assert "dimensions" not in local_fields
    assert "max_tokens" not in local_fields
    cognee_fields = set(CogneeEmbeddingSettings.model_fields)
    assert {"provider", "model", "dimensions", "max_tokens"} <= cognee_fields
    assert "local_runtime" in cognee_fields
