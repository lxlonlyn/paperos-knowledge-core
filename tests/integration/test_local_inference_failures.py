from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest

from paperos_core.config import load_settings
from paperos_core.errors import (
    LocalInferenceConfigurationError,
    LocalInferenceUnavailableError,
)
from paperos_core.paths import build_data_paths
from paperos_core.runtime.local_inference import (
    LocalInferenceClient,
    LocalInferenceRuntime,
)
from paperos_core.storage import StorageInitializer


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _settings(run_root: Path):
    return load_settings(
        environ={**os.environ, "PAPEROS_DATA_DIR": str(run_root)}
    )


def _runtime(settings):
    paths = build_data_paths(settings.data_dir)
    StorageInitializer(paths).initialize()
    local = settings.local_inference
    client = LocalInferenceClient(
        f"http://{local.host}:{local.port}", local.request_timeout_seconds
    )
    return LocalInferenceRuntime(settings, paths, client), client


def _with_real_model_paths(settings, configured_data_dir: Path):
    local = settings.local_inference
    return settings.model_copy(
        update={
            "local_inference": local.model_copy(
                update={
                    "embedding": local.embedding.model_copy(
                        update={
                            "model_path": configured_data_dir
                            / local.embedding.model_path
                        }
                    ),
                    "reranker": local.reranker.model_copy(
                        update={
                            "model_path": configured_data_dir
                            / local.reranker.model_path
                        }
                    ),
                    "query_expansion": local.query_expansion.model_copy(
                        update={
                            "model_path": configured_data_dir
                            / local.query_expansion.model_path
                        }
                    ),
                }
            )
        }
    )


@pytest.mark.asyncio
async def test_missing_model_fails_without_download(gate1_run_dir: Path) -> None:
    settings = _settings(gate1_run_dir / "missing-model")
    local = settings.local_inference.model_copy(
        update={
            "port": _free_port(),
            "embedding": settings.local_inference.embedding.model_copy(
                update={"model_path": Path("models/embedding/does-not-exist.gguf")}
            ),
        }
    )
    settings = settings.model_copy(update={"local_inference": local})
    runtime, client = _runtime(settings)
    try:
        with pytest.raises(LocalInferenceConfigurationError, match="does not exist"):
            await runtime.start()
        assert runtime.running is False
    finally:
        await runtime.stop()
        await client.aclose()


@pytest.mark.asyncio
async def test_missing_node_entry_is_actionable(gate1_run_dir: Path) -> None:
    settings = _settings(gate1_run_dir / "missing-node-entry")
    settings = settings.model_copy(
        update={
            "config_path": gate1_run_dir / "absent-repository" / "config" / "paperos.toml",
            "local_inference": settings.local_inference.model_copy(
                update={"port": _free_port()}
            ),
        }
    )
    runtime, client = _runtime(settings)
    try:
        with pytest.raises(LocalInferenceConfigurationError, match="entry is missing"):
            await runtime.start()
    finally:
        await runtime.stop()
        await client.aclose()


@pytest.mark.asyncio
async def test_port_conflict_fails_without_attaching(gate1_run_dir: Path) -> None:
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        port = int(occupied.getsockname()[1])
        settings = _settings(gate1_run_dir / "port-conflict")
        settings = settings.model_copy(
            update={
                "local_inference": settings.local_inference.model_copy(
                    update={"port": port}
                )
            }
        )
        runtime, client = _runtime(settings)
        try:
            with pytest.raises(LocalInferenceUnavailableError, match="already in use"):
                await runtime.start()
            assert runtime.running is False
        finally:
            await runtime.stop()
            await client.aclose()


@pytest.mark.asyncio
async def test_real_node_early_exit_is_reported(
    gate1_run_dir: Path, configured_data_dir: Path
) -> None:
    settings = _with_real_model_paths(
        _settings(gate1_run_dir / "node-early-exit"), configured_data_dir
    )
    invalid_model = configured_data_dir / "test-corpus" / "manifest.json"
    local = settings.local_inference.model_copy(
        update={
            "port": _free_port(),
            "startup_timeout_seconds": 30,
            "embedding": settings.local_inference.embedding.model_copy(
                update={"model_path": invalid_model}
            ),
        }
    )
    settings = settings.model_copy(update={"local_inference": local})
    runtime, client = _runtime(settings)
    try:
        with pytest.raises(LocalInferenceUnavailableError) as raised:
            await runtime.start()
        assert raised.value.details.get("exit_code") is not None
        assert raised.value.details.get("exited_before_timeout") is True
        assert runtime.running is False
    finally:
        await runtime.stop()
        await client.aclose()


@pytest.mark.asyncio
async def test_real_model_readiness_timeout_cleans_child(
    gate1_run_dir: Path, configured_data_dir: Path
) -> None:
    settings = _with_real_model_paths(
        _settings(gate1_run_dir / "health-timeout"), configured_data_dir
    )
    local = settings.local_inference.model_copy(
        update={"port": _free_port(), "startup_timeout_seconds": 1}
    )
    settings = settings.model_copy(update={"local_inference": local})
    runtime, client = _runtime(settings)
    try:
        with pytest.raises(LocalInferenceUnavailableError, match="within 1 seconds"):
            await runtime.start()
        assert runtime.running is False
        assert runtime.process is not None and runtime.process.returncode is not None
    finally:
        await runtime.stop()
        await client.aclose()
