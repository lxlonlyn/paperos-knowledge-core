"""Application-owned lifecycle for the private Node inference child process."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from paperos_core.adapters.cognee.runtime_config import CogneeRuntimeConfigReader
from paperos_core.config import RuntimeSettings, resolve_local_model_path
from paperos_core.errors import (
    LocalInferenceConfigurationError,
    LocalInferenceRuntimeIncompatibleError,
    LocalInferenceUnavailableError,
)
from paperos_core.locations import SERVICES_ROOT
from paperos_core.paths import DataPaths
from paperos_core.runtime.local_inference.client import LocalInferenceClient

LOCAL_INFERENCE_PROTOCOL_VERSION = 1
LOCAL_RERANKER_MAX_TOKENS = 4096
LOCAL_RERANKER_MODEL_NAME = "qwen3-reranker-0.6b"


@dataclass(frozen=True, slots=True)
class LocalRuntimeUsage:
    embedding: bool
    reranker: bool

    @property
    def required(self) -> bool:
        return self.embedding or self.reranker


def local_runtime_usage(
    settings: RuntimeSettings, cognee_config: CogneeRuntimeConfigReader
) -> LocalRuntimeUsage:
    local = settings.local_inference
    if not local.enabled:
        return LocalRuntimeUsage(embedding=False, reranker=False)
    resolved = cognee_config.read()
    return LocalRuntimeUsage(
        embedding=resolved.embedding_targets(local.host, local.port),
        reranker=settings.retrieval.rerank_enabled,
    )


class LocalInferenceRuntime:
    def __init__(
        self,
        settings: RuntimeSettings,
        paths: DataPaths,
        client: LocalInferenceClient,
        cognee_config: CogneeRuntimeConfigReader,
    ) -> None:
        self.settings = settings
        self.paths = paths
        self.client = client
        self.cognee_config = cognee_config
        self.process: asyncio.subprocess.Process | None = None
        self._log_stream: BinaryIO | None = None
        self._owned = False
        self._shutdown_token: str | None = None

    @property
    def endpoint(self) -> str:
        local = self.settings.local_inference
        return f"http://{local.host}:{local.port}"

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None
    def cleanup_stale_record(self) -> None:
        """Discard machine-local PID state before each application start."""
        (self.paths.jobs / "local-inference-process.json").unlink(missing_ok=True)


    @property
    def running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    @property
    def required(self) -> bool:
        """Start the child when local embedding or the local reranker is enabled."""
        return local_runtime_usage(self.settings, self.cognee_config).required

    def _model_path(self, configured: Path, *, label: str) -> Path:
        result = resolve_local_model_path(self.settings, configured)
        if not result.is_file():
            raise LocalInferenceConfigurationError(
                f"Configured local {label} model file does not exist.",
                affected=result,
            )
        return result

    def _service_root(self) -> Path:
        root = SERVICES_ROOT / "local_models"
        server = root / "dist" / "server.js"
        if not server.is_file():
            raise LocalInferenceConfigurationError(
                "Compiled local inference entry is missing; in services/local_models "
                "run `npm ci` and then `npm run build`.",
                affected=server,
            )
        return root

    def _process_environment(self) -> dict[str, str]:
        """Build the child environment without overriding ambient CUDA by default."""

        environment = dict(os.environ)
        configured = self.settings.local_inference.cuda_devices
        if configured:
            environment["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(device) for device in configured
            )
        return environment

    async def start(self) -> dict[str, Any]:
        if self.running and self._owned:
            return await self.client.health()
        local = self.settings.local_inference
        if not self.required:
            return {"status": "disabled"}
        reused = await self._reuse_healthy_endpoint(local.host, local.port)
        if reused is not None:
            return reused
        await self._assert_port_available(local.host, local.port)
        service_root = self._service_root()
        cognee = self.cognee_config.read()
        usage = local_runtime_usage(self.settings, self.cognee_config)
        embedding_enabled = usage.embedding
        model_path = (
            self._model_path(local.embedding_model_path, label="embedding")
            if embedding_enabled
            else None
        )
        reranker_enabled = usage.reranker
        reranker_path = (
            self._model_path(local.reranker_model_path, label="reranker")
            if reranker_enabled
            else None
        )
        log_path = self.paths.logs / "local-inference.log"
        process_path = self.paths.jobs / "local-inference-process.json"
        self._log_stream = log_path.open("ab", buffering=0)
        self._shutdown_token = secrets.token_urlsafe(32)
        environment = self._process_environment()
        environment.update(
            {
                "NODE_LLAMA_CPP_SKIP_DOWNLOAD": "true",
                "PAPEROS_LOCAL_INFERENCE_HOST": local.host,
                "PAPEROS_LOCAL_INFERENCE_PORT": str(local.port),
                "PAPEROS_EMBEDDING_ENABLED": "true" if embedding_enabled else "false",
                "PAPEROS_EMBEDDING_MODEL_NAME": cognee.embedding_model,
                "PAPEROS_EMBEDDING_DIMENSIONS": str(cognee.embedding_dimensions),
                "PAPEROS_EMBEDDING_MAX_TOKENS": str(cognee.embedding_max_tokens),
                "PAPEROS_RERANKER_ENABLED": "true" if reranker_enabled else "false",
                "PAPEROS_RERANKER_MODEL_NAME": LOCAL_RERANKER_MODEL_NAME,
                **(
                    {"PAPEROS_EMBEDDING_MODEL_PATH": str(model_path)}
                    if model_path is not None
                    else {}
                ),
                **(
                    {"PAPEROS_RERANKER_MODEL_PATH": str(reranker_path)}
                    if reranker_path is not None
                    else {}
                ),
                "PAPEROS_RERANKER_MAX_TOKENS": str(LOCAL_RERANKER_MAX_TOKENS),
                "PAPEROS_SHUTDOWN_TOKEN": self._shutdown_token,
            }
        )
        try:
            self.process = await asyncio.create_subprocess_exec(
                "node",
                "dist/server.js",
                cwd=service_root,
                env=environment,
                stdout=self._log_stream,
                stderr=asyncio.subprocess.STDOUT,
            )
        except OSError as exc:
            self._shutdown_token = None
            self._close_log()
            raise LocalInferenceUnavailableError(
                f"Unable to start local inference: {exc}", affected=service_root
            ) from exc
        self._owned = True
        self._write_process_record(
            process_path,
            {
                "pid": self.process.pid,
                "status": "starting",
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": str(log_path),
                "endpoint": self.endpoint,
                "cuda_devices": local.cuda_devices,
            },
        )
        last_error: Exception | None = None
        for _ in range(local.startup_timeout):
            if self.process.returncode is not None:
                break
            try:
                health = await self.client.health()
                record = json.loads(process_path.read_text(encoding="utf-8"))
                record["status"] = "running"
                self._write_process_record(process_path, record)
                return health
            except LocalInferenceUnavailableError as exc:
                last_error = exc
                await asyncio.sleep(1)
        exited_before_timeout = self.process.returncode is not None
        await self.stop()
        exit_code = self.process.returncode
        raise LocalInferenceUnavailableError(
            "Local inference did not become healthy within "
            f"{local.startup_timeout} seconds.",
            affected=log_path,
            details={
                "last_error": str(last_error) if last_error else None,
                "exit_code": exit_code,
                "exited_before_timeout": exited_before_timeout,
            },
            retryable=False,
        )

    async def _assert_port_available(self, host: str, port: int) -> None:
        try:
            _reader, writer = await asyncio.open_connection(host, port)
        except OSError:
            return
        writer.close()
        await writer.wait_closed()
        raise LocalInferenceUnavailableError(
            f"Cannot start local inference because {host}:{port} is already in use.",
            affected=f"{host}:{port}",
            retryable=False,
        )

    async def _reuse_healthy_endpoint(
        self, host: str, port: int
    ) -> dict[str, Any] | None:
        """Reuse an already-healthy local inference service instead of rebinding."""
        try:
            _reader, writer = await asyncio.open_connection(host, port)
        except OSError:
            return None
        writer.close()
        await writer.wait_closed()
        try:
            health = await self.client.health()
        except LocalInferenceUnavailableError as exc:
            raise LocalInferenceUnavailableError(
                f"Cannot start local inference because {host}:{port} is already in use.",
                affected=f"{host}:{port}",
                retryable=False,
            ) from exc
        if health.get("status") == "healthy":
            self._validate_reuse_identity(health, affected=f"{host}:{port}")
            self._owned = False
            return health
        raise LocalInferenceUnavailableError(
            f"Cannot start local inference because {host}:{port} is already in use.",
            affected=f"{host}:{port}",
            retryable=False,
        )

    def _expected_runtime_identity(self) -> dict[str, Any]:
        local = self.settings.local_inference
        cognee = self.cognee_config.read()
        usage = local_runtime_usage(self.settings, self.cognee_config)
        embedding_path = (
            self._model_path(local.embedding_model_path, label="embedding")
            if usage.embedding
            else None
        )
        reranker_path = (
            self._model_path(local.reranker_model_path, label="reranker")
            if usage.reranker
            else None
        )
        return {
            "protocol_version": LOCAL_INFERENCE_PROTOCOL_VERSION,
            "embedding": {
                "enabled": usage.embedding,
                "model": {
                    "name": cognee.embedding_model,
                    "file": self._model_file_identity(embedding_path),
                },
                "dimensions": cognee.embedding_dimensions,
                "max_tokens": cognee.embedding_max_tokens,
            },
            "reranker": {
                "enabled": usage.reranker,
                "model": {
                    "name": LOCAL_RERANKER_MODEL_NAME,
                    "file": self._model_file_identity(reranker_path),
                },
                "max_tokens": LOCAL_RERANKER_MAX_TOKENS,
            },
            "cuda_visible_devices": self._process_environment().get(
                "CUDA_VISIBLE_DEVICES", ""
            ),
        }

    @staticmethod
    def _model_file_identity(path: Path | None) -> dict[str, str] | None:
        if path is None:
            return None
        metadata = path.stat()
        return {
            "resolved_path": str(path.resolve()),
            "file_size": str(metadata.st_size),
            "mtime_ns": str(metadata.st_mtime_ns),
        }

    def _validate_reuse_identity(
        self, health: dict[str, Any], *, affected: str | None = None
    ) -> None:
        if health.get("runtime_identity") == self._expected_runtime_identity():
            return
        raise LocalInferenceRuntimeIncompatibleError(
            "Existing local inference runtime does not match the current configuration.",
            affected=affected or self.endpoint,
            details={"reason": "runtime_identity_mismatch"},
        )

    async def stop(self) -> None:
        if not self._owned:
            return
        if self.process is not None and self.process.returncode is None:
            try:
                if self._shutdown_token is None:
                    raise LocalInferenceUnavailableError(
                        "Local inference shutdown token is unavailable.",
                        affected=self.endpoint,
                    )
                await self.client.shutdown(self._shutdown_token)
                await asyncio.wait_for(self.process.wait(), timeout=20)
            except (LocalInferenceUnavailableError, TimeoutError):
                if self.process.returncode is None:
                    try:
                        self.process.terminate()
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(self.process.wait(), timeout=20)
                    except TimeoutError:
                        self.process.kill()
                        await self.process.wait()
        process_path = self.paths.jobs / "local-inference-process.json"
        self._mark_stopped(process_path)
        self._close_log()
        self._owned = False
        self._shutdown_token = None

    def _close_log(self) -> None:
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None

    def _mark_stopped(self, process_path: Path) -> None:
        if not process_path.is_file():
            return
        record = json.loads(process_path.read_text(encoding="utf-8"))
        record.update(
            {
                "status": "stopped",
                "stopped_at": datetime.now(UTC).isoformat(),
                "exit_code": self.process.returncode if self.process else None,
            }
        )
        self._write_process_record(process_path, record)

    @staticmethod
    def _write_process_record(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
