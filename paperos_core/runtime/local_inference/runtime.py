"""Application-owned lifecycle for the private Node inference child process."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from paperos_core.config import RuntimeSettings
from paperos_core.errors import (
    LocalInferenceConfigurationError,
    LocalInferenceUnavailableError,
)
from paperos_core.paths import DataPaths
from paperos_core.runtime.local_inference.client import LocalInferenceClient


class LocalInferenceRuntime:
    def __init__(
        self,
        settings: RuntimeSettings,
        paths: DataPaths,
        client: LocalInferenceClient,
    ) -> None:
        self.settings = settings
        self.paths = paths
        self.client = client
        self.process: asyncio.subprocess.Process | None = None
        self._log_stream: BinaryIO | None = None
        self._owned = False

    @property
    def endpoint(self) -> str:
        local = self.settings.local_inference
        return f"http://{local.host}:{local.port}"

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process is not None else None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    def _model_path(self, configured: Path, *, label: str) -> Path:
        configured = configured.expanduser()
        if configured.is_absolute():
            result = configured
        else:
            base = self.settings.configured_data_dir or self.paths.root
            result = base / configured
        result = result.resolve(strict=False)
        if not result.is_file():
            raise LocalInferenceConfigurationError(
                f"Configured local {label} model file does not exist.",
                affected=result,
            )
        return result

    def _service_root(self) -> Path:
        if self.settings.config_path is None:
            raise LocalInferenceConfigurationError(
                "A project config path is required to locate local inference."
            )
        root = self.settings.config_path.parent.parent / "services" / "local_models"
        server = root / "dist" / "server.js"
        if not server.is_file():
            raise LocalInferenceConfigurationError(
                "Compiled local inference entry is missing; run `npm run build` "
                "in services/local_models without installing dependencies.",
                affected=server,
            )
        return root

    async def start(self) -> dict[str, Any]:
        if self.running and self._owned:
            return await self.client.health()
        local = self.settings.local_inference
        await self._assert_port_available(local.host, local.port)
        service_root = self._service_root()
        model_path = self._model_path(local.embedding.model_path, label="embedding")
        reranker_path = self._model_path(local.reranker.model_path, label="reranker")
        query_expansion_path = self._model_path(
            local.query_expansion.model_path, label="query expansion"
        )
        log_path = self.paths.logs / "local-inference.log"
        process_path = self.paths.jobs / "local-inference-process.json"
        self._log_stream = log_path.open("ab", buffering=0)
        environment = dict(os.environ)
        environment.update(
            {
                "NODE_LLAMA_CPP_SKIP_DOWNLOAD": "true",
                "PAPEROS_LOCAL_INFERENCE_HOST": local.host,
                "PAPEROS_LOCAL_INFERENCE_PORT": str(local.port),
                "PAPEROS_EMBEDDING_MODEL_PATH": str(model_path),
                "PAPEROS_EMBEDDING_DIMENSIONS": str(local.embedding.dimensions),
                "PAPEROS_EMBEDDING_MAX_TOKENS": str(local.embedding.max_tokens),
                "PAPEROS_RERANKER_MODEL_PATH": str(reranker_path),
                "PAPEROS_RERANKER_MAX_TOKENS": "4096",
                "PAPEROS_QUERY_EXPANSION_MODEL_PATH": str(query_expansion_path),
                "PAPEROS_QUERY_EXPANSION_MAX_TOKENS": str(
                    local.query_expansion.max_output_tokens
                ),
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
            },
        )
        last_error: Exception | None = None
        for _ in range(local.startup_timeout_seconds):
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
        exit_code = self.process.returncode
        await self.stop()
        raise LocalInferenceUnavailableError(
            "Local inference did not become healthy within "
            f"{local.startup_timeout_seconds} seconds.",
            affected=log_path,
            details={
                "last_error": str(last_error) if last_error else None,
                "exit_code": exit_code,
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

    async def stop(self) -> None:
        if not self._owned:
            return
        if self.process is not None and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=20)
            except TimeoutError:
                self.process.kill()
                await self.process.wait()
        process_path = self.paths.jobs / "local-inference-process.json"
        self._mark_stopped(process_path)
        self._close_log()
        self._owned = False

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
