"""Client and controlled lifecycle for the local model gateway."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import urlparse

import httpx

from paperos_core.adapters.models.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    QueryExpansionRequest,
    QueryExpansionResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
)
from paperos_core.config import PaperOSConfig
from paperos_core.errors import (
    ModelGatewayConfigurationError,
    ModelGatewayResponseError,
    ModelGatewayUnavailableError,
)
from paperos_core.paths import DataPaths


class LocalModelGatewayClient:
    def __init__(self, endpoint: str, timeout_seconds: int) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.client = httpx.AsyncClient(
            base_url=self.endpoint,
            timeout=timeout_seconds,
            trust_env=False,
        )

    async def health(self) -> dict[str, Any]:
        try:
            response = await self.client.get("/health")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelGatewayUnavailableError(
                f"Local model gateway health check failed: {exc}",
                affected=self.endpoint,
            ) from exc
        if not isinstance(payload, dict) or payload.get("status") != "healthy":
            raise ModelGatewayResponseError(
                "Local model gateway returned an invalid health response.",
                affected=self.endpoint,
            )
        return payload

    async def embed(self, texts: list[str], *, expected_dimensions: int) -> list[list[float]]:
        if not texts or any(not value.strip() for value in texts):
            raise ValueError("Embedding input must contain non-empty text")
        request = EmbeddingRequest(input=texts)
        try:
            response = await self.client.post(
                "/v1/embeddings", json=request.model_dump(mode="json")
            )
            response.raise_for_status()
            payload = EmbeddingResponse.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelGatewayResponseError(
                f"Local embedding request failed: {exc}",
                affected=f"{self.endpoint}/v1/embeddings",
            ) from exc
        ordered = sorted(payload.data, key=lambda item: item.index)
        if len(ordered) != len(texts) or any(
            len(item.embedding) != expected_dimensions for item in ordered
        ):
            raise ModelGatewayResponseError(
                "Local embedding response has an unexpected count or dimension.",
                affected=f"{self.endpoint}/v1/embeddings",
                details={
                    "expected_count": len(texts),
                    "expected_dimensions": expected_dimensions,
                },
            )
        return [item.embedding for item in ordered]

    async def rerank(
        self,
        query: str,
        candidate_ids: list[str],
        texts: list[str],
        *,
        limit: int,
    ) -> list[RerankResult]:
        request = RerankRequest(
            query=query,
            candidate_ids=candidate_ids,
            texts=texts,
            limit=limit,
        )
        try:
            response = await self.client.post("/v1/rerank", json=request.model_dump(mode="json"))
            response.raise_for_status()
            payload = RerankResponse.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelGatewayResponseError(
                f"Local reranking request failed: {exc}",
                affected=f"{self.endpoint}/v1/rerank",
            ) from exc
        returned_ids = [item.candidate_id for item in payload.results]
        if len(returned_ids) != len(set(returned_ids)) or not set(returned_ids).issubset(
            candidate_ids
        ):
            raise ModelGatewayResponseError(
                "Local reranker returned invalid candidate IDs.",
                affected=f"{self.endpoint}/v1/rerank",
            )
        return payload.results

    async def expand_query(self, query: str, *, profile: str) -> QueryExpansionResponse:
        request = QueryExpansionRequest(query=query, profile=profile)
        try:
            response = await self.client.post(
                "/v1/query-expansion", json=request.model_dump(mode="json")
            )
            response.raise_for_status()
            return QueryExpansionResponse.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise ModelGatewayResponseError(
                f"Local query-expansion request failed: {exc}",
                affected=f"{self.endpoint}/v1/query-expansion",
            ) from exc

    async def aclose(self) -> None:
        await self.client.aclose()


class LocalModelGatewayProcess:
    def __init__(
        self,
        config: PaperOSConfig,
        paths: DataPaths,
        client: LocalModelGatewayClient,
    ) -> None:
        self.config = config
        self.paths = paths
        self.client = client
        self.process: asyncio.subprocess.Process | None = None
        self._log_stream: BinaryIO | None = None
        self._owned = False

    def _model_path(self, configured: Path, *, label: str) -> Path:
        configured = configured.expanduser()
        if configured.is_absolute():
            result = configured
        else:
            base = self.config.configured_data_dir or self.paths.root
            result = base / configured
        result = result.resolve(strict=False)
        if not result.is_file():
            raise ModelGatewayConfigurationError(
                f"Configured local {label} model file does not exist.",
                affected=result,
            )
        return result

    def _service_root(self) -> Path:
        if self.config.config_path is None:
            raise ModelGatewayConfigurationError(
                "A project config path is required to locate the local model service."
            )
        root = self.config.config_path.parent.parent / "services" / "local_models"
        server = root / "dist" / "server.js"
        if not server.is_file():
            raise ModelGatewayConfigurationError(
                "Compiled local model gateway is missing; run `npm run build` "
                "in services/local_models without installing dependencies.",
                affected=server,
            )
        return root

    def _endpoint(self) -> tuple[str, int]:
        parsed = urlparse(self.config.models.gateway_endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or parsed.port is None
        ):
            raise ModelGatewayConfigurationError(
                "The managed model gateway endpoint must be local HTTP with an explicit port.",
                affected=self.config.models.gateway_endpoint,
            )
        return parsed.hostname, parsed.port

    async def start(self) -> dict[str, Any]:
        try:
            return await self.client.health()
        except ModelGatewayUnavailableError:
            pass
        host, port = self._endpoint()
        service_root = self._service_root()
        model_path = self._model_path(self.config.models.embedding.model_path, label="embedding")
        reranker_path = self._model_path(self.config.models.reranker.model_path, label="reranker")
        query_expansion_path = self._model_path(
            self.config.models.query_expansion.model_path,
            label="query expansion",
        )
        log_path = self.paths.logs / "model-gateway.log"
        process_path = self.paths.jobs / "model-gateway-process.json"
        self._log_stream = log_path.open("ab", buffering=0)
        environment = dict(os.environ)
        environment.update(
            {
                "NODE_LLAMA_CPP_SKIP_DOWNLOAD": "true",
                "PAPEROS_MODEL_GATEWAY_HOST": host,
                "PAPEROS_MODEL_GATEWAY_PORT": str(port),
                "PAPEROS_EMBEDDING_MODEL_PATH": str(model_path),
                "PAPEROS_EMBEDDING_DIMENSIONS": str(self.config.models.embedding.dimensions),
                "PAPEROS_EMBEDDING_MAX_TOKENS": str(self.config.models.embedding.max_tokens),
                "PAPEROS_RERANKER_MODEL_PATH": str(reranker_path),
                "PAPEROS_RERANKER_MAX_TOKENS": "4096",
                "PAPEROS_QUERY_EXPANSION_MODEL_PATH": str(query_expansion_path),
                "PAPEROS_QUERY_EXPANSION_MAX_TOKENS": str(
                    self.config.models.query_expansion.max_output_tokens
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
            self._log_stream.close()
            self._log_stream = None
            raise ModelGatewayUnavailableError(
                f"Unable to start local model gateway: {exc}",
                affected=service_root,
            ) from exc
        self._owned = True
        self._write_process_record(
            process_path,
            {
                "pid": self.process.pid,
                "status": "starting",
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": str(log_path),
                "endpoint": self.config.models.gateway_endpoint,
                "model_path": str(model_path),
                "reranker_model_path": str(reranker_path),
                "query_expansion_model_path": str(query_expansion_path),
            },
        )
        last_error: Exception | None = None
        for _ in range(180):
            if self.process.returncode is not None:
                break
            try:
                health = await self.client.health()
                record = json.loads(process_path.read_text(encoding="utf-8"))
                record["status"] = "running"
                self._write_process_record(process_path, record)
                return health
            except ModelGatewayUnavailableError as exc:
                last_error = exc
                await asyncio.sleep(1)
        await self.stop()
        raise ModelGatewayUnavailableError(
            "Local model gateway did not become healthy within 180 seconds.",
            affected=log_path,
            details={"last_error": str(last_error) if last_error else None},
        )

    async def serve_forever(
        self,
        *,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        """Run the repository Node gateway in the foreground until signalled."""
        configured_host, configured_port = self._endpoint()
        selected_host = host or configured_host
        selected_port = port or configured_port
        if not selected_host.strip():
            raise ModelGatewayConfigurationError(
                "Local model gateway host must not be empty.", affected=selected_host
            )
        if not 1 <= selected_port <= 65535:
            raise ModelGatewayConfigurationError(
                "Local model gateway port must be between 1 and 65535.",
                affected=str(selected_port),
            )
        await self._assert_port_available(selected_host, selected_port)
        service_root = self._service_root()
        model_path = self._model_path(
            self.config.models.embedding.model_path, label="embedding"
        )
        reranker_path = self._model_path(
            self.config.models.reranker.model_path, label="reranker"
        )
        query_expansion_path = self._model_path(
            self.config.models.query_expansion.model_path,
            label="query expansion",
        )
        endpoint = f"http://{selected_host}:{selected_port}"
        environment = dict(os.environ)
        environment.update(
            {
                "NODE_LLAMA_CPP_SKIP_DOWNLOAD": "true",
                "PAPEROS_MODEL_GATEWAY_HOST": selected_host,
                "PAPEROS_MODEL_GATEWAY_PORT": str(selected_port),
                "PAPEROS_EMBEDDING_MODEL_PATH": str(model_path),
                "PAPEROS_EMBEDDING_DIMENSIONS": str(
                    self.config.models.embedding.dimensions
                ),
                "PAPEROS_EMBEDDING_MAX_TOKENS": str(
                    self.config.models.embedding.max_tokens
                ),
                "PAPEROS_RERANKER_MODEL_PATH": str(reranker_path),
                "PAPEROS_RERANKER_MAX_TOKENS": "4096",
                "PAPEROS_QUERY_EXPANSION_MODEL_PATH": str(query_expansion_path),
                "PAPEROS_QUERY_EXPANSION_MAX_TOKENS": str(
                    self.config.models.query_expansion.max_output_tokens
                ),
            }
        )
        process_path = self.paths.jobs / "model-gateway-process.json"
        try:
            self.process = await asyncio.create_subprocess_exec(
                "node",
                "dist/server.js",
                cwd=service_root,
                env=environment,
            )
        except OSError as exc:
            raise ModelGatewayUnavailableError(
                f"Unable to start local model gateway: {exc}",
                affected=service_root,
            ) from exc
        self._owned = True
        self._write_process_record(
            process_path,
            {
                "pid": self.process.pid,
                "status": "starting",
                "started_at": datetime.now(UTC).isoformat(),
                "log_path": None,
                "endpoint": endpoint,
                "model_path": str(model_path),
                "reranker_model_path": str(reranker_path),
                "query_expansion_model_path": str(query_expansion_path),
            },
        )
        loop = asyncio.get_running_loop()

        def forward(received: signal.Signals) -> None:
            if self.process is not None and self.process.returncode is None:
                self.process.send_signal(received)

        installed_signals: list[signal.Signals] = []
        for received in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(received, forward, received)
                installed_signals.append(received)
            except NotImplementedError:
                pass
        try:
            health = await self._wait_for_health(endpoint, timeout_seconds=180)
            record = json.loads(process_path.read_text(encoding="utf-8"))
            record.update({"status": "running", "health": health})
            self._write_process_record(process_path, record)
            print(f"Model gateway listening on {endpoint}", flush=True)
            return_code = await self.process.wait()
            if return_code != 0:
                raise ModelGatewayUnavailableError(
                    f"Local model gateway exited with code {return_code}.",
                    affected=endpoint,
                    retryable=False,
                )
        finally:
            for received in installed_signals:
                loop.remove_signal_handler(received)
            if self.process is not None and self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=20)
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
            self._mark_stopped(process_path)
            self._owned = False

    async def _assert_port_available(self, host: str, port: int) -> None:
        try:
            _reader, writer = await asyncio.open_connection(host, port)
        except OSError:
            return
        writer.close()
        await writer.wait_closed()
        raise ModelGatewayUnavailableError(
            f"Cannot start model gateway because {host}:{port} is already in use.",
            affected=f"{host}:{port}",
            retryable=False,
        )

    async def _wait_for_health(
        self, endpoint: str, *, timeout_seconds: int
    ) -> dict[str, Any]:
        last_error: str | None = None
        async with httpx.AsyncClient(
            base_url=endpoint, timeout=2, trust_env=False
        ) as client:
            for _ in range(timeout_seconds):
                if self.process is None or self.process.returncode is not None:
                    break
                try:
                    response = await client.get("/health")
                    response.raise_for_status()
                    payload = response.json()
                    if isinstance(payload, dict) and payload.get("status") == "healthy":
                        return payload
                    last_error = "health payload did not report healthy"
                except (httpx.HTTPError, ValueError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                await asyncio.sleep(1)
        return_code = self.process.returncode if self.process is not None else None
        raise ModelGatewayUnavailableError(
            f"Local model gateway did not become healthy within {timeout_seconds} seconds.",
            affected=endpoint,
            details={"last_error": last_error, "exit_code": return_code},
        )

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
        process_path = self.paths.jobs / "model-gateway-process.json"
        self._mark_stopped(process_path)
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None
        self._owned = False

    @staticmethod
    def _write_process_record(path: Path, payload: dict[str, Any]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
