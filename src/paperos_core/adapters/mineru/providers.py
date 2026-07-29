"""MinerU provider protocol and live MinerU Cloud v4 implementation."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

import httpx

from paperos_core.adapters.mineru.schemas import MinerUParseResult, MinerUTask
from paperos_core.config import MinerUConfig
from paperos_core.domain.documents import SourceFile
from paperos_core.errors import (
    MinerUAuthenticationError,
    MinerUConfigurationError,
    MinerUParseError,
    MinerUProviderError,
    MinerUQuotaError,
)

logger = logging.getLogger(__name__)
DEFAULT_MINERU_CLOUD_ENDPOINT = "https://mineru.net"
_RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}
_AUTH_CODES = {"A0202", "A0211"}
_QUOTA_CODES = {"-60018", "-60019"}


class MinerUProvider(Protocol):
    async def health_check(self) -> dict[str, Any]: ...

    async def submit_pdf(
        self, source: SourceFile, *, request_options: dict[str, Any]
    ) -> MinerUTask: ...

    async def get_task_status(self, task: MinerUTask) -> MinerUTask: ...

    async def fetch_result(
        self, task: MinerUTask, *, poll_history: list[dict[str, Any]]
    ) -> MinerUParseResult: ...

    async def aclose(self) -> None: ...


class MinerUCloudProvider:
    """Live token-authenticated local-file upload through MinerU Cloud v4."""

    name = "mineru_cloud"

    def __init__(self, config: MinerUConfig, *, max_retries: int = 3) -> None:
        self.config = config
        self.endpoint = (config.endpoint or DEFAULT_MINERU_CLOUD_ENDPOINT).rstrip("/")
        self.api_key = config.api_key_value()
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_seconds, connect=30.0),
            follow_redirects=True,
        )
        # Signed object-storage transfers can be much larger than control-plane
        # requests. Bypass ambient HTTP proxies for this data plane so a proxy
        # cannot truncate immutable PDF uploads or parser archives.
        self._transfer_client = httpx.AsyncClient(
            timeout=httpx.Timeout(config.timeout_seconds, connect=30.0),
            follow_redirects=True,
            trust_env=False,
        )

    def _require_key(self) -> str:
        if not self.api_key:
            raise MinerUConfigurationError(
                "MinerU Cloud requires a non-empty mineru_ocr.api_key or its "
                f"{self.config.api_key_env} environment override.",
                affected="config/paperos.toml",
            )
        return self.api_key

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._require_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    async def health_check(self) -> dict[str, Any]:
        self._require_key()
        return {
            "provider": self.name,
            "endpoint": self.endpoint,
            "configured": True,
        }

    async def _request(
        self,
        method: str,
        url: str,
        *,
        client: httpx.AsyncClient | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        last_error: Exception | None = None
        request_client = client or self._client
        for attempt in range(1, self.max_retries + 1):
            try:
                response = await request_client.request(method, url, **kwargs)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                delay = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "MinerU transport attempt %s/%s failed: %s; retrying in %ss",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            if response.status_code in {401, 403}:
                raise MinerUAuthenticationError(
                    "MinerU rejected the configured API token.",
                    affected=url,
                    details={"http_status": response.status_code},
                )
            if response.status_code in _RETRYABLE_HTTP:
                last_error = MinerUProviderError(
                    f"MinerU returned retryable HTTP {response.status_code}.",
                    affected=url,
                    details={"http_status": response.status_code},
                )
                if attempt < self.max_retries:
                    delay = min(2 ** (attempt - 1), 8)
                    logger.warning(
                        "MinerU HTTP attempt %s/%s returned %s; retrying in %ss",
                        attempt,
                        self.max_retries,
                        response.status_code,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
            if response.is_error:
                raise MinerUProviderError(
                    f"MinerU returned HTTP {response.status_code}: {response.text[:500]}",
                    affected=url,
                    retryable=False,
                    details={"http_status": response.status_code},
                )
            return response
        raise MinerUProviderError(
            f"MinerU request failed after {self.max_retries} attempts: {last_error}",
            affected=url,
        ) from last_error

    @staticmethod
    def _payload(response: httpx.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            raise MinerUProviderError(
                "MinerU returned a non-JSON API response.",
                affected=str(response.request.url),
                retryable=False,
            ) from exc
        if not isinstance(payload, dict):
            raise MinerUProviderError(
                "MinerU returned an invalid API response shape.",
                affected=str(response.request.url),
                retryable=False,
            )
        code = str(payload.get("code", ""))
        if payload.get("code") != 0:
            message = str(payload.get("msg") or "unknown MinerU API error")
            details = {"provider_code": code, "trace_id": payload.get("trace_id")}
            if code in _AUTH_CODES:
                raise MinerUAuthenticationError(
                    f"MinerU authentication failed: {message}", details=details
                )
            if code in _QUOTA_CODES:
                raise MinerUQuotaError(f"MinerU quota error: {message}", details=details)
            raise MinerUProviderError(
                f"MinerU API error {code}: {message}",
                retryable=code in {"-10001", "-60007", "-60009"},
                details=details,
            )
        return payload

    @staticmethod
    async def _file_content(path: Path) -> AsyncIterator[bytes]:
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                yield block

    async def submit_pdf(
        self, source: SourceFile, *, request_options: dict[str, Any]
    ) -> MinerUTask:
        backend = str(request_options.get("model_version") or self.config.preferred_backend)
        if backend == "auto":
            backend = "vlm"
        body = {
            "files": [
                {
                    "name": source.original_filename,
                    "data_id": source.id,
                    "is_ocr": bool(request_options.get("is_ocr", False)),
                }
            ],
            "model_version": backend,
            "enable_formula": bool(request_options.get("enable_formula", True)),
            "enable_table": bool(request_options.get("enable_table", True)),
            "language": str(request_options.get("language", "en")),
        }
        response = await self._request(
            "POST",
            f"{self.endpoint}/api/v4/file-urls/batch",
            headers=self._headers,
            json=body,
        )
        payload = self._payload(response)
        data = payload.get("data")
        if not isinstance(data, dict):
            raise MinerUProviderError("MinerU upload response is missing data.", retryable=False)
        batch_id = data.get("batch_id")
        file_urls = data.get("file_urls")
        if not isinstance(batch_id, str) or not isinstance(file_urls, list) or not file_urls:
            raise MinerUProviderError(
                "MinerU upload response is missing batch_id or file_urls.", retryable=False
            )
        upload_url = file_urls[0]
        if not isinstance(upload_url, str):
            raise MinerUProviderError("MinerU returned an invalid signed upload URL.")
        upload_response = await self._request(
            "PUT",
            upload_url,
            client=self._transfer_client,
            content=self._file_content(source.storage_path),
            headers={},
        )
        if upload_response.status_code not in {200, 201, 204}:
            raise MinerUProviderError(
                f"MinerU signed upload failed with HTTP {upload_response.status_code}.",
                affected=upload_url,
            )
        return MinerUTask(
            provider=self.name,
            task_id=batch_id,
            state="submitted",
            backend=backend,
            data_id=source.id,
            raw_metadata={
                "submission": payload,
                "request_options": body,
                "upload_http_status": upload_response.status_code,
            },
        )

    async def get_task_status(self, task: MinerUTask) -> MinerUTask:
        response = await self._request(
            "GET",
            f"{self.endpoint}/api/v4/extract-results/batch/{task.task_id}",
            headers=self._headers,
        )
        payload = self._payload(response)
        data = payload.get("data")
        results = data.get("extract_result") if isinstance(data, dict) else None
        if not isinstance(results, list) or not results:
            return task.model_copy(
                update={
                    "state": "pending",
                    "raw_metadata": {**task.raw_metadata, "status": payload},
                }
            )
        matching = next(
            (
                item
                for item in results
                if isinstance(item, dict) and item.get("data_id") == task.data_id
            ),
            results[0],
        )
        if not isinstance(matching, dict):
            raise MinerUProviderError("MinerU task result has an invalid shape.")
        state = str(matching.get("state") or "pending")
        return task.model_copy(
            update={
                "state": state,
                "result_archive_url": matching.get("full_zip_url"),
                "error_code": (
                    str(matching["err_code"]) if matching.get("err_code") is not None else None
                ),
                "error_message": matching.get("err_msg") or None,
                "progress": matching.get("extract_progress"),
                "raw_metadata": {
                    **task.raw_metadata,
                    "status": payload,
                    "result": matching,
                },
            }
        )

    async def fetch_result(
        self, task: MinerUTask, *, poll_history: list[dict[str, Any]]
    ) -> MinerUParseResult:
        if task.state != "done" or not task.result_archive_url:
            raise MinerUParseError(
                "Cannot fetch MinerU result before a completed task has an archive URL.",
                affected=task.task_id,
            )
        response = await self._request(
            "GET", task.result_archive_url, client=self._transfer_client
        )
        archive = response.content
        if len(archive) < 4 or not archive.startswith(b"PK"):
            raise MinerUParseError(
                "MinerU result is not a valid ZIP archive.", affected=task.result_archive_url
            )
        return MinerUParseResult(
            provider=self.name,
            provider_task_id=task.task_id,
            backend=task.backend,
            archive_bytes=archive,
            final_metadata=task.raw_metadata,
            poll_history=poll_history,
        )

    async def aclose(self) -> None:
        await self._client.aclose()
        await self._transfer_client.aclose()
