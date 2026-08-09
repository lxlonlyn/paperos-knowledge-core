"""HTTP client for the private loopback inference child process."""

from __future__ import annotations

from typing import Any

import httpx

from paperos_core.errors import (
    LocalInferenceResponseError,
    LocalInferenceUnavailableError,
)
from paperos_core.runtime.local_inference.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    RerankRequest,
    RerankResponse,
    RerankResult,
)


class LocalInferenceClient:
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
            raise LocalInferenceUnavailableError(
                f"Local inference health check failed: {exc}",
                affected=self.endpoint,
            ) from exc
        if not isinstance(payload, dict) or payload.get("status") != "healthy":
            raise LocalInferenceResponseError(
                "Local inference returned an invalid health response.",
                affected=self.endpoint,
            )
        return payload

    async def shutdown(self, token: str) -> None:
        """Ask the parent-owned child to shut down through its private protocol."""

        try:
            response = await self.client.post(
                "/internal/shutdown",
                headers={"x-paperos-shutdown-token": token},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LocalInferenceUnavailableError(
                f"Local inference shutdown request failed: {exc}",
                affected=f"{self.endpoint}/internal/shutdown",
            ) from exc

    async def embed(
        self, texts: list[str], *, expected_dimensions: int
    ) -> list[list[float]]:
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
            raise LocalInferenceResponseError(
                f"Local embedding request failed: {exc}",
                affected=f"{self.endpoint}/v1/embeddings",
            ) from exc
        ordered = sorted(payload.data, key=lambda item: item.index)
        if len(ordered) != len(texts) or any(
            len(item.embedding) != expected_dimensions for item in ordered
        ):
            raise LocalInferenceResponseError(
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
            response = await self.client.post(
                "/v1/rerank", json=request.model_dump(mode="json")
            )
            response.raise_for_status()
            payload = RerankResponse.model_validate(response.json())
        except (httpx.HTTPError, ValueError) as exc:
            raise LocalInferenceResponseError(
                f"Local reranking request failed: {exc}",
                affected=f"{self.endpoint}/v1/rerank",
            ) from exc
        returned_ids = [item.candidate_id for item in payload.results]
        if len(returned_ids) != len(set(returned_ids)) or not set(returned_ids).issubset(
            candidate_ids
        ):
            raise LocalInferenceResponseError(
                "Local reranker returned invalid candidate IDs.",
                affected=f"{self.endpoint}/v1/rerank",
            )
        return payload.results

    async def aclose(self) -> None:
        await self.client.aclose()
