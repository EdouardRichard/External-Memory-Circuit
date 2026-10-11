"""OpenAI-compatible remote embedding provider (006, T040).

FR-015 / research §1.5: a vendor-neutral /embeddings adapter. The credential
is referenced by environment variable NAME only (constitution V); the value
never enters this object. Any failure — HTTP error, timeout, malformed body
— degrades to an empty vector list (no matches) instead of raising into the
retrieval state machine.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx

from rag_mcp.providers.base import EmbeddingProvider

logger = logging.getLogger(__name__)


class RemoteAPIEmbeddingProvider(EmbeddingProvider):
    """Embedding via POST {endpoint}/embeddings (OpenAI-compatible)."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key_env: str | None = None,
        timeout_s: float | None = 2.0,
        dimension: int | None = None,
        transport: Any = None,
        limiter: Any = None,
    ) -> None:
        self._endpoint = (endpoint or "").rstrip("/")
        self._model = model
        self._api_key_env = api_key_env
        # None means "not supplied" (env var unset) -> constructor default.
        self._timeout_s = timeout_s if timeout_s is not None and timeout_s > 0 else 2.0
        self._dimension = dimension
        self._transport = transport
        # FR-009: per-capability concurrency ceiling; None means no limiting.
        self._limiter = limiter
        self.last_error: str | None = None
        self.calls = 0

    def _api_key(self) -> str:
        if not self._api_key_env:
            return ""
        return os.getenv(self._api_key_env, "")

    async def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=self._transport, timeout=self._timeout_s)

    async def _post_embeddings(self, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        """Perform the /embeddings HTTP call and return the parsed JSON body."""
        async with await self._client() as client:
            response = await client.post(
                f"{self._endpoint}/embeddings", json=payload, headers=headers
            )
            response.raise_for_status()
            return response.json()

    async def _call_with_limiter(self, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        """Enter the FR-009 limiter around the HTTP call only (None = unbounded)."""
        if self._limiter is None:
            return await self._post_embeddings(payload, headers)
        async with self._limiter:
            return await self._post_embeddings(payload, headers)

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self.calls += 1
        headers = {"Content-Type": "application/json"}
        key = self._api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {"model": self._model, "input": texts}
        try:
            body = await self._call_with_limiter(payload, headers)
            data = body.get("data", [])
            vectors: list[list[float]] = []
            for item in sorted(data, key=lambda d: d.get("index", 0)):
                vectors.append([float(x) for x in item["embedding"]])
            if vectors and self._dimension is None:
                self._dimension = len(vectors[0])
            if len(vectors) != len(texts):
                raise ValueError("embedding response item count mismatch")
            return vectors
        except Exception as exc:  # noqa: BLE001 - degrade, never raise into state machine
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("remote embedding failed: %s", self.last_error)
            return []

    async def embed_query(self, text: str) -> list[float]:
        vectors = await self.embed_texts([text])
        return vectors[0] if vectors else []

    def get_dimension(self) -> int:
        if self._dimension is None:
            return 0
        return self._dimension

    async def resolve_dimension(self) -> int:
        """Return the collection dimension, probing the endpoint once if unknown.

        A dimension declared in configuration (or already learned from an
        earlier call) is returned directly. Otherwise a single probe request
        determines it. Reuses ``embed_texts`` so the existing degradation path
        applies: any failure returns 0 with ``last_error`` set, and callers
        must not build a collection with 0.
        """
        if self._dimension:
            return self._dimension
        await self.embed_texts(["dimension probe"])  # 复用既有降级路径，失败返回 []
        return self._dimension or 0
