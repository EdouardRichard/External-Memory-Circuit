"""OpenAI-compatible remote reranker provider (006, T042).

FR-015 / research §1.5: a vendor-neutral /rerank adapter (Jina/Cohere
compatible shape). Credentials are referenced by environment variable NAME
only (constitution V). Any failure degrades to candidate passthrough —
never raising into the retrieval state machine.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from rag_mcp.providers.base import RerankerProvider

logger = logging.getLogger(__name__)


class RemoteAPIRerankerProvider(RerankerProvider):
    """Reranking via POST {endpoint}/rerank (OpenAI-compatible)."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key_env: str | None = None,
        timeout_s: float | None = 2.0,
        transport: Any = None,
        limiter: Any = None,
    ) -> None:
        self._endpoint = (endpoint or "").rstrip("/")
        self._model = model
        self._api_key_env = api_key_env
        # None means "not supplied" (env var unset) -> constructor default.
        self._timeout_s = timeout_s if timeout_s is not None and timeout_s > 0 else 2.0
        self._transport = transport
        # FR-009: per-capability concurrency ceiling; None means no limiting.
        self._limiter = limiter
        self.last_error: str | None = None
        self.calls = 0

    def _api_key(self) -> str:
        if not self._api_key_env:
            return ""
        return os.getenv(self._api_key_env, "")

    async def _post_rerank(self, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        """Perform the /rerank HTTP call and return the parsed JSON body."""
        async with httpx.AsyncClient(
            transport=self._transport, timeout=self._timeout_s
        ) as client:
            response = await client.post(
                f"{self._endpoint}/rerank", json=payload, headers=headers
            )
            response.raise_for_status()
            return response.json()

    async def _call_with_limiter(self, payload: dict[str, Any], headers: dict[str, str]) -> Any:
        """Enter the FR-009 limiter around the HTTP call only (None = unbounded)."""
        if self._limiter is None:
            return await self._post_rerank(payload, headers)
        async with self._limiter:
            return await self._post_rerank(payload, headers)

    def _normalize(self, score: float) -> float:
        """Clamp a relevance score into [0, 1] (FR-015 score normalization)."""
        if score is None:
            return 0.0
        return max(0.0, min(1.0, float(score)))

    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        self.calls += 1
        headers = {"Content-Type": "application/json"}
        key = self._api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        payload = {
            "model": self._model,
            "query": query,
            "documents": [c.get("text", "") for c in candidates],
            "top_n": max(1, top_k),
        }
        try:
            body = await self._call_with_limiter(payload, headers)
            results = body.get("results", [])
            scored: dict[int, float] = {
                int(r["index"]): self._normalize(r.get("relevance_score"))
                for r in results
            }
            reordered = []
            for i, candidate in enumerate(candidates):
                item = dict(candidate)
                item["rerank_score"] = scored.get(i, 0.0)
                reordered.append(item)
            reordered.sort(key=lambda c: c["rerank_score"], reverse=True)
            return reordered[: max(1, top_k)]
        except Exception as exc:  # noqa: BLE001 - passthrough degradation
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("remote reranker failed: %s", self.last_error)
            return candidates
