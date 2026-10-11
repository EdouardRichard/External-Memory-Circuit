"""Per-capability concurrency limiting (006, T046).

FR-009: each Provider capability (LLM / Embedding / Reranker) has an
independent concurrency ceiling (clarification Q2: LLM 4/8, Embedding 8/16,
Reranker 2/4). A bounded asyncio semaphore queues callers beyond the limit;
the three ceilings never interfere with one another.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from rag_mcp.config.provider_config import ProviderSettings


class ConcurrencyLimiter:
    """Async context-manager bound to one capability's concurrency ceiling.

    A semaphore is created per running event loop: a single process runs several loops
    (asyncio.run scopes, background threads), and reusing one asyncio.Semaphore across loops
    either raises "bound to a different event loop" or, worse, parks the waiter forever on a
    dead loop. Per-loop semaphores keep the limiter safe to share from the process-wide
    ProviderBundle.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"concurrency limit must be >= 1, got {limit}")
        self._limit = limit
        self._semaphores: dict[Any, asyncio.Semaphore] = {}

    @property
    def limit(self) -> int:
        return self._limit

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        semaphore = self._semaphores.get(loop)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self._limit)
            self._semaphores[loop] = semaphore
        return semaphore

    async def __aenter__(self) -> "ConcurrencyLimiter":
        await self._semaphore().acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self._semaphore().release()


class SyncConcurrencyLimiter:
    """Synchronous counterpart of :class:`ConcurrencyLimiter`, for threading callers.

    ``LLMClient`` is synchronous (it uses ``httpx.Client``, not
    ``AsyncClient``), so it cannot enter an ``asyncio.Semaphore``. One plain
    ``threading.Semaphore`` is the correct primitive there: unlike the async
    limiter — which must build a semaphore per running event loop, because an
    ``asyncio.Semaphore`` bound to a dead loop parks its waiters forever — a
    ``threading.Semaphore`` may be acquired and released by any thread and
    carries no loop affinity. A single instance can therefore be shared
    process-wide (the ProviderBundle) exactly like the async limiters.

    Same contract as the async limiter: ``limit < 1`` fails loudly, ``limit``
    is exposed, and the instance is a context manager.
    """

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"concurrency limit must be >= 1, got {limit}")
        self._limit = limit
        self._semaphore = threading.Semaphore(limit)

    @property
    def limit(self) -> int:
        return self._limit

    def __enter__(self) -> "SyncConcurrencyLimiter":
        self._semaphore.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._semaphore.release()


def build_limiters(
    providers: ProviderSettings,
) -> dict[str, ConcurrencyLimiter | SyncConcurrencyLimiter]:
    """Assemble one independent limiter per capability from run config.

    Embedding/reranker callers are async (``ConcurrencyLimiter``); the LLM
    client is synchronous (``SyncConcurrencyLimiter``). All three are still
    independent ceilings (FR-009).
    """
    return {
        "embedding": ConcurrencyLimiter(providers.embedding.concurrency_limit),
        "reranker": ConcurrencyLimiter(providers.reranker.concurrency_limit),
        "llm": SyncConcurrencyLimiter(providers.llm.concurrency_limit),
    }
