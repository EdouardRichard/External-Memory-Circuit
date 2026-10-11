"""FR-009：并发上限必须真正生效（此前 bundle.limiters 是死代码）."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from rag_mcp.providers.concurrency import ConcurrencyLimiter
from rag_mcp.providers.remote_api_embedding import RemoteAPIEmbeddingProvider
from rag_mcp.providers.remote_api_reranker import RemoteAPIRerankerProvider


class _Tracker:
    """记录同时在 HTTP 调用中的峰值."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self.release = asyncio.Event()
        self.admitted = 0

    def enter(self) -> None:
        self.current += 1
        self.admitted += 1
        self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        self.current -= 1


def _embed_transport(tracker: _Tracker) -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        tracker.enter()
        await tracker.release.wait()
        tracker.leave()
        return httpx.Response(200, json={"data": [{"embedding": [0.0, 0.0], "index": 0}]})

    return httpx.MockTransport(handler)


async def _drive_embedding(provider, tracker: _Tracker, count: int) -> None:
    tasks = [asyncio.create_task(provider.embed_texts([f"t{i}"])) for i in range(count)]
    await asyncio.sleep(0.1)          # 让所有任务推进到 HTTP 调用点
    tracker.release.set()
    await asyncio.gather(*tasks)


@pytest.mark.asyncio
async def test_embedding_respects_limiter() -> None:
    tracker = _Tracker()
    provider = RemoteAPIEmbeddingProvider(
        endpoint="http://api.test/v1",
        model="m",
        limiter=ConcurrencyLimiter(2),
        transport=_embed_transport(tracker),
    )
    await _drive_embedding(provider, tracker, 5)
    assert tracker.admitted == 5
    assert tracker.peak <= 2


@pytest.mark.asyncio
async def test_no_limiter_means_unbounded() -> None:
    tracker = _Tracker()
    provider = RemoteAPIEmbeddingProvider(
        endpoint="http://api.test/v1", model="m", transport=_embed_transport(tracker)
    )
    await _drive_embedding(provider, tracker, 5)
    assert tracker.admitted == 5
    assert tracker.peak >= 3


@pytest.mark.asyncio
async def test_reranker_respects_limiter() -> None:
    tracker = _Tracker()

    async def handler(request: httpx.Request) -> httpx.Response:
        tracker.enter()
        await tracker.release.wait()
        tracker.leave()
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})

    provider = RemoteAPIRerankerProvider(
        endpoint="http://api.test/v1",
        model="m",
        limiter=ConcurrencyLimiter(1),
        transport=httpx.MockTransport(handler),
    )
    tasks = [asyncio.create_task(provider.rerank("q", [{"text": "a"}], top_k=1)) for _ in range(4)]
    await asyncio.sleep(0.1)
    assert tracker.peak == 1
    tracker.release.set()
    await asyncio.gather(*tasks)


def test_factory_injects_limiter_for_remote_embedding() -> None:
    """factory 必须把对应 capability 的 limiter 传给 remote provider."""
    from rag_mcp.config.provider_config import load_provider_settings
    from rag_mcp.providers import factory

    providers = load_provider_settings(
        env={"EMBEDDING_PROVIDER_TYPE": "remote_api", "EMBEDDING_ENDPOINT": "https://e.test/v1"}
    )
    limiter = ConcurrencyLimiter(3)
    provider = factory._build_embedding(providers.embedding, limiter=limiter)
    assert provider._limiter is limiter


@pytest.mark.asyncio
async def test_dimension_probe_shares_the_limiter() -> None:
    """Review Focus #4：维度探测请求同样受上限约束，不出现"限制 2 却瞬时 3 个"."""
    tracker = _Tracker()
    provider = RemoteAPIEmbeddingProvider(
        endpoint="http://api.test/v1",
        model="m",
        limiter=ConcurrencyLimiter(2),
        transport=_embed_transport(tracker),
    )
    # 两个业务调用 + 一个维度探测并发
    tasks = [
        asyncio.create_task(provider.embed_texts(["a"])),
        asyncio.create_task(provider.embed_texts(["b"])),
        asyncio.create_task(provider.resolve_dimension()),
    ]
    await asyncio.sleep(0.1)
    assert tracker.peak <= 2
    tracker.release.set()
    await asyncio.gather(*tasks)
    assert provider.get_dimension() == 2      # 探测结果被缓存
