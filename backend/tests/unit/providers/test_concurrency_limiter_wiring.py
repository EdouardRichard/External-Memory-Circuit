"""FR-009：并发上限必须真正生效（此前 bundle.limiters 是死代码）."""

from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest

from rag_mcp.providers.concurrency import ConcurrencyLimiter, SyncConcurrencyLimiter
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


# ---------------------------------------------------------------------------
# Fix 3: ``LLMClient`` is synchronous (``httpx.Client``), so it cannot use the
# async ``ConcurrencyLimiter``. ``SyncConcurrencyLimiter`` is its thread-safe
# counterpart, and ``build_provider_bundle`` must hand it to the LLM client or
# ``LLM_CONCURRENCY_LIMIT`` stays inert (FR-009 / §3.5).
# ---------------------------------------------------------------------------


class _SyncTracker:
    """Thread-safe peak counter driven by real threads."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.current = 0
        self.peak = 0
        self.admitted = 0
        self.release = threading.Event()

    def enter(self) -> None:
        with self.lock:
            self.current += 1
            self.admitted += 1
            self.peak = max(self.peak, self.current)

    def leave(self) -> None:
        with self.lock:
            self.current -= 1


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _llm_handler(tracker: _SyncTracker):
    def handler(request: httpx.Request) -> httpx.Response:
        tracker.enter()
        tracker.release.wait(timeout=5)
        tracker.leave()
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"ok": true}'}}]}
        )

    return handler


def _install_transport(monkeypatch, handler) -> None:
    """Route ``LLMClient``'s internal ``httpx.Client`` through a MockTransport."""
    import rag_mcp.agents.llm_client as llm_client_module

    real_client = httpx.Client

    def _factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(llm_client_module.httpx, "Client", _factory)


def _spawn_llm_calls(client, count: int) -> tuple[list[threading.Thread], list]:
    results: list = []

    def call(index: int) -> None:
        results.append(client.chat_json("sys", {"i": index}))

    threads = [threading.Thread(target=call, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    return threads, results


def test_sync_limiter_caps_concurrent_holders() -> None:
    """``SyncConcurrencyLimiter(2)`` 用真实线程证明同时持有者不超过 2."""
    limiter = SyncConcurrencyLimiter(2)
    tracker = _SyncTracker()
    completed: list[int] = []

    def worker(index: int) -> None:
        with limiter:
            tracker.enter()
            tracker.release.wait(timeout=5)
            tracker.leave()
        completed.append(index)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(5)]
    for thread in threads:
        thread.start()
    try:
        assert _wait_for(lambda: tracker.admitted >= 2), "limiter never admitted two holders"
        # 观察窗口：上限若失效，peak 会在排队者涌入时继续涨到 5。
        time.sleep(0.2)
        assert tracker.peak == 2, f"limiter(2) admitted peak={tracker.peak}"
    finally:
        tracker.release.set()
        for thread in threads:
            thread.join(timeout=5)
    assert not any(thread.is_alive() for thread in threads)
    assert sorted(completed) == [0, 1, 2, 3, 4]   # 排队者最终都被放行
    assert limiter.limit == 2


def test_sync_limiter_rejects_non_positive_limit() -> None:
    """与异步 limiter 相同的入参校验：limit < 1 显式失败."""
    for bad in (0, -1):
        with pytest.raises(ValueError):
            SyncConcurrencyLimiter(bad)


def test_llm_client_respects_injected_sync_limiter(monkeypatch) -> None:
    """注入 limiter 后，LLMClient 的实际 HTTP 并发不超过上限."""
    from rag_mcp.agents.llm_client import LLMClient

    tracker = _SyncTracker()
    _install_transport(monkeypatch, _llm_handler(tracker))
    client = LLMClient(
        base_url="http://api.test/v1",
        api_key="",
        model="m",
        limiter=SyncConcurrencyLimiter(2),
    )
    threads, results = _spawn_llm_calls(client, 5)
    try:
        assert _wait_for(lambda: tracker.admitted >= 2)
        time.sleep(0.2)
        assert tracker.peak <= 2, f"limiter(2) admitted peak={tracker.peak}"
    finally:
        tracker.release.set()
        for thread in threads:
            thread.join(timeout=5)
    assert tracker.admitted == 5
    assert results == [{"ok": True}] * 5


def test_llm_client_without_limiter_is_unbounded(monkeypatch) -> None:
    """未注入 limiter 时行为不变：并发不受限（对照组）."""
    from rag_mcp.agents.llm_client import LLMClient

    tracker = _SyncTracker()
    _install_transport(monkeypatch, _llm_handler(tracker))
    client = LLMClient(base_url="http://api.test/v1", api_key="", model="m")
    threads, results = _spawn_llm_calls(client, 5)
    try:
        assert _wait_for(lambda: tracker.current >= 3), "unbounded client never ran 3 calls at once"
        assert tracker.peak >= 3
    finally:
        tracker.release.set()
        for thread in threads:
            thread.join(timeout=5)
    assert results == [{"ok": True}] * 5


def test_factory_build_llm_passes_limiter() -> None:
    """``_build_llm`` 必须接收并应用 limiter；不传时行为不变."""
    from types import SimpleNamespace

    from rag_mcp.config.provider_config import load_provider_settings
    from rag_mcp.providers import factory

    providers = load_provider_settings(
        env={"LLM_PROVIDER_TYPE": "remote_api", "LLM_ENDPOINT": "https://l.test/v1"}
    )
    settings = SimpleNamespace(llm_base_url="", llm_api_key="", llm_model="m")
    limiter = SyncConcurrencyLimiter(3)

    client = factory._build_llm(providers.llm, settings, limiter=limiter)
    assert client._limiter is limiter
    assert factory._build_llm(providers.llm, settings)._limiter is None


def test_build_provider_bundle_wires_llm_limiter(monkeypatch) -> None:
    """bundle 必须把 ``limiters["llm"]`` 交给同一个 LLMClient（不再是死代码）."""
    from rag_mcp.config import Settings
    from rag_mcp.providers import factory

    monkeypatch.setenv("LLM_PROVIDER_TYPE", "remote_api")
    monkeypatch.setenv("LLM_ENDPOINT", "https://l.test/v1")
    bundle = factory.build_provider_bundle(Settings())

    assert isinstance(bundle.limiters["llm"], SyncConcurrencyLimiter)
    assert bundle.llm._limiter is bundle.limiters["llm"]
