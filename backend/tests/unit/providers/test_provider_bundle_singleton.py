"""进程内唯一 Provider 组合（在线适配）."""

from __future__ import annotations

import pytest


def test_get_provider_bundle_is_cached() -> None:
    from rag_mcp.providers.factory import get_provider_bundle, reset_provider_bundle

    reset_provider_bundle()
    first = get_provider_bundle()
    second = get_provider_bundle()
    assert first is second


def test_reset_clears_cache() -> None:
    from rag_mcp.providers.factory import get_provider_bundle, reset_provider_bundle

    reset_provider_bundle()
    first = get_provider_bundle()
    reset_provider_bundle()
    assert get_provider_bundle() is not first


def test_cached_bundle_exposes_single_limiter_set(monkeypatch) -> None:
    """同一组合的 limiter 必须被所有使用者共享（FR-009）.

    两个能力都必须是 remote_api 才会携带 limiter（本地方案不带限流器），
    因此这里显式选择在线适配；本地 reranker 合法地没有 limiter，故加判空守卫。
    """
    from rag_mcp.providers.factory import get_provider_bundle, reset_provider_bundle

    monkeypatch.setenv("EMBEDDING_PROVIDER_TYPE", "remote_api")
    monkeypatch.setenv("EMBEDDING_ENDPOINT", "https://e.test/v1")
    monkeypatch.setenv("RERANKER_PROVIDER_TYPE", "remote_api")
    monkeypatch.setenv("RERANKER_ENDPOINT", "https://e.test/v1")
    reset_provider_bundle()
    bundle = get_provider_bundle()
    assert bundle.embedding._limiter is bundle.limiters["embedding"]
    if bundle.reranker._limiter is not None:
        assert bundle.reranker._limiter is bundle.limiters["reranker"]


def test_invalid_config_raises_on_get(monkeypatch) -> None:
    """无静默回退：非法配置在取用时显式失败（SC-004）."""
    from rag_mcp.providers.factory import get_provider_bundle, reset_provider_bundle

    monkeypatch.setenv("EMBEDDING_PROVIDER_TYPE", "remote_api")
    monkeypatch.delenv("EMBEDDING_ENDPOINT", raising=False)
    reset_provider_bundle()
    with pytest.raises(ValueError):
        get_provider_bundle()


def test_reset_is_idempotent() -> None:
    from rag_mcp.providers.factory import reset_provider_bundle

    reset_provider_bundle()
    reset_provider_bundle()  # 不抛


def test_limiter_is_usable_from_two_event_loops() -> None:
    """冒烟：同一个 limiter 对象可先后用于两个事件循环（不挂起、不抛错）.

    注意本用例**不能**证明循环绑定修复：limit=2 且只有两层嵌套获取，
    两次都命中信号量 fast path，``_loop`` 始终为 ``None``，因此修复前的实现
    同样通过（已实测复核）。真正的循环安全守卫是同一文件中的
    ``test_limiter_contention_across_two_event_loops``（它先让竞争者在 slow path
    上排队，把 ``_loop`` 绑定到首个循环）。
    """
    import asyncio

    from rag_mcp.providers.concurrency import ConcurrencyLimiter

    limiter = ConcurrencyLimiter(2)

    async def use_all():
        async with limiter:
            async with limiter:
                return "ok"

    assert asyncio.run(use_all()) == "ok"
    assert asyncio.run(use_all()) == "ok"      # 第二个循环：仅冒烟，非修复证明


def test_limiter_contention_across_two_event_loops() -> None:
    """超过上限的争用会把信号量绑定到某个循环；第二个循环仍须可用.

    这是 Change A 的真实 RED：嵌套获取在上限内只走信号量 fast path，
    永远不会绑定 ``_loop``（实测修复前该用例也通过）；只有争用才会走 slow
    path 并绑定循环。绑定后从第二个循环再次争用，旧实现抛
    ``RuntimeError: ... is bound to a different event loop``。
    """
    import asyncio

    from rag_mcp.providers.concurrency import ConcurrencyLimiter

    limiter = ConcurrencyLimiter(1)

    async def use_all():
        completed = []

        async def worker():
            async with limiter:
                completed.append("ok")

        # 先占住唯一许可（fast path），再让竞争者在 slow path 上真实排队，
        # 信号量因此把 _loop 绑定到本循环。
        async with limiter:
            parked = asyncio.ensure_future(worker())
            await asyncio.sleep(0)
            completed.append("ok")          # 持有者也确实拿到了许可
        await parked
        return sorted(completed)

    assert asyncio.run(use_all()) == ["ok", "ok"]
    assert asyncio.run(use_all()) == ["ok", "ok"]   # 第二个循环：绑定修复前抛 RuntimeError


@pytest.mark.asyncio
async def test_concurrent_first_callers_share_one_bundle(monkeypatch) -> None:
    """并发首次调用只构造一个组合，且返回同一对象."""
    import asyncio

    from rag_mcp.providers.factory import get_provider_bundle, reset_provider_bundle

    reset_provider_bundle()
    results = await asyncio.gather(*(asyncio.to_thread(get_provider_bundle) for _ in range(4)))
    assert all(r is results[0] for r in results)
