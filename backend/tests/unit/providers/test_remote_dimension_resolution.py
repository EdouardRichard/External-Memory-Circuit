"""远程 embedding 维度解析：可声明 + 首次调用后确定（在线适配）."""

from __future__ import annotations

import httpx
import pytest

from rag_mcp.providers.remote_api_embedding import RemoteAPIEmbeddingProvider


def _provider(handler, **kwargs):
    return RemoteAPIEmbeddingProvider(
        endpoint="http://api.test/v1",
        model="test-embedder",
        api_key_env=None,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_declared_dimension_never_probes() -> None:
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"data": [{"embedding": [0.0] * 7, "index": 0}]})

    provider = _provider(handler, dimension=1024)
    assert await provider.resolve_dimension() == 1024
    assert calls["n"] == 0


@pytest.mark.asyncio
async def test_probe_determines_and_caches_dimension() -> None:
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"data": [{"embedding": [0.0] * 1024, "index": 0}]})

    provider = _provider(handler)
    assert provider.get_dimension() == 0
    assert await provider.resolve_dimension() == 1024
    assert await provider.resolve_dimension() == 1024
    assert calls["n"] == 1  # 只探测一次
    assert provider.get_dimension() == 1024


@pytest.mark.asyncio
async def test_probe_failure_returns_zero_and_records_error() -> None:
    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    provider = _provider(handler)
    assert await provider.resolve_dimension() == 0
    assert provider.last_error is not None


@pytest.mark.asyncio
async def test_truncated_response_never_yields_partial_vectors() -> None:
    """Review Focus #2：服务端静默丢样本时不得产生"部分成功"的静默数据."""
    def handler(request):
        # 请求 3 条，只回 2 条
        return httpx.Response(200, json={"data": [
            {"embedding": [0.0, 1.0], "index": 0},
            {"embedding": [1.0, 0.0], "index": 1},
        ]})

    provider = _provider(handler)
    vectors = await provider.embed_texts(["a", "b", "c"])
    assert vectors == []                       # 整批降级，绝不返回长度不足的列表
    assert provider.last_error is not None
    assert "mismatch" in provider.last_error


@pytest.mark.asyncio
async def test_learned_dimension_from_business_call() -> None:
    """业务调用先发生时，维度随之确定，后续不再探测."""
    def handler(request):
        return httpx.Response(200, json={"data": [{"embedding": [0.0] * 512, "index": 0}]})

    provider = _provider(handler)
    await provider.embed_texts(["x"])
    assert provider.get_dimension() == 512
    assert await provider.resolve_dimension() == 512
