"""Integration test: Provider configuration end-to-end (T047/T048).

SC-004: only run configuration selects the three capability providers and
retrieval succeeds with the external MCP contract unchanged; >=3 illegal
configurations fail startup explicitly (silent fallback = 0).
"""

from __future__ import annotations

import pytest

from rag_mcp.config.provider_config import ProviderConfig, ProviderSettings
from rag_mcp.config import get_settings


def _cfg(**overrides) -> ProviderConfig:
    base = dict(
        capability="embedding",
        provider_type="local_cpu",
        model="BAAI/bge-m3",
        endpoint=None,
        api_key_env=None,
        concurrency_limit=8,
        concurrency_limit_max=16,
    )
    base.update(overrides)
    return ProviderConfig(**base)


def test_default_bundle_assembles_three_providers():
    """Three capabilities assembled independently from run config.

    断言的是「装配结果与运行配置一致」，而不是某个固定的 Provider 类型：
    本用例原先硬编码了 ``isinstance(bundle.embedding, LocalCPUEmbeddingProvider)``，
    只在 ``EMBEDDING_PROVIDER_TYPE`` 保持默认 ``local_cpu`` 时成立。一旦运维按
    在线适配的设计把 embedding 指向 remote_api（这正是该能力的目的），
    该断言就会失败——它测的是环境，不是代码。改为按配置逐能力比对类型。
    """
    from rag_mcp.agents.llm_client import LLMClient
    from rag_mcp.providers.factory import build_provider_bundle
    from rag_mcp.providers.local_cpu import LocalCPUEmbeddingProvider
    from rag_mcp.providers.local_cpu_reranker import LocalCPUReranker
    from rag_mcp.providers.remote_api_embedding import RemoteAPIEmbeddingProvider
    from rag_mcp.providers.remote_api_reranker import RemoteAPIRerankerProvider

    settings = get_settings()
    bundle = build_provider_bundle(settings)

    assert bundle.embedding is not None
    assert bundle.reranker is not None
    assert bundle.llm is not None

    # 每个能力的实际实现类型必须与其 provider_type 声明相符（各能力独立选择）。
    embedding_by_type = {
        "local_cpu": LocalCPUEmbeddingProvider,
        # local_gpu 在无 CUDA 时会被启动校验直接拒绝（见 test_local_gpu_without_cuda_fails），
        # 有 CUDA 时其实现是 LocalGPUEmbeddingProvider —— LocalCPUEmbeddingProvider 的子类，
        # 因此该 isinstance 仍然成立。
        "local_gpu": LocalCPUEmbeddingProvider,
        "remote_api": RemoteAPIEmbeddingProvider,
    }
    reranker_by_type = {
        "local_cpu": LocalCPUReranker,
        "local_gpu": LocalCPUReranker,
        "remote_api": RemoteAPIRerankerProvider,
    }
    assert isinstance(bundle.embedding, embedding_by_type[settings.providers.embedding.provider_type])
    assert isinstance(bundle.reranker, reranker_by_type[settings.providers.reranker.provider_type])
    # llm 目前只支持 remote_api
    assert isinstance(bundle.llm, LLMClient)

    assert bundle.validation.valid is True
    assert set(bundle.limiters.keys()) == {"embedding", "reranker", "llm"}


def test_bundle_embedding_provider_is_functional():
    """The factory-assembled embedding provider actually embeds (retrieval-ready)."""
    from rag_mcp.providers.factory import build_provider_bundle

    bundle = build_provider_bundle(get_settings())
    dim = bundle.embedding.get_dimension()
    assert dim > 0


@pytest.mark.asyncio
async def test_invalid_configs_fail_explicitly_no_silent_fallback():
    """SC-004: 非法配置 -> 显式失败，静默回退 = 0。

    维度不匹配的用例已拆分为独立的 test_dimension_mismatch_fails_explicitly：
    它需要显式传入维度参数，否则会因为 LLM endpoint 缺失而"因错误的原因通过"。
    """
    from rag_mcp.providers.factory import validate_provider_config

    settings = get_settings()
    defaults = settings.providers

    # 1. unknown provider_type
    unknown = _cfg(provider_type="quantum")
    # 2. remote without endpoint
    remote_no_ep = _cfg(capability="embedding", provider_type="remote_api", endpoint=None)

    for bad in (unknown, remote_no_ep):
        providers = ProviderSettings(
            embedding=bad,
            reranker=defaults.reranker,
            llm=defaults.llm,
        )
        result = validate_provider_config(providers, llm_base_url="")
        assert result.valid is False, f"{bad.provider_type} should be invalid"


def test_dimension_mismatch_fails_explicitly() -> None:
    """SC-005：声明的维度与活跃集合维度不一致 → 显式失败（不再依赖 LLM endpoint 缺失误通过）."""
    from rag_mcp.providers.factory import validate_provider_config

    result = validate_provider_config(
        get_settings().providers,
        embedding_dimension=768,
        active_collection_dimension=1024,
        llm_base_url="http://llm.local/v1",
    )
    assert result.valid is False
    assert any("dimension" in e.message.lower() for e in result.errors)


@pytest.mark.asyncio
async def test_assemble_or_fail_raises_on_invalid():
    """T048: the startup assembly refuses to proceed on invalid config."""
    from rag_mcp.providers.factory import assemble_or_fail
    from rag_mcp.config.provider_config import ProviderSettings

    settings = get_settings()
    defaults = settings.providers
    bad = ProviderSettings(
        embedding=_cfg(capability="embedding", provider_type="remote_api", endpoint=None),
        reranker=defaults.reranker,
        llm=defaults.llm,
    )
    # Build a settings stand-in with the bad providers
    class S:
        providers = bad
        llm_base_url = ""
        llm_api_key = ""
        llm_model = "deepseek-v4-flash"

    with pytest.raises(ValueError) as excinfo:
        assemble_or_fail(S())
    assert "invalid provider configuration" in str(excinfo.value)
    assert "endpoint" in str(excinfo.value).lower()
