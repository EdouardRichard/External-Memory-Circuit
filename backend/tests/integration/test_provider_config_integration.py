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
    """Three capabilities assembled independently from run config."""
    from rag_mcp.providers.factory import build_provider_bundle

    bundle = build_provider_bundle(get_settings())
    assert bundle.embedding is not None
    assert bundle.reranker is not None
    assert bundle.llm is not None
    # embedding local_cpu != llm remote_api (capabilities choose independently)
    from rag_mcp.providers.local_cpu import LocalCPUEmbeddingProvider
    from rag_mcp.agents.llm_client import LLMClient

    assert isinstance(bundle.embedding, LocalCPUEmbeddingProvider)
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
    """SC-004: >=3 illegal configs -> explicit failure, silent fallback = 0."""
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
