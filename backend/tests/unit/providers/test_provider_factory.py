"""Unit tests for the Provider factory validation (T037, RED first).

FR-010/FR-011/SC-004: the factory validates the three capability configs
(embedding / reranker / llm) at startup and fails loudly with correctable
errors for unknown provider_type, missing remote endpoint, and embedding
dimension mismatch — never silently falling back.
"""

from __future__ import annotations

import pytest

from rag_mcp.config.provider_config import (
    ProviderConfig,
    ProviderSettings,
    load_provider_settings,
)


def _valid_configs() -> ProviderSettings:
    return load_provider_settings(env={})


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


def _settings(embedding=None, reranker=None, llm=None) -> ProviderSettings:
    defaults = _valid_configs()
    return ProviderSettings(
        embedding=embedding or defaults.embedding,
        reranker=reranker or defaults.reranker,
        llm=llm or defaults.llm,
    )


def test_import_factory() -> None:
    from rag_mcp.providers.factory import (  # noqa: F401
        ProviderBundle,
        ValidationResult,
        validate_provider_config,
    )


def test_valid_default_config_passes() -> None:
    from rag_mcp.providers.factory import validate_provider_config

    result = validate_provider_config(_valid_configs(), llm_base_url="http://llm.local/v1")
    assert result.valid is True
    assert result.errors == []


def test_unknown_provider_type_fails() -> None:
    """FR-010: unknown provider_type -> explicit failure with correctable error."""
    from rag_mcp.providers.factory import validate_provider_config

    bad = _cfg(provider_type="quantum")
    settings = _settings(embedding=bad)
    result = validate_provider_config(settings, llm_base_url="http://llm.local/v1")
    assert result.valid is False
    assert any("quantum" in e.message for e in result.errors)


def test_remote_embedding_missing_endpoint_fails() -> None:
    """FR-010: remote_api without an endpoint -> explicit failure."""
    from rag_mcp.providers.factory import validate_provider_config

    bad = _cfg(capability="embedding", provider_type="remote_api", endpoint=None)
    settings = _settings(embedding=bad)
    result = validate_provider_config(settings, llm_base_url="")
    assert result.valid is False
    assert any("endpoint" in e.message.lower() for e in result.errors)


def test_remote_llm_endpoint_can_fall_back_to_base_url() -> None:
    """LLM remote may reuse llm_base_url when ENDPOINT is unset (research §1.4)."""
    from rag_mcp.providers.factory import validate_provider_config

    llm = _cfg(capability="llm", provider_type="remote_api", endpoint=None, model="deepseek-v4-flash")
    settings = _settings(llm=llm)
    result = validate_provider_config(settings, llm_base_url="http://llm.local/v1")
    assert result.valid is True


def test_dimension_mismatch_fails() -> None:
    """FR-011/FR-013: declared embedding dimension != active collection -> fail."""
    from rag_mcp.providers.factory import validate_provider_config

    result = validate_provider_config(
        _valid_configs(),
        embedding_dimension=768,
        active_collection_dimension=1024,
        llm_base_url="http://llm.local/v1",
    )
    assert result.valid is False
    assert any("dimension" in e.message.lower() for e in result.errors)
    assert any("768" in e.message and "1024" in e.message for e in result.errors)


def test_dimension_match_passes() -> None:
    from rag_mcp.providers.factory import validate_provider_config

    result = validate_provider_config(
        _valid_configs(),
        embedding_dimension=1024,
        active_collection_dimension=1024,
        llm_base_url="http://llm.local/v1",
    )
    assert result.valid is True


def test_validation_result_shape_matches_contract() -> None:
    """Validation result matches provider-config.schema.json validation shape."""
    from rag_mcp.providers.factory import validate_provider_config

    result = validate_provider_config(_valid_configs(), llm_base_url="http://llm.local/v1")
    assert {"valid", "errors"} <= set(result.to_contract_dict().keys())


def test_local_gpu_without_cuda_fails() -> None:
    """FR-010/Assumptions: local_gpu without CUDA -> explicit failure (no CPU fallback)."""
    from rag_mcp.providers.factory import validate_provider_config

    gpu = _cfg(capability="embedding", provider_type="local_gpu", model="BAAI/bge-m3")
    settings = _settings(embedding=gpu)
    result = validate_provider_config(settings, llm_base_url="http://llm.local/v1")
    if _cuda_available():
        pytest.skip("CUDA present; the no-GPU failure branch is not exercisable here")
    assert result.valid is False
    assert any("gpu" in e.message.lower() or "cuda" in e.message.lower() for e in result.errors)


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def test_build_embedding_forwards_declared_dimension() -> None:
    """factory 必须把声明维度传给远程 embedding provider（否则 EMBEDDING_DIMENSION 空转）."""
    from rag_mcp.providers import factory

    cfg = _cfg(
        provider_type="remote_api",
        endpoint="https://e.test/v1",
        dimension=1024,
    )
    provider = factory._build_embedding(cfg)
    assert provider.get_dimension() == 1024
    assert provider._dimension == 1024
