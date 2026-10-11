"""ProviderConfig dimension/timeout parsing (在线适配).

可声明 + 自动兜底：`*_DIMENSION`、`*_TIMEOUT_S` 均可选；空串等于未设置；
非法值必须显式失败（SC-004 不静默回退）。
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from rag_mcp.config.provider_config import ProviderConfig, load_provider_config


def test_dimension_and_timeout_default_to_none() -> None:
    cfg = load_provider_config("embedding", env={})
    assert cfg.dimension is None
    assert cfg.timeout_s is None


def test_dimension_and_timeout_parse_from_env() -> None:
    cfg = load_provider_config(
        "embedding",
        env={"EMBEDDING_DIMENSION": "1024", "EMBEDDING_TIMEOUT_S": "30"},
    )
    assert cfg.dimension == 1024
    assert cfg.timeout_s == 30.0


def test_empty_string_means_unset() -> None:
    cfg = load_provider_config(
        "embedding", env={"EMBEDDING_DIMENSION": "", "EMBEDDING_TIMEOUT_S": ""}
    )
    assert cfg.dimension is None
    assert cfg.timeout_s is None


def test_whitespace_only_means_unset() -> None:
    cfg = load_provider_config(
        "embedding", env={"EMBEDDING_DIMENSION": "   ", "EMBEDDING_TIMEOUT_S": "\t"}
    )
    assert cfg.dimension is None
    assert cfg.timeout_s is None


def test_positional_construction_with_seven_fields_still_works() -> None:
    cfg = ProviderConfig("embedding", "local_cpu", "BAAI/bge-m3", None, None, 8, 16)
    assert cfg.dimension is None
    assert cfg.timeout_s is None


def test_fractional_timeout_allowed() -> None:
    cfg = load_provider_config("reranker", env={"RERANKER_TIMEOUT_S": "2.5"})
    assert cfg.timeout_s == 2.5


@pytest.mark.parametrize("bad", ["0", "-1", "abc"])
def test_invalid_dimension_fails_loudly(bad: str) -> None:
    with pytest.raises(ValueError, match="DIMENSION"):
        load_provider_config("embedding", env={"EMBEDDING_DIMENSION": bad})


@pytest.mark.parametrize("bad", ["0", "-3", "abc"])
def test_invalid_timeout_fails_loudly(bad: str) -> None:
    with pytest.raises(ValueError, match="TIMEOUT_S"):
        load_provider_config("embedding", env={"EMBEDDING_TIMEOUT_S": bad})


def test_contract_dict_exposes_new_fields() -> None:
    cfg = load_provider_config(
        "embedding",
        env={"EMBEDDING_DIMENSION": "1024", "EMBEDDING_TIMEOUT_S": "30"},
    )
    payload = cfg.to_contract_dict()
    assert payload["dimension"] == 1024
    assert payload["timeout_s"] == 30.0


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "1e999"])
def test_non_finite_timeout_fails_loudly(bad: str) -> None:
    """非有限超时（nan/inf）必须显式失败，而不是放进契约。

    背景：nan 能通过 float() 且不满足 <= 0，加载后会让 to_contract_dict() 的值违反本变更
    自己的 schema 约束（provider-config.schema.json 的 exclusiveMinimum: 0）。
    """
    with pytest.raises(ValueError, match="TIMEOUT_S"):
        load_provider_config("embedding", env={"EMBEDDING_TIMEOUT_S": bad})


def test_declared_embedding_dimension_is_wired_into_remote_provider() -> None:
    """EMBEDDING_DIMENSION 必须真正传到远程 Provider（修复"已声明维度"空转）。

    使用 MockTransport：若维度确实由构造声明，探测就不会发出任何 HTTP 请求。
    """
    from rag_mcp.config.provider_config import load_provider_settings
    from rag_mcp.providers import factory

    calls = {"n": 0}

    def handler(request):  # pragma: no cover - 只有在探测路径才会被调用
        calls["n"] += 1
        return httpx.Response(200, json={"data": [{"embedding": [0.0], "index": 0}]})

    providers = load_provider_settings(
        env={
            "EMBEDDING_PROVIDER_TYPE": "remote_api",
            "EMBEDDING_ENDPOINT": "https://e.test/v1",
            "EMBEDDING_DIMENSION": "1024",
        }
    )
    provider = factory._build_embedding(providers.embedding)
    provider._transport = httpx.MockTransport(handler)

    assert asyncio.run(provider.resolve_dimension()) == 1024
    assert calls["n"] == 0, "已声明维度时不应发出探测请求"
