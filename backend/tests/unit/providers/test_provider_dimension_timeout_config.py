"""ProviderConfig dimension/timeout parsing (在线适配).

可声明 + 自动兜底：`*_DIMENSION`、`*_TIMEOUT_S` 均可选；空串等于未设置；
非法值必须显式失败（SC-004 不静默回退）。
"""

from __future__ import annotations

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
