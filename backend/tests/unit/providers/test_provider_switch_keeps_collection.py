"""SC-3：仅切换 Provider 类型、模型标识不变时，集合名必须逐字符不变。

这是"现有向量数据可继续使用"的全部依据；若取名逻辑读取 provider_type，
本用例必须失败。
"""

from __future__ import annotations

import pytest

from rag_mcp.config.provider_config import load_provider_settings
from rag_mcp.providers.active_model import derive_index_version, derive_memory_collection

_MODEL = "BAAI/bge-m3"
_FROZEN_INDEX = "bge-m3_v1"
_FROZEN_COLLECTION = "memories_dense_BAAI_bge_m3_012_v2"


@pytest.mark.parametrize("provider_type", ["local_cpu", "local_gpu", "remote_api"])
def test_index_version_independent_of_provider_type(provider_type: str) -> None:
    settings = load_provider_settings(env={"EMBEDDING_PROVIDER_TYPE": provider_type, "EMBEDDING_MODEL": _MODEL})
    assert derive_index_version(settings.embedding.model) == _FROZEN_INDEX


@pytest.mark.parametrize("provider_type", ["local_cpu", "local_gpu", "remote_api"])
def test_memory_collection_independent_of_provider_type(provider_type: str) -> None:
    settings = load_provider_settings(env={"EMBEDDING_PROVIDER_TYPE": provider_type, "EMBEDDING_MODEL": _MODEL})
    assert derive_memory_collection(settings.embedding.model) == _FROZEN_COLLECTION


def test_switching_provider_type_keeps_names() -> None:
    """同一模型、不同 Provider 配置 → 同名（数据复用成立）."""
    local = load_provider_settings(env={"EMBEDDING_PROVIDER_TYPE": "local_cpu"})
    remote = load_provider_settings(
        env={
            "EMBEDDING_PROVIDER_TYPE": "remote_api",
            "EMBEDDING_ENDPOINT": "https://example.com/v1",
        }
    )
    assert local.embedding.model == remote.embedding.model
    assert derive_index_version(local.embedding.model) == derive_index_version(remote.embedding.model)
    assert derive_memory_collection(local.embedding.model) == derive_memory_collection(remote.embedding.model)
