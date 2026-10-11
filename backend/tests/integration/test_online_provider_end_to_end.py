"""SC-1/SC-2/SC-3：仅凭运行配置走在线 embedding + rerank 的端到端验收。

本文件是 Task 10 的验收证据，不重复实现细节，只回答三个问题：

1. 配置指向在线 API 时，Provider 组合是否**真的装配出远程实现**（SC-1 前半）；
2. 保持模型标识不变、仅切换 Provider 类型时，派生集合名是否**逐字符不变**
   （SC-3，即"既有向量继续可用"的唯一依据）；
3. 在线调用路径是否**不导入 sentence_transformers**（SC-2，不残留本地模型加载）。

凭据只以环境变量「名」引用；本文件不含任何凭据值。
"""

from __future__ import annotations

import json
import sys

import httpx
import pytest


def test_online_config_assembles_remote_providers(monkeypatch) -> None:
    """SC-1 前半：配置指向在线 API 时，bundle 装配出远程 Provider."""
    monkeypatch.setenv("EMBEDDING_PROVIDER_TYPE", "remote_api")
    monkeypatch.setenv("EMBEDDING_ENDPOINT", "https://online.test/v1")
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")
    monkeypatch.setenv("RERANKER_PROVIDER_TYPE", "remote_api")
    monkeypatch.setenv("RERANKER_ENDPOINT", "https://online.test/v1")

    from rag_mcp.config import get_settings
    from rag_mcp.providers.factory import get_provider_bundle, reset_provider_bundle
    from rag_mcp.providers.remote_api_embedding import RemoteAPIEmbeddingProvider
    from rag_mcp.providers.remote_api_reranker import RemoteAPIRerankerProvider

    reset_provider_bundle()
    bundle = get_provider_bundle(get_settings())
    assert isinstance(bundle.embedding, RemoteAPIEmbeddingProvider)
    assert isinstance(bundle.reranker, RemoteAPIRerankerProvider)
    # 配置逐项透传到装配结果：模型标识决定集合名，声明维度决定建集合维度。
    assert bundle.embedding._model == "BAAI/bge-m3"
    assert bundle.embedding._endpoint == "https://online.test/v1"
    assert bundle.embedding.get_dimension() == 1024
    assert bundle.reranker._endpoint == "https://online.test/v1"


def test_collection_name_unchanged_when_switching_provider(monkeypatch) -> None:
    """SC-3：数据复用的依据——同名集合.

    断言"切换 Provider 类型不改集合名"，因此对同一模型标识分别读两种
    Provider 配置，而不是只把期望值写死一遍。
    """
    from rag_mcp.providers.active_model import (
        active_embedding_model,
        derive_index_version,
        derive_memory_collection,
    )

    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("EMBEDDING_PROVIDER_TYPE", "local_cpu")
    local_version = derive_index_version(active_embedding_model())
    local_memory = derive_memory_collection(active_embedding_model())

    monkeypatch.setenv("EMBEDDING_PROVIDER_TYPE", "remote_api")
    monkeypatch.setenv("EMBEDDING_ENDPOINT", "https://online.test/v1")
    remote_version = derive_index_version(active_embedding_model())
    remote_memory = derive_memory_collection(active_embedding_model())

    assert local_version == remote_version == "bge-m3_v1"
    assert remote_memory == local_memory == "memories_dense_BAAI_bge_m3_012_v2"


@pytest.mark.asyncio
async def test_online_embed_and_rerank_never_import_sentence_transformers(monkeypatch) -> None:
    """SC-2：在线路径不加载本地模型."""
    sys.modules.pop("sentence_transformers", None)          # 消除用例顺序依赖
    from rag_mcp.providers.remote_api_embedding import RemoteAPIEmbeddingProvider
    from rag_mcp.providers.remote_api_reranker import RemoteAPIRerankerProvider

    def embed_handler(request):
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={"data": [{"embedding": [0.1] * 1024, "index": i} for i in range(len(payload["input"]))]},
        )

    def rerank_handler(request):
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={"results": [{"index": i, "relevance_score": 0.9 - i * 0.1} for i in range(len(payload["documents"]))]},
        )

    embedding = RemoteAPIEmbeddingProvider(
        endpoint="https://online.test/v1",
        model="BAAI/bge-m3",
        dimension=1024,
        transport=httpx.MockTransport(embed_handler),
    )
    reranker = RemoteAPIRerankerProvider(
        endpoint="https://online.test/v1",
        model="bge-reranker-v2-m3",
        transport=httpx.MockTransport(rerank_handler),
    )

    vectors = await embedding.embed_texts(["a", "b"])
    assert len(vectors) == 2 and len(vectors[0]) == 1024
    assert await embedding.resolve_dimension() == 1024

    reranked = await reranker.rerank("q", [{"text": "a"}, {"text": "b"}], top_k=2)
    assert reranked[0]["rerank_score"] >= reranked[1]["rerank_score"]

    assert "sentence_transformers" not in sys.modules
