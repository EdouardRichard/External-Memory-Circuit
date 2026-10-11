"""当前生效模型标识的唯一来源（在线适配，SC-3/SC-4）.

集合名只由模型字符串决定：切换 Provider 类型而模型标识不变时，
派生的索引版本与记忆集合名必须逐字符不变（现网向量数据可继续复用）。
"""

from __future__ import annotations

from dataclasses import replace

from rag_mcp.config.provider_config import load_provider_settings


def _settings_with_models(embedding_model: str, reranker_model: str):
    """Build a stand-in Settings whose providers carry the given model identifiers."""

    class _S:
        providers = None

    settings = _S()
    base = load_provider_settings(env={})
    settings.providers = type(base)(
        embedding=replace(base.embedding, model=embedding_model),
        reranker=replace(base.reranker, model=reranker_model),
        llm=base.llm,
    )
    return settings


def test_derive_index_version_preserves_frozen_convention() -> None:
    from rag_mcp.providers.active_model import derive_index_version

    assert derive_index_version("BAAI/bge-m3") == "bge-m3_v1"
    assert derive_index_version("sentence-transformers/all-MiniLM-L6-v2") == "all-MiniLM-L6-v2_v1"


def test_derive_memory_collection_matches_deployed_name() -> None:
    from rag_mcp.providers.active_model import derive_memory_collection

    assert derive_memory_collection("BAAI/bge-m3") == "memories_dense_BAAI_bge_m3_012_v2"


def test_distinct_models_derive_distinct_names() -> None:
    from rag_mcp.providers.active_model import derive_index_version, derive_memory_collection

    assert derive_index_version("BAAI/bge-m3") != derive_index_version("text-embedding-3-large")
    assert derive_memory_collection("BAAI/bge-m3") != derive_memory_collection("text-embedding-3-large")


def test_active_embedding_model_reads_provider_config() -> None:
    from rag_mcp.providers.active_model import active_embedding_model

    settings = _settings_with_models("online-embedder", "online-reranker")
    assert active_embedding_model(settings) == "online-embedder"


def test_active_reranker_model_reads_provider_config() -> None:
    from rag_mcp.providers.active_model import active_reranker_model

    settings = _settings_with_models("online-embedder", "online-reranker")
    assert active_reranker_model(settings) == "online-reranker"


def test_active_models_are_whitespace_insensitive() -> None:
    """前后空格视为环境变量噪声，不改变取名（Review Focus #1 的可预测性要求）."""
    from rag_mcp.providers.active_model import derive_index_version, derive_memory_collection

    canonical = "BAAI/bge-m3"
    assert derive_index_version(" BAAI/bge-m3") == derive_index_version(canonical)
    assert derive_index_version("BAAI/bge-m3 ") == derive_index_version(canonical)
    assert derive_index_version(" BAAI/bge-m3 ") == "bge-m3_v1"

    assert derive_memory_collection(" BAAI/bge-m3") == derive_memory_collection(canonical)
    assert derive_memory_collection("BAAI/bge-m3 ") == derive_memory_collection(canonical)
    assert derive_memory_collection(" BAAI/bge-m3 ") == "memories_dense_BAAI_bge_m3_012_v2"
