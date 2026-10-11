"""检索路径的集合名必须与入库路径一致（在线适配，SC-3）.

`RetrievalService._derive_index_version` 必须由**当前生效的 Provider 模型**
（``providers.embedding.model``）派生 Qdrant 集合索引版本，而不是历史遗留的
``settings.embedding_model``。取名口径与入库路径共用
``rag_mcp.providers.active_model.derive_index_version``，两者一旦分叉，
检索读的集合与入库写的集合不同，检索会静默返回空结果。
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace


def test_retrieval_and_ingestion_derive_same_index_version() -> None:
    """检索与入库对同一份 settings 派生出逐字符相同的 index_version."""
    from rag_mcp.config import get_settings
    from rag_mcp.providers.active_model import active_embedding_model
    from rag_mcp.services.ingestion_service import _derive_index_version
    from rag_mcp.services.retrieval_service import RetrievalService

    settings = get_settings()
    service = RetrievalService.__new__(RetrievalService)  # 只验证取名逻辑，不建会话
    service._settings = settings
    assert service._derive_index_version() == _derive_index_version(
        active_embedding_model(settings)
    )


def test_retrieval_ignores_settings_embedding_model() -> None:
    """settings.embedding_model 不再是取值来源.

    本环境里遗留字段与 provider 字段恰好都是 ``BAAI/bge-m3``，因此必须构造一份
    **两者不同**的 settings，否则该用例在改动前也会通过，证明不了任何事。
    """
    from rag_mcp.providers.active_model import derive_index_version
    from rag_mcp.services.retrieval_service import RetrievalService

    legacy_model = "BAAI/bge-m3"
    provider_model = "cloud-vendor/online-embed-large"
    # 哨兵：两个模型必须派生出不同的名字，否则本用例仍然区分不出取值来源。
    assert derive_index_version(provider_model) != derive_index_version(legacy_model)

    settings = SimpleNamespace(
        embedding_model=legacy_model,
        providers=SimpleNamespace(embedding=SimpleNamespace(model=provider_model)),
    )
    service = RetrievalService.__new__(RetrievalService)
    service._settings = settings

    assert service._derive_index_version() == "online-embed-large_v1"


def test_default_local_model_keeps_production_collection_name() -> None:
    """默认本地配置下行为不变：``BAAI/bge-m3`` → ``bge-m3_v1``（906 条现网向量可达）."""
    from rag_mcp.config import get_settings
    from rag_mcp.providers.active_model import derive_index_version
    from rag_mcp.services.retrieval_service import RetrievalService

    assert derive_index_version("BAAI/bge-m3") == "bge-m3_v1"

    service = RetrievalService.__new__(RetrievalService)
    service._settings = get_settings()
    assert service._derive_index_version() == "bge-m3_v1"


def test_derive_index_version_keeps_no_argument_signature() -> None:
    """保持无参方法名与签名（调用点 :311 / :778 / :1477 依赖它）."""
    from rag_mcp.services.retrieval_service import RetrievalService

    params = list(inspect.signature(RetrievalService._derive_index_version).parameters)
    assert params == ["self"]
