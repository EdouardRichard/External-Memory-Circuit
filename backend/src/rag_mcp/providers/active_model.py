"""当前生效的模型标识与集合取名（在线适配，SC-3）——集合名的唯一来源。

取名只依赖**模型字符串**，不得读取 ``provider_type``。
理由（SC-3，数据复用）：现网 Qdrant 已有生产向量数据

- 知识库：``chunks_hybrid_bge-m3_v1``
- 记忆库：``memories_dense_BAAI_bge_m3_012_v2``

运维把 embedding/reranker 从本地 Provider 切到 OpenAI 兼容的云端 Provider 时，
模型标识通常保持不变；只要派生出的集合名逐字符不变，既有向量就能继续使用。
一旦取名逻辑读入 ``provider_type``，切换 Provider 就会悄悄改掉集合名，
等于静默丢开现网数据。因此本模块只消费 ``providers.<cap>.model``。

模型标识的首尾空白（``EMBEDDING_MODEL`` 里的手误空格）按环境变量噪声归一化，
先 ``strip()`` 再取名；除此之外不做任何改写。
"""

from __future__ import annotations

from rag_mcp.config import get_settings

__all__ = [
    "active_embedding_model",
    "active_reranker_model",
    "derive_index_version",
    "derive_memory_collection",
]


def active_embedding_model(settings=None) -> str:
    """返回当前生效的 embedding 模型标识（``providers.embedding.model``）.

    ``settings`` 为 ``None`` 时读取 ``get_settings()``。刻意不读传统的
    ``settings.embedding_model``：那正是本次改造要停止依赖的旧设置。
    """
    source = settings or get_settings()
    return source.providers.embedding.model


def active_reranker_model(settings=None) -> str:
    """返回当前生效的 reranker 模型标识（``providers.reranker.model``）.

    ``settings`` 为 ``None`` 时读取 ``get_settings()``。
    """
    source = settings or get_settings()
    return source.providers.reranker.model


def derive_index_version(model: str) -> str:
    """由 embedding 模型标识派生索引版本（知识库集合名）.

    约定：``<模型短名>_v1``，短名取模型 ID 最后一个 ``/`` 之后的片段。
    例：``BAAI/bge-m3`` → ``bge-m3_v1``（既有实现
    ``ingestion_service._derive_index_version``，行为逐字符一致）。

    空白归一化：先 ``strip()`` 再去短名。运维在 ``EMBEDDING_MODEL`` 里多打一个
    首尾空格是环境变量噪声，不是另一个模型身份；不归一化会让尾随空格派生出
    ``bge-m3 _v1`` 这种非法 Qdrant 集合名，检索时只报一个看不懂的错误。
    """
    name = model.strip()
    short_name = name.rsplit("/", 1)[-1]
    return f"{short_name}_v1"


def derive_memory_collection(model: str) -> str:
    """由 embedding 模型标识派生记忆密集向量集合名.

    约定：``memories_dense_<归一化模型名>_012_v2``，归一化即把 ``/`` 与 ``-``
    都换成 ``_``。例：``BAAI/bge-m3`` → ``memories_dense_BAAI_bge_m3_012_v2``
    （既有实现 ``memory_projection_store._materialize_dense``，逐字符一致，
    与现网集合名相同）。

    空白归一化：先 ``strip()`` 再替换；理由同 ``derive_index_version``——
    首尾空格是环境变量噪声。``/``、``-`` 与空白之外的字符原样保留，不做大小写变换。
    """
    name = model.strip()
    normalized = _normalize_model_name(name)
    return f"memories_dense_{normalized}_012_v2"


def _normalize_model_name(model: str) -> str:
    """把模型标识归一化为可嵌入集合名的片段（``/`` 与 ``-`` → ``_``）."""
    return model.replace("/", "_").replace("-", "_")
