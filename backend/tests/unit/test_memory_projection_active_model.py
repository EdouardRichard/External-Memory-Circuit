"""记忆投影集合名与维度取自 Provider（在线适配，SC-3）.

The deployed collection ``memories_dense_BAAI_bge_m3_012_v2`` holds the live
memory vectors, so the projection must derive its collection name from the
active provider model via ``derive_memory_collection`` and resolve the
dimension through the shared ``resolve_collection_dimension`` helper. The old
``self.embedding.get_dimension()`` path returns 0 for a remote provider and
would create a zero-dimension collection.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from rag_mcp.providers.active_model import active_embedding_model, derive_memory_collection
from rag_mcp.services.memory_projection_store import MemoryProjectionStore


class _FakeQdrant:
    """Synchronous Qdrant metadata slice used by ``resolve_collection_dimension``."""

    def __init__(self, *, exists: bool = False, dimension: int | None = None) -> None:
        self._exists = exists
        self._dimension = dimension
        self.created: list[tuple[str, int]] = []
        self.exists_calls = 0

    def collection_exists(self, name: str) -> bool:
        self.exists_calls += 1
        return self._exists

    def get_collection_dimension(self, name: str) -> int | None:
        return self._dimension

    def create_collection(self, name: str, dimension: int) -> None:
        self.created.append((name, dimension))


class _FakeEmbedding:
    """Provider double: ``resolve_dimension`` is the only dimension source."""

    def __init__(self, *, resolved: int = 0, dimension: int = 0) -> None:
        self._resolved = resolved
        self._dimension = dimension
        self.resolve_calls = 0

    async def resolve_dimension(self) -> int:
        self.resolve_calls += 1
        return self._resolved

    def get_dimension(self) -> int:
        # The legacy provider path: 0 for a remote provider.
        return self._dimension


def _store(qdrant: _FakeQdrant, embedding: _FakeEmbedding) -> MemoryProjectionStore:
    """Build the store with authority checking bypassed (unit-level seam)."""

    store = MemoryProjectionStore(qdrant_store=qdrant, embedding_provider=embedding)

    async def _authorized(state, scope_id, event_id):
        return None

    store._authorize = _authorized
    return store


def test_memory_collection_name_uses_active_model(monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    assert derive_memory_collection(active_embedding_model()) == "memories_dense_BAAI_bge_m3_012_v2"


@pytest.mark.asyncio
async def test_materialize_dense_does_not_create_zero_dim_collection(monkeypatch) -> None:
    """集合不存在且维度不可确定 → 显式失败，不建 size=0 集合."""
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    qdrant = _FakeQdrant(exists=False)
    embedding = _FakeEmbedding(resolved=0, dimension=0)

    with pytest.raises(ValueError, match="EMBEDDING_DIMENSION_UNRESOLVED"):
        await _store(qdrant, embedding)._materialize_dense({"dense": {}}, uuid4(), 7)

    assert qdrant.created == []


@pytest.mark.asyncio
async def test_materialize_dense_creates_collection_from_active_model_and_resolved_dimension(monkeypatch) -> None:
    """回到旧实现（``get_settings().embedding_model`` + ``get_dimension()``）时本用例必须失败."""
    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    qdrant = _FakeQdrant(exists=False)
    embedding = _FakeEmbedding(resolved=1024, dimension=0)

    collection = await _store(qdrant, embedding)._materialize_dense({"dense": {}}, uuid4(), 7)

    assert collection == derive_memory_collection(active_embedding_model())
    assert collection == "memories_dense_BAAI_bge_m3_012_v2"
    assert qdrant.created == [(collection, 1024)]


@pytest.mark.asyncio
async def test_materialize_dense_existing_collection_dimension_wins(monkeypatch) -> None:
    """既有集合的维度优先，且不触碰 provider（避免重复探测/网络调用）."""
    monkeypatch.setenv("EMBEDDING_MODEL", "remote/model-x")
    qdrant = _FakeQdrant(exists=True, dimension=768)
    embedding = _FakeEmbedding(resolved=1024, dimension=0)

    collection = await _store(qdrant, embedding)._materialize_dense({"dense": {}}, uuid4(), 7)

    assert collection == "memories_dense_remote_model_x_012_v2"
    assert qdrant.created == []
    assert embedding.resolve_calls == 0
