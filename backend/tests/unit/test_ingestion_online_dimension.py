"""入库路径：模型标识取自 Provider，维度无法确定时显式失败（在线适配）."""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Brief Step 1: helper-level tests (Task 2 / Task 3 already implemented)
# ---------------------------------------------------------------------------


def test_derive_index_version_delegates_and_keeps_signature() -> None:
    from rag_mcp.services.ingestion_service import _derive_index_version
    from rag_mcp.providers.active_model import derive_index_version

    assert _derive_index_version("BAAI/bge-m3") == "bge-m3_v1"
    assert _derive_index_version("BAAI/bge-m3") == derive_index_version("BAAI/bge-m3")


def test_active_embedding_model_overrides_settings_embedding_model(monkeypatch) -> None:
    """provider 配置是唯一来源，不再读 settings.embedding_model."""
    from rag_mcp.config import get_settings
    from rag_mcp.providers.active_model import active_embedding_model

    monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("EMBEDDING_PROVIDER_TYPE", "remote_api")
    monkeypatch.setenv("EMBEDDING_ENDPOINT", "https://e.test/v1")
    assert active_embedding_model(get_settings()) == "BAAI/bge-m3"


@pytest.mark.asyncio
async def test_unresolvable_dimension_never_creates_zero_dim_collection() -> None:
    """绝不出现 create_hybrid_collection(name, 0)（Review Focus #3）."""
    from rag_mcp.providers.dimensions import resolve_collection_dimension

    class _Store:
        created: list[tuple[str, int]] = []

        def collection_exists(self, name: str) -> bool:
            return False

        def get_collection_dimension(self, name: str) -> int | None:
            return None

        def create_hybrid_collection(self, name: str, dimension: int) -> None:
            self.created.append((name, dimension))

    class _Provider:
        def get_dimension(self) -> int:
            return 0

        async def resolve_dimension(self) -> int:
            return 0

    store = _Store()
    with pytest.raises(ValueError, match="EMBEDDING_DIMENSION_UNRESOLVED"):
        dimension = await resolve_collection_dimension("chunks_hybrid_x_v1", _Provider(), store)
        store.create_hybrid_collection("chunks_hybrid_x_v1", dimension)
    assert store.created == []


# ---------------------------------------------------------------------------
# Service-layer wiring regression (Task 6's own work).
#
# The tests above pass as soon as Tasks 2/3 land; these drive IngestionService
# itself so they fail while the service still reads ``settings.embedding_model``
# or probes the provider for an existing collection.
# ---------------------------------------------------------------------------


class _StopPipeline(Exception):
    """Sentinel raised by a fake collaborator to halt the pipeline deterministically."""


class _LegacySettings:
    """Settings whose legacy field deliberately disagrees with the active Provider.

    ``embedding_model`` is what the service must STOP using; ``providers.embedding``
    is the only legal source. If both agreed the regression below would be blind.
    """

    embedding_model = "legacy/local-model"  # must never name the collection
    max_conversion_seconds = 30
    data_root = "./data/uploads"

    class providers:
        class embedding:
            model = "BAAI/bge-m3"

        class reranker:
            model = "BAAI/bge-reranker-v2-m3"


class _FakeSession:
    async def flush(self) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None

    def add(self, obj) -> None:
        return None

    def add_all(self, objs) -> None:
        return None


class _FakeSource:
    status = "pending"
    processing_error = None
    knowledge_scope_id = 42
    source_id = 7
    filename = "doc.md"
    format = "markdown"


class _FakeRun:
    status = "pending"
    started_at = None
    completed_at = None
    error_message = None
    stages: list = []


class _FakeRegistry:
    def is_converter(self, fmt: str) -> bool:
        return False

    def is_binary(self, fmt: str) -> bool:
        return False

    def to_text(self, raw: bytes, fmt: str, filename: str) -> str:
        return raw.decode("utf-8")


class _FakeSparseEncoder:
    def fit(self, texts) -> None:
        return None

    def encode(self, text: str) -> dict:
        return {"indices": [0], "values": [1.0]}


class _ResolvableProvider:
    """Remote-style provider: get_dimension() is unknown (0) until probed."""

    def __init__(self, dim: int = 1024) -> None:
        self._dim = dim
        self.probes = 0

    async def embed_texts(self, texts):
        return [[0.0] * self._dim for _ in texts]

    async def embed_query(self, text: str):
        return [0.0] * self._dim

    def get_dimension(self) -> int:
        return 0

    async def resolve_dimension(self) -> int:
        self.probes += 1
        return self._dim


class _RecordingStore:
    """Minimal synchronous Qdrant slice; records collection creation."""

    def __init__(self, exists: bool = False, existing_dim: int | None = None) -> None:
        self.created: list[tuple[str, int]] = []
        self._exists = exists
        self._existing_dim = existing_dim

    def collection_exists(self, name: str) -> bool:
        return self._exists

    def get_collection_dimension(self, name: str) -> int | None:
        return self._existing_dim

    def create_hybrid_collection(self, name: str, dimension: int) -> None:
        self.created.append((name, dimension))
        raise _StopPipeline


def _async_return(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner


def _build_service(monkeypatch, provider, store):
    from rag_mcp.services import ingestion_service as mod

    monkeypatch.setattr(mod, "get_settings", lambda: _LegacySettings())
    service = mod.IngestionService(_FakeSession(), provider, store)
    monkeypatch.setattr(service, "_load_source", _async_return(_FakeSource()))
    monkeypatch.setattr(service, "_pending_initial_run_id", _async_return(None))
    monkeypatch.setattr(service, "_create_processing_run", _async_return(_FakeRun()))
    monkeypatch.setattr(service, "_read_raw_bytes", _async_return(b"# Title\n\nbody"))
    monkeypatch.setattr(
        service,
        "_parse_content",
        lambda *args, **kwargs: [
            {
                "content_text": "hello world",
                "chunk_type": "section",
                "start_line": 1,
                "end_line": 1,
            }
        ],
    )
    monkeypatch.setattr(
        mod.FormatHandlerRegistry, "instance", staticmethod(lambda: _FakeRegistry())
    )
    monkeypatch.setattr(
        "rag_mcp.indexing.sparse_encoder.BM25SparseEncoder", _FakeSparseEncoder
    )
    return service


@pytest.mark.asyncio
async def test_pipeline_names_collection_from_active_provider_model(monkeypatch) -> None:
    """集合名与维度都取自生效 Provider，而不是 settings.embedding_model.

    RED before the service-layer change: the collection would be
    ``chunks_hybrid_local-model_v1`` (legacy field) with dimension ``get_dimension()``=0.
    """
    provider = _ResolvableProvider(dim=1024)
    store = _RecordingStore(exists=False)
    service = _build_service(monkeypatch, provider, store)

    with pytest.raises(_StopPipeline):
        await service.ingest(7)

    assert store.created == [("chunks_hybrid_bge-m3_v1", 1024)]


@pytest.mark.asyncio
async def test_pipeline_never_creates_zero_dimension_collection(monkeypatch) -> None:
    """维度无法解析时显式失败，且绝不调用 create_hybrid_collection(name, 0)."""
    provider = _ResolvableProvider(dim=0)  # resolve_dimension() -> 0
    store = _RecordingStore(exists=False)
    service = _build_service(monkeypatch, provider, store)

    with pytest.raises(ValueError, match="EMBEDDING_DIMENSION_UNRESOLVED"):
        await service.ingest(7)

    assert store.created == []


@pytest.mark.asyncio
async def test_existing_collection_never_probes_provider(monkeypatch) -> None:
    """已存在的集合不得触发维度解析，更不得探测 provider（Controller ruling）.

    只断言 ``provider.probes == 0`` 不够：它藏在 ``if collection_exists`` 的
    else 分支之外，现行实现根本走不到那里；一旦将来把
    ``resolve_collection_dimension`` 改成无条件调用，而集合维度又能读出来，
    探测计数仍是 0，用例依旧全绿。因此直接在调用点钉住契约：
    ``ingestion_service.resolve_collection_dimension`` 不得被调用。
    """
    provider = _ResolvableProvider(dim=1024)
    store = _RecordingStore(exists=True, existing_dim=None)
    service = _build_service(monkeypatch, provider, store)

    from rag_mcp.services import ingestion_service as mod

    resolution_calls: list[tuple] = []

    async def _forbidden_resolution(*args, **kwargs):
        resolution_calls.append(args)
        raise AssertionError("provider must not be probed for an existing collection")

    monkeypatch.setattr(mod, "resolve_collection_dimension", _forbidden_resolution)

    async def _stop(scope_id):
        raise _StopPipeline

    monkeypatch.setattr(service, "_get_next_version_number", _stop)

    with pytest.raises(_StopPipeline):
        await service.ingest(7)

    assert resolution_calls == []
    assert provider.probes == 0
    assert store.created == []
