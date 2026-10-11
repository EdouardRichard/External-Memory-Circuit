"""resolve_collection_dimension 三分支（在线适配，Review Focus #3）."""

from __future__ import annotations

import pytest


class _Store:
    def __init__(self, exists: bool, dimension: int | None):
        self._exists, self._dimension = exists, dimension

    def collection_exists(self, name: str) -> bool:
        return self._exists

    def get_collection_dimension(self, name: str) -> int | None:
        return self._dimension


class _Provider:
    def __init__(self, dimension: int = 0):
        self._dimension = dimension
        self.probed = 0

    def get_dimension(self) -> int:
        return self._dimension

    async def resolve_dimension(self) -> int:
        self.probed += 1
        return self._dimension


@pytest.mark.asyncio
async def test_existing_collection_dimension_wins_without_probing() -> None:
    from rag_mcp.providers.dimensions import resolve_collection_dimension

    provider = _Provider(dimension=999)
    got = await resolve_collection_dimension("c", provider, _Store(exists=True, dimension=1024))
    assert got == 1024
    assert provider.probed == 0


@pytest.mark.asyncio
async def test_new_collection_uses_provider_resolved_dimension() -> None:
    from rag_mcp.providers.dimensions import resolve_collection_dimension

    provider = _Provider(dimension=1536)
    got = await resolve_collection_dimension("c", provider, _Store(exists=False, dimension=None))
    assert got == 1536
    assert provider.probed == 1


@pytest.mark.asyncio
async def test_unresolvable_dimension_fails_loudly() -> None:
    from rag_mcp.providers.dimensions import resolve_collection_dimension

    with pytest.raises(ValueError, match="EMBEDDING_DIMENSION_UNRESOLVED"):
        await resolve_collection_dimension("c", _Provider(dimension=0), _Store(exists=False, dimension=None))


@pytest.mark.asyncio
async def test_existing_collection_with_unknown_dimension_falls_back_to_provider() -> None:
    """集合存在但维度读不出来（异常/旧集合）→ 退回 provider，而不是 0."""
    from rag_mcp.providers.dimensions import resolve_collection_dimension

    provider = _Provider(dimension=1024)
    got = await resolve_collection_dimension("c", provider, _Store(exists=True, dimension=None))
    assert got == 1024
