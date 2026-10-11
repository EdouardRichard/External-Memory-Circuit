"""Collection dimension resolution shared by indexing and projection paths.

A dense collection needs a positive vector dimension before it can be
created. Existing collections carry their own dimension, a new collection
depends on the embedding provider, and when neither can supply a positive
value the caller must fail loudly rather than create a 0-dimension
collection (which Qdrant would reject or, worse, leave unusable).
"""

from __future__ import annotations

from typing import Protocol

from rag_mcp.providers.base import EmbeddingProvider


class DimensionStore(Protocol):
    """The read-only slice of ``QdrantStore`` this helper needs (sync)."""

    def collection_exists(self, name: str) -> bool: ...

    def get_collection_dimension(self, name: str) -> int | None: ...


async def resolve_collection_dimension(
    collection: str,
    provider: EmbeddingProvider,
    qdrant_store: DimensionStore,
) -> int:
    """Determine a collection's vector dimension: existing first, else provider.

    Three branches:

    1. The collection exists and its dense dimension reads back as a positive
       integer: that dimension wins and the provider is never touched.
    2. Otherwise (new collection, or an existing one whose dimension cannot
       be introspected): ask ``await provider.resolve_dimension()``.
    3. If the resolved value is not positive, raise ``ValueError`` with the
       literal ``EMBEDDING_DIMENSION_UNRESOLVED`` so the operator can set
       ``EMBEDDING_DIMENSION`` explicitly. Never return 0.

    Args:
        collection: Target collection name.
        provider: Embedding provider exposing ``resolve_dimension()``.
        qdrant_store: Store with synchronous ``collection_exists`` and
            ``get_collection_dimension``.

    Returns:
        A positive vector dimension.

    Raises:
        ValueError: When no positive dimension can be determined.
    """
    if qdrant_store.collection_exists(collection):
        existing = qdrant_store.get_collection_dimension(collection)
        if existing is not None and existing > 0:
            return int(existing)

    resolved = await provider.resolve_dimension()
    if resolved is None or resolved <= 0:
        raise ValueError(
            f"EMBEDDING_DIMENSION_UNRESOLVED: cannot determine dimension for "
            f"collection {collection!r}; set EMBEDDING_DIMENSION explicitly"
        )
    return int(resolved)
