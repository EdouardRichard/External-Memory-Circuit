"""Provider abstraction layer for Embedding, Reranker, and LLM backends.

Defines abstract base classes that all provider implementations must satisfy.
Only LocalCPUEmbeddingProvider is implemented for 001; others are stubs for
future features (002 Reranker, 005 Agentic Retrieval).

Blueprint §17, §18.1
"""

from abc import ABC, abstractmethod
from typing import Any


class EmbeddingProvider(ABC):
    """Abstract interface for text embedding providers."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of float vectors, same order as input.
        """
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text.

        Args:
            text: Query string to embed.

        Returns:
            Float vector.
        """
        ...

    async def resolve_dimension(self) -> int:
        """Return a dimension usable for creating a collection.

        A return value > 0 means the dimension is determined (declared in
        configuration, previously probed, or introspected from the loaded
        model). A return value of 0 means it is UNKNOWN: callers MUST NOT
        pass 0 to collection creation and must fail explicitly instead.
        """
        return self.get_dimension()


class RerankerProvider(ABC):
    """Abstract interface for reranking providers. Stub for 002."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Rerank candidates by relevance to query.

        Args:
            query: The search query.
            candidates: List of candidate dicts with at least 'text' key.
            top_k: Number of top results to return.

        Returns:
            Reranked list of candidate dicts with added 'rerank_score' key.
        """
        ...


class LLMProvider(ABC):
    """Abstract interface for LLM structured completion. Stub for 005."""

    @abstractmethod
    async def structured_complete(
        self,
        prompt: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate a structured completion conforming to the given schema.

        Args:
            prompt: The prompt/instruction for the LLM.
            schema: JSON Schema dict describing the expected output structure.

        Returns:
            Dict conforming to the provided schema.
        """
        ...
