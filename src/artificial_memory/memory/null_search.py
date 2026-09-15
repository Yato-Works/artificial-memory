"""Fallback NullVectorSearchEngine when vector search dependencies are not installed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from artificial_memory.core.models import Memory


@dataclass
class VectorSearchResult:
    """Result of a vector search."""

    memory_id: int
    score: float = 0.0
    distance: float = 0.0
    metadata: dict = field(default_factory=dict)


class NullVectorSearchEngine:
    """Fallback vector search engine when FAISS or sentence-transformers are not installed."""

    def __init__(self, store: Any = None, *args: Any, **kwargs: Any):
        self.store = store

    def add_memory(self, memory: Memory) -> int:
        return 0

    def remove_memory(self, memory_id: int) -> bool:
        return False

    def search(
        self,
        query: str,
        topic_id: int | None = None,
        memory_type: str | None = None,
        k: int = 10,
        threshold: float = 0.0,
    ) -> list[VectorSearchResult]:
        return []

    def hybrid_search(
        self,
        query: str,
        topic_id: int | None = None,
        k: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[VectorSearchResult]:
        return []

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_memories": 0,
            "available": False,
            "message": "Vector dependencies (sentence-transformers / faiss) not installed.",
        }

    def _rebuild_index(self) -> None:
        pass
