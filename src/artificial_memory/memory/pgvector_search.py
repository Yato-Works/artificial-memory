from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from artificial_memory.core.models import Memory, MemoryType
from artificial_memory.memory.embeddings import get_sentence_transformer
from artificial_memory.metrics.collector import get_metrics_collector
from artificial_memory.storage.postgres_store import PostgresMemoryStore


@dataclass
class VectorSearchResult:
    """Result of a vector search."""
    memory_id: int
    score: float
    distance: float
    metadata: dict = field(default_factory=dict)


class PgVectorSearchEngine:
    """pgvector-based vector search engine for memories using PostgreSQL."""

    def __init__(
        self,
        store: PostgresMemoryStore,
        model_name: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
    ):
        self.store = store
        self.model = get_sentence_transformer(model_name)
        self.dimension = dimension
        self.collector = get_metrics_collector()

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to normalized vector."""
        vector = self.model.encode([text], normalize_embeddings=True)
        return vector.astype(np.float32).flatten()

    def add_memory(self, memory: Memory) -> int:
        """Add a memory to the vector index by computing and storing embedding."""
        # Create text representation for embedding
        text_parts = []
        if memory.memory_type == MemoryType.DECISION:
            text_parts.append(f"Decision: {memory.content}")
        elif memory.memory_type == MemoryType.SEMANTIC:
            text_parts.append(f"Semantic: {memory.content}")
        elif memory.memory_type == MemoryType.EPISODE:
            text_parts.append(f"Episode: {memory.content}")
        elif memory.memory_type == MemoryType.TIMELINE:
            text_parts.append(f"Timeline: {memory.content}")
        else:
            text_parts.append(memory.content)

        text = " ".join(text_parts)
        vector = self._encode_text(text)

        # Store embedding in PostgreSQL
        self.store.update_memory_embedding(memory.id, vector.tolist())

        return memory.id

    def update_memory(self, memory: Memory) -> int:
        """Update a memory's embedding in the vector index."""
        return self.add_memory(memory)

    def remove_memory(self, memory_id: int) -> bool:
        """Remove a memory's embedding (set to NULL)."""
        with self.store._pool.connection() as conn:
            cursor = conn.execute(
                "UPDATE memories SET embedding = NULL WHERE id = %s",
                (memory_id,)
            )
            return cursor.rowcount > 0

    def rebuild_index(self):
        """Rebuild embeddings for all memories."""
        memories = self.store.get_memories(limit=10000)
        for mem in memories:
            self.add_memory(mem)

    def search(
        self,
        query: str,
        topic_id: int | None = None,
        memory_type: str | None = None,
        k: int = 10,
        threshold: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search for similar memories using pgvector similarity."""
        query_vector = self._encode_text(query).tolist()

        results = self.store.vector_search(
            query_embedding=query_vector,
            topic_id=topic_id,
            memory_type=memory_type,
            k=k,
            threshold=threshold,
        )

        return [
            VectorSearchResult(
                memory_id=memory.id,
                score=similarity,
                distance=1.0 - similarity,
                metadata={}
            )
            for memory, similarity in results
        ]

    def hybrid_search(
        self,
        query: str,
        topic_id: int | None = None,
        k: int = 10,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ) -> list[VectorSearchResult]:
        """Hybrid search combining vector and keyword search."""
        # Vector search
        vector_results = self.search(query, topic_id=topic_id, k=k*2)

        # Keyword search
        keyword_results = self._keyword_search(query, topic_id=topic_id, k=k*2)

        # Combine scores
        combined = {}
        for r in vector_results:
            combined[r.memory_id] = combined.get(r.memory_id, 0) + r.score * vector_weight
        for r in keyword_results:
            combined[r.memory_id] = combined.get(r.memory_id, 0) + r.score * keyword_weight

        # Sort and return top k
        sorted_results = sorted(combined.items(), key=lambda x: -x[1])
        results = []
        for memory_id, score in sorted_results[:k]:
            results.append(VectorSearchResult(
                memory_id=memory_id,
                score=score,
                distance=1.0 - score,
                metadata={"hybrid": True}
            ))

        return results

    def _keyword_search(self, query: str, topic_id: int | None = None, k: int = 10) -> list[VectorSearchResult]:
        """Simple keyword-based search using PostgreSQL full-text search."""
        query_words = set(query.lower().split())
        memories = self.store.get_memories(topic_id=topic_id, limit=1000)

        results = []
        for mem in memories:
            content_words = set(mem.content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                score = overlap / len(query_words | content_words)  # Jaccard similarity
                results.append(VectorSearchResult(
                    memory_id=mem.id,
                    score=score,
                    distance=1.0 - score,
                    metadata={"keyword_match": overlap}
                ))

        results.sort(key=lambda x: -x.score)
        return results[:k]

    def get_stats(self) -> dict:
        """Get index statistics from PostgreSQL."""
        with self.store._pool.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as total, COUNT(embedding) as with_embedding FROM memories"
            ).fetchone()

        return {
            "total_memories": row["total"],
            "memories_with_embeddings": row["with_embedding"],
            "dimension": self.dimension,
            "index_type": "pgvector HNSW",
        }


def create_pgvector_search_engine(
    store: PostgresMemoryStore,
    model_name: str = "all-MiniLM-L6-v2",
) -> PgVectorSearchEngine:
    """Factory function to create pgvector search engine."""
    return PgVectorSearchEngine(store, model_name)
