from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import faiss
import numpy as np

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Memory, MemoryType
from artificial_memory.memory.embeddings import get_sentence_transformer
from artificial_memory.metrics.collector import get_metrics_collector


@dataclass
class VectorSearchResult:
    """Result of a vector search."""
    memory_id: int
    score: float
    distance: float
    metadata: dict = field(default_factory=dict)


class VectorSearchEngine:
    """FAISS-based vector search engine for memories."""

    def __init__(
        self,
        store: MemoryStore,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: Path | None = None,
        dimension: int = 384,
    ):
        self.store = store
        self.model = get_sentence_transformer(model_name)
        self.dimension = dimension
        self.index_path = index_path or Path("vector_index")
        self.index_path.mkdir(parents=True, exist_ok=True)

        # FAISS index
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity with normalized vectors)
        self.id_to_memory_id: dict[int, int] = {}  # FAISS index -> memory_id
        self.memory_id_to_index: dict[int, int] = {}  # memory_id -> FAISS index

        # Metadata storage
        self.metadata_file = self.index_path / "metadata.json"
        self.index_file = self.index_path / "faiss.index"

        self._load_or_create_index()
        self.collector = get_metrics_collector()

    def _load_or_create_index(self):
        """Load existing index or create new one.

        Tolerates a missing or corrupt metadata file by starting fresh:
        an empty metadata.json (e.g. from an interrupted non-atomic write
        in an earlier process) must never crash engine construction.
        """
        try:
            if self.index_file.exists() and self.metadata_file.exists():
                index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, encoding="utf-8") as f:
                    data = json.load(f)
                if index.ntotal == len(data.get("memory_id_to_index", {})):
                    self.index = index
                    self.id_to_memory_id = {
                        int(k): v for k, v in data.get("id_to_memory_id", {}).items()
                    }
                    self.memory_id_to_index = {
                        int(k): v for k, v in data.get("memory_id_to_index", {}).items()
                    }
                    return
        except (json.JSONDecodeError, OSError, KeyError, ValueError, RuntimeError):
            pass
        # Corrupt or inconsistent state: rebuild from scratch
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_to_memory_id = {}
        self.memory_id_to_index = {}

    def _save_index(self):
        """Save index and metadata to disk (atomically)."""
        # Write both files to temporary names first, then atomically replace,
        # so a crash mid-write can never leave an empty/partial metadata file.
        tmp_meta = self.metadata_file.with_suffix(".json.tmp")
        tmp_index = self.index_file.with_suffix(".index.tmp")
        faiss.write_index(self.index, str(tmp_index))
        with open(tmp_meta, "w", encoding="utf-8") as f:
            json.dump({
                'id_to_memory_id': self.id_to_memory_id,
                'memory_id_to_index': self.memory_id_to_index,
                'dimension': self.dimension,
                'total_vectors': self.index.ntotal,
                'updated_at': datetime.now().isoformat(),
            }, f)
        self._atomic_replace(tmp_index, self.index_file)
        self._atomic_replace(tmp_meta, self.metadata_file)

    @staticmethod
    def _atomic_replace(src: Path, dst: Path) -> None:
        """``os.replace`` with retry for transient Windows AV file locks.

        On Windows, antivirus scanners briefly hold a shared-read lock on
        freshly written files; ``os.replace`` then fails with WinError 5
        (Access denied) even though the process has full permissions.  The
        lock clears within milliseconds, so a short bounded retry converts
        the flaky failure into the intended atomic swap.
        """
        last_exc: OSError | None = None
        for attempt in range(5):
            try:
                os.replace(str(src), str(dst))
                return
            except PermissionError as exc:  # WinError 5 (transient AV lock)
                last_exc = exc
                time.sleep(0.05 * (2 ** attempt))  # 50ms, 100ms, 200ms, 400ms
        raise last_exc  # type: ignore[misc]

    def _encode_text(self, text: str) -> np.ndarray:
        """Encode text to normalized vector."""
        vector = self.model.encode([text], normalize_embeddings=True)
        return vector.astype(np.float32)

    def add_memory(self, memory: Memory) -> int:
        """Add a memory to the vector index."""
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

        # Add to FAISS index
        faiss_idx = self.index.ntotal
        self.index.add(vector)

        # Update mappings
        self.id_to_memory_id[faiss_idx] = memory.id
        self.memory_id_to_index[memory.id] = faiss_idx

        self._save_index()
        return faiss_idx

    def update_memory(self, memory: Memory) -> int:
        """Update a memory in the vector index (remove old, add new)."""
        if memory.id in self.memory_id_to_index:
            # FAISS doesn't support direct update, so we rebuild
            # For now, just add new version
            pass
        return self.add_memory(memory)

    def remove_memory(self, memory_id: int) -> bool:
        """Remove a memory from the index (requires rebuild)."""
        if memory_id in self.memory_id_to_index:
            # FAISS doesn't support direct removal, need rebuild
            self._rebuild_index()
            return True
        return False

    def _rebuild_index(self):
        """Rebuild the entire index from current memories."""
        # Get all memories from store
        memories = self.store.get_memories(limit=10000)

        # Rebuild index
        self.index = faiss.IndexFlatIP(self.dimension)
        self.id_to_memory_id = {}
        self.memory_id_to_index = {}

        for mem in memories:
            self.add_memory(mem)

        self._save_index()

    def search(
        self,
        query: str,
        topic_id: int | None = None,
        memory_type: str | None = None,
        k: int = 10,
        threshold: float = 0.0,
    ) -> list[VectorSearchResult]:
        """Search for similar memories using vector similarity."""
        if self.index.ntotal == 0:
            return []

        query_vector = self._encode_text(query)

        # Search
        search_k = min(k * 3, self.index.ntotal)
        distances, indices = self.index.search(query_vector, search_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if dist < threshold:
                continue

            memory_id = self.id_to_memory_id.get(int(idx))
            if memory_id is None:
                continue

            # Filter by topic if specified
            if topic_id is not None:
                memory = self.store.get_memory(memory_id)
                if not memory or memory.topic_id != topic_id:
                    continue

            # Filter by memory type if specified
            if memory_type is not None:
                memory = self.store.get_memory(memory_id)
                if not memory or memory.memory_type.value != memory_type:
                    continue

            results.append(VectorSearchResult(
                memory_id=memory_id,
                score=float(dist),
                distance=float(dist),
                metadata={"faiss_index": int(idx)}
            ))

            if len(results) >= k:
                break

        return results

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

        # Keyword search (simple implementation)
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
                distance=1.0 - score,  # Approximate
                metadata={"hybrid": True}
            ))

        return results

    def _keyword_search(self, query: str, topic_id: int | None = None, k: int = 10) -> list[VectorSearchResult]:
        """Simple keyword-based search."""
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
        """Get index statistics."""
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": "IndexFlatIP",
            "memory_mappings": len(self.id_to_memory_id),
        }


def create_vector_search_engine(
    store: MemoryStore,
    model_name: str = "all-MiniLM-L6-v2",
    index_path: Path | None = None,
) -> VectorSearchEngine:
    """Factory function to create vector search engine."""
    return VectorSearchEngine(store, model_name, index_path)
