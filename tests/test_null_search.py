"""Tests for NullVectorSearchEngine fallback."""

from artificial_memory.memory.null_search import NullVectorSearchEngine


def test_null_vector_search_engine():
    engine = NullVectorSearchEngine()

    assert engine.search("test") == []
    assert engine.hybrid_search("test") == []
    assert engine.add_memory(None) == 0  # type: ignore
    assert engine.remove_memory(1) is False

    stats = engine.get_stats()
    assert stats["available"] is False
    assert stats["total_memories"] == 0
