"""Tests for Phase 8: Distributed Runtime - PostgreSQL/pgvector backend."""

import importlib.util
import os
from datetime import datetime

import pytest

# Check if PostgreSQL dependencies are available
HAS_POSTGRES = (
    importlib.util.find_spec("psycopg") is not None
    and importlib.util.find_spec("pgvector") is not None
)

if HAS_POSTGRES:
    from artificial_memory.core.models import (
        AssociationType,
        Conversation,
        Memory,
        MemoryType,
        Project,
        ResolutionLevel,
        Topic,
    )
    from artificial_memory.memory.pgvector_search import (
        create_pgvector_search_engine,
    )
    from artificial_memory.storage.postgres_store import (
        PostgresConfig,
        PostgresMemoryStore,
    )

# PostgreSQL test configuration - uses environment variables or defaults
POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.environ.get("POSTGRES_DB", "artificial_memory_test")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "postgres")

DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"


def _postgres_server_available() -> bool:
    """Return True if a PostgreSQL server can actually be reached.

    Without this check, tests hang for a long time (or fail obscurely) when
    no local PostgreSQL server is running.
    """
    if not HAS_POSTGRES:
        return False
    try:
        import psycopg

        psycopg.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname="postgres",
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            connect_timeout=3,
        ).close()
        return True
    except Exception:
        return False


POSTGRES_AVAILABLE = _postgres_server_available()


@pytest.mark.skipif(
    not POSTGRES_AVAILABLE,
    reason="PostgreSQL server not available (start with docker-compose or set POSTGRES_HOST etc.)",
)
class TestPostgresMemoryStore:
    """Test PostgreSQL Memory Store."""

    @pytest.fixture(scope="class")
    def store(self):
        """Create a test database connection."""
        config = PostgresConfig(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        store = PostgresMemoryStore(config)
        yield store
        store.close()

    @pytest.fixture
    def sample_data(self, store):
        """Create sample project, topic, and conversation."""
        project = Project(name="test-project")
        project = store.create_project(project)

        topic = Topic(project_id=project.id, name="Test", path="Projects/test-project/Test")
        topic = store.create_topic(topic)

        conv = Conversation(
            topic_id=topic.id,
            title="Test Conversation",
            started_at=datetime.now(),
        )
        conv = store.create_conversation(conv)

        return {
            "project": project,
            "topic": topic,
            "conversation": conv,
        }

    def test_create_project(self, store):
        project = Project(name="test-project-2")
        project = store.create_project(project)

        assert project.id is not None
        assert project.name == "test-project-2"

        # Retrieve and verify
        retrieved = store.get_project(project.id)
        assert retrieved is not None
        assert retrieved.name == "test-project-2"

    def test_create_topic(self, store):
        project = Project(name="test-project-3")
        project = store.create_project(project)

        topic = Topic(project_id=project.id, name="Test Topic", path="Projects/test/Topic")
        topic = store.create_topic(topic)

        assert topic.id is not None
        assert topic.name == "Test Topic"
        assert topic.path == "Projects/test/Topic"

        # Retrieve by path
        retrieved = store.get_topic_by_path("Projects/test/Topic")
        assert retrieved is not None
        assert retrieved.id == topic.id

    def test_create_conversation(self, store, sample_data):
        conv = Conversation(
            topic_id=sample_data["topic"].id,
            title="Test Conversation 2",
            started_at=datetime.now(),
        )
        conv = store.create_conversation(conv)

        assert conv.id is not None
        assert conv.title == "Test Conversation 2"

    def test_create_memory(self, store, sample_data):
        mem = Memory(
            topic_id=sample_data["topic"].id,
            memory_type=MemoryType.SEMANTIC,
            content="Test memory content for PostgreSQL",
            resolution=ResolutionLevel.SEMANTIC,
            importance=0.8,
            confidence=0.9,
            source_conversation_id=sample_data["conversation"].id,
        )
        mem = store.create_memory(mem)

        assert mem.id is not None
        assert mem.content == "Test memory content for PostgreSQL"

        # Retrieve and verify
        retrieved = store.get_memory(mem.id)
        assert retrieved is not None
        assert retrieved.content == mem.content

    def test_memory_operations(self, store, sample_data):
        # Create multiple memories
        memories = []
        for i in range(3):
            mem = Memory(
                topic_id=sample_data["topic"].id,
                memory_type=MemoryType.SEMANTIC,
                content=f"Memory {i}",
                resolution=ResolutionLevel.SEMANTIC,
                importance=0.5 + i * 0.1,
                confidence=0.9,
                source_conversation_id=sample_data["conversation"].id,
            )
            mem = store.create_memory(mem)
            memories.append(mem)

        # List memories
        all_memories = store.get_memories(topic_id=sample_data["topic"].id, limit=10)
        assert len(all_memories) >= 3

        # Filter by type
        semantic_memories = store.get_memories(
            topic_id=sample_data["topic"].id,
            memory_type=MemoryType.SEMANTIC,
            limit=10
        )
        assert len(semantic_memories) >= 3

        # Update memory
        memories[0].importance = 0.95
        updated = store.update_memory(memories[0])
        assert updated.importance == 0.95

        # Delete memory
        deleted = store.delete_memory(memories[1].id)
        assert deleted is True

        # Verify deletion
        remaining = store.get_memories(topic_id=sample_data["topic"].id, limit=10)
        assert len(remaining) == 2

    def test_memory_versions(self, store, sample_data):
        mem = Memory(
            topic_id=sample_data["topic"].id,
            memory_type=MemoryType.SEMANTIC,
            content="Original content",
            resolution=ResolutionLevel.RAW,
            importance=0.5,
            confidence=0.9,
        )
        mem = store.create_memory(mem)

        # Add versions
        from artificial_memory.core.models import MemoryVersion
        for res in [ResolutionLevel.LIGHT, ResolutionLevel.EPISODE, ResolutionLevel.SEMANTIC]:
            version = MemoryVersion(
                memory_id=mem.id,
                resolution=res,
                content=f"Compressed to {res.name}",
                compression_ratio=2.0,
            )
            version = store.add_memory_version(version)
            assert version.id is not None

        # Get versions
        versions = store.get_memory_versions(mem.id)
        assert len(versions) == 3

        # Get specific version
        semantic_version = store.get_memory_version(mem.id, ResolutionLevel.SEMANTIC)
        assert semantic_version is not None
        assert semantic_version.resolution == ResolutionLevel.SEMANTIC

    def test_associations(self, store, sample_data):
        mem1 = Memory(
            topic_id=sample_data["topic"].id,
            memory_type=MemoryType.SEMANTIC,
            content="Memory 1",
            resolution=ResolutionLevel.SEMANTIC,
        )
        mem1 = store.create_memory(mem1)

        mem2 = Memory(
            topic_id=sample_data["topic"].id,
            memory_type=MemoryType.SEMANTIC,
            content="Memory 2",
            resolution=ResolutionLevel.SEMANTIC,
        )
        mem2 = store.create_memory(mem2)

        # Create association
        from artificial_memory.core.models import Association
        assoc = Association(
            source_memory_id=mem1.id,
            target_memory_id=mem2.id,
            association_type=AssociationType.RELATED,
            strength=0.8,
        )
        assoc = store.create_association(assoc)

        assert assoc.id is not None
        assert assoc.strength == 0.8

        # Get related memories
        related = store.get_related_memories(mem1.id)
        assert len(related) >= 1
        assert related[0][0].id == mem2.id


@pytest.mark.skipif(
    not POSTGRES_AVAILABLE,
    reason="PostgreSQL server not available (start with docker-compose or set POSTGRES_HOST etc.)",
)
class TestPgVectorSearchEngine:
    """Test pgvector-based vector search engine."""

    @pytest.fixture(scope="class")
    def store(self):
        config = PostgresConfig(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
        )
        store = PostgresMemoryStore(config)
        yield store
        store.close()

    @pytest.fixture(scope="class")
    def vector_engine(self, store):
        return create_pgvector_search_engine(store)

    @pytest.fixture
    def sample_memories(self, store):
        project = Project(name="vector-test")
        project = store.create_project(project)

        topic = Topic(project_id=project.id, name="Vector Test", path="Projects/vector-test/Topic")
        topic = store.create_topic(topic)

        memories = []
        contents = [
            "PostgreSQL was chosen for horizontal scaling",
            "The team decided to use Redis for caching",
            "Docker is used for containerization",
            "Kubernetes will orchestrate the services",
            "Python is the primary programming language",
        ]

        for content in contents:
            mem = Memory(
                topic_id=topic.id,
                memory_type=MemoryType.SEMANTIC,
                content=content,
                resolution=ResolutionLevel.SEMANTIC,
                importance=0.8,
                confidence=0.9,
            )
            mem = store.create_memory(mem)
            memories.append(mem)

        return memories, topic.id

    def test_add_memory_to_vector_index(self, vector_engine, sample_memories):
        memories, topic_id = sample_memories

        for mem in memories:
            idx = vector_engine.add_memory(mem)
            assert idx == mem.id

    def test_vector_search(self, vector_engine, sample_memories, store):
        memories, topic_id = sample_memories

        # Add all memories to vector index
        for mem in memories:
            vector_engine.add_memory(mem)

        # Search for database-related content
        results = vector_engine.search(
            query="database scaling postgresql",
            topic_id=topic_id,
            k=5,
            threshold=0.3
        )

        assert len(results) > 0
        # Should find the PostgreSQL memory
        found_pg = False
        for r in results:
            mem = store.get_memory(r.memory_id)
            if "PostgreSQL" in mem.content:
                found_pg = True
                break
        assert found_pg, "Should find PostgreSQL memory"

    def test_hybrid_search(self, vector_engine, sample_memories, store):
        memories, topic_id = sample_memories

        for mem in memories:
            vector_engine.add_memory(mem)

        results = vector_engine.hybrid_search(
            query="docker container",
            topic_id=topic_id,
            k=5
        )

        assert len(results) > 0
        found_docker = False
        for r in results:
            mem = store.get_memory(r.memory_id)
            if "Docker" in mem.content:
                found_docker = True
                break
        assert found_docker, "Should find Docker memory"

    def test_get_stats(self, vector_engine):
        stats = vector_engine.get_stats()

        assert "total_memories" in stats
        assert "memories_with_embeddings" in stats
        assert "dimension" in stats
        assert stats["dimension"] == 384


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
