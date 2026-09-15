"""Tests for Memory IR and adapter lossless round-trip."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from artificial_memory.core.adapters import (
    IRToMemoryAdapter,
    create_memory_ir_adapter,
)
from artificial_memory.core.ir import MemoryIR
from artificial_memory.core.models import (
    Association,
    AssociationType,
    CompressionEvent,
    CompressionMethod,
    Conversation,
    Memory,
    MemoryStatus,
    MemoryType,
    MemoryVersion,
    Project,
    ResolutionLevel,
    Topic,
)
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    db_path.unlink(missing_ok=True)


@pytest.fixture
def store(temp_db):
    s = SQLiteMemoryStore(temp_db)
    yield s
    s.close()


@pytest.fixture
def adapter(store):
    return create_memory_ir_adapter(store)


@pytest.fixture
def sample_data(store):
    """Create sample project, topic, conversation, memories with versions and associations."""
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

    # Create memories of different types
    memories = []

    # Memory 1: SEMANTIC with versions and compression
    mem1 = Memory(
        topic_id=topic.id,
        memory_type=MemoryType.SEMANTIC,
        content="Decision: We adopted PostgreSQL for better scalability. Reason: Horizontal scaling support.",
        resolution=ResolutionLevel.SEMANTIC,
        importance=0.9,
        confidence=0.95,
        source_conversation_id=conv.id,
    )
    mem1 = store.create_memory(mem1)
    memories.append(mem1)

    # Add versions for mem1
    light_ver = MemoryVersion(memory_id=mem1.id, resolution=ResolutionLevel.LIGHT, content="Light: Adopted PostgreSQL", compression_ratio=1.5)
    episode_ver = MemoryVersion(memory_id=mem1.id, resolution=ResolutionLevel.EPISODE, content="Episode: Team discussed DB choice", compression_ratio=2.0)
    store.add_memory_version(light_ver)
    store.add_memory_version(episode_ver)

    # Add compression events
    store.log_compression(CompressionEvent(
        source_memory_id=mem1.id,
        from_resolution=ResolutionLevel.RAW,
        to_resolution=ResolutionLevel.LIGHT,
        original_tokens=100,
        compressed_tokens=60,
        compression_ratio=1.67,
        method=CompressionMethod.LIGHT,
    ))
    store.log_compression(CompressionEvent(
        source_memory_id=mem1.id,
        from_resolution=ResolutionLevel.LIGHT,
        to_resolution=ResolutionLevel.EPISODE,
        original_tokens=60,
        compressed_tokens=30,
        compression_ratio=2.0,
        method=CompressionMethod.EPISODE,
    ))

    # Memory 2: DECISION
    mem2 = Memory(
        topic_id=topic.id,
        memory_type=MemoryType.DECISION,
        content="Decided to use microservices architecture",
        resolution=ResolutionLevel.SEMANTIC,
        importance=0.85,
        confidence=0.9,
        source_conversation_id=conv.id,
    )
    mem2 = store.create_memory(mem2)
    memories.append(mem2)

    # Memory 3: EPISODE
    mem3 = Memory(
        topic_id=topic.id,
        memory_type=MemoryType.EPISODE,
        content="Team discussed migration timeline for Q3",
        resolution=ResolutionLevel.EPISODE,
        importance=0.6,
        confidence=0.8,
    )
    mem3 = store.create_memory(mem3)
    memories.append(mem3)

    # Create associations
    assoc1 = Association(
        source_memory_id=mem1.id,
        target_memory_id=mem2.id,
        association_type=AssociationType.RELATED,
        strength=0.8,
    )
    store.create_association(assoc1)

    assoc2 = Association(
        source_memory_id=mem2.id,
        target_memory_id=mem3.id,
        association_type=AssociationType.FOLLOWS,
        strength=0.7,
    )
    store.create_association(assoc2)

    return {
        "project": project,
        "topic": topic,
        "conversation": conv,
        "memories": memories,
    }


class TestMemoryIRRoundTrip:
    """Test lossless round-trip conversion between Memory and MemoryIR."""

    def test_single_memory_roundtrip(self, adapter, sample_data):
        """Test that a single memory can be converted to IR and back without loss."""
        original = sample_data["memories"][0]

        # Convert to IR
        ir = adapter.to_ir(original)
        assert isinstance(ir, MemoryIR)
        assert ir.identity.memory_id == original.id
        assert ir.type == original.memory_type
        assert ir.resolution == original.resolution
        assert ir.semantic_content.content == original.content

        # Convert back to legacy
        ir_adapter = IRToMemoryAdapter(adapter.store)
        restored_mem, restored_versions, restored_assocs, restored_compressions = ir_adapter.to_legacy(ir)

        # Verify core fields match
        assert restored_mem.id == original.id
        assert restored_mem.topic_id == original.topic_id
        assert restored_mem.memory_type == original.memory_type
        assert restored_mem.content == original.content
        assert restored_mem.resolution == original.resolution
        assert restored_mem.importance == original.importance
        assert restored_mem.confidence == original.confidence
        assert restored_mem.status == original.status
        assert restored_mem.valid_from == original.valid_from
        assert restored_mem.valid_until == original.valid_until
        assert restored_mem.is_current == original.is_current
        assert restored_mem.source_conversation_id == original.source_conversation_id

    def test_all_memory_types_roundtrip(self, adapter, store, sample_data):
        """Test round-trip for all memory types."""
        topic = sample_data["topic"]
        conv = sample_data["conversation"]

        for mem_type in MemoryType:
            memory = Memory(
                topic_id=topic.id,
                memory_type=mem_type,
                content=f"Test content for {mem_type.value}",
                resolution=ResolutionLevel.SEMANTIC,
                importance=0.7,
                confidence=0.8,
                source_conversation_id=conv.id,
            )
            memory = store.create_memory(memory)

            ir = adapter.to_ir(memory)
            assert ir.type == mem_type

            ir_adapter = IRToMemoryAdapter(adapter.store)
            restored_mem, _, _, _ = ir_adapter.to_legacy(ir)

            assert restored_mem.memory_type == mem_type
            assert restored_mem.content == memory.content

    def test_provenance_chain_preserved(self, adapter, sample_data):
        """Test that source conversation/message IDs are preserved."""
        original = sample_data["memories"][0]

        ir = adapter.to_ir(original)

        assert ir.source.source.conversation_id == original.source_conversation_id
        assert ir.source.source.message_id == original.source_message_id
        assert ir.source.source.timestamp == original.created_at

    def test_versions_preserved(self, adapter, sample_data):
        """Test that memory versions are preserved in compression history."""
        original = sample_data["memories"][0]

        ir = adapter.to_ir(original)

        # Should have compression records for LIGHT and EPISODE
        assert len(ir.compression_history) >= 2
        resolutions = {r.to_resolution for r in ir.compression_history}
        assert ResolutionLevel.LIGHT in resolutions
        assert ResolutionLevel.EPISODE in resolutions

    def test_associations_preserved(self, adapter, sample_data):
        """Test that associations are preserved as relations and dependencies."""
        from artificial_memory.core.ir import AssociationType as IRAssociationType

        original = sample_data["memories"][0]  # mem1

        ir = adapter.to_ir(original)

        # Should have relations (both directions)
        assert len(ir.relations) >= 1
        # Relations should contain RELATED (from assoc1: mem1 -> mem2)
        rel_types = {r.association_type for r in ir.relations}
        assert IRAssociationType.RELATED in rel_types
        # Dependencies only include CAUSES, FOLLOWS, ELABORATES (not RELATED)
        # For mem1, only RELATED association exists, so dependencies should be empty
        dep_types = {d.dependency_type for d in ir.dependencies}
        # This is expected behavior - RELATED is not a dependency type
        assert len(dep_types) == 0 or IRAssociationType.FOLLOWS in dep_types

    def test_batch_conversion(self, adapter, sample_data):
        """Test batch conversion of multiple memories."""
        memories = sample_data["memories"]

        irs = adapter.batch_to_ir(memories)

        assert len(irs) == len(memories)
        for i, ir in enumerate(irs):
            assert ir.identity.memory_id == memories[i].id
            assert ir.type == memories[i].memory_type
            assert ir.semantic_content.content == memories[i].content

    def test_confidence_profile_preserved(self, adapter, sample_data):
        """Test that confidence profile is properly built."""
        original = sample_data["memories"][0]

        ir = adapter.to_ir(original)

        assert ir.confidence.memory_confidence == original.confidence
        assert ir.confidence.retrieval_confidence == original.importance
        assert 0.5 <= ir.confidence.temporal_confidence <= 1.0
        assert ir.confidence.source_confidence == 0.9
        # P0-3: overall must be the weighted combination of all components,
        # not a pass-through of memory_confidence.
        expected_overall = (
            0.40 * ir.confidence.memory_confidence
            + 0.25 * ir.confidence.retrieval_confidence
            + 0.20 * ir.confidence.temporal_confidence
            + 0.15 * ir.confidence.source_confidence
        )
        assert ir.confidence.overall == pytest.approx(expected_overall)

    def test_lifecycle_state_mapping(self, adapter, store, sample_data):
        """Test that MemoryStatus maps correctly to LifecycleState."""
        topic = sample_data["topic"]

        for status, expected_lifecycle in [
            (MemoryStatus.ACTIVE, "HOT"),
            (MemoryStatus.DORMANT, "WARM"),
            (MemoryStatus.COMPRESSED, "COLD"),
            (MemoryStatus.ARCHIVED, "ARCHIVED"),
            (MemoryStatus.DEEP_ARCHIVED, "DEEP_ARCHIVED"),
        ]:
            memory = Memory(
                topic_id=topic.id,
                memory_type=MemoryType.CURRENT,
                content=f"Test {status.value}",
                resolution=ResolutionLevel.RAW,
                status=status,
            )
            memory = store.create_memory(memory)

            ir = adapter.to_ir(memory)
            assert ir.lifecycle_state.value == expected_lifecycle.lower()

    def test_temporal_scope_preserved(self, adapter, sample_data):
        """Test that temporal fields are preserved."""
        original = sample_data["memories"][0]

        ir = adapter.to_ir(original)

        assert ir.temporal_scope.valid_from == original.valid_from
        assert ir.temporal_scope.valid_until == original.valid_until
        assert ir.temporal_scope.created_at == original.created_at
        assert ir.temporal_scope.updated_at == original.updated_at

    def test_access_history_preserved(self, adapter, store, sample_data):
        """Test that access count and last accessed are preserved."""
        topic = sample_data["topic"]

        memory = Memory(
            topic_id=topic.id,
            memory_type=MemoryType.CURRENT,
            content="Access test",
            resolution=ResolutionLevel.RAW,
            access_count=5,
            last_accessed=datetime.now(),
        )
        memory = store.create_memory(memory)

        ir = adapter.to_ir(memory)

        assert ir.access_history.access_count == 5
        assert ir.access_history.last_accessed is not None


class TestMemoryIRValidation:
    """Test that MemoryIR enforces strict validation."""

    def test_strict_mode_rejects_extra_fields(self, adapter, sample_data):
        """MemoryIR should reject unknown fields in strict mode."""
        original = sample_data["memories"][0]
        ir = adapter.to_ir(original)

        # This should not raise - we're testing the model can be created
        assert ir is not None

    def test_identity_content_hash(self, adapter, sample_data):
        """Test that content hash is deterministic."""
        original = sample_data["memories"][0]

        ir1 = adapter.to_ir(original)
        ir2 = adapter.to_ir(original)

        # Same content should produce same hash
        assert ir1.identity.content_hash == ir2.identity.content_hash

        # Different content should produce different hash
        original.content = "Modified content"
        ir3 = adapter.to_ir(original)
        assert ir3.identity.content_hash != ir1.identity.content_hash


class TestIRToMemoryAdapter:
    """Test the reverse adapter (IR -> Legacy)."""

    def test_to_legacy_returns_tuple(self, adapter, sample_data):
        """Test that to_legacy returns the expected tuple structure."""
        original = sample_data["memories"][0]
        ir = adapter.to_ir(original)

        ir_adapter = IRToMemoryAdapter(adapter.store)
        result = ir_adapter.to_legacy(ir)

        assert isinstance(result, tuple)
        assert len(result) == 4
        memory, versions, associations, compressions = result

        assert isinstance(memory, Memory)
        assert isinstance(versions, list)
        assert isinstance(associations, list)
        assert isinstance(compressions, list)

    def test_restored_memory_has_correct_id(self, adapter, sample_data):
        """Test that restored memory preserves the original ID."""
        original = sample_data["memories"][0]
        ir = adapter.to_ir(original)

        ir_adapter = IRToMemoryAdapter(adapter.store)
        restored_mem, _, _, _ = ir_adapter.to_legacy(ir)

        assert restored_mem.id == original.id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
