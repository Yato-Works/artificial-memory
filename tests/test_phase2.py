"""Tests for Compiler Pipeline and Runtime Facade."""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from artificial_memory.compiler import create_compiler_pipeline
from artificial_memory.core.models import (
    Conversation,
    Message,
    MessageRole,
    Project,
    Topic,
)
from artificial_memory.runtime import ArtificialMemoryRuntime, RuntimeConfig
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    # Windows: ensure file is closed before deletion
    import gc
    gc.collect()
    import time
    time.sleep(0.1)
    try:
        db_path.unlink(missing_ok=True)
    except PermissionError:
        pass  # Ignore on Windows if still locked


@pytest.fixture
def store(temp_db):
    s = SQLiteMemoryStore(temp_db)
    yield s
    s.close()


@pytest.fixture
def sample_conversation(store):
    """Create a sample conversation with messages."""
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

    # Add messages
    msg1 = Message(
        conversation_id=conv.id,
        role=MessageRole.USER,
        content="We need to choose a database for the new project",
        sequence_num=1,
    )
    msg2 = Message(
        conversation_id=conv.id,
        role=MessageRole.ASSISTANT,
        content="Let's adopt PostgreSQL for horizontal scaling. The reason is multi-region support.",
        sequence_num=2,
    )
    msg3 = Message(
        conversation_id=conv.id,
        role=MessageRole.USER,
        content="What about SQLite for simplicity?",
        sequence_num=3,
    )
    msg4 = Message(
        conversation_id=conv.id,
        role=MessageRole.ASSISTANT,
        content="SQLite is good for local dev but we decided on PostgreSQL for production. This decision is final.",
        sequence_num=4,
    )

    store.add_message(msg1)
    store.add_message(msg2)
    store.add_message(msg3)
    store.add_message(msg4)

    return conv, topic


class TestCompilerPipeline:
    """Test the new Compiler Pipeline."""

    def test_pipeline_creation(self):
        """Test that pipeline can be created with default stages."""
        pipeline = create_compiler_pipeline()
        assert pipeline is not None
        assert len(pipeline.stages) == 10
        stage_names = [s.name for s in pipeline.stages]
        assert "lexical_analysis" in stage_names
        assert "structural_analysis" in stage_names
        assert "semantic_extraction" in stage_names
        assert "fact_decision_intent" in stage_names
        assert "episode_construction" in stage_names
        assert "temporal_linking" in stage_names
        assert "provenance_linking" in stage_names
        assert "classification" in stage_names
        assert "compression" in stage_names
        assert "optimization" in stage_names

    def test_pipeline_compiles_conversation(self, store, sample_conversation):
        """Test that pipeline produces MemoryIR from conversation."""
        conv, topic = sample_conversation
        messages = store.get_messages(conv.id)

        pipeline = create_compiler_pipeline()
        memory_irs = pipeline.compile(conv, messages)

        # Should produce multiple memory IRs
        # Expected: 2 semantic (decisions/important), 1 decision, 1 timeline, 1 light episode = 5
        assert len(memory_irs) >= 4

        # Check types
        types = {mem.type.value for mem in memory_irs}
        assert "semantic" in types or "decision" in types

        # All should have provenance
        for mem in memory_irs:
            assert mem.source.source.conversation_id == conv.id
            assert mem.identity.memory_id == 0  # Not yet stored

    def test_pipeline_with_diagnostics(self, store, sample_conversation):
        """Test that pipeline returns diagnostics."""
        conv, topic = sample_conversation
        messages = store.get_messages(conv.id)

        pipeline = create_compiler_pipeline()
        memory_irs, diagnostics = pipeline.compile_with_diagnostics(conv, messages)

        # Should have some diagnostics (at least INFO level)
        assert isinstance(diagnostics, list)


class TestRuntimeFacade:
    """Test the Runtime Facade."""

    def test_runtime_creation(self, temp_db):
        """Test that runtime can be created and closed."""
        config = RuntimeConfig(database_path=str(temp_db))
        runtime = ArtificialMemoryRuntime(config)

        # Should have all components initialized
        assert runtime.store is not None
        assert runtime.conversation_manager is not None
        assert runtime.recall_engine is not None
        assert runtime.context_builder is not None
        assert runtime.memory_compiler is not None
        # Note: pipeline_wrapper is in CompilerPipelineWrapper, not directly in runtime
        assert runtime.association_engine is not None
        assert runtime.temporal_engine is not None
        assert runtime.confidence_engine is not None
        assert runtime.style_engine is not None
        assert runtime.human_recall_engine is not None
        assert runtime.vector_search_engine is not None
        assert runtime.llm_manager is not None

        runtime.close()
        # Give time for DB connections to close on Windows
        import time
        time.sleep(0.1)

    def test_remember_recall_cycle(self, temp_db):
        """Test basic remember/recall through facade."""
        config = RuntimeConfig(database_path=str(temp_db))
        runtime = ArtificialMemoryRuntime(config)

        import asyncio

        async def test_cycle():
            # Remember a fact
            mem_ir = await runtime.remember(
                content="PostgreSQL was chosen for the project database",
                topic="Projects/Test/Database",
                memory_type="semantic",
            )
            assert mem_ir.identity.memory_id > 0
            assert mem_ir.type.value == "semantic"

            # Recall it
            recall_result = await runtime.recall(
                query="database choice",
                topic="Projects/Test/Database",
            )
            assert recall_result.memories_retrieved >= 1
            assert len(recall_result.memories) >= 1

            # Check memory content
            recalled = recall_result.memories[0]
            assert "PostgreSQL" in recalled.semantic_content.content

        asyncio.run(test_cycle())
        runtime.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
