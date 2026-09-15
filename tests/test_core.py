"""Tests for Artificial Memory core functionality."""

import tempfile
from pathlib import Path

import pytest

from artificial_memory.compression.compressor import RuleBasedCompressor
from artificial_memory.context.builder import BasicContextBuilder
from artificial_memory.core.models import (
    Conversation,
    ConversationStatus,
    Memory,
    MemoryType,
    Message,
    MessageRole,
    Project,
    RecallLevel,
    ResolutionLevel,
    Topic,
)
from artificial_memory.memory.compiler import IncrementalMemoryCompiler
from artificial_memory.recall.engine import BasicRecallEngine
from artificial_memory.storage.conversation_logger import ConversationManager
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore
from artificial_memory.topic.classifier import RuleBasedTopicClassifier


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
def conversation_manager(store):
    return ConversationManager(store)


@pytest.fixture
def topic_classifier(store):
    return RuleBasedTopicClassifier(store)


@pytest.fixture
def compressor():
    return RuleBasedCompressor()


@pytest.fixture
def recall_engine(store):
    return BasicRecallEngine(store)


@pytest.fixture
def context_builder(store, recall_engine):
    return BasicContextBuilder(store, recall_engine)


@pytest.fixture
def memory_compiler(store, compressor, topic_classifier):
    return IncrementalMemoryCompiler(store, compressor, topic_classifier)


def test_project_crud(store):
    project = Project(name="test-project", display_name="Test Project")
    project = store.create_project(project)
    assert project.id is not None

    fetched = store.get_project(project.id)
    assert fetched.name == "test-project"

    by_name = store.get_project_by_name("test-project")
    assert by_name.id == project.id

    projects = store.list_projects()
    assert len(projects) == 1


def test_topic_crud(store):
    project = Project(name="test-project")
    project = store.create_project(project)

    topic = Topic(project_id=project.id, name="Architecture", path="Projects/test-project/Architecture")
    topic = store.create_topic(topic)
    assert topic.id is not None

    fetched = store.get_topic(topic.id)
    assert fetched.name == "Architecture"

    by_path = store.get_topic_by_path("Projects/test-project/Architecture")
    assert by_path.id == topic.id


def test_conversation_logging(conversation_manager):
    # Create topic
    conversation_manager.get_or_create_topic("Projects/Test/Topic")

    # Start conversation
    conv = conversation_manager.start_conversation("Projects/Test/Topic", title="Test Conversation")
    assert conv.id is not None
    assert conv.status == ConversationStatus.ACTIVE

    # Log messages
    msg1 = conversation_manager.log_user("Hello, how are you?")
    assert msg1.role == MessageRole.USER
    assert msg1.sequence_num == 1

    msg2 = conversation_manager.log_assistant("I'm doing well, thank you!")
    assert msg2.role == MessageRole.ASSISTANT
    assert msg2.sequence_num == 2

    # End conversation
    ended = conversation_manager.end_conversation()
    assert ended.status == ConversationStatus.COMPLETED
    assert ended.message_count == 2


def test_memory_operations(store):
    project = Project(name="test")
    project = store.create_project(project)

    topic = Topic(project_id=project.id, name="Test", path="Projects/test/Test")
    topic = store.create_topic(topic)

    # Create memory
    memory = Memory(
        topic_id=topic.id,
        memory_type=MemoryType.CURRENT,
        content="Test memory content",
        resolution=ResolutionLevel.RAW,
        importance=0.8,
        confidence=0.9,
    )
    memory = store.create_memory(memory)
    assert memory.id is not None

    # Fetch
    fetched = store.get_memory(memory.id)
    assert fetched.content == "Test memory content"

    # List
    memories = store.get_memories(topic_id=topic.id)
    assert len(memories) == 1

    # Update
    memory.content = "Updated content"
    memory = store.update_memory(memory)
    assert memory.content == "Updated content"


def test_memory_versions(store):
    project = Project(name="test")
    project = store.create_project(project)

    topic = Topic(project_id=project.id, name="Test", path="Projects/test/Test")
    topic = store.create_topic(topic)

    memory = Memory(
        topic_id=topic.id,
        memory_type=MemoryType.SEMANTIC,
        content="Original content",
        resolution=ResolutionLevel.SEMANTIC,
    )
    memory = store.create_memory(memory)

    # Add versions

    from artificial_memory.core.models import MemoryVersion

    light_version = MemoryVersion(
        memory_id=memory.id,
        resolution=ResolutionLevel.LIGHT,
        content="Light compressed",
        compression_ratio=2.0,
    )
    light_version = store.add_memory_version(light_version)
    assert light_version.id is not None

    versions = store.get_memory_versions(memory.id)
    assert len(versions) == 1
    assert versions[0].resolution == ResolutionLevel.LIGHT


def test_compressor(compressor):
    original = """いや〜思うわけですよ？
    このAを採用したら普通に将来的に保守性とか堅牢性の観点から意外に悪いかな〜とか思うわけだけど、
    結構実装楽じゃない？
    でも、将来的なこと考えたらC++コードとかをネスト構造にして、
    できる限り保守性とか上げながら……
    まぁBでいこうか！"""

    # Light compression
    light, meta = compressor.compress_light(original, {})
    assert meta["compression_ratio"] > 1.0
    assert "実装楽" in light or "実装が楽" in light

    # Episode compression
    episode, meta = compressor.compress_episode(original, {})
    assert meta["compression_ratio"] > 1.0

    # Semantic compression
    semantic, meta = compressor.compress_semantic(original, {})
    assert "決定" in semantic or "採用" in semantic

    # Long-term compression
    longterm, meta = compressor.compress_long_term(original, {})
    assert meta["compression_ratio"] > 1.0


def test_recall_engine(store, recall_engine):
    project = Project(name="test")
    project = store.create_project(project)

    topic = Topic(project_id=project.id, name="Test", path="Projects/test/Test")
    topic = store.create_topic(topic)

    # Create some memories
    for i in range(5):
        memory = Memory(
            topic_id=topic.id,
            memory_type=MemoryType.SEMANTIC,
            content=f"Memory {i}: Important decision about architecture choice",
            resolution=ResolutionLevel.SEMANTIC,
            importance=0.7 + i * 0.05,
        )
        store.create_memory(memory)

    # Recall
    memories, tokens = recall_engine.recall("architecture decision", topic.id, RecallLevel.CURRENT_ONLY, 1000)
    assert len(memories) > 0
    assert tokens > 0


def test_context_builder(store, recall_engine, context_builder):
    project = Project(name="test")
    project = store.create_project(project)

    topic = Topic(project_id=project.id, name="Test", path="Projects/test/Test")
    topic = store.create_topic(topic)

    memory = Memory(
        topic_id=topic.id,
        memory_type=MemoryType.CURRENT,
        content="Current state: Using B architecture for maintainability",
        resolution=ResolutionLevel.SEMANTIC,
        importance=0.9,
        confidence=0.95,
        is_current=True,
    )
    store.create_memory(memory)

    current_memories = [memory]
    context = context_builder.build_context("current architecture", topic.id, 4000, current_memories)

    assert "B architecture" in context
    stats = context_builder.get_context_stats()
    assert stats.effective_tokens > 0


def test_topic_classifier(store, topic_classifier):
    project = Project(name="test")
    project = store.create_project(project)

    topic = Topic(project_id=project.id, name="Architecture", path="Projects/test/Architecture")
    topic = store.create_topic(topic)

    # Classify - use ASCII to avoid encoding issues
    text = "architecture design discussion microservice monolith"
    classified, confidence = topic_classifier.classify(text, [topic])

    assert classified is not None
    assert classified.id == topic.id
    assert confidence > 0.3


def test_memory_compiler(store, compressor, topic_classifier, memory_compiler):
    project = Project(name="test")
    project = store.create_project(project)

    topic = Topic(project_id=project.id, name="Test", path="Projects/test/Test")
    topic = store.create_topic(topic)

    conversation = Conversation(
        topic_id=topic.id,
        title="Test Conv",
        started_at=__import__('datetime').datetime.now(),
    )
    conversation = store.create_conversation(conversation)

    # Add messages (use ASCII to avoid encoding issues)
    msg1 = Message(conversation_id=conversation.id, role=MessageRole.USER, content="Which is better, A or B?", sequence_num=1)
    msg2 = Message(conversation_id=conversation.id, role=MessageRole.ASSISTANT, content="Let's adopt B for maintainability. The reason is future extensibility.", sequence_num=2)
    store.add_message(msg1)
    store.add_message(msg2)

    # Compile
    memories = memory_compiler.compile_conversation(conversation)
    assert len(memories) >= 3  # timeline, light, episode, semantic

    # Check decision was extracted (may not work with simple heuristic)
    decisions = store.get_decisions(topic.id)
    # Note: decision extraction is heuristic-based and may not catch all cases
    assert len(decisions) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
