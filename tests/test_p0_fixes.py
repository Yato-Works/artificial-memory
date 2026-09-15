"""Tests for P0 correctness fixes (V1 Release Hardening, Phase 0).

Covers:
- P0-3: ConfidenceProfile.overall combines all four components
- P0-4: BeliefEngine persists beliefs across restarts
- P0-5: EvolutionEvent is persisted to storage
- P0-6: ProvenanceChain.compilation_chain populated by compiler
- P0-7: MemoryVersion records policy_version / algorithm_version (+ migration)
"""
from __future__ import annotations

import sqlite3
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from artificial_memory.compiler import CompilerPipeline
from artificial_memory.compression.compressor import RuleBasedCompressor
from artificial_memory.core.ir import ConfidenceProfile
from artificial_memory.core.models import (
    Conversation,
    MemoryVersion,
    Message,
    MessageRole,
    ResolutionLevel,
)
from artificial_memory.memory.belief import BeliefEngine
from artificial_memory.memory.evolution import (
    EvolutionOperationType,
    MemoryEvolutionEngine,
)
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmp:
        s = SQLiteMemoryStore(Path(tmp) / "test.db")
        yield s
        s.close()


def _ensure_topic(store) -> int:
    from artificial_memory.core.models import Project, Topic
    project = store.create_project(Project(name=f"test-project-{uuid.uuid4().hex[:12]}"))
    topic = store.create_topic(Topic(
        project_id=project.id, name="test", path=f"{project.name}/test"
    ))
    return topic.id


def _make_memory(store, content: str) -> int:
    from artificial_memory.core.models import Memory, MemoryType
    mem = Memory(
        topic_id=_ensure_topic(store),
        memory_type=MemoryType.SEMANTIC,
        content=content,
        importance=0.8,
        confidence=0.9,
    )
    return store.create_memory(mem).id


# ==================== P0-3: ConfidenceProfile ====================

def test_confidence_overall_combines_all_components():
    cp = ConfidenceProfile(
        memory_confidence=1.0,
        retrieval_confidence=1.0,
        temporal_confidence=1.0,
        source_confidence=1.0,
        overall=0.0,  # ignored - must be recomputed
    )
    assert cp.overall == pytest.approx(1.0)


def test_confidence_overall_weighted_sum():
    cp = ConfidenceProfile(
        memory_confidence=0.8,     # * 0.40 = 0.32
        retrieval_confidence=0.2,  # * 0.25 = 0.05
        temporal_confidence=1.0,   # * 0.20 = 0.20
        source_confidence=0.0,     # * 0.15 = 0.00
        overall=0.9,               # passed value must be ignored
    )
    assert cp.overall == pytest.approx(0.57)


# ==================== P0-4: Belief persistence ====================

def test_belief_survives_restart(store):
    mid = _make_memory(store, "We chose PostgreSQL for scaling")
    engine = BeliefEngine(store)
    belief = engine.create_or_update_belief(
        "The project database is PostgreSQL", [mid]
    )
    original_id = belief.id

    # Simulate restart: new engine over the same store
    engine2 = BeliefEngine(store)
    loaded = engine2.get_belief(original_id)
    assert loaded is not None
    assert loaded.proposition == "The project database is PostgreSQL"
    assert loaded.supporting_evidence == [mid]


def test_belief_update_persists_new_evidence(store):
    mid1 = _make_memory(store, "PostgreSQL chosen")
    mid2 = _make_memory(store, "pgvector added for search")
    engine = BeliefEngine(store)
    engine.create_or_update_belief("DB is PostgreSQL", [mid1])

    engine2 = BeliefEngine(store)  # restart
    engine2.create_or_update_belief("DB is PostgreSQL", [mid2])

    persisted = store.get_all_belief_states()
    assert len(persisted) == 1
    assert set(persisted[0].supporting_evidence) == {mid1, mid2}


# ==================== P0-5: Evolution events ====================

def test_evolution_event_persisted(store):
    mid = _make_memory(store, "Original content about caching strategy")
    engine = MemoryEvolutionEngine(store, RuleBasedCompressor())
    engine.revise_memory(mid, "Use Redis for cache eviction policy")

    events = store.get_evolution_events(memory_id=mid)
    assert len(events) >= 1
    assert events[0].operation == EvolutionOperationType.REVISE.value


# ==================== P0-6: compilation_chain ====================

def test_compilation_chain_populated():
    conversation = Conversation(topic_id=1, title="t", started_at=datetime.now())
    messages = [
        Message(conversation_id=1, role=MessageRole.USER,
                content="We decided to use PostgreSQL because we need horizontal scaling.",
                sequence_num=1, created_at=datetime.now()),
        Message(conversation_id=1, role=MessageRole.ASSISTANT,
                content="Understood: PostgreSQL chosen for horizontal scaling.",
                sequence_num=2, created_at=datetime.now()),
    ]
    pipeline = CompilerPipeline()
    ir_list = pipeline.compile(conversation, messages)

    for ir in ir_list:
        assert ir.source.source.conversation_id == conversation.id
        assert len(ir.source.compilation_chain) > 0
        stages = [link.stage for link in ir.source.compilation_chain]
        assert "lexical_analysis" in stages
        assert "compression" in stages


# ==================== P0-7: policy/algorithm version + migration ====================

def test_memory_version_records_policy_and_algorithm(store):
    mid = _make_memory(store, "version policy test content")
    version = MemoryVersion(
        memory_id=mid,
        resolution=ResolutionLevel.LIGHT,
        content="compressed",
        compression_ratio=0.5,
    )
    saved = store.add_memory_version(version)
    loaded = store.get_memory_version(mid, ResolutionLevel.LIGHT)
    assert loaded.policy_version == "decision-v1"
    assert loaded.algorithm_version == "rule-compressor-v1"
    assert saved.policy_version == "decision-v1"


def test_migration_on_legacy_database(tmp_path):
    """Opening a pre-V1 database (old memory_versions schema) must migrate."""
    legacy_db = tmp_path / "legacy.db"

    # Build a legacy database with the OLD memory_versions schema
    conn = sqlite3.connect(legacy_db)
    conn.executescript("""
        CREATE TABLE memory_versions (
            id INTEGER PRIMARY KEY,
            memory_id INTEGER NOT NULL,
            resolution INTEGER NOT NULL,
            content TEXT NOT NULL,
            compression_ratio REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            source TEXT
        );
        INSERT INTO memory_versions (memory_id, resolution, content, compression_ratio, created_at, source)
        VALUES (1, 1, 'legacy content', 0.5, '2025-01-01T00:00:00', 'auto');
    """)
    conn.commit()

    # Run only the migration logic on the legacy connection
    s = SQLiteMemoryStore.__new__(SQLiteMemoryStore)
    s.db_path = legacy_db
    conn.row_factory = sqlite3.Row
    s._conn = conn
    s._migrate_schema()

    columns = {row["name"] for row in conn.execute("PRAGMA table_info(memory_versions)")}
    assert "policy_version" in columns
    assert "algorithm_version" in columns
    row = conn.execute("SELECT policy_version, algorithm_version FROM memory_versions").fetchone()
    assert row["policy_version"] == "decision-v1"
    assert row["algorithm_version"] == "rule-compressor-v1"
    conn.close()


