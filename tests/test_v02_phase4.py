"""Tests for AM v0.2.0 Phase 4: Context Allocator.

Covers (plan #18 and the Phase 4 implementation list):

* token budgeting (budget is never exceeded, raises on invalid budgets),
* full/compressed/tag representations (desired level per priority, stored
  MemoryVersion usage, deterministic TAG synthesis),
* relevance scoring (relevance floor, contradiction penalty, importance),
* priority allocation (greedy downgrade under pressure, OMIT reasons,
  utility-per-token accounting, Phase 3 integration).
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from artificial_memory.context.allocator import (
    AllocatorConfig,
    ContextAllocator,
    RepresentationLevel,
)
from artificial_memory.core.models import (
    Conversation,
    Memory,
    MemoryType,
    MemoryVersion,
    Project,
    ResolutionLevel,
    Topic,
)
from artificial_memory.memory.reconstruction import MemoryReconstructor
from artificial_memory.storage.sqlite_store import SQLiteMemoryStore


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    import gc
    import time

    gc.collect()
    time.sleep(0.1)
    try:
        db_path.unlink(missing_ok=True)
    except PermissionError:
        pass


@pytest.fixture
def store(temp_db):
    s = SQLiteMemoryStore(temp_db)
    yield s
    s.close()


@pytest.fixture
def topic_id(store):
    project = store.create_project(Project(name="v02-phase4"))
    topic = store.create_topic(
        Topic(project_id=project.id, name="Test", path="Projects/v02-phase4/Test")
    )
    return topic.id


@pytest.fixture
def conversation_id(store, topic_id):
    conv = store.create_conversation(Conversation(topic_id=topic_id, title="chat"))
    return conv.id


@pytest.fixture
def make_memory(store, topic_id, conversation_id):
    def _make(content: str, **kwargs) -> Memory:
        return store.create_memory(
            Memory(
                topic_id=topic_id,
                memory_type=kwargs.pop("memory_type", MemoryType.SEMANTIC),
                content=content,
                source_conversation_id=kwargs.pop("source_conversation_id", conversation_id),
                **kwargs,
            )
        )

    return _make


LONG_CONTENT = (
    "The user adopted Rust for systems programming, migrated the CLI tooling, "
    "rewrote the build scripts, and now maintains three production services."
)

# ==================== Configuration ====================


class TestAllocatorConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"relevance_weight": 1.5},
            {"relevance_floor": -0.1},
            {"full_threshold": 0.4, "compressed_threshold": 0.6},  # unordered
            {"tag_max_tokens": 0},
            {"retention": {"full": 2.0}},
        ],
    )
    def test_rejects_invalid_values(self, kwargs):
        with pytest.raises(ValueError):
            AllocatorConfig(**kwargs)

    def test_retention_lookup(self):
        config = AllocatorConfig()
        assert config.retention_for(RepresentationLevel.FULL) == 1.0
        assert config.retention_for(RepresentationLevel.OMIT) == 0.0


# ==================== Desired representation ====================


class TestDesiredRepresentation:
    def test_high_priority_wants_full(self, store, make_memory):
        memory = make_memory("The user adopted Rust.", importance=0.95, confidence=0.99)
        allocator = ContextAllocator(store)
        result = allocator.allocate("The user adopted Rust", [memory], max_tokens=500)

        assert result.distribution == {"full": 1}
        part = result.parts[0]
        assert part.desired_level is RepresentationLevel.FULL
        assert part.content == memory.content

    def test_medium_priority_wants_compressed(self, store, make_memory):
        memory = make_memory("The user adopted Rust for tooling.", importance=0.5)
        config = AllocatorConfig(compressed_threshold=0.6, full_threshold=0.9)
        allocator = ContextAllocator(store, config)
        result = allocator.allocate("The user adopted Rust for tooling", [memory], max_tokens=500)
        # No stored versions exist, so COMPRESSED degrades to a TAG.
        assert result.distribution == {"tag": 1}
        assert result.parts[0].desired_level is RepresentationLevel.COMPRESSED
        assert result.parts[0].reason == "degraded from compressed to tag to fit the budget"

    def test_low_priority_wants_tag(self, store, make_memory):
        memory = make_memory("The user adopted Rust.", importance=0.2, confidence=0.5)
        config = AllocatorConfig(
            full_threshold=0.85, compressed_threshold=0.8, summary_threshold=0.7
        )
        allocator = ContextAllocator(store, config)
        result = allocator.allocate("Rust", [memory], max_tokens=500)
        assert result.parts[0].level is RepresentationLevel.TAG
        assert result.parts[0].desired_level is RepresentationLevel.TAG


# ==================== Stored versions ====================


class TestStoredVersions:
    def test_compressed_uses_stored_version(self, store, make_memory):
        memory = make_memory(LONG_CONTENT)
        compressed = "User adopted Rust; maintains services."
        store.add_memory_version(
            MemoryVersion(
                memory_id=memory.id,
                resolution=ResolutionLevel.LIGHT,
                content=compressed,
            )
        )
        allocator = ContextAllocator(store)
        level, content = allocator._render(memory, RepresentationLevel.COMPRESSED)
        assert level is RepresentationLevel.COMPRESSED
        assert content == compressed

    def test_summary_uses_two_steps_down(self, store, make_memory):
        memory = make_memory(LONG_CONTENT)
        store.add_memory_version(
            MemoryVersion(
                memory_id=memory.id,
                resolution=ResolutionLevel.EPISODE,
                content="Rust adoption story.",
            )
        )
        allocator = ContextAllocator(store)
        level, content = allocator._render(memory, RepresentationLevel.SUMMARY)
        assert level is RepresentationLevel.SUMMARY
        assert content == "Rust adoption story."

    def test_compressed_falls_back_to_summary_when_unstored(self, store, make_memory):
        memory = make_memory(LONG_CONTENT)
        store.add_memory_version(
            MemoryVersion(
                memory_id=memory.id,
                resolution=ResolutionLevel.EPISODE,  # two steps below RAW
                content="Rust adoption story.",
            )
        )
        allocator = ContextAllocator(store)
        level, content = allocator._render(memory, RepresentationLevel.COMPRESSED)
        # No LIGHT version stored: COMPRESSED degrades; SUMMARY (steps=2)
        # reaches the stored EPISODE version, and the part reports the
        # effective level rather than the requested one.
        assert level is RepresentationLevel.SUMMARY
        assert content == "Rust adoption story."

    def test_all_degrades_to_tag_when_nothing_stored(self, store, make_memory):
        memory = make_memory(LONG_CONTENT)
        allocator = ContextAllocator(store)
        level, content = allocator._render(memory, RepresentationLevel.COMPRESSED)
        assert level is RepresentationLevel.TAG
        assert content.startswith(f"#{memory.id}")

    def test_tag_is_deterministic_and_small(self, store, make_memory):
        memory = make_memory("Rust adoption story with CLI tooling.")
        allocator = ContextAllocator(store)
        first = allocator._tag(memory)
        second = allocator._tag(memory)
        assert first == second
        assert first.startswith(f"#{memory.id}")
        assert allocator.token_counter.count(first) <= 14


# ==================== Budgeting and allocation ====================


class TestBudgetAllocation:
    def test_budget_is_never_exceeded(self, store, make_memory):
        candidates = [make_memory(LONG_CONTENT, importance=0.9) for _ in range(5)]
        allocator = ContextAllocator(store)
        result = allocator.allocate("Rust systems programming", candidates, max_tokens=120)

        assert result.used_tokens <= 120
        assert result.parts or result.omitted  # something was decided
        assert all(part.tokens > 0 for part in result.included)

    def test_downgrade_under_budget_pressure(self, store, make_memory):
        memory = make_memory(LONG_CONTENT, importance=0.95, confidence=0.99)
        allocator = ContextAllocator(store)
        # Enough for a compressed version but not the full text.
        store.add_memory_version(
            MemoryVersion(
                memory_id=memory.id,
                resolution=ResolutionLevel.LIGHT,
                content="User adopted Rust; maintains services.",
            )
        )
        result = allocator.allocate("Rust", [memory], max_tokens=18)
        part = result.parts[0]
        assert part.level is RepresentationLevel.COMPRESSED
        assert part.desired_level is RepresentationLevel.FULL
        assert part.reason == "degraded from full to compressed to fit the budget"

    def test_when_nothing_fits_everything_is_omitted(self, store, make_memory):
        memory = make_memory(LONG_CONTENT)
        allocator = ContextAllocator(store)
        result = allocator.allocate("Rust", [memory], max_tokens=2)
        assert result.parts == []
        assert result.used_tokens == 0
        assert result.omitted[0].level is RepresentationLevel.OMIT
        assert result.omitted[0].reason == "budget exhausted"

    def test_below_relevance_floor_is_omitted(self, store, make_memory):
        relevant = make_memory("The user adopted Rust for tooling.")
        irrelevant = make_memory("The cat slept on the keyboard all afternoon.")
        allocator = ContextAllocator(store)
        result = allocator.allocate("Rust tooling", [relevant, irrelevant], max_tokens=400)

        assert relevant.id in result.memory_ids()
        assert irrelevant.id not in result.memory_ids()
        omitted = {part.memory_id: part for part in result.omitted}
        assert omitted[irrelevant.id].reason == "below relevance floor"

    def test_non_current_is_omitted(self, store, make_memory):
        memory = make_memory("The user adopted Rust.", is_current=False)
        allocator = ContextAllocator(store)
        result = allocator.allocate("Rust", [memory], max_tokens=400)
        assert result.omitted[0].reason == "superseded / not current"

    def test_higher_priority_gets_full_lower_gets_summary(self, store, make_memory):
        vital = make_memory(LONG_CONTENT, importance=1.0, confidence=1.0)
        minor = make_memory("Minor Rust note about formatting.", importance=0.1, confidence=0.5)
        config = AllocatorConfig(
            full_threshold=0.85, compressed_threshold=0.75, summary_threshold=0.55
        )
        allocator = ContextAllocator(store, config)
        result = allocator.allocate("Rust", [vital, minor], max_tokens=1000)

        by_id = {part.memory_id: part for part in result.parts}
        assert by_id[vital.id].level is RepresentationLevel.FULL
        assert by_id[minor.id].level is RepresentationLevel.TAG

    def test_raises_on_invalid_budget(self, store, make_memory):
        memory = make_memory("The user adopted Rust.")
        allocator = ContextAllocator(store)
        with pytest.raises(ValueError):
            allocator.allocate("Rust", [memory], max_tokens=0)

    def test_empty_candidates(self, store):
        result = ContextAllocator(store).allocate("anything", [], max_tokens=100)
        assert result.parts == []
        assert result.considered == 0
        assert result.used_tokens == 0


# ==================== Scoring and auditability ====================


class TestScoring:
    def test_contradiction_lowers_priority(self, store, make_memory):
        memory = make_memory("The user deploys on Fridays.", importance=0.8)
        rival = make_memory("The user never deploys on Fridays.")

        allocator = ContextAllocator(store)
        when = datetime.now()
        before = allocator.allocate("deploys", [memory], max_tokens=500, now=when).parts[0]

        from artificial_memory.memory.contradiction_edges import ContradictionEdgeManager

        ContradictionEdgeManager(store).create_edge(memory.id, rival.id, severity=0.9)
        after = allocator.allocate("deploys", [memory], max_tokens=500, now=when).parts[0]

        assert after.priority < before.priority

    def test_part_accounting_tracks_utility_per_token(self, store, make_memory):
        memory = make_memory("The user adopted Rust.", importance=0.9)
        result = ContextAllocator(store).allocate("Rust", [memory], max_tokens=500)

        part = result.parts[0]
        assert part.useful_information > 0.0
        assert part.utility_per_token == pytest.approx(
            part.useful_information / part.tokens
        )
        assert result.avg_utility_per_token > 0.0

    def test_allocation_is_deterministic(self, store, make_memory):
        candidates = [
            make_memory("The user adopted Rust for tooling.", importance=0.8),
            make_memory("The user prefers dark mode.", importance=0.4),
        ]
        allocator = ContextAllocator(store)
        when = datetime.now()
        first = allocator.allocate("Rust", candidates, max_tokens=60, now=when).to_dict()
        second = allocator.allocate("Rust", candidates, max_tokens=60, now=when).to_dict()
        assert first == second

    def test_to_dict_and_to_text_are_serializable(self, store, make_memory):
        memory = make_memory("The user adopted Rust for tooling.")
        result = ContextAllocator(store).allocate("Rust tooling", [memory], max_tokens=500)

        payload = result.to_dict()
        assert payload["distribution"]["full"] == 1
        assert payload["parts"][0]["memory_id"] == memory.id

        text = result.to_text()
        assert "[FULL]" in text
        assert "Rust" in text


# ==================== Phase 3 integration ====================


class TestReconstructionIntegration:
    def test_allocate_from_reconstruction_package(self, store, make_memory):
        from artificial_memory.core.models import Association, AssociationType

        a = make_memory("The user started learning Rust.", importance=0.9)
        b = make_memory("Built a first CLI application in Rust.")
        c = make_memory("Continued using it for later projects.")
        store.create_association(
            Association(source_memory_id=a.id, target_memory_id=b.id,
                        association_type=AssociationType.FOLLOWS, strength=0.9)
        )
        store.create_association(
            Association(source_memory_id=b.id, target_memory_id=c.id,
                        association_type=AssociationType.FOLLOWS, strength=0.9)
        )

        package = MemoryReconstructor(store).reconstruct(
            "Which programming language has the user increasingly adopted?"
        )
        allocation = ContextAllocator(store).allocate_from_reconstruction(package, max_tokens=200)

        assert allocation.considered >= 1
        assert set(allocation.memory_ids()) <= {a.id, b.id, c.id}
        assert allocation.used_tokens <= 200
        assert allocation.avg_utility_per_token > 0.0

    def test_reconstructed_chains_fit_in_small_budget(self, store, make_memory):
        from artificial_memory.core.models import Association, AssociationType

        a = make_memory(
            "The user spent months learning Rust and rebuilt the entire backend "
            "tooling with it, documenting every step for the team.",
            importance=0.95,
            confidence=0.95,
        )
        b = make_memory("Built a first CLI application in Rust.")
        c = make_memory("Continued using it for later projects.")
        store.create_association(
            Association(source_memory_id=a.id, target_memory_id=b.id,
                        association_type=AssociationType.FOLLOWS, strength=0.9)
        )
        store.create_association(
            Association(source_memory_id=b.id, target_memory_id=c.id,
                        association_type=AssociationType.FOLLOWS, strength=0.9)
        )
        package = MemoryReconstructor(store).reconstruct("learning Rust")

        # Tiny budget: high-priority evidence degrades instead of vanishing.
        allocation = ContextAllocator(store).allocate_from_reconstruction(package, max_tokens=18)
        assert allocation.memory_ids(), "chain evidence should survive as degraded parts"
        assert allocation.used_tokens <= 18
        assert all(part.level is not RepresentationLevel.FULL for part in allocation.included)



