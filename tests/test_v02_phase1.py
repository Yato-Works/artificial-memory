"""Tests for AM v0.2.0 Phase 1: Memory Evolution Core.

Covers:
* Memory state vectors (importance / confidence / currentness /
  future_utility / contradiction_risk) - deterministic computation
* New lifecycle operations: REINFORCE, REINTERPRET, RESTORE, KEEP, REJECT
* Evolution policy layer: proposal -> validation -> commit -> audit event
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from artificial_memory.compression.compressor import RuleBasedCompressor
from artificial_memory.core.models import (
    Association,
    AssociationType,
    Conversation,
    Memory,
    MemoryStatus,
    MemoryType,
    Project,
    ResolutionLevel,
    Topic,
)
from artificial_memory.memory.evolution import (
    EvolutionOperationType,
    MemoryEvolutionEngine,
)
from artificial_memory.memory.evolution_policy import (
    EvolutionGovernor,
    EvolutionPolicyConfig,
    EvolutionProposal,
    PolicyVerdict,
)
from artificial_memory.memory.state_signals import (
    StateSignalCalculator,
    StateSignalConfig,
)
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
def compressor():
    return RuleBasedCompressor()


@pytest.fixture
def engine(store, compressor):
    return MemoryEvolutionEngine(store, compressor)


@pytest.fixture
def governor(engine):
    return EvolutionGovernor(engine)


@pytest.fixture
def calculator(store):
    return StateSignalCalculator(store)


@pytest.fixture
def sample_memory(store):
    """Create a project / topic / conversation / one active memory."""
    project = store.create_project(Project(name="v02-phase1"))
    topic = store.create_topic(
        Topic(project_id=project.id, name="Test", path="Projects/v02-phase1/Test")
    )
    conv = store.create_conversation(Conversation(topic_id=topic.id, title="Conv"))
    memory = store.create_memory(
        Memory(
            topic_id=topic.id,
            memory_type=MemoryType.SEMANTIC,
            content="User prefers Rust for CLI tooling.",
            resolution=ResolutionLevel.SEMANTIC,
            importance=0.7,
            confidence=0.8,
            source_conversation_id=conv.id,
        )
    )
    return memory


# ==================== State vector ====================


class TestStateVector:
    def test_fresh_memory_baseline(self, calculator, sample_memory):
        vector = calculator.compute_state_vector(sample_memory)
        assert 0.0 <= vector.currentness <= 1.0
        assert 0.0 <= vector.future_utility <= 1.0
        assert vector.contradiction_risk == 0.0
        assert vector.importance == 0.7
        assert vector.confidence == 0.8

    def test_access_increases_usage_signal(self, calculator, sample_memory):
        before = calculator.compute_state_vector(sample_memory)
        sample_memory.touch()
        sample_memory.touch()
        after = calculator.compute_state_vector(sample_memory)
        assert after.signals["usage"] > before.signals["usage"]

    def test_expired_memory_has_zero_temporal_signal(self, calculator, sample_memory):
        now = datetime.now()
        before = calculator.compute_state_vector(sample_memory, now=now)
        assert before.signals["temporal"] == 1.0  # no window -> always relevant
        sample_memory.valid_until = now - timedelta(days=1)
        after = calculator.compute_state_vector(sample_memory, now=now)
        assert after.signals["temporal"] == 0.0
        assert after.currentness < before.currentness

    def test_future_validity_window_is_not_yet_relevant(self, calculator, sample_memory):
        now = datetime.now()
        sample_memory.valid_from = now + timedelta(days=7)
        vector = calculator.compute_state_vector(sample_memory, now=now)
        assert vector.signals["temporal"] == 0.0

    def test_contradiction_risk_from_association(self, store, calculator, sample_memory):
        other = store.create_memory(
            Memory(
                topic_id=sample_memory.topic_id,
                memory_type=MemoryType.SEMANTIC,
                content="User prefers Python for CLI tooling.",
                resolution=ResolutionLevel.SEMANTIC,
            )
        )
        store.create_association(
            Association(
                source_memory_id=sample_memory.id,
                target_memory_id=other.id,
                association_type=AssociationType.CONTRADICTS,
                strength=0.9,
            )
        )
        vector = calculator.compute_state_vector(sample_memory)
        assert vector.contradiction_risk == pytest.approx(0.9)
        # currentness penalized by contradiction
        assert vector.currentness < 1.0

    def test_recency_decays_with_age(self, calculator, sample_memory):
        now = datetime.now()
        fresh = calculator.compute_state_vector(sample_memory, now=now)
        old = calculator.compute_state_vector(sample_memory, now=now + timedelta(days=90))
        assert old.signals["recency"] < fresh.signals["recency"]
        assert old.currentness < fresh.currentness

    def test_ranking_by_future_utility(self, calculator, sample_memory, store):
        low = store.create_memory(
            Memory(
                topic_id=sample_memory.topic_id,
                memory_type=MemoryType.SEMANTIC,
                content="Low importance note.",
                importance=0.1,
                confidence=0.3,
            )
        )
        ranked = calculator.rank_by_future_utility([low, sample_memory])
        assert ranked[0][0].id == sample_memory.id
        assert ranked[0][1].future_utility >= ranked[1][1].future_utility

    def test_config_rejects_invalid_weights(self):
        with pytest.raises(ValueError):
            StateSignalConfig(utility_importance_weight=1.5)
        with pytest.raises(ValueError):
            StateSignalConfig(recency_half_life_days=0)


# ==================== Lifecycle operations ====================


class TestReinforce:
    def test_reinforce_raises_confidence(self, engine, store, sample_memory):
        old_confidence = sample_memory.confidence
        result = engine.reinforce_memory(sample_memory.id, amount=0.1)
        assert result.confidence == pytest.approx(old_confidence + 0.1)
        assert result.access_count == 1

        persisted = store.get_memory(sample_memory.id)
        assert persisted is not None
        assert persisted.confidence == pytest.approx(old_confidence + 0.1)

    def test_reinforce_is_capped_at_one(self, engine, sample_memory):
        result = engine.reinforce_memory(sample_memory.id, amount=0.5)
        result = engine.reinforce_memory(result.id, amount=0.5)
        assert result.confidence <= 1.0

    def test_reinforce_rejects_negative_amount(self, engine, sample_memory):
        with pytest.raises(ValueError):
            engine.reinforce_memory(sample_memory.id, amount=-0.1)

    def test_reinforce_logs_audit_event(self, engine, store, sample_memory):
        engine.reinforce_memory(sample_memory.id, reason="corroborated by user")
        events = store.get_evolution_events(memory_id=sample_memory.id)
        assert any(e.operation == "reinforce" for e in events)


class TestReinterpret:
    def test_reinterpret_creates_linked_semantic_memory(self, engine, store, sample_memory):
        interpretation = "User has systems-programming experience."
        result = engine.reinterpret_memory(sample_memory.id, interpretation, confidence=0.75)
        assert result.id != sample_memory.id
        assert result.content == interpretation
        assert result.memory_type == MemoryType.SEMANTIC
        assert result.resolution == ResolutionLevel.SEMANTIC

        associations = store.get_associations(sample_memory.id)
        assert any(
            a.target_memory_id == result.id
            and a.association_type == AssociationType.ELABORATES
            for a in associations
        )

    def test_reinterpret_preserves_evidence(self, engine, store, sample_memory):
        original = store.get_memory(sample_memory.id)
        engine.reinterpret_memory(sample_memory.id, "A new reading of the evidence.")
        still_there = store.get_memory(sample_memory.id)
        assert still_there is not None
        assert still_there.content == original.content

    def test_reinterpret_supersede_archives_source(self, engine, store, sample_memory):
        engine.reinterpret_memory(sample_memory.id, "Updated reading.", supersede=True)
        source = store.get_memory(sample_memory.id)
        assert source.status == MemoryStatus.ARCHIVED
        assert source.is_current is False

    def test_reinterpret_rejects_empty_interpretation(self, engine, sample_memory):
        with pytest.raises(ValueError):
            engine.reinterpret_memory(sample_memory.id, "   ")

    def test_reinterpret_logs_audit_event(self, engine, store, sample_memory):
        new_memory = engine.reinterpret_memory(sample_memory.id, "Interpretation.")
        events = store.get_evolution_events(memory_id=sample_memory.id)
        event = next(e for e in events if e.operation == "reinterpret")
        assert event.target_memory_id == new_memory.id


class TestRestore:
    def test_restore_recovers_more_detailed_version(self, engine, store, sample_memory):
        from artificial_memory.core.models import MemoryVersion

        sample_memory.status = MemoryStatus.ARCHIVED
        sample_memory.is_current = False
        sample_memory.resolution = ResolutionLevel.LONG_TERM
        sample_memory.content = "User likes Rust. (long-term summary)"
        store.update_memory(sample_memory)

        store.add_memory_version(
            MemoryVersion(
                memory_id=sample_memory.id,
                resolution=ResolutionLevel.EPISODE,
                content="User said they like Rust for building CLI tools.",
            )
        )

        restored = engine.restore_memory(sample_memory.id)
        assert restored.status == MemoryStatus.ACTIVE
        assert restored.is_current is True
        assert restored.resolution == ResolutionLevel.EPISODE
        assert restored.content == "User said they like Rust for building CLI tools."

    def test_restore_without_detailed_version_reactivates(self, engine, store, sample_memory):
        sample_memory.status = MemoryStatus.ARCHIVED
        sample_memory.is_current = False
        store.update_memory(sample_memory)
        restored = engine.restore_memory(
            sample_memory.id, target_resolution=ResolutionLevel.RAW
        )
        assert restored.status == MemoryStatus.ACTIVE
        assert restored.resolution == ResolutionLevel.SEMANTIC  # unchanged

    def test_restore_respects_target_resolution(self, engine, store, sample_memory):
        from artificial_memory.core.models import MemoryVersion

        store.add_memory_version(
            MemoryVersion(
                memory_id=sample_memory.id,
                resolution=ResolutionLevel.EPISODE,
                content="Episode detail.",
            )
        )
        store.add_memory_version(
            MemoryVersion(
                memory_id=sample_memory.id,
                resolution=ResolutionLevel.RAW,
                content="Raw verbatim detail.",
            )
        )
        sample_memory.resolution = ResolutionLevel.LONG_TERM
        store.update_memory(sample_memory)

        restored = engine.restore_memory(
            sample_memory.id, target_resolution=ResolutionLevel.EPISODE
        )
        # The most detailed version within the target is chosen (maximum
        # detail recovery, capped at the requested resolution).
        assert restored.resolution == ResolutionLevel.RAW
        assert restored.content == "Raw verbatim detail."


class TestKeepReject:
    def test_keep_is_non_destructive_and_audited(self, engine, store, sample_memory):
        original_content = sample_memory.content
        result = engine.keep_memory(sample_memory.id, reason="still relevant")
        assert result.content == original_content
        events = store.get_evolution_events(memory_id=sample_memory.id)
        assert any(e.operation == "keep" for e in events)

    def test_reject_makes_memory_dormant_not_deleted(self, engine, store, sample_memory):
        result = engine.reject_memory(sample_memory.id, reason="superseded by new preference")
        assert result.status == MemoryStatus.DORMANT
        assert result.is_current is False
        # Evidence remains auditable (forgetting != deletion)
        assert store.get_memory(sample_memory.id) is not None
        events = store.get_evolution_events(memory_id=sample_memory.id)
        assert any(e.operation == "reject" for e in events)

    def test_operations_reject_unknown_memory(self, engine):
        for call in (
            lambda: engine.reinforce_memory(99999),
            lambda: engine.reinterpret_memory(99999, "x"),
            lambda: engine.restore_memory(99999),
            lambda: engine.keep_memory(99999),
            lambda: engine.reject_memory(99999),
        ):
            with pytest.raises(ValueError):
                call()


# ==================== Evolution policy layer ====================


class TestEvolutionGovernor:
    def test_approved_reinforce_is_committed_and_audited(self, governor, store, sample_memory):
        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.REINFORCE,
                memory_id=sample_memory.id,
                reason="User reiterated the preference",
                metadata={"amount": 0.1},
            )
        )
        assert decision.verdict == PolicyVerdict.APPROVE
        assert decision.committed is True
        assert decision.resulting_memory_id == sample_memory.id

        persisted = store.get_memory(sample_memory.id)
        assert persisted.confidence == pytest.approx(0.9)
        events = store.get_evolution_events(memory_id=sample_memory.id)
        assert any(e.operation == "reinforce" for e in events)

    def test_missing_memory_is_rejected(self, governor):
        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.REINFORCE,
                memory_id=99999,
                reason="check",
            )
        )
        assert decision.verdict == PolicyVerdict.REJECT
        assert any("not found" in r for r in decision.reasons)
        assert decision.committed is False

    def test_missing_reason_is_rejected(self, governor, sample_memory):
        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.REINFORCE,
                memory_id=sample_memory.id,
                reason="   ",
            )
        )
        assert decision.verdict == PolicyVerdict.REJECT
        assert any("reason" in r for r in decision.reasons)

    def test_destructive_operation_requires_higher_confidence(self, governor, sample_memory):
        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.REJECT,
                memory_id=sample_memory.id,
                reason="not worth keeping",
                confidence=0.4,  # below min_confidence_destructive (0.6)
            )
        )
        assert decision.verdict == PolicyVerdict.REJECT
        assert any("below required" in r for r in decision.reasons)


    def test_archived_target_requires_restore_first(self, governor, store, sample_memory):
        sample_memory.status = MemoryStatus.ARCHIVED
        sample_memory.is_current = False
        store.update_memory(sample_memory)

        blocked = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.REINFORCE,
                memory_id=sample_memory.id,
                reason="remembered again",
            )
        )
        assert blocked.verdict == PolicyVerdict.REJECT
        assert any("archived" in r for r in blocked.reasons)

        allowed = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.RESTORE,
                memory_id=sample_memory.id,
                reason="needed again",
            )
        )
        assert allowed.verdict == PolicyVerdict.APPROVE
        restored = store.get_memory(sample_memory.id)
        assert restored.status == MemoryStatus.ACTIVE

    def test_reinterpret_requires_interpretation(self, governor, sample_memory):
        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.REINTERPRET,
                memory_id=sample_memory.id,
                reason="new angle",
                evidence="",
                interpretation="",
            )
        )
        assert decision.verdict == PolicyVerdict.REJECT

    def test_approved_reinterpret_creates_new_memory(self, governor, store, sample_memory):
        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.REINTERPRET,
                memory_id=sample_memory.id,
                reason="Generalize tool preference into experience",
                evidence="User prefers Rust for CLI tooling.",
                interpretation="User has systems-programming experience.",
                confidence=0.7,
            )
        )
        assert decision.approved
        assert decision.resulting_memory_id not in (None, sample_memory.id)
        new_memory = store.get_memory(decision.resulting_memory_id)
        assert new_memory is not None
        assert new_memory.content == "User has systems-programming experience."


    def test_keep_always_allowed_for_active_memory(self, governor, sample_memory):
        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.KEEP,
                memory_id=sample_memory.id,
                reason="explicitly retained",
            )
        )
        assert decision.approved

    def test_contradict_without_parameters_rejected(self, governor, sample_memory):
        # Since Phase 2 CONTRADICT is dispatchable, but requires
        # other_memory_id in metadata.
        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.CONTRADICT,
                memory_id=sample_memory.id,
                reason="conflict detected",
            )
        )
        assert decision.verdict == PolicyVerdict.REJECT
        assert any("other_memory_id" in r for r in decision.reasons)

    def test_out_of_range_confidence_rejected(self, governor, sample_memory):
        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.REINFORCE,
                memory_id=sample_memory.id,
                reason="check bounds",
                confidence=1.5,
            )
        )
        assert decision.verdict == PolicyVerdict.REJECT
        assert any("outside [0, 1]" in r for r in decision.reasons)

    def test_cooldown_blocks_rapid_mutations(self, engine, sample_memory):
        governor = EvolutionGovernor(engine, EvolutionPolicyConfig(cooldown_seconds=3600))
        first = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.REINFORCE,
                memory_id=sample_memory.id,
                reason="first",
            )
        )
        assert first.approved
        second = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.REINFORCE,
                memory_id=sample_memory.id,
                reason="too soon",
            )
        )
        assert second.verdict == PolicyVerdict.REJECT
        assert any("Cooldown" in r for r in second.reasons)






