"""Tests for AM v0.2.0 Phase 2: Temporal and Contradiction Layer.

Covers:
* Temporal validity (invalidation, supersession bounds, state-at-time queries)
* Supersession chains (explicit SUPERSEDES edges, chain walking)
* Contradiction edges (explicit CONTRADICTS edges, temporal resolution)
* Lifecycle state transitions (validated transition graph)
* Provenance chains and memory.explain()
* Governor dispatch for CONTRADICT / SUPERSEDE / TRANSITION
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from artificial_memory.compression.compressor import RuleBasedCompressor
from artificial_memory.core.models import (
    AssociationType,
    Conversation,
    Memory,
    MemoryStatus,
    MemoryType,
    Project,
    Topic,
)
from artificial_memory.memory.contradiction_edges import ContradictionEdgeManager
from artificial_memory.memory.evolution import (
    EvolutionOperationType,
    MemoryEvolutionEngine,
)
from artificial_memory.memory.evolution_policy import (
    EvolutionGovernor,
    EvolutionProposal,
    PolicyVerdict,
)
from artificial_memory.memory.lifecycle_transitions import (
    InvalidTransitionError,
    LifecycleTransitionEngine,
)
from artificial_memory.memory.provenance import ProvenanceChainBuilder
from artificial_memory.memory.supersession import SupersessionManager
from artificial_memory.memory.temporal_validity import TemporalValidityManager
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
def engine(store):
    return MemoryEvolutionEngine(store, RuleBasedCompressor())


@pytest.fixture
def governor(engine):
    return EvolutionGovernor(engine)


@pytest.fixture
def topic_id(store):
    project = store.create_project(Project(name="v02-phase2"))
    topic = store.create_topic(
        Topic(project_id=project.id, name="Test", path="Projects/v02-phase2/Test")
    )
    return topic.id


@pytest.fixture
def conversation_id(store, topic_id):
    conv = store.create_conversation(Conversation(topic_id=topic_id, title="Design chat"))
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


# ==================== Temporal validity ====================


class TestTemporalValidity:
    def test_invalidate_sets_valid_until_and_logs_event(self, store, make_memory):
        memory = make_memory("User prefers Python.")
        manager = TemporalValidityManager(store)
        when = datetime.now()

        result = manager.invalidate(memory.id, when, reason="preference changed")
        assert result.valid_until is not None
        assert abs((result.valid_until - when).total_seconds()) < 1

        events = store.get_evolution_events(memory_id=memory.id)
        event = next(e for e in events if e.operation == "transition")
        assert event.metadata["kind"] == "temporal_invalidation"

    def test_invalidate_is_monotonic(self, store, make_memory):
        memory = make_memory("User prefers Python.")
        manager = TemporalValidityManager(store)
        early = datetime.now() - timedelta(days=1)
        late = datetime.now()

        manager.invalidate(memory.id, late)
        result = manager.invalidate(memory.id, early)  # earlier must win
        assert result.valid_until is not None
        assert result.valid_until <= early

    def test_is_valid_at_and_filter(self, store, make_memory):
        now = datetime.now()
        m1 = make_memory("Active preference.")
        m2 = make_memory("Expired preference.", valid_from=now - timedelta(days=10),
                         valid_until=now - timedelta(days=1))
        manager = TemporalValidityManager(store)

        assert manager.is_valid_at(m1, now) is True
        assert manager.is_valid_at(m2, now) is False
        assert manager.is_valid_at(m2, now - timedelta(days=5)) is True

        valid = manager.filter_valid_at([m1, m2], now)
        assert [m.id for m in valid] == [m1.id]

    def test_apply_supersession_bounds_sets_both_windows(self, store, make_memory):
        old = make_memory("User prefers Python.")
        new = make_memory("User prefers Rust.")
        manager = TemporalValidityManager(store)
        transition = datetime.now()

        result_old, result_new = manager.apply_supersession_bounds(
            old.id, new.id, transition
        )
        assert result_old.valid_until is not None
        assert result_new.valid_from is not None
        assert result_old.valid_until <= transition
        assert result_new.valid_from >= transition - timedelta(seconds=1)

    def test_apply_supersession_bounds_is_idempotent(self, store, make_memory):
        old = make_memory("User prefers Python.")
        new = make_memory("User prefers Rust.")
        manager = TemporalValidityManager(store)
        transition = datetime.now()

        manager.apply_supersession_bounds(old.id, new.id, transition)
        before = store.get_memory(old.id)
        manager.apply_supersession_bounds(old.id, new.id, transition)
        after = store.get_memory(old.id)
        assert before.valid_until == after.valid_until

    def test_operations_reject_unknown_memory(self, store):
        manager = TemporalValidityManager(store)
        with pytest.raises(ValueError):
            manager.invalidate(99999, datetime.now())
        with pytest.raises(ValueError):
            manager.apply_supersession_bounds(99999, 99998, datetime.now())


# ==================== Supersession ====================


class TestSupersession:
    def test_supersede_archives_old_and_creates_new(self, store, make_memory):
        old = make_memory("User prefers Python.", confidence=0.9)
        manager = SupersessionManager(store)

        result_old, result_new = manager.supersede_memory(
            old.id, "User prefers Rust.", reason="stated in new chat"
        )
        assert result_old.status == MemoryStatus.ARCHIVED
        assert result_old.is_current is False
        assert result_new.status == MemoryStatus.ACTIVE
        assert result_new.content == "User prefers Rust."

        # Evidence preserved (never overwritten destructively)
        assert store.get_memory(old.id) is not None
        assert store.get_memory(old.id).content == "User prefers Python."

    def test_supersede_links_edge_and_applies_temporal_bounds(self, store, make_memory):
        old = make_memory("User prefers Python.")
        manager = SupersessionManager(store)

        _old, new = manager.supersede_memory(old.id, "User prefers Rust.")

        associations = store.get_associations(old.id, AssociationType.SUPERSEDES)
        assert any(a.source_memory_id == new.id and a.target_memory_id == old.id for a in associations)

        persisted_old = store.get_memory(old.id)
        persisted_new = store.get_memory(new.id)
        assert persisted_old.valid_until is not None
        assert persisted_new.valid_from is not None

    def test_supersede_logs_audit_event(self, store, make_memory):
        old = make_memory("User prefers Python.")
        manager = SupersessionManager(store)
        _old, new = manager.supersede_memory(old.id, "User prefers Rust.")

        events = store.get_evolution_events(memory_id=old.id)
        event = next(e for e in events if e.operation == "supersede")
        assert event.target_memory_id == new.id

    def test_chain_walking_both_directions(self, store, make_memory):
        m1 = make_memory("Prefers Python (2024).")
        manager = SupersessionManager(store)
        _o1, m2 = manager.supersede_memory(m1.id, "Prefers Rust (2025).")
        _o2, m3 = manager.supersede_memory(m2.id, "Prefers Go (2026).")

        forward = manager.get_supersession_chain(m1.id, direction="forward")
        assert [m.id for m in forward] == [m2.id, m3.id]

        backward = manager.get_supersession_chain(m3.id, direction="backward")
        assert [m.id for m in backward] == [m2.id, m1.id]

        head = manager.get_current_head(m1.id)
        assert head.id == m3.id

    def test_empty_content_rejected(self, store, make_memory):
        memory = make_memory("User prefers Python.")
        manager = SupersessionManager(store)
        with pytest.raises(ValueError):
            manager.supersede_memory(memory.id, "   ")


# ==================== Contradiction edges ====================


class TestContradictionEdges:
    def test_create_edge_logs_events_on_both_sides(self, store, make_memory):
        a = make_memory("User prefers Python.")
        b = make_memory("User prefers Rust.")
        manager = ContradictionEdgeManager(store)

        association = manager.create_edge(a.id, b.id, severity=0.9)
        assert association.association_type == AssociationType.CONTRADICTS
        assert association.strength == pytest.approx(0.9)

        # Both directions are queryable (store scans source AND target)
        assert len(manager.get_edges(a.id)) == 1
        assert len(manager.get_edges(b.id)) == 1

        for memory_id in (a.id, b.id):
            events = store.get_evolution_events(memory_id=memory_id)
            assert any(e.operation == "contradict" for e in events)

    def test_self_contradiction_rejected(self, store, make_memory):
        memory = make_memory("User prefers Python.")
        manager = ContradictionEdgeManager(store)
        with pytest.raises(ValueError):
            manager.create_edge(memory.id, memory.id)

    def test_severity_clamped(self, store, make_memory):
        a = make_memory("Fact A.")
        b = make_memory("Fact B.")
        manager = ContradictionEdgeManager(store)
        association = manager.create_edge(a.id, b.id, severity=5.0)
        assert association.strength == 1.0

    def test_apply_temporal_resolution_bounds_older_and_newer(self, store, make_memory):
        older = make_memory("User prefers Python.")
        newer = make_memory("User prefers Rust.")
        manager = ContradictionEdgeManager(store)
        manager.create_edge(older.id, newer.id)
        transition = datetime.now()

        result_older, result_newer = manager.apply_temporal_resolution(
            newer.id, older.id, transition  # argument order should not matter
        )
        assert result_older.id == older.id
        assert result_older.valid_until is not None
        assert result_newer.id == newer.id
        assert result_newer.valid_from is not None

    def test_unresolved_count_drops_after_resolution(self, store, make_memory):
        a = make_memory("User prefers Python.")
        b = make_memory("User prefers Rust.")
        manager = ContradictionEdgeManager(store)
        manager.create_edge(a.id, b.id)
        assert manager.unresolved_count(a.id) == 1

        # Temporal resolution separates the validity windows: the
        # contradiction is resolved without deleting either record.
        manager.apply_temporal_resolution(a.id, b.id, datetime.now())
        assert manager.unresolved_count(a.id) == 0
        assert store.get_memory(a.id) is not None
        assert store.get_memory(b.id) is not None


# ==================== Lifecycle transitions ====================


class TestLifecycleTransitions:
    def test_valid_transition_applies_and_logs(self, store, make_memory):
        memory = make_memory("Some fact.")
        engine = LifecycleTransitionEngine(store)

        result = engine.transition(memory.id, MemoryStatus.DORMANT, reason="inactive")
        assert result.status == MemoryStatus.DORMANT
        assert result.is_current is False

        events = store.get_evolution_events(memory_id=memory.id)
        event = next(e for e in events if e.operation == "transition")
        assert event.metadata["from_status"] == "active"
        assert event.metadata["to_status"] == "dormant"

    def test_invalid_transition_raises(self, store, make_memory):
        memory = make_memory("Some fact.")
        engine = LifecycleTransitionEngine(store)
        with pytest.raises(InvalidTransitionError):
            engine.transition(memory.id, MemoryStatus.DEEP_ARCHIVED)

    def test_archived_to_active_and_deep_archived(self, store, make_memory):
        memory = make_memory("Some fact.")
        engine = LifecycleTransitionEngine(store)

        engine.transition(memory.id, MemoryStatus.ARCHIVED)
        result = engine.transition(memory.id, MemoryStatus.DEEP_ARCHIVED, reason="rarely used")
        assert result.status == MemoryStatus.DEEP_ARCHIVED
        assert result.is_current is False

        revived = engine.transition(memory.id, MemoryStatus.ACTIVE, reason="needed again")
        assert revived.status == MemoryStatus.ACTIVE
        assert revived.is_current is True

    def test_noop_transition_returns_unchanged(self, store, make_memory):
        memory = make_memory("Some fact.")
        engine = LifecycleTransitionEngine(store)
        result = engine.transition(memory.id, MemoryStatus.ACTIVE)
        assert result.status == MemoryStatus.ACTIVE

    def test_transition_history(self, store, make_memory):
        memory = make_memory("Some fact.")
        engine = LifecycleTransitionEngine(store)
        engine.transition(memory.id, MemoryStatus.ARCHIVED)
        engine.transition(memory.id, MemoryStatus.DEEP_ARCHIVED)

        history = engine.transition_history(memory.id)
        assert [(h["from_status"], h["to_status"]) for h in history] == [
            ("active", "archived"),
            ("archived", "deep_archived"),
        ]

    def test_unknown_memory_raises(self, store):
        engine = LifecycleTransitionEngine(store)
        with pytest.raises(ValueError):
            engine.transition(99999, MemoryStatus.ARCHIVED)


# ==================== Governor dispatch (Phase 2 ops) ====================


class TestGovernorPhase2:
    def test_contradict_dispatch_creates_edge(self, governor, store, make_memory):
        a = make_memory("User prefers Python.")
        b = make_memory("User prefers Rust.")

        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.CONTRADICT,
                memory_id=a.id,
                reason="conflicting preferences observed",
                confidence=0.9,
                metadata={"other_memory_id": b.id, "severity": 0.9},
            )
        )
        assert decision.approved
        assert decision.committed is True
        assert len(store.get_associations(a.id, AssociationType.CONTRADICTS)) == 1

    def test_contradict_missing_parameter_rejected(self, governor, make_memory):
        a = make_memory("User prefers Python.")
        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.CONTRADICT,
                memory_id=a.id,
                reason="conflict",
                metadata={},
            )
        )
        assert decision.verdict == PolicyVerdict.REJECT
        assert any("other_memory_id" in r for r in decision.reasons)

    def test_contradict_self_rejected(self, governor, make_memory):
        a = make_memory("User prefers Python.")
        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.CONTRADICT,
                memory_id=a.id,
                reason="self conflict",
                metadata={"other_memory_id": a.id},
            )
        )
        assert decision.verdict == PolicyVerdict.REJECT

    def test_supersede_dispatch_preserves_history(self, governor, store, make_memory):
        old = make_memory("User prefers Python.")

        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.SUPERSEDE,
                memory_id=old.id,
                reason="newer explicit preference",
                evidence="User prefers Rust.",
                confidence=0.9,
            )
        )
        assert decision.approved
        assert decision.resulting_memory_id not in (None, old.id)

        superseded = store.get_memory(old.id)
        assert superseded.status == MemoryStatus.ARCHIVED
        assert superseded.content == "User prefers Python."  # evidence preserved
        replacement = store.get_memory(decision.resulting_memory_id)
        assert replacement.content == "User prefers Rust."

    def test_supersede_requires_evidence(self, governor, make_memory):
        old = make_memory("User prefers Python.")
        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.SUPERSEDE,
                memory_id=old.id,
                reason="replacement",
                evidence="",
                confidence=0.9,
            )
        )
        assert decision.verdict == PolicyVerdict.REJECT

    def test_transition_dispatch(self, governor, store, make_memory):
        memory = make_memory("Some fact.")
        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.TRANSITION,
                memory_id=memory.id,
                reason="no longer active",
                metadata={"to_status": "archived"},
            )
        )
        assert decision.approved
        assert store.get_memory(memory.id).status == MemoryStatus.ARCHIVED

    def test_transition_invalid_status_rejected_in_validation(self, governor, make_memory):
        memory = make_memory("Some fact.")
        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.TRANSITION,
                memory_id=memory.id,
                reason="bad status",
                metadata={"to_status": "deleted"},
            )
        )
        assert decision.verdict == PolicyVerdict.REJECT

    def test_transition_on_archived_memory_is_archive_safe(
        self, governor, store, make_memory
    ):
        memory = make_memory("Some fact.")
        memory.status = MemoryStatus.ARCHIVED
        store.update_memory(memory)

        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.TRANSITION,
                memory_id=memory.id,
                reason="deepen the archive",
                metadata={"to_status": "deep_archived"},
            )
        )
        assert decision.approved
        assert store.get_memory(memory.id).status == MemoryStatus.DEEP_ARCHIVED


# ==================== Provenance chains ====================


class TestProvenance:
    def test_chain_walks_evidence_lineage(self, store, make_memory):
        from artificial_memory.core.models import Association

        evidence = make_memory("I used Rust for a small project.")
        builder = ProvenanceChainBuilder(store)

        # Interpretation derived from evidence (ELABORATES: evidence -> interpretation)
        interpretation = make_memory("User is becoming a regular Rust developer.")
        store.create_association(Association(
            source_memory_id=evidence.id,
            target_memory_id=interpretation.id,
            association_type=AssociationType.ELABORATES,
            strength=0.8,
        ))

        chain = builder.build_chain(interpretation.id)
        assert [n.memory_id for n in chain.nodes] == [interpretation.id, evidence.id]
        assert chain.nodes[0].relationship == "self"
        assert chain.nodes[1].relationship == "elaborates"
        assert chain.nodes[1].depth == 1

    def test_chain_includes_conversation_provenance(self, store, make_memory):
        memory = make_memory("User prefers Rust.")
        builder = ProvenanceChainBuilder(store)
        chain = builder.build_chain(memory.id)

        node = chain.nodes[0]
        assert node.source_conversation_id is not None
        assert node.conversation_title == "Design chat"

    def test_chain_walks_supersession_lineage(self, store, make_memory):
        manager = SupersessionManager(store)
        original = make_memory("Prefers Python (2024).")
        _old, replacement = manager.supersede_memory(original.id, "Prefers Rust (2025).")

        builder = ProvenanceChainBuilder(store)
        chain = builder.build_chain(replacement.id)
        assert [n.memory_id for n in chain.nodes] == [replacement.id, original.id]
        assert chain.nodes[1].relationship == "supersedes"

    def test_explain_combines_all_layers(self, store, make_memory):
        a = make_memory("User prefers Python.")
        b = make_memory("User prefers Rust.")
        edge_manager = ContradictionEdgeManager(store)
        edge_manager.create_edge(a.id, b.id)

        builder = ProvenanceChainBuilder(store)
        report = builder.explain(a.id)

        assert report["memory"]["id"] == a.id
        assert report["validity"]["valid_from"] is None or isinstance(
            report["validity"]["valid_from"], str
        )
        assert len(report["provenance"]["nodes"]) >= 1
        assert report["contradictions"][0]["other_memory_id"] == b.id
        assert any(e["operation"] == "contradict" for e in report["audit_history"])

    def test_explain_unknown_memory_raises(self, store):
        builder = ProvenanceChainBuilder(store)
        with pytest.raises(ValueError):
            builder.explain(99999)






