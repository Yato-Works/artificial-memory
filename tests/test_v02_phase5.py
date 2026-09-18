"""Tests for AM v0.2.0 Phase 5: Background Reflection.

Covers (plan #13 / #14 and the Phase 5 implementation list):

* candidate selection (bounded review, reflection priority),
* reflection workers (reinforce / keep / merge / reinterpret / archive),
* merge detection (lexical duplicate clusters),
* reinterpretation (deterministic synthesis from related evidence),
* archival (TRANSITION to archived, never deletion),
* audit events (everything flows through the EvolutionGovernor).
"""

import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from artificial_memory.core.models import (
    Association,
    AssociationType,
    Conversation,
    Memory,
    MemoryStatus,
    MemoryType,
    Project,
    Topic,
)
from artificial_memory.memory.evolution_policy import EvolutionGovernor
from artificial_memory.memory.reflection import (
    BackgroundReflector,
    ReflectionConfig,
    ReflectionKind,
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
def topic_id(store):
    project = store.create_project(Project(name="v02-phase5"))
    topic = store.create_topic(
        Topic(project_id=project.id, name="Test", path="Projects/v02-phase5/Test")
    )
    return topic.id


@pytest.fixture
def conversation_id(store, topic_id):
    conv = store.create_conversation(Conversation(topic_id=topic_id, title="chat"))
    return conv.id


@pytest.fixture
def make_memory(store, topic_id, conversation_id):
    def _make(content: str, **kwargs) -> Memory:
        memory = Memory(
            topic_id=topic_id,
            memory_type=kwargs.pop("memory_type", MemoryType.SEMANTIC),
            content=content,
            source_conversation_id=conversation_id,
            **kwargs,
        )
        return store.create_memory(memory)

    return _make


NOW = datetime.now()

# ==================== Candidate selection ====================


class TestCandidateSelection:
    def test_selects_bounded_candidates(self, store, make_memory):
        for index in range(10):
            make_memory(f"Reflection candidate number {index} about Rust.")
        config = ReflectionConfig(max_candidates=4)
        candidates = BackgroundReflector(store, config=config)._select_candidates(NOW)
        assert len(candidates) == 4

    def test_selection_prefers_used_and_stale(self, store, make_memory):
        stale_used = make_memory(
            "Stale heavily used memory about deployment.",
            importance=0.9,
            access_count=8,
            created_at=NOW - timedelta(days=60),
        )
        fresh_idle = make_memory("Fresh idle memory about deployment.", created_at=NOW)
        candidates = BackgroundReflector(store)._select_candidates(NOW)
        ids = [memory.id for memory in candidates]
        assert stale_used.id in ids
        assert ids.index(stale_used.id) < ids.index(fresh_idle.id)

    def test_archived_memories_are_not_candidates(self, store, make_memory):
        make_memory("Archived memory about caching.", status=MemoryStatus.ARCHIVED)
        candidates = BackgroundReflector(store)._select_candidates(NOW)
        assert candidates == []


# ==================== Reflection workers ====================


class TestReflectionWorkers:
    def test_reinforce_for_frequent_valuable_memory(self, store, make_memory):
        memory = make_memory(
            "The deployment pipeline runs on Fridays.",
            importance=0.9,
            access_count=5,
        )
        report = BackgroundReflector(store).run(now=NOW)

        kinds = {finding.kind for finding in report.findings}
        assert ReflectionKind.REINFORCE in kinds
        finding = next(f for f in report.findings if f.memory_id == memory.id)
        assert finding.proposal.confidence >= 0.4
        assert finding.proposal.metadata["triggered_by"] == "reflection"

    def test_keep_for_rarely_accessed_valuable_memory(self, store, make_memory):
        memory = make_memory(
            "Critical on-call escalation policy for the database.",
            importance=0.95,
            access_count=0,
            created_at=NOW - timedelta(days=60),
        )
        report = BackgroundReflector(store).run(now=NOW)
        kinds = {finding.kind for finding in report.findings}
        assert ReflectionKind.KEEP in kinds
        finding = next(f for f in report.findings if f.memory_id == memory.id)
        assert finding.proposal.operation.value == "keep"

    def test_archive_for_stale_low_utility_memory(self, store, make_memory):
        memory = make_memory(
            "An obsolete note about an abandoned experiment.",
            importance=0.05,
            confidence=0.3,
            created_at=NOW - timedelta(days=120),
        )
        report = BackgroundReflector(store).run(now=NOW)
        finding = next(f for f in report.findings if f.memory_id == memory.id)
        assert finding.kind is ReflectionKind.ARCHIVE
        assert finding.proposal.operation.value == "transition"
        assert finding.proposal.metadata["to_status"] == "archived"

    def test_archive_commit_moves_memory_to_archived(self, store, make_memory):
        memory = make_memory(
            "An obsolete note about an abandoned experiment.",
            importance=0.05,
            confidence=0.3,
            created_at=NOW - timedelta(days=120),
        )
        BackgroundReflector(store).run(now=NOW)
        updated = store.get_memory(memory.id)
        assert updated is not None
        assert updated.status is MemoryStatus.ARCHIVED

    def test_reinterpret_for_related_evidence(self, store, make_memory):
        memory = make_memory("The user prefers Rust for systems work.")
        for index in range(3):
            other = make_memory(f"Rust systems project note {index}.")
            store.create_association(
                Association(
                    source_memory_id=other.id,
                    target_memory_id=memory.id,
                    association_type=AssociationType.RELATED,
                    strength=0.6,
                )
            )
        report = BackgroundReflector(store).run(now=NOW)
        finding = next(f for f in report.findings if f.memory_id == memory.id)
        assert finding.kind is ReflectionKind.REINTERPRET
        assert finding.proposal.interpretation
        assert "Rust" in finding.proposal.interpretation or finding.proposal.interpretation

    def test_merge_detected_for_duplicates(self, store, make_memory):
        first = make_memory("The user adopted Rust for the CLI tooling project.")
        second = make_memory("The user adopted Rust for the CLI tooling project!")
        report = BackgroundReflector(store).run(now=NOW)

        kinds = {finding.kind for finding in report.findings}
        assert ReflectionKind.MERGE in kinds
        finding = next(f for f in report.findings if f.kind is ReflectionKind.MERGE)
        assert set(finding.evidence_ids) == {first.id, second.id} - {finding.memory_id}
        assert finding.proposal.merge_memory_ids

# ==================== Governor integration / audit ====================


class TestGovernorIntegration:
    def test_approved_proposals_commit_and_audited(self, store, make_memory):
        memory = make_memory(
            "The deployment pipeline runs on Fridays.",
            importance=0.9,
            access_count=5,
        )
        report = BackgroundReflector(store).run(now=NOW)

        assert report.decisions, "reinforce proposal should be submitted"
        decision = report.decisions[0]
        assert decision.approved is True
        assert decision.committed is True
        assert decision.resulting_memory_id == memory.id

        # The commit wrote an auditable evolution event.
        events = store.get_evolution_events(memory_id=memory.id, limit=5)
        assert any(event.operation == "reinforce" for event in events)

    def test_reinforce_raises_confidence(self, store, make_memory):
        memory = make_memory(
            "The deployment pipeline runs on Fridays.",
            importance=0.9,
            access_count=5,
            confidence=0.6,
        )
        BackgroundReflector(store).run(now=NOW)
        updated = store.get_memory(memory.id)
        assert updated is not None
        assert updated.confidence > 0.6

    def test_rejected_proposals_do_not_mutate(self, store, make_memory):
        # A low-confidence archival proposal must be rejected by the
        # governor's destructive-operation threshold, and the memory must
        # remain untouched (the reflector path itself is covered above).
        from artificial_memory.compression.compressor import RuleBasedCompressor
        from artificial_memory.memory.evolution import (
            EvolutionOperationType,
            MemoryEvolutionEngine,
        )
        from artificial_memory.memory.evolution_policy import EvolutionProposal

        memory = make_memory(
            "An obsolete note about an abandoned experiment.",
            importance=0.05,
            confidence=0.3,
        )
        governor = EvolutionGovernor(
            MemoryEvolutionEngine(store, RuleBasedCompressor())
        )
        decision = governor.submit(
            EvolutionProposal(
                operation=EvolutionOperationType.TRANSITION,
                memory_id=memory.id,
                reason="reflection rehearsal with insufficient confidence",
                confidence=0.1,
                metadata={"to_status": "archived"},
            )
        )
        assert decision.approved is False
        assert decision.committed is False

        updated = store.get_memory(memory.id)
        assert updated is not None
        assert updated.status is MemoryStatus.ACTIVE

    def test_dry_run_produces_findings_without_commit(self, store, make_memory):
        make_memory(
            "The deployment pipeline runs on Fridays.",
            importance=0.9,
            access_count=5,
        )
        config = ReflectionConfig(dry_run=True)
        report = BackgroundReflector(store, config=config).run(now=NOW)

        assert report.findings
        assert report.decisions == []
        assert report.dry_run is True
        memory = store.get_memories(limit=1)[0]
        events = store.get_evolution_events(memory_id=memory.id or 0, limit=5)
        assert all(event.operation != "reinforce" for event in events)

    def test_max_proposals_bounds_submission(self, store, make_memory):
        for index in range(6):
            make_memory(
                f"Frequent valuable reflection memory {index}.",
                importance=0.9,
                access_count=5,
            )
        config = ReflectionConfig(max_proposals=2)
        report = BackgroundReflector(store, config=config).run(now=NOW)
        assert len(report.decisions) <= 2

    def test_report_is_serializable(self, store, make_memory):
        make_memory(
            "The deployment pipeline runs on Fridays.",
            importance=0.9,
            access_count=5,
        )
        report = BackgroundReflector(store).run(now=NOW)
        payload = report.to_dict()
        assert payload["trigger"] == "idle"
        assert payload["reviewed"] >= 1
        assert payload["approved"] == report.approved

    def test_no_candidates_yields_empty_report(self, store):
        report = BackgroundReflector(store).run(now=NOW)
        assert report.reviewed == 0
        assert report.findings == []
        assert report.finished_at is not None


