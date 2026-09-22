"""Unit tests for Streaming Dialogue Runner & Conflict State Management (Phase 5)."""

import pytest

from artificial_memory.core.ir.conflict_state import (
    ConflictResolutionStatus,
    ConflictSeverity,
    ConflictStateManager,
)
from artificial_memory.core.ir.structured import IRRelation, IRStatus, StructuredIR
from artificial_memory.core.models import ResolutionLevel
from artificial_memory.research.benchmarks.streaming_runner import (
    StreamingDialogueRunner,
    StreamingTurn,
)


def test_conflict_detection_and_formatting() -> None:
    """Test explicit conflict detection and context formatting."""
    mgr = ConflictStateManager()

    ir1 = StructuredIR(
        entity="Production Database",
        property="engine",
        value="PostgreSQL",
        source="Bob",
        relation=IRRelation.ASSERTS,
        status=IRStatus.ACTIVE,
        raw_content="Bob: We decided on PostgreSQL.",
    )
    ir2 = StructuredIR(
        entity="Production Database",
        property="engine",
        value="Redis",
        source="Alice",
        relation=IRRelation.ASSERTS,
        status=IRStatus.ACTIVE,
        raw_content="Alice: I strongly suggest Redis.",
    )

    conflicts = mgr.detect_and_register([ir1, ir2])
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.entity == "Production Database"
    assert c.property == "engine"
    assert c.status == ConflictResolutionStatus.CONFLICTED
    assert c.severity == ConflictSeverity.HIGH

    # Verify context formatting includes explicit uncertainty directives
    context_str = c.format_for_context_ir()
    assert "[STATUS]: CONFLICTED" in context_str
    assert "Redis" in context_str
    assert "PostgreSQL" in context_str
    assert "[SYSTEM DIRECTIVE]" in context_str

    # Test resolution
    resolved = mgr.resolve_conflict("Production Database", "engine", "PostgreSQL")
    assert resolved is True
    assert c.status == ConflictResolutionStatus.RESOLVED
    assert c.resolved_value == "PostgreSQL"


def test_streaming_dialogue_evolution() -> None:
    """Test multi-turn streaming dialogue with real-time memory evolution."""
    runner = StreamingDialogueRunner()

    turns = [
        StreamingTurn(
            turn_id=1,
            speaker="Alice",
            utterance="Alice: We are starting Project Orion. The team lead is Sarah.",
        ),
        StreamingTurn(
            turn_id=2,
            speaker="Bob",
            utterance="Bob: For Project Orion, our backend framework is FastAPI.",
            probe_query="What is the backend framework for Project Orion?",
            expected_answer="FastAPI",
        ),
        StreamingTurn(
            turn_id=3,
            speaker="Sarah",
            utterance="Sarah: Actually, let's migrate Project Orion backend framework to Go Gin.",
            probe_query="What did we migrate Project Orion backend framework to?",
            expected_answer="Go Gin",
        ),
        StreamingTurn(
            turn_id=4,
            speaker="Alice",
            utterance="Alice: Sounds great, Go Gin will give us high throughput.",
        ),
        StreamingTurn(
            turn_id=5,
            speaker="Bob",
            utterance="Bob: Also, remember our deployment target is AWS ECS.",
        ),
        StreamingTurn(
            turn_id=6,
            speaker="Sarah",
            utterance="Sarah: Let's review the framework again. Go Gin is working solidly.",
            probe_query="What is the current backend framework for Project Orion?",
            expected_answer="Go Gin",
        ),
    ]

    report = runner.run_session(turns)

    assert report.total_turns == 6
    assert report.probes_evaluated == 3
    assert report.probe_accuracy == 1.0
    assert report.final_memories_count >= 3

    # Check evolution record
    orion_records = [rec for k, rec in runner.lifecycle_records.items() if "orion" in k]
    assert len(orion_records) >= 1
    # Check that Go Gin was tracked as current value
    fw_rec = next((r for r in orion_records if "gin" in r.current_value.lower() or "fastapi" in r.initial_value.lower()), None)
    assert fw_rec is not None
    assert fw_rec.current_value == "Go Gin"
    assert fw_rec.access_count >= 1
