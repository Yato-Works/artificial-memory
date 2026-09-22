"""Streaming Memory Dialogue Runtime & Lifecycle Benchmark.

Phase 5-A: Evaluates memory evolution, progressive resolution decay,
and real-time consistency over multi-turn continuous conversations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any

from artificial_memory.compiler.ir_extractor import UniversalIRExtractor
from artificial_memory.core.ir.conflict_state import ConflictStateManager
from artificial_memory.core.ir.structured import StructuredIR
from artificial_memory.core.models import ResolutionLevel
from artificial_memory.recall.ir_resolver import UniversalIRResolver


@dataclass
class StreamingTurn:
    """A single turn in a continuous multi-turn dialogue."""
    turn_id: int
    speaker: str
    utterance: str
    probe_query: str | None = None
    expected_answer: str | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryEvolutionRecord:
    """Tracks how a memory unit evolves over dialogue turns."""
    memory_id: str
    entity: str
    property: str
    initial_value: str
    current_value: str
    initial_turn: int
    last_accessed_turn: int
    resolution: ResolutionLevel = ResolutionLevel.RAW
    access_count: int = 1
    decay_score: float = 1.0


@dataclass
class StreamingSessionReport:
    """Summary report of a streaming dialogue benchmark session."""
    total_turns: int
    probes_evaluated: int
    probe_accuracy: float
    conflicts_detected: int
    conflicts_resolved: int
    final_memories_count: int
    resolution_distribution: dict[str, int]
    turn_logs: list[dict[str, Any]] = field(default_factory=list)


class StreamingDialogueRunner:
    """Executes a multi-turn streaming dialogue, evolving memory in real-time."""

    def __init__(self) -> None:
        self.extractor = UniversalIRExtractor()
        self.resolver = UniversalIRResolver()
        self.conflict_mgr = ConflictStateManager()
        self.active_memories: list[StructuredIR] = []
        self.lifecycle_records: dict[str, MemoryEvolutionRecord] = {}
        self.current_turn: int = 0

    def process_turn(self, turn: StreamingTurn) -> dict[str, Any]:
        """Ingest a single dialogue turn, update memory state, and test probe if present."""
        self.current_turn = turn.turn_id
        timestamp_str = turn.timestamp or (datetime(2026, 1, 1) + timedelta(hours=turn.turn_id)).isoformat()

        # 1. Deterministic IR extraction from turn utterance
        extracted = self.extractor.extract(
            text=turn.utterance,
            default_source=turn.speaker,
        )

        for ir in extracted:
            self.active_memories.append(ir)
            mem_key = f"{ir.entity.lower()}::{ir.property.lower()}"
            if mem_key not in self.lifecycle_records:
                self.lifecycle_records[mem_key] = MemoryEvolutionRecord(
                    memory_id=mem_key,
                    entity=ir.entity,
                    property=ir.property,
                    initial_value=ir.value,
                    current_value=ir.value,
                    initial_turn=self.current_turn,
                    last_accessed_turn=self.current_turn,
                    resolution=ResolutionLevel.RAW,
                )
            else:
                rec = self.lifecycle_records[mem_key]
                rec.current_value = ir.value
                rec.last_accessed_turn = self.current_turn
                rec.access_count += 1

        # 2. Update conflict states
        new_conflicts = self.conflict_mgr.detect_and_register(self.active_memories)

        # 3. Simulate progressive decay / consolidation every 5 turns
        if self.current_turn > 0 and self.current_turn % 5 == 0:
            self._apply_progressive_decay()

        # 4. Evaluate probe query if present
        probe_result: dict[str, Any] | None = None
        if turn.probe_query:
            resolved = self.resolver.resolve(
                query=turn.probe_query,
                ir_records=self.active_memories,
            )
            context_ir = resolved.context_text
            is_correct = False
            if turn.expected_answer:
                is_correct = turn.expected_answer.lower() in context_ir.lower()

            probe_result = {
                "query": turn.probe_query,
                "expected": turn.expected_answer,
                "context_ir": context_ir,
                "is_correct": is_correct,
            }

        return {
            "turn_id": self.current_turn,
            "extracted_count": len(extracted),
            "new_conflicts": len(new_conflicts),
            "probe_result": probe_result,
        }

    def _apply_progressive_decay(self) -> None:
        """Simulate human-like progressive resolution decay based on recency and access count."""
        for rec in self.lifecycle_records.values():
            age_turns = self.current_turn - rec.last_accessed_turn
            # High access count promotes to higher abstract resolution (SEMANTIC / LONG_TERM)
            # Long inactivity demotes or compresses
            if rec.access_count >= 3:
                rec.resolution = ResolutionLevel.SEMANTIC
            elif age_turns > 10:
                rec.resolution = ResolutionLevel.LIGHT
            elif age_turns > 20:
                rec.resolution = ResolutionLevel.EPISODE

    def run_session(self, turns: list[StreamingTurn]) -> StreamingSessionReport:
        """Run an entire streaming dialogue session."""
        turn_logs = []
        probes_evaluated = 0
        probes_correct = 0

        for turn in turns:
            log = self.process_turn(turn)
            turn_logs.append(log)
            if log["probe_result"]:
                probes_evaluated += 1
                if log["probe_result"]["is_correct"]:
                    probes_correct += 1

        res_dist: dict[str, int] = {}
        for rec in self.lifecycle_records.values():
            name = rec.resolution.name
            res_dist[name] = res_dist.get(name, 0) + 1

        accuracy = (probes_correct / probes_evaluated) if probes_evaluated > 0 else 1.0

        return StreamingSessionReport(
            total_turns=len(turns),
            probes_evaluated=probes_evaluated,
            probe_accuracy=accuracy,
            conflicts_detected=len(self.conflict_mgr._conflicts),
            conflicts_resolved=sum(1 for c in self.conflict_mgr._conflicts.values() if c.status.value == "resolved"),
            final_memories_count=len(self.active_memories),
            resolution_distribution=res_dist,
            turn_logs=turn_logs,
        )
