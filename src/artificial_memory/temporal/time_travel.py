from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import (
    Decision,
    Memory,
    MemoryStatus,
    MemoryType,
    ResolutionLevel,
)
from artificial_memory.memory.belief import Belief, BeliefEngine


class TemporalQueryType(StrEnum):
    STATE_AT = "state_at"           # What was the memory state at time T?
    BELIEFS_AT = "beliefs_at"       # What were the beliefs at time T?
    CONTEXT_AT = "context_at"       # What context would have been built at time T?
    CHANGES_BETWEEN = "changes_between"  # What changed between T1 and T2?
    TIMELINE = "timeline"           # Full timeline of changes


@dataclass
class TemporalState:
    """Represents the state of memory at a specific timestamp."""
    timestamp: datetime
    active_memories: list[Memory]
    active_decisions: list[Decision]
    memory_count: int
    topic_id: int | None = None


@dataclass
class TemporalChange:
    """Represents a change between two timestamps."""
    timestamp: datetime
    change_type: str  # "added", "removed", "modified", "status_changed"
    memory_id: int
    memory_type: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalQuery:
    """A temporal query request."""
    query_type: TemporalQueryType
    timestamp: datetime | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    topic_id: int | None = None
    include_archived: bool = False


@dataclass
class TimeTravelResult:
    """Result of a time travel query."""
    query: TemporalQuery
    state: TemporalState | None = None
    beliefs: list[Belief] = field(default_factory=list)
    context: str | None = None
    changes: list[TemporalChange] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)


class TimeTravelEngine:
    """Enables time travel queries on the memory system.

    Can answer:
    - What did the system know at time T?
    - What did the system believe at time T?
    - What context would have been constructed at time T?
    - What changed between T1 and T2?
    """

    def __init__(
        self,
        store: MemoryStore,
        belief_engine: BeliefEngine,
        recall_engine=None,
        context_builder=None,
    ):
        self.store = store
        self.belief_engine = belief_engine
        self.recall_engine = recall_engine
        self.context_builder = context_builder

    def travel_to(self, query: TemporalQuery) -> TimeTravelResult:
        """Execute a time travel query."""
        result = TimeTravelResult(query=query)

        if query.query_type == TemporalQueryType.STATE_AT:
            result.state = self._get_state_at(query.timestamp, query.topic_id, query.include_archived)

        elif query.query_type == TemporalQueryType.BELIEFS_AT:
            result.beliefs = self.belief_engine.get_belief_at(query.timestamp)

        elif query.query_type == TemporalQueryType.CONTEXT_AT:
            if self.context_builder and query.timestamp:
                result.context = self._reconstruct_context_at(
                    query.timestamp, query.topic_id
                )

        elif query.query_type == TemporalQueryType.CHANGES_BETWEEN:
            if query.start_time and query.end_time:
                result.changes = self._get_changes_between(
                    query.start_time, query.end_time, query.topic_id
                )

        elif query.query_type == TemporalQueryType.TIMELINE:
            result.timeline = self._get_timeline(
                query.topic_id, query.start_time, query.end_time
            )

        return result

    def _get_state_at(
        self,
        timestamp: datetime,
        topic_id: int | None = None,
        include_archived: bool = False,
    ) -> TemporalState:
        """Get memory state at a specific timestamp."""
        # Get all memories that existed at this timestamp
        if topic_id:
            memories = self.store.get_memories(topic_id=topic_id, limit=1000)
        else:
            memories = self.store.get_memories(limit=1000)

        active_memories = []
        active_decisions = []

        for memory in memories:
            # Check if memory existed at timestamp
            if memory.created_at <= timestamp:
                # Check if not archived/deleted before timestamp
                if not self._was_archived_before(memory, timestamp):
                    if include_archived or memory.status != MemoryStatus.ARCHIVED:
                        active_memories.append(memory)

                        if memory.memory_type == MemoryType.DECISION:
                            # Also check decisions
                            decisions = self.store.get_decisions(
                                memory.topic_id, current_only=False
                            )
                            for dec in decisions:
                                if dec.decided_at <= timestamp:
                                    if not dec.valid_until or dec.valid_until > timestamp:
                                        active_decisions.append(dec)

        return TemporalState(
            timestamp=timestamp,
            active_memories=active_memories,
            active_decisions=active_decisions,
            memory_count=len(active_memories),
            topic_id=memories[0].topic_id if memories else None,
        )

    def _was_archived_before(self, memory: Memory, timestamp: datetime) -> bool:
        """Check if memory was archived before timestamp."""
        # Check compression history for archive events
        compression_events = self.store.get_compression_history(memory.id)
        for event in compression_events:
            if event.to_resolution >= ResolutionLevel.LONG_TERM:
                if event.created_at <= timestamp:
                    return True

        # Check if status was changed to archived
        if memory.status == MemoryStatus.ARCHIVED:
            if memory.updated_at <= timestamp:
                return True

        return False

    def _reconstruct_context_at(
        self,
        timestamp: datetime,
        topic_id: int | None = None,
    ) -> str:
        """Reconstruct what context would have been built at timestamp."""
        if not self.context_builder:
            return ""

        # Get state at timestamp
        state = self._get_state_at(timestamp, topic_id)

        # Build context using only memories that existed at that time
        # Filter memories to those that existed at timestamp
        current_memories = [
            m for m in state.active_memories
            if m.created_at <= timestamp and not self._was_archived_before(m, timestamp)
        ]

        if not current_memories:
            return "No memories available at this time."

        # Use a generic query to build context
        query = "current context"
        context = self.context_builder.build_context(
            query, topic_id, max_tokens=4000, current_memories=current_memories
        )

        return f"[Context at {timestamp.isoformat()}]\n{context}"

    def _get_changes_between(
        self,
        start: datetime,
        end: datetime,
        topic_id: int | None = None,
    ) -> list[TemporalChange]:
        """Get all changes between two timestamps."""
        changes = []

        if topic_id:
            memories = self.store.get_memories(topic_id=topic_id, limit=1000)
        else:
            memories = self.store.get_memories(limit=1000)

        for memory in memories:
            # Check if created in range
            if start <= memory.created_at <= end:
                changes.append(TemporalChange(
                    timestamp=memory.created_at,
                    change_type="added",
                    memory_id=memory.id,
                    memory_type=memory.memory_type.value,
                    details={"content_preview": memory.content[:100]},
                ))

            # Check if archived in range
            if memory.status in [MemoryStatus.ARCHIVED, MemoryStatus.DEEP_ARCHIVED]:
                if memory.updated_at and start <= memory.updated_at <= end:
                    changes.append(TemporalChange(
                        timestamp=memory.updated_at,
                        change_type="status_changed",
                        memory_id=memory.id,
                        memory_type=memory.memory_type.value,
                        details={"new_status": memory.status.value},
                    ))

            # Check compression events in range
            events = self.store.get_compression_history(memory.id)
            for event in events:
                if start <= event.created_at <= end:
                    changes.append(TemporalChange(
                        timestamp=event.created_at,
                        change_type="compressed",
                        memory_id=memory.id,
                        memory_type=memory.memory_type.value,
                        details={
                            "from": event.from_resolution.name,
                            "to": event.to_resolution.name,
                            "ratio": event.compression_ratio,
                        },
                    ))

        # Sort by timestamp
        changes.sort(key=lambda c: c.timestamp)
        return changes

    def _get_timeline(
        self,
        topic_id: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Get timeline of events for a topic."""
        timeline = []

        if topic_id:
            memories = self.store.get_memories(topic_id=topic_id, limit=1000)
        else:
            memories = self.store.get_memories(limit=1000)

        for memory in memories:
            # Filter by time range
            if start and memory.created_at < start:
                continue
            if end and memory.created_at > end:
                continue

            timeline.append({
                "timestamp": memory.created_at.isoformat(),
                "event_type": "memory_created",
                "memory_id": memory.id,
                "type": memory.memory_type.value,
                "resolution": memory.resolution.name,
                "preview": memory.content[:100],
            })

            # Add compression events
            events = self.store.get_compression_history(memory.id)
            for event in events:
                if start and event.created_at < start:
                    continue
                if end and event.created_at > end:
                    continue

                timeline.append({
                    "timestamp": event.created_at.isoformat(),
                    "event_type": "memory_compressed",
                    "memory_id": memory.id,
                    "from_resolution": event.from_resolution.name,
                    "to_resolution": event.to_resolution.name,
                    "ratio": event.compression_ratio,
                })

        # Add decision events
        if topic_id:
            decisions = self.store.get_decisions(topic_id, current_only=False)
            for dec in decisions:
                if start and dec.decided_at < start:
                    continue
                if end and dec.decided_at > end:
                    continue

                timeline.append({
                    "timestamp": dec.decided_at.isoformat(),
                    "event_type": "decision_made",
                    "decision_id": dec.id,
                    "decision": dec.decision_text[:100],
                })

        # Sort by timestamp
        timeline.sort(key=lambda x: x["timestamp"])
        return timeline


def create_time_travel_engine(
    store: MemoryStore,
    belief_engine: BeliefEngine,
    recall_engine=None,
    context_builder=None,
) -> TimeTravelEngine:
    return TimeTravelEngine(store, belief_engine, recall_engine, context_builder)
