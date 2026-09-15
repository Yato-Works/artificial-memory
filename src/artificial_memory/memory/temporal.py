from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Decision, Memory


@dataclass
class TemporalQuery:
    """Query for temporal state."""
    timestamp: datetime
    topic_id: int | None = None
    include_archived: bool = False


@dataclass
class TemporalState:
    """State of the world at a specific timestamp."""
    timestamp: datetime
    active_memories: list
    active_decisions: list
    valid_from_snapshot: dict
    valid_until_snapshot: dict


class TemporalEngine:
    """Engine for querying temporal state of memories."""

    def __init__(self, store: MemoryStore):
        self.store = store

    def get_state_at(self, timestamp: datetime, topic_id: int | None = None,
                     include_archived: bool = False) -> TemporalState:
        """Get the state of all memories at a specific timestamp."""

        # Get memories that were valid at this timestamp
        if topic_id:
            memories = self.store.get_memories(topic_id=topic_id, limit=1000)
        else:
            # Get all memories across topics
            topics = self.store.list_topics()
            memories = []
            for topic in topics:
                memories.extend(self.store.get_memories(topic_id=topic.id, limit=1000))

        # Filter by temporal validity
        active_memories = []
        for mem in memories:
            if self._was_valid_at(mem, timestamp):
                if include_archived or mem.status != "archived" and mem.status != "deep_archived":
                    active_memories.append(mem)

        # Get decisions valid at timestamp
        if topic_id:
            decisions = self.store.get_decisions(topic_id, current_only=False)
        else:
            decisions = []
            for topic in self.store.list_topics():
                decisions.extend(self.store.get_decisions(topic.id, current_only=False))

        active_decisions = [d for d in decisions if self._was_valid_at_decision(d, timestamp)]

        # Build snapshots
        valid_from = {}
        valid_until = {}

        for mem in active_memories:
            if mem.valid_from:
                key = mem.valid_from.isoformat()
                if key not in valid_from:
                    valid_from[key] = []
                valid_from[key].append({
                    "memory_id": mem.id,
                    "type": mem.memory_type.value,
                    "content_preview": mem.content[:100]
                })

            if mem.valid_until:
                key = mem.valid_until.isoformat()
                if key not in valid_until:
                    valid_until[key] = []
                valid_until[key].append({
                    "memory_id": mem.id,
                    "type": mem.memory_type.value,
                    "content_preview": mem.content[:100]
                })

        return TemporalState(
            timestamp=timestamp,
            active_memories=active_memories,
            active_decisions=active_decisions,
            valid_from_snapshot=valid_from,
            valid_until_snapshot=valid_until,
        )

    def get_state_for_topic_at(self, topic_id: int, timestamp: datetime) -> TemporalState:
        """Get state for a specific topic at a timestamp."""
        return self.get_state_at(timestamp, topic_id)

    def get_changes_between(self, start: datetime, end: datetime,
                           topic_id: int | None = None) -> dict:
        """Get changes between two timestamps."""
        state_start = self.get_state_at(start, topic_id, include_archived=True)
        state_end = self.get_state_at(end, topic_id, include_archived=True)

        start_ids = {m.id for m in state_start.active_memories}
        end_ids = {m.id for m in state_end.active_memories}

        added = [m for m in state_end.active_memories if m.id not in start_ids]
        removed = [m for m in state_start.active_memories if m.id not in end_ids]

        # Check for modified memories (same ID but different content/status)
        modified = []
        start_by_id = {m.id: m for m in state_start.active_memories}
        for m in state_end.active_memories:
            if m.id in start_by_id:
                orig = start_by_id[m.id]
                if orig.content != m.content or orig.status != m.status:
                    modified.append({"memory_id": m.id, "from": orig, "to": m})

        return {
            "period": {"start": start.isoformat(), "end": end.isoformat()},
            "added": [{"id": m.id, "type": m.memory_type.value, "preview": m.content[:100]} for m in added],
            "removed": [{"id": m.id, "type": m.memory_type.value, "preview": m.content[:100]} for m in removed],
            "modified": modified,
        }

    def get_timeline_for_topic(self, topic_id: int,
                               start: datetime | None = None,
                               end: datetime | None = None) -> list[dict]:
        """Get timeline of changes for a topic."""
        memories = self.store.get_memories(topic_id=topic_id, limit=1000)

        events = []
        for mem in memories:
            if start and mem.created_at < start:
                continue
            if end and mem.created_at > end:
                continue

            events.append({
                "timestamp": mem.created_at.isoformat(),
                "memory_id": mem.id,
                "type": mem.memory_type.value,
                "resolution": mem.resolution.name,
                "status": mem.status.value,
                "preview": mem.content[:150],
                "valid_from": mem.valid_from.isoformat() if mem.valid_from else None,
                "valid_until": mem.valid_until.isoformat() if mem.valid_until else None,
            })

        # Sort by timestamp
        events.sort(key=lambda e: e["timestamp"])
        return events

    def _was_valid_at(self, memory: Memory, timestamp: datetime) -> bool:
        """Check if memory was valid at timestamp."""
        if memory.valid_from and memory.valid_from > timestamp:
            return False
        if memory.valid_until and memory.valid_until <= timestamp:
            return False
        return True

    def _was_valid_at_decision(self, decision: Decision, timestamp: datetime) -> bool:
        """Check if decision was valid at timestamp."""
        if decision.valid_from and decision.valid_from > timestamp:
            return False
        if decision.valid_until and decision.valid_until <= timestamp:
            return False
        return True

    def query_by_time_range(self, start: datetime, end: datetime,
                           topic_id: int | None = None,
                           memory_type: str | None = None) -> list[Memory]:
        """Query memories by time range."""
        memories = self.store.get_memories(topic_id=topic_id, limit=1000)

        filtered = []
        for mem in memories:
            if mem.created_at >= start and mem.created_at <= end:
                if memory_type and mem.memory_type.value != memory_type:
                    continue
                filtered.append(mem)

        return filtered


def create_temporal_engine(store: MemoryStore) -> TemporalEngine:
    """Factory function to create temporal engine."""
    return TemporalEngine(store)
