"""Supersession chains (AM v0.2.0 Phase 2).

Supersession replaces a memory's *current interpretation* while preserving
the historical evidence. The explicit edge is::

    (new_memory) --SUPERSEDES--> (old_memory)

so a chain can be walked in both directions:

* ``forward``  - successors: memories that supersede this one (toward newest)
* ``backward`` - predecessors: what this memory superseded (toward oldest)

The old memory is archived and non-current, its validity ends at the
transition time, and the new memory's validity starts there (plan #21).
Nothing is deleted.
"""

from __future__ import annotations

from datetime import datetime

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import (
    Association,
    AssociationType,
    Memory,
    MemoryStatus,
    MemoryType,
)
from artificial_memory.memory.evolution import (
    EvolutionEvent,
    EvolutionOperationType,
    log_evolution_event,
)
from artificial_memory.memory.temporal_validity import TemporalValidityManager


class SupersessionManager:
    """Creates and walks supersession chains."""

    def __init__(self, store: MemoryStore, triggered_by: str = "policy"):
        self.store = store
        self.temporal = TemporalValidityManager(store)
        self.triggered_by = triggered_by

    # ==================== Mutations ====================

    def supersede_memory(
        self,
        old_memory_id: int,
        new_content: str,
        confidence: float = 0.8,
        reason: str = "",
        memory_type: MemoryType | None = None,
        triggered_by: str | None = None,
    ) -> tuple[Memory, Memory]:
        """Supersede a memory with new content, preserving the old evidence.

        Returns ``(old_memory, new_memory)``. The old memory is archived and
        non-current; a SUPERSEDES edge links the two; temporal bounds are
        applied at the transition time.
        """
        old = self.store.get_memory(old_memory_id)
        if not old:
            raise ValueError(f"Memory {old_memory_id} not found")
        if not new_content.strip():
            raise ValueError("New content must be a non-empty string")

        now = datetime.now()
        confidence = max(0.0, min(1.0, confidence))

        new = Memory(
            topic_id=old.topic_id,
            memory_type=memory_type or old.memory_type,
            content=new_content,
            resolution=old.resolution,
            importance=old.importance,
            confidence=confidence,
            status=MemoryStatus.ACTIVE,
            valid_from=now,
            is_current=True,
            source_conversation_id=old.source_conversation_id,
            source_message_id=old.source_message_id,
        )
        new = self.store.create_memory(new)

        if new.id is not None and old.id is not None:
            self.store.create_association(Association(
                source_memory_id=new.id,
                target_memory_id=old.id,
                association_type=AssociationType.SUPERSEDES,
                strength=0.9,
            ))

        old.status = MemoryStatus.ARCHIVED
        old.is_current = False
        old.updated_at = now
        old = self.store.update_memory(old)

        self.temporal.apply_supersession_bounds(
            old.id if old.id is not None else 0,
            new.id if new.id is not None else 0,
            now,
            reason=reason or "Superseded by newer evidence",
        )

        event = EvolutionEvent(
            memory_id=old.id if old.id is not None else 0,
            operation=EvolutionOperationType.SUPERSEDE,
            source_memory_ids=[old.id] if old.id is not None else [],
            target_memory_id=new.id,
            description=reason or "Superseded by newer evidence",
            old_content=old.content,
            new_content=new_content,
            metadata={
                "new_memory_id": new.id,
                "transition_time": now.isoformat(),
                "confidence": confidence,
            },
            triggered_by=triggered_by or self.triggered_by,
        )
        log_evolution_event(self.store, event)

        return old, new

    # ==================== Chain queries ====================

    def get_successors(self, memory_id: int) -> list[Memory]:
        """Memories that supersede this one (ordered by creation time)."""
        associations = self.store.get_associations(memory_id, AssociationType.SUPERSEDES)
        ids = [
            a.source_memory_id
            for a in associations
            if a.target_memory_id == memory_id
        ]
        memories = [m for m in (self.store.get_memory(i) for i in ids) if m is not None]
        memories.sort(key=lambda m: (m.created_at, m.id or 0))
        return memories

    def get_predecessors(self, memory_id: int) -> list[Memory]:
        """Memories this one superseded (ordered by creation time)."""
        associations = self.store.get_associations(memory_id, AssociationType.SUPERSEDES)
        ids = [
            a.target_memory_id
            for a in associations
            if a.source_memory_id == memory_id
        ]
        memories = [m for m in (self.store.get_memory(i) for i in ids) if m is not None]
        memories.sort(key=lambda m: (m.created_at, m.id or 0))
        return memories

    def get_supersession_chain(
        self,
        memory_id: int,
        direction: str = "forward",
        max_depth: int = 10,
    ) -> list[Memory]:
        """Walk the SUPERSEDES chain from ``memory_id``.

        ``forward`` follows successors toward the newest version;
        ``backward`` follows predecessors toward the oldest evidence.
        Cycles are guarded with a visited set.
        """
        if direction not in ("forward", "backward"):
            raise ValueError("direction must be 'forward' or 'backward'")

        chain: list[Memory] = []
        visited: set[int] = {memory_id}
        current_id = memory_id
        for _ in range(max_depth):
            neighbours = (
                self.get_successors(current_id)
                if direction == "forward"
                else self.get_predecessors(current_id)
            )
            neighbours = [n for n in neighbours if n.id is not None and n.id not in visited]
            if not neighbours:
                break
            nxt = neighbours[0]
            if nxt.id is None:
                break
            visited.add(nxt.id)
            chain.append(nxt)
            current_id = nxt.id
        return chain

    def get_current_head(self, memory_id: int, max_depth: int = 10) -> Memory:
        """Follow the chain forward to the newest version of this memory."""
        chain = self.get_supersession_chain(memory_id, direction="forward", max_depth=max_depth)
        if chain:
            return chain[-1]
        memory = self.store.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")
        return memory


def create_supersession_manager(
    store: MemoryStore,
    triggered_by: str = "policy",
) -> SupersessionManager:
    return SupersessionManager(store, triggered_by)


__all__ = [
    "SupersessionManager",
    "create_supersession_manager",
]

