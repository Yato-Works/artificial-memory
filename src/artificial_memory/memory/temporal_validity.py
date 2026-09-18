"""Temporal validity layer (AM v0.2.0 Phase 2).

Every persistent memory supports temporal interpretation. AM distinguishes
creation / update / validity-start / validity-end times (plan #20), so that
questions like "What does the user currently prefer?" and "What did the user
prefer last year?" are answerable without destroying historical evidence.

Plan #21 governs supersession bounds: when evidence supports a change of
state, the runtime infers temporal bounds instead of overwriting history::

    M1.valid_until = t
    M2.valid_from  = t

All mutations are logged as TRANSITION evolution events (auditability).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Memory
from artificial_memory.memory.evolution import (
    EvolutionEvent,
    EvolutionOperationType,
    log_evolution_event,
)


@dataclass
class ValidityInterval:
    """The validity window of a memory."""

    valid_from: datetime | None
    valid_until: datetime | None

    def contains(self, timestamp: datetime) -> bool:
        """Half-open window check: [valid_from, valid_until)."""
        if self.valid_from is not None and timestamp < self.valid_from:
            return False
        if self.valid_until is not None and timestamp >= self.valid_until:
            return False
        return True


class TemporalValidityManager:
    """Manages temporal validity windows on memories."""

    def __init__(self, store: MemoryStore, triggered_by: str = "temporal"):
        self.store = store
        self.triggered_by = triggered_by

    # ==================== Queries ====================

    def is_valid_at(self, memory: Memory, timestamp: datetime) -> bool:
        """True while ``timestamp`` is inside the memory's validity window.

        The window is half-open ``[valid_from, valid_until)``: at the exact
        handover instant of a supersession (old.valid_until == new.valid_from
        == t) the successor is valid and the predecessor is not, so the two
        records never overlap.
        """
        if memory.valid_from is not None and timestamp < memory.valid_from:
            return False
        if memory.valid_until is not None and timestamp >= memory.valid_until:
            return False
        return True

    def validity_interval(self, memory: Memory) -> ValidityInterval:
        return ValidityInterval(valid_from=memory.valid_from, valid_until=memory.valid_until)

    def filter_valid_at(self, memories: list[Memory], timestamp: datetime) -> list[Memory]:
        """Return only the memories valid at ``timestamp``."""
        return [memory for memory in memories if self.is_valid_at(memory, timestamp)]

    # ==================== Mutations ====================

    def invalidate(
        self,
        memory_id: int,
        valid_until: datetime,
        reason: str = "",
        triggered_by: str | None = None,
    ) -> Memory:
        """End a memory's validity at ``valid_until``.

        Monotonic: an earlier (or equal) existing ``valid_until`` wins, so
        invalidation never extends a memory's life.
        """
        memory = self.store.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")

        if memory.valid_until is not None and memory.valid_until <= valid_until:
            return memory  # already invalidated no later than requested

        previous = memory.valid_until
        memory.valid_until = valid_until
        memory.updated_at = datetime.now()
        memory = self.store.update_memory(memory)

        event = EvolutionEvent(
            memory_id=memory_id,
            operation=EvolutionOperationType.TRANSITION,
            description=reason or "Temporal invalidation",
            metadata={
                "kind": "temporal_invalidation",
                "previous_valid_until": previous.isoformat() if previous else None,
                "new_valid_until": valid_until.isoformat(),
                "reason": reason,
            },
            triggered_by=triggered_by or self.triggered_by,
        )
        log_evolution_event(self.store, event)

        return memory

    def apply_supersession_bounds(
        self,
        old_memory_id: int,
        new_memory_id: int,
        transition_time: datetime,
        reason: str = "",
        triggered_by: str | None = None,
    ) -> tuple[Memory, Memory]:
        """Infer temporal bounds for a state change (plan #21).

        ``old.valid_until = transition_time`` and ``new.valid_from =
        transition_time`` so both records coexist with a clean handover.
        Idempotent: existing tighter bounds are preserved.
        """
        old = self.store.get_memory(old_memory_id)
        if not old:
            raise ValueError(f"Memory {old_memory_id} not found")
        new = self.store.get_memory(new_memory_id)
        if not new:
            raise ValueError(f"Memory {new_memory_id} not found")

        changed = False
        if old.valid_until is None or old.valid_until > transition_time:
            old.valid_until = transition_time
            changed = True
        if new.valid_from is None or new.valid_from > transition_time:
            new.valid_from = transition_time
            changed = True

        if not changed:
            return old, new

        now = datetime.now()
        old.updated_at = now
        new.updated_at = now
        old = self.store.update_memory(old)
        new = self.store.update_memory(new)

        event = EvolutionEvent(
            memory_id=old.id if old.id is not None else 0,
            operation=EvolutionOperationType.TRANSITION,
            source_memory_ids=[old.id] if old.id is not None else [],
            target_memory_id=new.id,
            description=reason or "Supersession temporal bounds applied",
            metadata={
                "kind": "supersession_bounds",
                "new_memory_id": new.id,
                "transition_time": transition_time.isoformat(),
                "reason": reason,
            },
            triggered_by=triggered_by or self.triggered_by,
        )
        log_evolution_event(self.store, event)

        return old, new


def create_temporal_validity_manager(
    store: MemoryStore,
    triggered_by: str = "temporal",
) -> TemporalValidityManager:
    return TemporalValidityManager(store, triggered_by)


__all__ = [
    "TemporalValidityManager",
    "ValidityInterval",
    "create_temporal_validity_manager",
]

