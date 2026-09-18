"""Lifecycle state transitions (AM v0.2.0 Phase 2).

Memory status changes are explicit, validated transitions - not ad-hoc
field writes. Every transition is:

* checked against an allowed-transition graph,
* applied to the store,
* logged as a TRANSITION evolution event with from/to statuses.

Allowed graph (MemoryStatus)::

    ACTIVE      -> DORMANT | COMPRESSED | ARCHIVED
    COMPRESSED  -> DORMANT | ARCHIVED | ACTIVE
    DORMANT     -> ACTIVE | ARCHIVED
    ARCHIVED    -> DEEP_ARCHIVED | ACTIVE
    DEEP_ARCHIVED -> ARCHIVED | ACTIVE

The ARCHIVED -> ACTIVE and DEEP_ARCHIVED -> ACTIVE edges correspond to
RESTORE / REACTIVATE; they exist here so temporal-state tooling can move
memories back into the active set explicitly.
"""

from __future__ import annotations

from datetime import datetime

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Memory, MemoryStatus
from artificial_memory.memory.evolution import (
    EvolutionEvent,
    EvolutionOperationType,
    log_evolution_event,
)

ALLOWED_TRANSITIONS: dict[MemoryStatus, set[MemoryStatus]] = {
    MemoryStatus.ACTIVE: {
        MemoryStatus.DORMANT,
        MemoryStatus.COMPRESSED,
        MemoryStatus.ARCHIVED,
    },
    MemoryStatus.COMPRESSED: {
        MemoryStatus.DORMANT,
        MemoryStatus.ARCHIVED,
        MemoryStatus.ACTIVE,
    },
    MemoryStatus.DORMANT: {
        MemoryStatus.ACTIVE,
        MemoryStatus.ARCHIVED,
    },
    MemoryStatus.ARCHIVED: {
        MemoryStatus.DEEP_ARCHIVED,
        MemoryStatus.ACTIVE,
    },
    MemoryStatus.DEEP_ARCHIVED: {
        MemoryStatus.ARCHIVED,
        MemoryStatus.ACTIVE,
    },
}


class InvalidTransitionError(ValueError):
    """Raised when a status transition is not allowed by the lifecycle graph."""


class LifecycleTransitionEngine:
    """Validates and applies explicit lifecycle state transitions."""

    def __init__(self, store: MemoryStore, triggered_by: str = "policy"):
        self.store = store
        self.triggered_by = triggered_by

    # ==================== Mutations ====================

    def transition(
        self,
        memory_id: int,
        to_status: MemoryStatus,
        reason: str = "",
        triggered_by: str | None = None,
    ) -> Memory:
        """Move a memory to ``to_status`` if the graph allows it.

        ``is_current`` follows the target status: only ACTIVE and COMPRESSED
        representations remain in the current retrieval set.
        """
        memory = self.store.get_memory(memory_id)
        if not memory:
            raise ValueError(f"Memory {memory_id} not found")

        from_status = memory.status
        if to_status == from_status:
            return memory  # no-op transition

        allowed = ALLOWED_TRANSITIONS.get(from_status, set())
        if to_status not in allowed:
            raise InvalidTransitionError(
                f"Transition {from_status.value} -> {to_status.value} is not allowed "
                f"(allowed from {from_status.value}: "
                f"{sorted(s.value for s in allowed)})"
            )

        memory.status = to_status
        memory.is_current = to_status in (MemoryStatus.ACTIVE, MemoryStatus.COMPRESSED)
        memory.updated_at = datetime.now()
        memory = self.store.update_memory(memory)

        event = EvolutionEvent(
            memory_id=memory_id,
            operation=EvolutionOperationType.TRANSITION,
            description=reason or f"State transition {from_status.value} -> {to_status.value}",
            metadata={
                "kind": "lifecycle_transition",
                "from_status": from_status.value,
                "to_status": to_status.value,
                "is_current": memory.is_current,
                "reason": reason,
            },
            triggered_by=triggered_by or self.triggered_by,
        )
        log_evolution_event(self.store, event)

        return memory

    # ==================== Queries ====================

    def transition_history(self, memory_id: int) -> list[dict[str, str]]:
        """The memory's state-transition history from the audit log."""
        # ``get_evolution_events`` exists on persistent store backends;
        # in-memory or minimal stores may not implement it.
        get_events = getattr(self.store, "get_evolution_events", None)
        if get_events is None:
            return []
        try:
            events = get_events(
                memory_id=memory_id, operation=EvolutionOperationType.TRANSITION.value
            )
        except (AttributeError, TypeError):
            return []
        history = []
        for event in events:
            kind = event.metadata.get("kind", "")
            if kind == "lifecycle_transition":
                history.append({
                    "from_status": event.metadata.get("from_status", ""),
                    "to_status": event.metadata.get("to_status", ""),
                    "reason": event.metadata.get("reason", ""),
                    "at": event.created_at.isoformat(),
                    "_id": event.id or 0,
                })
        # Order by insertion id: created_at has coarse resolution on some
        # OS clocks, so equal timestamps would make ordering unstable.
        history.sort(key=lambda h: h["_id"])
        for item in history:
            del item["_id"]
        return history

    def is_transition_allowed(self, from_status: MemoryStatus, to_status: MemoryStatus) -> bool:
        return to_status in ALLOWED_TRANSITIONS.get(from_status, set())


def create_lifecycle_transition_engine(
    store: MemoryStore,
    triggered_by: str = "policy",
) -> LifecycleTransitionEngine:
    return LifecycleTransitionEngine(store, triggered_by)


__all__ = [
    "ALLOWED_TRANSITIONS",
    "InvalidTransitionError",
    "LifecycleTransitionEngine",
    "create_lifecycle_transition_engine",
]

