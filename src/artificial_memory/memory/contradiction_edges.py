"""Explicit contradiction edges (AM v0.2.0 Phase 2).

Plan #21: contradictions are represented **explicitly** as CONTRADICTS edges
and must never silently overwrite historical evidence. Where sufficient
evidence supports a genuine change of state, temporal bounds are inferred
instead (``apply_temporal_resolution``).

A contradiction may represent change rather than corruption: changing
preferences, skills, circumstances, or conflicting sources. Both records
coexist with timestamps, and the runtime decides which is relevant to a
query.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import (
    Association,
    AssociationType,
    Memory,
)
from artificial_memory.memory.evolution import (
    EvolutionEvent,
    EvolutionOperationType,
    log_evolution_event,
)
from artificial_memory.memory.temporal_validity import TemporalValidityManager


@dataclass
class ContradictionEdge:
    """An explicit CONTRADICTS edge between two memories."""

    association: Association
    memory_a: Memory
    memory_b: Memory

    @property
    def severity(self) -> float:
        return self.association.strength


class ContradictionEdgeManager:
    """Creates, lists, and resolves explicit contradiction edges."""

    def __init__(self, store: MemoryStore, triggered_by: str = "policy"):
        self.store = store
        self.temporal = TemporalValidityManager(store)
        self.triggered_by = triggered_by

    # ==================== Mutations ====================

    def create_edge(
        self,
        memory_a_id: int,
        memory_b_id: int,
        severity: float = 0.8,
        description: str = "",
        triggered_by: str | None = None,
    ) -> Association:
        """Create an explicit CONTRADICTS edge between two memories.

        Both sides receive a CONTRADICT evolution event so the edge is
        discoverable from either memory's audit trail.
        """
        memory_a = self.store.get_memory(memory_a_id)
        if not memory_a:
            raise ValueError(f"Memory {memory_a_id} not found")
        memory_b = self.store.get_memory(memory_b_id)
        if not memory_b:
            raise ValueError(f"Memory {memory_b_id} not found")
        if memory_a.id == memory_b.id:
            raise ValueError("A memory cannot contradict itself")

        severity = max(0.0, min(1.0, severity))
        if memory_a.id is None or memory_b.id is None:
            raise ValueError("Both memories must be persisted before linking")

        association = self.store.create_association(Association(
            source_memory_id=memory_a.id,
            target_memory_id=memory_b.id,
            association_type=AssociationType.CONTRADICTS,
            strength=severity,
        ))

        base_metadata = {
            "other_memory_id": memory_b.id,
            "severity": severity,
            "edge_id": association.id,
            "description": description,
        }
        for side, memory, other in (
            ("a", memory_a, memory_b),
            ("b", memory_b, memory_a),
        ):
            metadata = dict(base_metadata)
            metadata["side"] = side
            metadata["other_memory_id"] = other.id
            event = EvolutionEvent(
                memory_id=memory.id if memory.id is not None else 0,
                operation=EvolutionOperationType.CONTRADICT,
                target_memory_id=other.id,
                description=description or f"Contradicts memory {other.id}",
                metadata=metadata,
                triggered_by=triggered_by or self.triggered_by,
            )
            log_evolution_event(self.store, event)

        return association

    def apply_temporal_resolution(
        self,
        memory_a_id: int,
        memory_b_id: int,
        transition_time: datetime,
        reason: str = "",
        triggered_by: str | None = None,
    ) -> tuple[Memory, Memory]:
        """Resolve a contradiction as a state change (plan #21).

        The earlier-created memory is treated as the prior state: its
        validity ends at ``transition_time``; the later-created memory's
        validity starts there. Both records are preserved.
        """
        memory_a = self.store.get_memory(memory_a_id)
        if not memory_a:
            raise ValueError(f"Memory {memory_a_id} not found")
        memory_b = self.store.get_memory(memory_b_id)
        if not memory_b:
            raise ValueError(f"Memory {memory_b_id} not found")

        # Older = earlier creation; on identical timestamps (coarse OS clock)
        # the lower autoincrement id is treated as older (deterministic).
        if (memory_a.created_at, memory_a.id or 0) <= (memory_b.created_at, memory_b.id or 0):
            older, newer = memory_a, memory_b
        else:
            older, newer = memory_b, memory_a

        older_id = older.id if older.id is not None else 0
        newer_id = newer.id if newer.id is not None else 0
        older, newer = self.temporal.apply_supersession_bounds(
            older_id,
            newer_id,
            transition_time,
            reason=reason or "Contradiction resolved as temporal change of state",
            triggered_by=triggered_by or self.triggered_by,
        )
        return older, newer

    # ==================== Queries ====================

    def get_edges(self, memory_id: int) -> list[ContradictionEdge]:
        """All CONTRADICTS edges touching ``memory_id`` (either direction)."""
        associations = self.store.get_associations(memory_id, AssociationType.CONTRADICTS)
        edges: list[ContradictionEdge] = []
        for association in associations:
            memory_a = self.store.get_memory(association.source_memory_id)
            memory_b = self.store.get_memory(association.target_memory_id)
            if memory_a is None or memory_b is None:
                continue
            edges.append(ContradictionEdge(
                association=association, memory_a=memory_a, memory_b=memory_b
            ))
        return edges

    def unresolved_count(self, memory_id: int, now: datetime | None = None) -> int:
        """Edges whose two sides are simultaneously valid (unresolved).

        A contradiction is unresolved while both records' validity windows
        overlap at ``now``. ``apply_temporal_resolution`` separates the
        windows, which resolves the contradiction without deleting either
        record.
        """
        now = now or datetime.now()
        unresolved = 0
        for edge in self.get_edges(memory_id):
            if self.temporal.is_valid_at(edge.memory_a, now) and self.temporal.is_valid_at(
                edge.memory_b, now
            ):
                unresolved += 1
        return unresolved


def create_contradiction_edge_manager(
    store: MemoryStore,
    triggered_by: str = "policy",
) -> ContradictionEdgeManager:
    return ContradictionEdgeManager(store, triggered_by)


__all__ = [
    "ContradictionEdge",
    "ContradictionEdgeManager",
    "create_contradiction_edge_manager",
]

