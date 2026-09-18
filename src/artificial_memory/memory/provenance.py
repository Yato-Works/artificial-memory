"""Provenance chains (AM v0.2.0 Phase 2).

Plan #22: every persistent interpretation has an evidence chain::

    Semantic Memory
          |
        Episode
          |
     Conversation
          |
    Original Message

A memory should answer "Where did this come from?" and "Which evidence
supports this interpretation?" without modifying any record.

The chain is walked from the evidence links recorded at formation / evolution
time (ELABORATES interpretation links, SUPERSEDES lineage) down to the source
conversation and message. ``explain()`` combines provenance with validity,
contradiction, supersession, and audit history - the plan's ``memory.explain()``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import AssociationType, Memory
from artificial_memory.memory.contradiction_edges import ContradictionEdgeManager
from artificial_memory.memory.supersession import SupersessionManager

# Association types that carry *evidence lineage* (target derived from source).
EVIDENCE_LINK_TYPES = (AssociationType.ELABORATES, AssociationType.SUMMARIZES)


@dataclass
class ProvenanceNode:
    """One hop in a memory's provenance chain."""

    memory_id: int
    content_preview: str
    resolution: str
    relationship: str  # how this node relates to the child ("self", link type)
    depth: int
    source_conversation_id: int | None = None
    source_message_id: int | None = None
    conversation_title: str | None = None
    message_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "content_preview": self.content_preview,
            "resolution": self.resolution,
            "relationship": self.relationship,
            "depth": self.depth,
            "source_conversation_id": self.source_conversation_id,
            "source_message_id": self.source_message_id,
            "conversation_title": self.conversation_title,
            "message_count": self.message_count,
        }


@dataclass
class ProvenanceChain:
    """The full provenance chain of a memory."""

    memory_id: int
    nodes: list[ProvenanceNode] = field(default_factory=list)
    built_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "depth": len(self.nodes),
            "nodes": [node.to_dict() for node in self.nodes],
            "built_at": self.built_at.isoformat(),
        }


class ProvenanceChainBuilder:
    """Builds provenance chains and full explanations for memories."""

    def __init__(self, store: MemoryStore):
        self.store = store
        self.supersession = SupersessionManager(store)
        self.contradictions = ContradictionEdgeManager(store)

    # ==================== Provenance chain ====================

    def build_chain(self, memory_id: int, max_depth: int = 10) -> ProvenanceChain:
        """Walk the evidence lineage from a memory toward its sources.

        Evidence parents are memories linked *into* this one via
        ELABORATES / SUMMARIZES (the parent supplied the evidence) plus the
        SUPERSEDES predecessor lineage. Each node is annotated with its
        source conversation / message where recorded.
        """
        chain = ProvenanceChain(memory_id=memory_id)
        visited: set[int] = set()
        frontier: list[tuple[int, str, int]] = [(memory_id, "self", 0)]

        while frontier and len(chain.nodes) < max_depth:
            current_id, relationship, depth = frontier.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            memory = self.store.get_memory(current_id)
            if memory is None:
                continue
            chain.nodes.append(self._to_node(memory, relationship, depth))

            # Evidence lineage, two directions:
            #  * ELABORATES / SUMMARIZES: source supplied the evidence for the
            #    target -> parents of the current node are the sources.
            #  * SUPERSEDES: source (new) replaced target (old) -> the
            #    predecessor of the current node is the target.
            associations = self.store.get_associations(current_id)
            for association in associations:
                link_type = association.association_type
                if (
                    association.target_memory_id == current_id
                    and (link_type in EVIDENCE_LINK_TYPES or link_type == AssociationType.SUPERSEDES)
                ):
                    frontier.append(
                        (association.source_memory_id, link_type.value, depth + 1)
                    )
                elif (
                    association.source_memory_id == current_id
                    and link_type == AssociationType.SUPERSEDES
                    and association.target_memory_id != current_id
                ):
                    frontier.append(
                        (association.target_memory_id, link_type.value, depth + 1)
                    )

        return chain

    def _to_node(self, memory: Memory, relationship: str, depth: int) -> ProvenanceNode:
        conversation_title: str | None = None
        message_count: int | None = None
        if memory.source_conversation_id is not None:
            conversation = self.store.get_conversation(memory.source_conversation_id)
            if conversation is not None:
                conversation_title = conversation.title
                try:
                    message_count = self.store.get_message_count(memory.source_conversation_id)
                except AttributeError:
                    message_count = None

        return ProvenanceNode(
            memory_id=memory.id if memory.id is not None else 0,
            content_preview=memory.content[:120],
            resolution=memory.resolution.name,
            relationship=relationship,
            depth=depth,
            source_conversation_id=memory.source_conversation_id,
            source_message_id=memory.source_message_id,
            conversation_title=conversation_title,
            message_count=message_count,
        )

    # ==================== Explanation ====================

    def explain(self, memory_id: int) -> dict[str, Any]:
        """Full explanation for a memory (plan API: ``memory.explain()``).

        Combines: the record itself, its validity window, provenance chain,
        supersession lineage (predecessors / successors), contradiction
        edges, and the append-only evolution (audit) history.
        """
        memory = self.store.get_memory(memory_id)
        if memory is None:
            raise ValueError(f"Memory {memory_id} not found")

        # ``get_evolution_events`` exists on persistent store backends;
        # in-memory or minimal stores may not implement it.
        get_events = getattr(self.store, "get_evolution_events", None)
        if get_events is None:
            events: list[Any] = []
        else:
            try:
                events = get_events(memory_id=memory_id, limit=100)
            except (AttributeError, TypeError):
                events = []

        edges = self.contradictions.get_edges(memory_id)

        return {
            "memory": {
                "id": memory.id,
                "type": memory.memory_type.value,
                "content_preview": memory.content[:200],
                "resolution": memory.resolution.name,
                "importance": memory.importance,
                "confidence": memory.confidence,
                "status": memory.status.value,
                "is_current": memory.is_current,
                "created_at": memory.created_at.isoformat(),
                "updated_at": memory.updated_at.isoformat(),
            },
            "validity": {
                "valid_from": memory.valid_from.isoformat() if memory.valid_from else None,
                "valid_until": memory.valid_until.isoformat() if memory.valid_until else None,
            },
            "provenance": self.build_chain(memory_id).to_dict(),
            "supersession": {
                "predecessors": [m.id for m in self.supersession.get_predecessors(memory_id)],
                "successors": [m.id for m in self.supersession.get_successors(memory_id)],
                "current_head": self.supersession.get_current_head(memory_id).id,
            },
            "contradictions": [
                {
                    "other_memory_id": (
                        edge.memory_b.id
                        if edge.memory_a.id == memory_id
                        else edge.memory_a.id
                    ),
                    "severity": edge.severity,
                }
                for edge in edges
            ],
            "audit_history": [
                {
                    "id": event.id,
                    "operation": event.operation,
                    "description": event.description,
                    "triggered_by": event.triggered_by,
                    "at": event.created_at.isoformat(),
                }
                for event in events
            ],
        }


def create_provenance_chain_builder(store: MemoryStore) -> ProvenanceChainBuilder:
    return ProvenanceChainBuilder(store)


__all__ = [
    "EVIDENCE_LINK_TYPES",
    "ProvenanceChain",
    "ProvenanceChainBuilder",
    "ProvenanceNode",
    "create_provenance_chain_builder",
]


