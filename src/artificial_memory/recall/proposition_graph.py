"""Unified Proposition Graph (Overdrive Core Phase 2).

Maintains an interconnected associative graph of propositions, entities,
state transitions, and temporal relationships.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Sequence

from artificial_memory.core.ir.proposition import PropositionStatus, StateHistory, UnifiedProposition
from artificial_memory.core.ir.structured import StructuredIR


@dataclass
class GraphEdge:
    """A directed/undirected edge in the Proposition Graph."""
    source_id: str
    target_id: str
    edge_type: str  # "entity_subject", "entity_object", "supersedes", "temporal_delta", "coreference"
    weight: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)


class UnifiedPropositionGraph:
    """Unified Graph combining Propositions, Entities, State Histories, and Temporal Relations."""

    def __init__(self) -> None:
        self.propositions: dict[str, UnifiedProposition] = {}
        self.entities: dict[str, set[str]] = defaultdict(set)  # entity_name -> set of prop_ids
        self.state_histories: dict[tuple[str, str], StateHistory] = {}  # (entity, attribute) -> StateHistory
        self.adjacency: dict[str, list[GraphEdge]] = defaultdict(list)

    def add_proposition(self, prop: UnifiedProposition) -> None:
        """Add a proposition and link it to its constituent entities and state histories."""
        self.propositions[prop.id] = prop

        # Index subject and object
        subj_clean = prop.subject.lower().strip()
        obj_clean = prop.object.lower().strip()

        if subj_clean:
            self.entities[subj_clean].add(prop.id)
            self.adjacency[prop.id].append(GraphEdge(prop.id, f"entity:{subj_clean}", "entity_subject"))
            self.adjacency[f"entity:{subj_clean}"].append(GraphEdge(f"entity:{subj_clean}", prop.id, "subject_of"))

        if obj_clean and len(obj_clean) > 2:
            self.entities[obj_clean].add(prop.id)
            self.adjacency[prop.id].append(GraphEdge(prop.id, f"entity:{obj_clean}", "entity_object"))
            self.adjacency[f"entity:{obj_clean}"].append(GraphEdge(f"entity:{obj_clean}", prop.id, "object_of"))

        # Update StateHistory for update-able attributes
        if prop.predicate in ["uses", "owns", "works_at", "lives_in", "preference"]:
            key = (subj_clean, prop.predicate)
            if key not in self.state_histories:
                self.state_histories[key] = StateHistory(prop.subject, prop.predicate)
            
            # Check if this supersedes previous snapshots
            prev = self.state_histories[key].get_latest()
            if prev and prev.value.lower() != prop.object.lower():
                # Add supersession edge
                self.adjacency[prev.proposition_id].append(
                    GraphEdge(prev.proposition_id, prop.id, "supersedes", weight=1.5)
                )
                self.propositions[prev.proposition_id].status = PropositionStatus.SUPERSEDED

            self.state_histories[key].add_snapshot(
                value=prop.object,
                timestamp=prop.time_scope,
                session_id=prop.session_id,
                prop_id=prop.id,
            )

    def build_from_records(self, records: Sequence[StructuredIR]) -> None:
        """Deterministically convert StructuredIR records into UnifiedPropositions."""
        for i, r in enumerate(records):
            subj = r.entity or (r.source if r.source != "general" else "user")
            pred = r.property or r.relation.value
            obj = r.value or r.raw_content

            # Parse session ID from raw_content if present (e.g. "[D4:3 on 2023/05/22]")
            m_sess = re.search(r"\[(D\d+:\d+|session_\d+)", r.raw_content)
            sess_id = m_sess.group(1) if m_sess else f"rec-{i}"

            prop_id = f"prop-{i}"
            status = PropositionStatus.SUPERSEDED if r.status.value == "superseded" else PropositionStatus.ACTIVE

            prop = UnifiedProposition(
                id=prop_id,
                subject=subj,
                predicate=pred,
                object=obj,
                time_scope=r.time_scope,
                session_id=sess_id,
                source=r.source or "user",
                status=status,
                confidence=r.confidence,
                raw_text=r.raw_content,
            )
            self.add_proposition(prop)

    def get_propositions_for_entity(self, entity_name: str) -> list[UnifiedProposition]:
        """Retrieve all propositions involving an entity name or alias."""
        ent_lower = entity_name.lower().strip()
        prop_ids = self.entities.get(ent_lower, set())
        # Check substring matches
        if not prop_ids:
            for k, ids in self.entities.items():
                if ent_lower in k or k in ent_lower:
                    prop_ids.update(ids)
        return [self.propositions[pid] for pid in prop_ids if pid in self.propositions]

    def traverse_neighbors(self, prop_id: str, max_depth: int = 1) -> list[UnifiedProposition]:
        """Traverse connected propositions across entity and supersession edges."""
        visited: set[str] = {prop_id}
        frontier: list[str] = [prop_id]
        result: list[UnifiedProposition] = []

        for _ in range(max_depth):
            next_frontier: list[str] = []
            for curr in frontier:
                for edge in self.adjacency.get(curr, []):
                    target = edge.target_id
                    if target.startswith("entity:"):
                        # Walk from entity node to connected propositions
                        for back_edge in self.adjacency.get(target, []):
                            p_target = back_edge.target_id
                            if p_target not in visited and p_target in self.propositions:
                                visited.add(p_target)
                                next_frontier.append(p_target)
                                result.append(self.propositions[p_target])
                    elif target in self.propositions and target not in visited:
                        visited.add(target)
                        next_frontier.append(target)
                        result.append(self.propositions[target])
            frontier = next_frontier

        return result
