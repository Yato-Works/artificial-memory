from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import BeliefState, Memory


class BeliefStatus(StrEnum):
    """Status of a belief."""
    ACCEPTED = "accepted"      # Current best understanding
    CONTESTED = "contested"    # Has contradictory evidence
    REJECTED = "rejected"      # Superseded by better evidence
    SUPERSEDED = "superseded"  # Explicitly replaced by newer belief


@dataclass
class Belief:
    """A belief - the system's current interpretation of evidence.

    Belief != Memory
    - Memory = recorded evidence (never deleted, only compressed)
    - Belief = current interpretation (can change, tracks confidence)
    """
    id: int | None = None
    proposition: str = ""           # What is believed: "Database is PostgreSQL"
    proposition_hash: str = ""      # For deduplication
    status: BeliefStatus = BeliefStatus.ACCEPTED
    confidence: float = 0.5         # 0-1 overall confidence
    supporting_evidence: list[int] = field(default_factory=list)  # Memory IDs
    contradicting_evidence: list[int] = field(default_factory=list)  # Memory IDs
    source_memories: list[int] = field(default_factory=list)  # All related memories
    valid_from: datetime = field(default_factory=datetime.now)
    valid_until: datetime | None = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BeliefConflict:
    """A detected contradiction between beliefs/evidence."""
    belief_id: int
    conflicting_memory_id: int
    conflict_type: str = "direct"  # "direct", "temporal", "source"
    severity: float = 0.5  # 0-1
    description: str = ""
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution: str | None = None


class BeliefEngine:
    """Manages beliefs - the system's current understanding separate from raw evidence.

    Key principles:
    - Evidence (Memory) is immutable; Beliefs are mutable interpretations
    - Beliefs track confidence based on evidence quality, recency, source reliability
    - Contradictions are explicit, not hidden
    - Temporal beliefs: "what did we believe at time T?"
    """

    def __init__(self, store: MemoryStore):
        self.store = store
        # In-memory read-through cache; authoritative state lives in the DB.
        # P0-4: beliefs are persisted via belief_states table (JSONB in Postgres).
        self._beliefs: dict[int, Belief] = {}
        self._belief_counter = 0
        self._conflicts: list[BeliefConflict] = []
        self._load_persisted_beliefs()

    # ==================== Persistence (P0-4) ====================

    def _load_persisted_beliefs(self) -> None:
        """Load persisted beliefs from the store into the in-memory cache."""
        try:
            states = self.store.get_all_belief_states()
        except AttributeError:
            # Store backend without belief persistence (e.g. legacy store)
            return
        for state in states:
            belief = Belief(
                id=state.id,
                proposition=state.proposition,
                proposition_hash=state.belief_key,
                status=BeliefStatus(state.status),
                confidence=state.confidence,
                supporting_evidence=list(state.supporting_evidence),
                contradicting_evidence=list(state.contradicting_evidence),
                source_memories=list(state.source_memories),
                valid_from=state.valid_from,
                valid_until=state.valid_until,
                created_at=state.created_at,
                updated_at=state.updated_at,
                metadata={"revision": state.revision},
            )
            self._beliefs[belief.id] = belief
            if belief.id is not None and belief.id > self._belief_counter:
                self._belief_counter = belief.id

    def _persist_belief(self, belief: Belief) -> None:
        """Persist a single belief to the store (best effort)."""
        try:
            state = self.store.get_belief_state_by_key(belief.proposition_hash) \
                if hasattr(self.store, "get_belief_state_by_key") else None
            revision = (state.revision + 1) if state else (belief.metadata.get("revision", 0))
            state = BeliefState(
                id=state.id if state else None,
                belief_key=belief.proposition_hash,
                proposition=belief.proposition,
                status=belief.status.value,
                confidence=belief.confidence,
                supporting_evidence=belief.supporting_evidence,
                contradicting_evidence=belief.contradicting_evidence,
                source_memories=belief.source_memories,
                valid_from=belief.valid_from,
                valid_until=belief.valid_until,
                revision=revision,
                created_at=belief.created_at,
                updated_at=belief.updated_at,
            )
            saved = self.store.upsert_belief_state(state)
            # Keep the DB-assigned id authoritative (survives restarts)
            if saved.id is not None:
                belief.id = saved.id
                self._beliefs[saved.id] = belief
                if saved.id > self._belief_counter:
                    self._belief_counter = saved.id
            belief.metadata["revision"] = revision
        except AttributeError:
            pass  # Store backend without belief persistence

    # ==================== Belief Management ====================

    def create_or_update_belief(
        self,
        proposition: str,
        supporting_memory_ids: list[int],
        confidence: float | None = None,
    ) -> Belief:
        """Create or update a belief from evidence."""
        prop_hash = self._hash_proposition(proposition)

        # Check if belief exists
        existing = self._find_belief_by_hash(prop_hash)

        if existing:
            # Update existing belief
            return self._update_belief(existing, supporting_memory_ids, confidence)
        else:
            # Create new belief
            return self._create_belief(proposition, prop_hash, supporting_memory_ids, confidence)

    def _create_belief(
        self,
        proposition: str,
        prop_hash: str,
        supporting_memory_ids: list[int],
        confidence: float | None,
    ) -> Belief:
        self._belief_counter += 1

        belief = Belief(
            id=self._belief_counter,
            proposition=proposition,
            proposition_hash=prop_hash,
            supporting_evidence=supporting_memory_ids,
            source_memories=supporting_memory_ids,
            confidence=confidence or self._compute_initial_confidence(supporting_memory_ids),
            status=BeliefStatus.ACCEPTED,
            valid_from=datetime.now(),
        )

        self._beliefs[belief.id] = belief
        self._persist_belief(belief)
        return belief

    def _update_belief(
        self,
        belief: Belief,
        new_supporting_ids: list[int],
        confidence: float | None,
    ) -> Belief:
        # Add new evidence
        for mid in new_supporting_ids:
            if mid not in belief.supporting_evidence:
                belief.supporting_evidence.append(mid)
            if mid not in belief.source_memories:
                belief.source_memories.append(mid)

        # Recompute confidence
        belief.confidence = confidence or self._compute_confidence(belief)
        belief.updated_at = datetime.now()

        self._persist_belief(belief)
        return belief

    def _compute_initial_confidence(self, memory_ids: list[int]) -> float:
        if not memory_ids:
            return 0.5

        memories = [self.store.get_memory(mid) for mid in memory_ids]
        memories = [m for m in memories if m]

        if not memories:
            return 0.5

        # Weight by importance, confidence, recency
        total_weight = 0.0
        weighted_conf = 0.0

        for mem in memories:
            weight = mem.importance * mem.confidence
            # Recency factor
            days_old = (datetime.now() - mem.created_at).days
            recency = max(0.3, 1.0 - days_old / 365)
            weight *= recency

            total_weight += weight
            weighted_conf += weight * mem.confidence

        return weighted_conf / total_weight if total_weight > 0 else 0.5

    def _compute_confidence(self, belief: Belief) -> float:
        """Compute belief confidence from evidence."""
        if not belief.supporting_evidence:
            return 0.3

        memories = [self.store.get_memory(mid) for mid in belief.supporting_evidence]
        memories = [m for m in memories if m]

        if not memories:
            return 0.3

        # Penalize for contradicting evidence
        contradiction_penalty = len(belief.contradicting_evidence) * 0.15

        total_weight = 0.0
        weighted_conf = 0.0

        for mem in memories:
            weight = mem.importance * mem.confidence
            days_old = (datetime.now() - mem.created_at).days
            recency = max(0.3, 1.0 - days_old / 365)
            weight *= recency

            total_weight += weight
            weighted_conf += weight * mem.confidence

        base_conf = weighted_conf / total_weight if total_weight > 0 else 0.5
        return max(0.0, min(1.0, base_conf - contradiction_penalty))

    def _hash_proposition(self, proposition: str) -> str:
        import hashlib
        return hashlib.sha256(proposition.lower().strip().encode()).hexdigest()[:16]

    def _find_belief_by_hash(self, prop_hash: str) -> Belief | None:
        for belief in self._beliefs.values():
            if belief.proposition_hash == prop_hash:
                return belief
        return None

    # ==================== Contradiction Detection ====================

    def detect_contradictions(self, memory_ids: list[int]) -> list[BeliefConflict]:
        """Detect contradictions between a new memory and existing beliefs."""
        conflicts = []

        for mid in memory_ids:
            memory = self.store.get_memory(mid)
            if not memory:
                continue

            # Check against existing beliefs
            for belief in self._beliefs.values():
                conflict = self._check_contradiction(belief, memory)
                if conflict:
                    conflicts.append(conflict)
                    self._conflicts.append(conflict)

        return conflicts

    def _check_contradiction(self, belief: Belief, memory: Memory) -> BeliefConflict | None:
        """Check if a memory contradicts a belief."""
        # Simple keyword-based contradiction detection
        # In production, use NLI (Natural Language Inference) model

        belief_text = belief.proposition.lower()
        memory_text = memory.content.lower()

        # Check for explicit negation patterns
        negation_pairs = [
            ("is", "is not"), ("was", "was not"), ("will", "will not"),
            ("adopted", "rejected"), ("chose", "did not choose"),
            ("true", "false"), ("yes", "no"), ("decided", "undecided"),
        ]

        for pos, neg in negation_pairs:
            if pos in belief_text and neg in memory_text:
                # Potential contradiction
                return BeliefConflict(
                    belief_id=belief.id,
                    conflicting_memory_id=memory.id,
                    conflict_type="direct",
                    severity=0.7,
                    description=f"Belief '{belief.proposition}' contradicted by memory {memory.id}",
                )

        # Check for entity-value contradictions (e.g., "DB is PostgreSQL" vs "DB is SQLite")
        # This would need entity extraction - simplified for now
        return None

    def resolve_conflict(self, conflict_id: int, resolution: str) -> bool:
        """Mark a conflict as resolved."""
        for conflict in self._conflicts:
            if conflict.conflicting_memory_id == conflict_id:
                conflict.resolved = True
                conflict.resolution = resolution
                return True
        return False

    # ==================== Temporal Beliefs ====================

    def get_belief_at(self, timestamp: datetime) -> list[Belief]:
        """Get all beliefs that were valid at a given timestamp (Time Travel)."""
        result = []
        for belief in self._beliefs.values():
            if belief.valid_from <= timestamp:
                if belief.valid_until is None or belief.valid_until > timestamp:
                    # Create a snapshot with confidence at that time
                    snapshot = Belief(
                        id=belief.id,
                        proposition=belief.proposition,
                        proposition_hash=belief.proposition_hash,
                        status=belief.status,
                        confidence=self._compute_confidence_at(belief, timestamp),
                        supporting_evidence=belief.supporting_evidence.copy(),
                        contradicting_evidence=belief.contradicting_evidence.copy(),
                        valid_from=belief.valid_from,
                        valid_until=belief.valid_until,
                    )
                    result.append(snapshot)
        return result

    def _compute_confidence_at(self, belief: Belief, timestamp: datetime) -> float:
        """Compute confidence as it would have been at timestamp."""
        # Only consider evidence that existed at that time
        relevant_evidence = []
        for mid in belief.supporting_evidence:
            mem = self.store.get_memory(mid)
            if mem and mem.created_at <= timestamp:
                relevant_evidence.append(mem)

        if not relevant_evidence:
            return 0.3

        total_weight = 0.0
        weighted_conf = 0.0

        for mem in relevant_evidence:
            weight = mem.importance * mem.confidence
            days_old = (timestamp - mem.created_at).days
            recency = max(0.3, 1.0 - days_old / 365)
            weight *= recency

            total_weight += weight
            weighted_conf += weight * mem.confidence

        base_conf = weighted_conf / total_weight if total_weight > 0 else 0.5

        # Subtract contradiction penalty for evidence existing at that time
        contradiction_penalty = len(belief.contradicting_evidence) * 0.15
        return max(0.0, min(1.0, base_conf - contradiction_penalty))

    # ==================== Queries ====================

    def get_belief(self, belief_id: int) -> Belief | None:
        return self._beliefs.get(belief_id)

    def get_beliefs_by_topic(self, topic_id: int) -> list[Belief]:
        """Get beliefs related to a topic (via evidence memory topic)."""
        result = []
        for belief in self._beliefs.values():
            for mid in belief.source_memories:
                mem = self.store.get_memory(mid)
                if mem and mem.topic_id == topic_id:
                    result.append(belief)
                    break
        return result

    def get_conflicts(self, belief_id: int | None = None) -> list[BeliefConflict]:
        if belief_id:
            return [c for c in self._conflicts if c.belief_id == belief_id]
        return self._conflicts

    def supersede_belief(self, old_belief_id: int, new_proposition: str,
                         supporting_memory_ids: list[int]) -> Belief:
        """Explicitly supersede a belief with a new one."""
        old_belief = self._beliefs.get(old_belief_id)
        if not old_belief:
            raise ValueError(f"Belief {old_belief_id} not found")

        old_belief.status = BeliefStatus.SUPERSEDED
        old_belief.valid_until = datetime.now()
        old_belief.updated_at = datetime.now()
        self._persist_belief(old_belief)

        # Create new belief
        return self.create_or_update_belief(new_proposition, supporting_memory_ids)

    def get_belief_provenance(self, belief_id: int) -> dict[str, Any]:
        """Get full provenance for a belief."""
        belief = self._beliefs.get(belief_id)
        if not belief:
            return {}

        supporting = []
        for mid in belief.supporting_evidence:
            mem = self.store.get_memory(mid)
            if mem:
                supporting.append({
                    "memory_id": mem.id,
                    "type": mem.memory_type.value,
                    "resolution": mem.resolution.name,
                    "confidence": mem.confidence,
                    "content_preview": mem.content[:100],
                })

        contradicting = []
        for mid in belief.contradicting_evidence:
            mem = self.store.get_memory(mid)
            if mem:
                contradicting.append({
                    "memory_id": mem.id,
                    "content_preview": mem.content[:100],
                })

        return {
            "belief_id": belief.id,
            "proposition": belief.proposition,
            "status": belief.status.value,
            "confidence": belief.confidence,
            "supporting_evidence": supporting,
            "contradicting_evidence": contradicting,
            "valid_from": belief.valid_from.isoformat(),
            "valid_until": belief.valid_until.isoformat() if belief.valid_until else None,
        }


def create_belief_engine(store: MemoryStore) -> BeliefEngine:
    return BeliefEngine(store)
