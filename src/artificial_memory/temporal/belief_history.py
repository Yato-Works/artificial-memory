from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.memory.belief import Belief, BeliefEngine, BeliefStatus


@dataclass
class HistoricalBelief:
    """A belief as it existed at a specific point in time."""
    belief_id: int
    proposition: str
    status: BeliefStatus
    confidence: float
    valid_from: datetime
    snapshot_timestamp: datetime
    supporting_evidence: list[int] = field(default_factory=list)  # Memory IDs
    contradicting_evidence: list[int] = field(default_factory=list)
    valid_until: datetime | None = None


@dataclass
class BeliefEvolution:
    """Tracks how a belief evolved over time."""
    belief_id: int
    proposition: str
    current_status: BeliefStatus
    current_confidence: float
    snapshots: list[HistoricalBelief] = field(default_factory=list)


@dataclass
class BeliefTimeline:
    """Timeline of all beliefs for a topic."""
    topic_id: int
    belief_evolutions: list[BeliefEvolution] = field(default_factory=list)
    total_beliefs: int = 0
    contested_count: int = 0
    superseded_count: int = 0


class BeliefHistoryEngine:
    """Reconstructs historical beliefs from memory evidence.

    Given the current memory state and temporal information,
    reconstructs what the system believed at any point in time.
    """

    def __init__(
        self,
        store: MemoryStore,
        belief_engine: BeliefEngine,
        time_travel_engine=None,
    ):
        self.store = store
        self.belief_engine = belief_engine
        self.time_travel_engine = time_travel_engine

    def get_beliefs_at(
        self,
        timestamp: datetime,
        topic_id: int | None = None,
    ) -> list[HistoricalBelief]:
        """Get all beliefs as they existed at a specific timestamp."""
        # Get current beliefs
        if topic_id:
            current_beliefs = self.belief_engine.get_beliefs_by_topic(topic_id)
        else:
            current_beliefs = list(self.belief_engine._beliefs.values())

        historical_beliefs = []

        for belief in current_beliefs:
            # Reconstruct belief state at timestamp
            historical = self._reconstruct_belief_at(belief, timestamp)
            if historical:
                historical_beliefs.append(historical)

        return historical_beliefs

    def _reconstruct_belief_at(
        self,
        belief: Belief,
        timestamp: datetime,
    ) -> HistoricalBelief | None:
        """Reconstruct a belief's state at a specific timestamp."""
        # Check if belief existed at timestamp
        if belief.valid_from > timestamp:
            return None  # Belief didn't exist yet

        if belief.valid_until and belief.valid_until <= timestamp:
            # Belief was already superseded/rejected by timestamp
            # Use the status it had at that time
            pass  # We'll compute the historical confidence

        # Filter evidence to only what existed at timestamp
        supporting_at_time = [
            mid for mid in belief.supporting_evidence
            if self._memory_existed_at(mid, timestamp)
        ]

        contradicting_at_time = [
            mid for mid in belief.contradicting_evidence
            if self._memory_existed_at(mid, timestamp)
        ]

        # Compute confidence at that time
        confidence_at_time = self._compute_confidence_at(
            supporting_at_time, contradicting_at_time, timestamp
        )

        # Determine status at that time
        status_at_time = self._determine_status_at(
            belief, supporting_at_time, contradicting_at_time, timestamp
        )

        return HistoricalBelief(
            belief_id=belief.id,
            proposition=belief.proposition,
            status=status_at_time,
            confidence=confidence_at_time,
            supporting_evidence=supporting_at_time,
            contradicting_evidence=contradicting_at_time,
            valid_from=belief.valid_from,
            valid_until=belief.valid_until,
            snapshot_timestamp=timestamp,
        )

    def _memory_existed_at(self, memory_id: int, timestamp: datetime) -> bool:
        """Check if a memory existed at a given timestamp."""
        memory = self.store.get_memory(memory_id)
        if not memory:
            return False

        if memory.created_at > timestamp:
            return False

        # Check if it was archived before timestamp
        if memory.status in ['archived', 'deep_archived']:
            if memory.updated_at and memory.updated_at <= timestamp:
                return False

        # Check compression history
        events = self.store.get_compression_history(memory_id)
        for event in events:
            if event.to_resolution.value >= 4:  # LONG_TERM or DEEP_LONG_TERM
                if event.created_at <= timestamp:
                    return False

        return True

    def _compute_confidence_at(
        self,
        supporting: list[int],
        contradicting: list[int],
        timestamp: datetime,
    ) -> float:
        """Compute belief confidence at a specific timestamp."""
        if not supporting:
            return 0.3

        total_weight = 0.0
        weighted_conf = 0.0

        for mid in supporting:
            memory = self.store.get_memory(mid)
            if not memory or not self._memory_existed_at(mid, timestamp):
                continue

            # Weight by importance, confidence, and recency at that time
            weight = memory.importance * memory.confidence
            days_old = (timestamp - memory.created_at).days
            recency = max(0.3, 1.0 - days_old / 180)
            weight *= recency

            total_weight += weight
            weighted_conf += weight * memory.confidence

        base_conf = weighted_conf / total_weight if total_weight > 0 else 0.5

        # Subtract contradiction penalty
        contradiction_penalty = len(contradicting) * 0.15
        return max(0.0, min(1.0, base_conf - contradiction_penalty))

    def _determine_status_at(
        self,
        belief: Belief,
        supporting: list[int],
        contradicting: list[int],
        timestamp: datetime,
    ) -> BeliefStatus:
        """Determine belief status at a specific timestamp."""
        if belief.valid_until and belief.valid_until <= timestamp:
            return BeliefStatus.SUPERSEDED

        if contradicting and len(contradicting) >= len(supporting) * 0.5:
            return BeliefStatus.CONTESTED

        if not supporting:
            return BeliefStatus.REJECTED

        return BeliefStatus.ACCEPTED

    def get_belief_evolution(
        self,
        belief_id: int,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> BeliefEvolution | None:
        """Get the full evolution history of a belief."""
        belief = self.belief_engine.get_belief(belief_id)
        if not belief:
            return None

        # Generate snapshots at key points
        snapshots = []

        # Start from belief creation
        current_time = belief.valid_from
        end_time = end or datetime.now()

        if start:
            current_time = max(current_time, start)

        # Sample at intervals
        while current_time <= end_time:
            snapshot = self._reconstruct_belief_at(belief, current_time)
            if snapshot:
                snapshots.append(snapshot)
            current_time += timedelta(days=30)  # Monthly snapshots

        # Add final snapshot
        if snapshots and snapshots[-1].snapshot_timestamp != end_time:
            final = self._reconstruct_belief_at(belief, end_time)
            if final:
                snapshots.append(final)

        return BeliefEvolution(
            belief_id=belief.id,
            proposition=belief.proposition,
            snapshots=snapshots,
            current_status=belief.status,
            current_confidence=belief.confidence,
        )

    def get_belief_timeline(
        self,
        topic_id: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> BeliefTimeline:
        """Get timeline of all beliefs for a topic."""
        if topic_id:
            beliefs = self.belief_engine.get_beliefs_by_topic(topic_id)
        else:
            beliefs = list(self.belief_engine._beliefs.values())

        evolutions = []
        for belief in beliefs:
            evolution = self.get_belief_evolution(belief.id, start, end)
            if evolution:
                evolutions.append(evolution)

        contested = sum(1 for e in evolutions if e.current_status == BeliefStatus.CONTESTED)
        superseded = sum(1 for e in evolutions if e.current_status == BeliefStatus.SUPERSEDED)

        return BeliefTimeline(
            topic_id=topic_id or 0,
            belief_evolutions=evolutions,
            total_beliefs=len(evolutions),
            contested_count=contested,
            superseded_count=superseded,
        )

    def compare_beliefs(
        self,
        timestamp_a: datetime,
        timestamp_b: datetime,
        topic_id: int | None = None,
    ) -> dict[str, Any]:
        """Compare beliefs at two different timestamps."""
        beliefs_a = self.get_beliefs_at(timestamp_a, topic_id)
        beliefs_b = self.get_beliefs_at(timestamp_b, topic_id)

        # Create maps for comparison
        map_a = {b.proposition: b for b in beliefs_a}
        map_b = {b.proposition: b for b in beliefs_b}

        all_props = set(map_a.keys()) | set(map_b.keys())

        changes = []
        for prop in all_props:
            b_a = map_a.get(prop)
            b_b = map_b.get(prop)

            if b_a and not b_b:
                changes.append({
                    "type": "belief_lost",
                    "proposition": prop,
                    "confidence_at_a": b_a.confidence,
                    "status_at_a": b_a.status.value,
                })
            elif not b_a and b_b:
                changes.append({
                    "type": "belief_gained",
                    "proposition": prop,
                    "confidence_at_b": b_b.confidence,
                    "status_at_b": b_b.status.value,
                })
            elif b_a.confidence != b_b.confidence:
                changes.append({
                    "type": "confidence_changed",
                    "proposition": prop,
                    "confidence_a": b_a.confidence,
                    "confidence_b": b_b.confidence,
                    "delta": b_b.confidence - b_a.confidence,
                })
            elif b_a.status != b_b.status:
                changes.append({
                    "type": "status_changed",
                    "proposition": prop,
                    "status_a": b_a.status.value,
                    "status_b": b_b.status.value,
                })

        return {
            "timestamp_a": timestamp_a.isoformat(),
            "timestamp_b": timestamp_b.isoformat(),
            "total_beliefs_a": len(beliefs_a),
            "total_beliefs_b": len(beliefs_b),
            "changes": changes,
        }


def create_belief_history_engine(
    store: MemoryStore,
    belief_engine: BeliefEngine,
    time_travel_engine=None,
) -> BeliefHistoryEngine:
    return BeliefHistoryEngine(store, belief_engine, time_travel_engine)
