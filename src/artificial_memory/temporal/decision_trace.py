from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import (
    Decision,
    MessageRole,
)
from artificial_memory.memory.belief import BeliefEngine


@dataclass
class DecisionTrace:
    """Complete trace of a decision's lifecycle."""
    decision_id: int
    decision_text: str
    proposed_at: datetime
    decided_at: datetime
    status: str  # "proposed", "accepted", "rejected", "superseded"
    current_status: str
    confidence: float

    # Evidence chain
    supporting_memories: list[int] = field(default_factory=list)
    contradicting_memories: list[int] = field(default_factory=list)

    # Provenance
    source_conversation_id: int | None = None
    source_message_id: int | None = None

    # Evolution
    revisions: list[dict[str, Any]] = field(default_factory=list)
    superseded_by: int | None = None

    # Impact
    dependent_memories: list[int] = field(default_factory=list)
    affected_beliefs: list[int] = field(default_factory=list)


@dataclass
class DecisionTraceEvent:
    """An event in a decision's lifecycle."""
    timestamp: datetime
    event_type: str  # "proposed", "supported", "challenged", "decided", "revised", "superseded"
    description: str
    actor: str  # "user", "assistant", "system"
    memory_id: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionTraceReport:
    """Full report on a decision's trace."""
    decision_id: int
    decision_text: str
    current_status: str
    confidence: float
    timeline: list[DecisionTraceEvent] = field(default_factory=list)
    evidence_chain: dict[str, list[int]] = field(default_factory=lambda: {"supporting": [], "contradicting": []})
    superseded_by: int | None = None
    impact_analysis: dict[str, Any] | None = None


class DecisionTracer:
    """Traces the full lifecycle of decisions from proposal to current status."""

    def __init__(
        self,
        store: MemoryStore,
        belief_engine: BeliefEngine,
    ):
        self.store = store
        self.belief_engine = belief_engine

    def trace_decision(self, decision_id: int) -> DecisionTraceReport:
        """Generate a complete trace report for a decision."""
        decision = self.store.get_decisions(0, current_only=False)
        decision = next((d for d in decision if d.id == decision_id), None)

        if not decision:
            raise ValueError(f"Decision {decision_id} not found")

        # Build timeline
        timeline = self._build_timeline(decision)

        # Get evidence chain
        evidence = self._get_evidence_chain(decision)

        # Get impact analysis
        impact = self._analyze_impact(decision)

        return DecisionTraceReport(
            decision_id=decision.id,
            decision_text=decision.decision_text,
            timeline=timeline,
            evidence_chain=evidence,
            current_status="current" if decision.is_current else decision.status.value if hasattr(decision, 'status') else "unknown",
            confidence=decision.confidence,
            superseded_by=self._find_superseding_decision(decision),
            impact_analysis=impact,
        )

    def _build_timeline(self, decision: Decision) -> list[DecisionTraceEvent]:
        """Build timeline of events for a decision."""
        events = []

        # Decision creation
        events.append(DecisionTraceEvent(
            timestamp=decision.decided_at,
            event_type="decided",
            description=f"Decision made: {decision.decision_text[:100]}",
            actor="assistant",
            memory_id=decision.memory_id,
        ))

        # Get supporting memories
        if decision.memory_id:
            mem = self.store.get_memory(decision.memory_id)
            if mem and mem.source_conversation_id:
                # Find related messages
                messages = self.store.get_messages(mem.source_conversation_id)
                for msg in messages:
                    if msg.role == MessageRole.ASSISTANT and msg.id != mem.source_message_id:
                        if self._is_related_to_decision(msg.content, decision.decision_text):
                            events.append(DecisionTraceEvent(
                                timestamp=msg.created_at,
                                event_type="supported",
                                description=f"Supporting argument: {msg.content[:100]}",
                                actor=msg.role.value,
                                memory_id=msg.id,
                            ))

        # Check for challenges/contradictions
        if decision.memory_id:
            associations = self.store.get_associations(decision.memory_id)
            for assoc in associations:
                if assoc.association_type.value == "contradicts":
                    other_id = assoc.target_memory_id if assoc.source_memory_id == decision.memory_id else assoc.source_memory_id
                    other_mem = self.store.get_memory(other_id)
                    if other_mem:
                        events.append(DecisionTraceEvent(
                            timestamp=other_mem.created_at,
                            event_type="challenged",
                            description=f"Contradicted by: {other_mem.content[:100]}",
                            actor="system",
                            memory_id=other_id,
                        ))

        # Check for supersession
        superseded_by = self._find_superseding_decision(decision)
        if superseded_by:
            sup_dec = self.store.get_decisions(0, current_only=False)
            sup_dec = next((d for d in sup_dec if d.id == superseded_by), None)
            if sup_dec:
                events.append(DecisionTraceEvent(
                    timestamp=sup_dec.decided_at,
                    event_type="superseded",
                    description=f"Superseded by: {sup_dec.decision_text[:100]}",
                    actor="system",
                    memory_id=sup_dec.memory_id,
                ))

        events.sort(key=lambda e: e.timestamp)
        return events

    def _is_related_to_decision(self, text: str, decision_text: str) -> bool:
        """Check if a message is related to a decision."""
        text_lower = text.lower()
        decision_lower = decision_text.lower()

        # Check for keyword overlap
        keywords = set(decision_lower.split())
        text_words = set(text_lower.split())
        overlap = len(keywords & text_words)

        return overlap >= 2

    def _get_evidence_chain(self, decision: Decision) -> dict[str, list[int]]:
        """Get the evidence chain for a decision."""
        return {
            "supporting": decision.supporting_memory_ids if hasattr(decision, 'supporting_memory_ids') else [],
            "contradicting": decision.contradicting_memory_ids if hasattr(decision, 'contradicting_memory_ids') else [],
        }

    def _find_superseding_decision(self, decision: Decision) -> int | None:
        """Find if this decision was superseded by a later one."""
        all_decisions = self.store.get_decisions(0, current_only=False)
        for d in all_decisions:
            if d.id == decision.id:
                continue
            if d.decided_at > decision.decided_at:
                # Check if same topic/entity
                if self._decisions_related(decision, d):
                    return d.id
        return None

    def _decisions_related(self, d1: Decision, d2: Decision) -> bool:
        """Check if two decisions are about the same topic."""
        text1 = d1.decision_text.lower()
        text2 = d2.decision_text.lower()

        # Extract key entities
        words1 = set(text1.split())
        words2 = set(text2.split())

        # Significant overlap suggests same topic
        overlap = len(words1 & words2)
        return overlap >= 3

    def _analyze_impact(self, decision: Decision) -> dict[str, Any]:
        """Analyze the impact of a decision."""
        impact = {
            "dependent_memories": [],
            "affected_beliefs": [],
            "cascade_risk": 0.0,
        }

        if decision.memory_id:
            # Find memories that depend on this decision
            associations = self.store.get_associations(decision.memory_id)
            for assoc in associations:
                if assoc.association_type.value in ["causes", "follows", "elaborates"]:
                    target_id = assoc.target_memory_id if assoc.source_memory_id == decision.memory_id else assoc.source_memory_id
                    impact["dependent_memories"].append(target_id)

        # Check beliefs affected
        if self.belief_engine:
            for belief in self.belief_engine._beliefs.values():
                if decision.memory_id in belief.supporting_evidence:
                    impact["affected_beliefs"].append(belief.id)

        # Calculate cascade risk
        total_affected = len(impact["dependent_memories"]) + len(impact["affected_beliefs"])
        impact["cascade_risk"] = min(1.0, total_affected * 0.1)

        return impact

    def get_all_decision_traces(self, topic_id: int | None = None) -> list[DecisionTraceReport]:
        """Get traces for all decisions in a topic."""
        decisions = self.store.get_decisions(topic_id, current_only=False)

        reports = []
        for decision in decisions:
            try:
                report = self.trace_decision(decision.id)
                reports.append(report)
            except Exception:
                continue

        return reports

    def compare_decisions(
        self,
        decision_id_a: int,
        decision_id_b: int,
    ) -> dict[str, Any]:
        """Compare two decisions."""
        trace_a = self.trace_decision(decision_id_a)
        trace_b = self.trace_decision(decision_id_b)

        return {
            "decision_a": {
                "id": trace_a.decision_id,
                "text": trace_a.decision_text,
                "confidence": trace_a.confidence,
                "status": trace_a.current_status,
            },
            "decision_b": {
                "id": trace_b.decision_id,
                "text": trace_b.decision_text,
                "confidence": trace_b.confidence,
                "status": trace_b.current_status,
            },
            "timeline_overlap": self._check_timeline_overlap(trace_a, trace_b),
            "evidence_overlap": self._check_evidence_overlap(trace_a, trace_b),
            "conflict": self._check_conflict(trace_a, trace_b),
        }

    def _check_timeline_overlap(
        self,
        trace_a: DecisionTraceReport,
        trace_b: DecisionTraceReport,
    ) -> bool:
        """Check if two decisions' timelines overlap."""
        if not trace_a.timeline or not trace_b.timeline:
            return False

        time_a = [e.timestamp for e in trace_a.timeline]
        time_b = [e.timestamp for e in trace_b.timeline]

        return max(time_a) >= min(time_b) and max(time_b) >= min(time_a)

    def _check_evidence_overlap(
        self,
        trace_a: DecisionTraceReport,
        trace_b: DecisionTraceReport,
    ) -> dict[str, int]:
        """Check evidence overlap between decisions."""
        ev_a = trace_a.evidence_chain
        ev_b = trace_b.evidence_chain

        return {
            "supporting_overlap": len(set(ev_a.get("supporting", [])) & set(ev_b.get("supporting", []))),
            "contradicting_overlap": len(set(ev_a.get("contradicting", [])) & set(ev_b.get("contradicting", []))),
        }

    def _check_conflict(
        self,
        trace_a: DecisionTraceReport,
        trace_b: DecisionTraceReport,
    ) -> bool:
        """Check if two decisions conflict."""
        # Simple heuristic: if one supersedes the other
        if trace_a.superseded_by == trace_b.decision_id:
            return True
        if trace_b.superseded_by == trace_a.decision_id:
            return True

        # Check for contradictory evidence
        return False


def create_decision_tracer(
    store: MemoryStore,
    belief_engine: BeliefEngine,
) -> DecisionTracer:
    return DecisionTracer(store, belief_engine)
