from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Memory
from artificial_memory.recall.engine import BasicRecallEngine


class StalenessReason(StrEnum):
    """Reasons why a memory is considered stale."""
    TIME_EXPIRED = "time_expired"           # Past valid_until
    NO_RECENT_ACCESS = "no_recent_access"   # Not accessed in N days
    SUPERSEDED = "superseded"               # Newer version exists
    CONTRADICTED = "contradicted"           # Contradicted by newer evidence
    LOW_CONFIDENCE = "low_confidence"       # Confidence dropped
    ORPHANED = "orphaned"                   # No associations, never recalled
    TEMPORAL_DRIFT = "temporal_drift"       # World state changed


@dataclass
class StalenessSignal:
    """A signal contributing to staleness."""
    reason: StalenessReason
    weight: float  # 0-1
    description: str
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class StalenessAssessment:
    """Complete staleness assessment for a memory."""
    memory_id: int
    is_stale: bool
    staleness_score: float  # 0-1, higher = more stale
    signals: list[StalenessSignal] = field(default_factory=list)
    recommended_action: str = "keep"
    assessed_at: datetime = field(default_factory=datetime.now)


class StaleMemoryDetector:
    """Detects stale memories using multiple signals.

    A memory is stale when it's no longer reliable as a source of truth,
    not merely old. Multiple signals are combined for robust detection.
    """

    def __init__(
        self,
        store: MemoryStore,
        recall_engine: BasicRecallEngine,
        config: dict[str, Any] | None = None,
    ):
        self.store = store
        self.recall_engine = recall_engine
        self.config = config or {}

        # Default thresholds
        self.max_age_days = self.config.get("max_age_days", 365)
        self.max_inactive_days = self.config.get("max_inactive_days", 90)
        self.min_access_count = self.config.get("min_access_count", 1)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.4)
        self.staleness_threshold = self.config.get("staleness_threshold", 0.6)

    def assess_memory(self, memory: Memory) -> StalenessAssessment:
        """Assess staleness of a single memory."""
        signals = []

        # Signal 1: Time expired
        if memory.valid_until and memory.valid_until < datetime.now():
            days_overdue = (datetime.now() - memory.valid_until).days
            signals.append(StalenessSignal(
                reason=StalenessReason.TIME_EXPIRED,
                weight=min(1.0, days_overdue / 30),
                description=f"Memory expired {days_overdue} days ago",
                evidence={"valid_until": memory.valid_until.isoformat() if memory.valid_until else None},
            ))

        # Signal 2: No recent access
        last_access = memory.last_accessed or memory.created_at
        days_inactive = (datetime.now() - last_access).days
        if days_inactive > self.max_inactive_days:
            signals.append(StalenessSignal(
                reason=StalenessReason.NO_RECENT_ACCESS,
                weight=min(1.0, days_inactive / (self.max_inactive_days * 2)),
                description=f"Not accessed for {days_inactive} days",
                evidence={"last_accessed": last_access.isoformat(), "days_inactive": days_inactive},
            ))

        # Signal 3: Low access count
        if memory.access_count < self.min_access_count:
            signals.append(StalenessSignal(
                reason=StalenessReason.ORPHANED,
                weight=0.5,
                description=f"Only accessed {memory.access_count} time(s)",
                evidence={"access_count": memory.access_count},
            ))

        # Signal 4: Low confidence
        if memory.confidence < self.confidence_threshold:
            signals.append(StalenessSignal(
                reason=StalenessReason.LOW_CONFIDENCE,
                weight=(self.confidence_threshold - memory.confidence) / self.confidence_threshold,
                description=f"Confidence {memory.confidence:.2f} below threshold {self.confidence_threshold}",
                evidence={"confidence": memory.confidence},
            ))

        # Signal 5: Superseded by newer memory
        superseded_by = self._find_superseding_memory(memory)
        if superseded_by:
            signals.append(StalenessSignal(
                reason=StalenessReason.SUPERSEDED,
                weight=0.8,
                description=f"Superseded by memory {superseded_by.id}",
                evidence={"superseded_by": superseded_by.id, "newer_content": superseded_by.content[:100]},
            ))

        # Signal 6: Contradicted
        contradictions = self._find_contradictions(memory)
        if contradictions:
            signals.append(StalenessSignal(
                reason=StalenessReason.CONTRADICTED,
                weight=min(1.0, len(contradictions) * 0.3),
                description=f"Contradicted by {len(contradictions)} other memory(s)",
                evidence={"contradiction_count": len(contradictions)},
            ))

        # Signal 7: Temporal drift (world state changed)
        if self._check_temporal_drift(memory):
            signals.append(StalenessSignal(
                reason=StalenessReason.TEMPORAL_DRIFT,
                weight=0.4,
                description="Referenced external state may have changed",
                evidence={},
            ))

        # Compute composite score
        if signals:
            total_weight = sum(s.weight for s in signals)
            weighted_sum = sum(s.weight for s in signals)
            staleness_score = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            staleness_score = 0.0

        is_stale = staleness_score >= self.staleness_threshold

        # Determine recommended action
        if staleness_score >= 0.8:
            recommended_action = "archive_or_delete"
        elif staleness_score >= 0.6:
            recommended_action = "review_and_update"
        elif staleness_score >= 0.4:
            recommended_action = "monitor"
        else:
            recommended_action = "keep"

        return StalenessAssessment(
            memory_id=memory.id,
            is_stale=is_stale,
            staleness_score=staleness_score,
            signals=signals,
            recommended_action=recommended_action,
        )

    def _find_superseding_memory(self, memory: Memory) -> Memory | None:
        """Find a newer memory that supersedes this one."""
        # Look for memories of same type in same topic with higher version
        memories = self.store.get_memories(
            topic_id=memory.topic_id,
            memory_type=memory.memory_type,
            limit=50
        )

        for m in memories:
            if m.id == memory.id:
                continue
            # Check if newer and same entity
            if m.created_at > memory.created_at:
                # Simple heuristic: similar content but newer
                if self._similar_content(memory.content, m.content):
                    return m
        return None

    def _find_contradictions(self, memory: Memory) -> list[Memory]:
        """Find memories that contradict this one."""
        associations = self.store.get_associations(memory.id)
        contradictions = []

        for assoc in associations:
            if assoc.association_type.value == "contradicts":
                other_id = assoc.target_memory_id if assoc.source_memory_id == memory.id else assoc.source_memory_id
                other = self.store.get_memory(other_id)
                if other:
                    contradictions.append(other)

        return contradictions

    def _check_temporal_drift(self, memory: Memory) -> bool:
        """Check if external references may have drifted."""
        # Check for references to external systems that may have changed
        # e.g., API versions, library versions, service endpoints
        drift_keywords = [
            "version", "v1.", "v2.", "api", "endpoint", "service",
            "library", "framework", "deprecated", "migrate"
        ]

        content_lower = memory.content.lower()
        return any(kw in content_lower for kw in drift_keywords)

    def _similar_content(self, a: str, b: str) -> bool:
        """Check if two contents are about the same thing."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())

        if not words_a or not words_b:
            return False

        overlap = len(words_a & words_b)
        union = len(words_a | words_b)

        return overlap / union > 0.5

    def scan_topic(
        self,
        topic_id: int,
        limit: int = 100,
        include_keep: bool = False,
    ) -> list[StalenessAssessment]:
        """Scan all memories in a topic for staleness."""
        memories = self.store.get_memories(topic_id=topic_id, limit=limit)

        assessments = []
        for memory in memories:
            assessment = self.assess_memory(memory)
            if assessment.is_stale or include_keep:
                assessments.append(assessment)

        # Sort by staleness score descending
        assessments.sort(key=lambda a: -a.staleness_score)
        return assessments

    def get_stale_summary(self, topic_id: int) -> dict[str, Any]:
        """Get summary of stale memories in a topic."""
        assessments = self.scan_topic(topic_id, include_keep=True)

        total = len(assessments)
        stale = [a for a in assessments if a.is_stale]

        by_action = defaultdict(int)
        by_reason = defaultdict(int)

        for a in stale:
            by_action[a.recommended_action] += 1
            for s in a.signals:
                by_reason[s.reason.value] += 1

        return {
            "topic_id": topic_id,
            "total_memories": total,
            "stale_count": len(stale),
            "stale_percentage": len(stale) / total * 100 if total > 0 else 0,
            "by_action": dict(by_action),
            "by_reason": dict(by_reason),
            "avg_staleness_score": sum(a.staleness_score for a in stale) / len(stale) if stale else 0,
        }


def create_stale_memory_detector(
    store: MemoryStore,
    recall_engine: BasicRecallEngine,
    config: dict[str, Any] | None = None,
) -> StaleMemoryDetector:
    return StaleMemoryDetector(store, recall_engine, config)
