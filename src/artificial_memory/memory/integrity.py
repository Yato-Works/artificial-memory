from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from artificial_memory.compression.compressor import RuleBasedCompressor
from artificial_memory.core.interfaces import MemoryStore
from artificial_memory.core.models import Memory, ResolutionLevel


class IntegrityIssueType(StrEnum):
    """Types of integrity issues."""
    SEMANTIC_DRIFT = "semantic_drift"           # Meaning changed during compression
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"  # Time-related contradictions
    PROVENANCE_BROKEN = "provenance_broken"     # Source chain broken
    COMPRESSION_ARTIFACT = "compression_artifact"  # Information lost in compression
    CONTRADICTION = "contradiction"              # Conflicts with other memories
    STALE = "stale"                             # Outdated information
    ORPHANED = "orphaned"                       # No associations, no recall


class IntegritySeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class IntegrityIssue:
    """A detected integrity issue."""
    id: str
    memory_id: int
    issue_type: IntegrityIssueType
    severity: IntegritySeverity
    description: str
    details: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    resolution: str | None = None


@dataclass
class IntegrityReport:
    """Full integrity report for a memory or topic."""
    memory_id: int | None = None
    topic_id: int | None = None
    overall_score: float = 1.0  # 0-1
    issues: list[IntegrityIssue] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def has_critical(self) -> bool:
        return any(i.severity == IntegritySeverity.CRITICAL for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == IntegritySeverity.WARNING for i in self.issues)


class IntegrityMetrics:
    """Computes various integrity metrics for memories."""

    def __init__(self, store: MemoryStore, compressor: RuleBasedCompressor):
        self.store = store
        self.compressor = compressor

    # ==================== Semantic Preservation ====================

    def semantic_preservation_score(self, memory: Memory) -> float:
        """Score how well original meaning is preserved across compressions.

        Compares original content with all compressed versions.
        """
        if memory.resolution == ResolutionLevel.RAW:
            return 1.0

        versions = self.store.get_memory_versions(memory.id)
        if not versions:
            return 0.5  # Unknown

        # Compare each version to original
        original_content = memory.content
        if not original_content:
            return 0.0

        scores = []
        for version in versions:
            score = self._compare_semantic_similarity(original_content, version.content)
            scores.append(score)

        return sum(scores) / len(scores) if scores else 0.5

    def _compare_semantic_similarity(self, text_a: str, text_b: str) -> float:
        """Simple semantic similarity using word overlap.

        In production, use embeddings or NLI model.
        """
        if not text_a or not text_b:
            return 0.0

        words_a = set(text_a.lower().split())
        words_b = set(text_b.lower().split())

        if not words_a or not words_b:
            return 0.0

        intersection = words_a & words_b
        union = words_a | words_b

        return len(intersection) / len(union)

    # ==================== Temporal Consistency ====================

    def temporal_consistency_score(self, memory: Memory) -> float:
        """Check for temporal inconsistencies."""
        score = 1.0

        # Check valid_from/valid_until consistency
        if memory.valid_from and memory.valid_until:
            if memory.valid_from > memory.valid_until:
                score -= 0.3

        # Check if memory claims to be current but has expired validity
        if memory.is_current and memory.valid_until:
            if memory.valid_until < datetime.now():
                score -= 0.4

        # Check created_at vs valid_from
        if memory.valid_from and memory.created_at:
            if memory.valid_from < memory.created_at:
                score -= 0.2

        return max(0.0, score)

    # ==================== Provenance Integrity ====================

    def provenance_integrity_score(self, memory: Memory) -> float:
        """Check if provenance chain is complete."""
        score = 1.0

        # Check source conversation exists
        if memory.source_conversation_id:
            conv = self.store.get_conversation(memory.source_conversation_id)
            if not conv:
                score -= 0.5
        else:
            score -= 0.3

        # Check source message exists
        if memory.source_message_id:
            if memory.source_conversation_id:
                messages = self.store.get_messages(memory.source_conversation_id)
                msg_ids = {m.id for m in messages}
                if memory.source_message_id not in msg_ids:
                    score -= 0.4
        else:
            score -= 0.2

        return max(0.0, score)

    # ==================== Compression Quality ====================

    def compression_quality_score(self, memory: Memory) -> float:
        """Evaluate compression quality across all versions."""
        versions = self.store.get_memory_versions(memory.id)
        if not versions:
            return 0.5

        scores = []
        for version in versions:
            # Check compression ratio is reasonable
            if version.compression_ratio:
                # Good compression: 1.5x to 10x
                if 1.5 <= version.compression_ratio <= 10:
                    scores.append(1.0)
                elif version.compression_ratio < 1.5:
                    scores.append(0.7)  # Barely compressed
                elif version.compression_ratio > 20:
                    scores.append(0.5)  # Over-compressed, likely loss
                else:
                    scores.append(0.8)
            else:
                scores.append(0.6)

        return sum(scores) / len(scores) if scores else 0.5

    # ==================== Contradiction Consistency ====================

    def contradiction_consistency_score(self, memory: Memory) -> float:
        """Check for contradictions with other memories."""
        # This would use ContradictionDetector in practice
        # Simplified: check associations for CONTRADICTS
        associations = self.store.get_associations(memory.id)

        contradiction_count = sum(
            1 for a in associations
            if a.association_type.value == "contradicts"
        )

        if contradiction_count == 0:
            return 1.0
        elif contradiction_count <= 2:
            return 0.8
        elif contradiction_count <= 5:
            return 0.6
        else:
            return 0.3

    # ==================== Composite Score ====================

    def compute_overall_integrity(self, memory: Memory) -> IntegrityReport:
        """Compute complete integrity report for a memory."""
        metrics = {
            "semantic_preservation": self.semantic_preservation_score(memory),
            "temporal_consistency": self.temporal_consistency_score(memory),
            "provenance_integrity": self.provenance_integrity_score(memory),
            "compression_quality": self.compression_quality_score(memory),
            "contradiction_consistency": self.contradiction_consistency_score(memory),
        }

        # Overall score is weighted average
        weights = {
            "semantic_preservation": 0.3,
            "temporal_consistency": 0.15,
            "provenance_integrity": 0.2,
            "compression_quality": 0.15,
            "contradiction_consistency": 0.2,
        }

        overall = sum(metrics[k] * weights[k] for k in metrics)

        # Generate issues from low scores
        issues = []
        for metric, score in metrics.items():
            if score < 0.5:
                severity = IntegritySeverity.CRITICAL if score < 0.3 else IntegritySeverity.WARNING
                issues.append(IntegrityIssue(
                    id=f"{metric}_{memory.id}_{datetime.now().timestamp()}",
                    memory_id=memory.id,
                    issue_type=self._metric_to_issue_type(metric),
                    severity=severity,
                    description=f"{metric.replace('_', ' ').title()} score is {score:.2f}",
                    details={"score": score, "threshold": 0.5},
                ))
            elif score < 0.7:
                issues.append(IntegrityIssue(
                    id=f"{metric}_{memory.id}_{datetime.now().timestamp()}",
                    memory_id=memory.id,
                    issue_type=self._metric_to_issue_type(metric),
                    severity=IntegritySeverity.WARNING,
                    description=f"{metric.replace('_', ' ').title()} score is {score:.2f} (below optimal)",
                    details={"score": score, "threshold": 0.7},
                ))

        return IntegrityReport(
            memory_id=memory.id,
            overall_score=overall,
            issues=issues,
            metrics=metrics,
        )

    def _metric_to_issue_type(self, metric: str) -> IntegrityIssueType:
        mapping = {
            "semantic_preservation": IntegrityIssueType.SEMANTIC_DRIFT,
            "temporal_consistency": IntegrityIssueType.TEMPORAL_INCONSISTENCY,
            "provenance_integrity": IntegrityIssueType.PROVENANCE_BROKEN,
            "compression_quality": IntegrityIssueType.COMPRESSION_ARTIFACT,
            "contradiction_consistency": IntegrityIssueType.CONTRADICTION,
        }
        return mapping.get(metric, IntegrityIssueType.SEMANTIC_DRIFT)

    def scan_topic(self, topic_id: int, limit: int = 100) -> list[IntegrityReport]:
        """Scan all memories in a topic."""
        memories = self.store.get_memories(topic_id=topic_id, limit=limit)
        return [self.compute_overall_integrity(m) for m in memories]


def create_integrity_metrics(store: MemoryStore, compressor: RuleBasedCompressor) -> IntegrityMetrics:
    return IntegrityMetrics(store, compressor)
