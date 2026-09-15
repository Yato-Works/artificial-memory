from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from artificial_memory.core.interfaces import MemoryStore, RecallEngine
from artificial_memory.core.models import Memory, MemoryStatus, RecallLevel
from artificial_memory.metrics.collector import get_metrics_collector


class ConfidenceLevel(StrEnum):
    """Confidence levels as defined in design doc section 22."""
    CERTAIN = "certain"        # 0.95+ - "〜でした。"
    CONFIDENT = "confident"    # 0.75-0.95 - "〜だったと思います。"
    UNCERTAIN = "uncertain"    # 0.5-0.75 - "〜だった記憶があります。"
    VAGUE = "vague"            # < 0.5 - "〜だった気がしますが、曖昧です。"


@dataclass
class ConfidenceScore:
    """Confidence score with breakdown."""
    overall: float  # 0.0 to 1.0
    level: ConfidenceLevel
    memory_confidence: float = 0.0
    retrieval_confidence: float = 0.0
    temporal_confidence: float = 0.0
    source_confidence: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)

    def to_natural_language(self) -> str:
        """Convert to natural language prefix."""
        if self.level == ConfidenceLevel.CERTAIN:
            return "〜でした。"
        elif self.level == ConfidenceLevel.CONFIDENT:
            return "〜だったと思います。"
        elif self.level == ConfidenceLevel.UNCERTAIN:
            return "〜だった記憶があります。"
        else:
            return "〜だった気がしますが、曖昧です。"


class ConfidenceEngine:
    """Engine for computing confidence scores for memories and responses."""

    def __init__(self, store: MemoryStore, recall_engine: RecallEngine):
        self.store = store
        self.recall_engine = recall_engine
        self.collector = get_metrics_collector()

    def compute_memory_confidence(self, memory: Memory) -> float:
        """Compute confidence for a single memory."""
        # Base confidence from memory's own confidence
        base = memory.confidence

        # Adjust by status
        status_modifiers = {
            MemoryStatus.ACTIVE: 1.0,
            MemoryStatus.DORMANT: 0.9,
            MemoryStatus.COMPRESSED: 0.8,
            MemoryStatus.ARCHIVED: 0.6,
            MemoryStatus.DEEP_ARCHIVED: 0.4,
        }
        status_mod = status_modifiers.get(memory.status, 1.0)

        # Adjust by resolution (lower resolution = less detail = lower confidence)
        resolution_modifiers = {
            0: 1.0,   # RAW
            1: 0.95,  # LIGHT
            2: 0.9,   # EPISODE
            3: 0.85,  # SEMANTIC
            4: 0.75,  # LONG_TERM
            5: 0.65,  # DEEP_LONG_TERM
        }
        resolution_mod = resolution_modifiers.get(memory.resolution.value, 1.0)

        # Adjust by age (older = less confident)
        age_days = (datetime.now() - memory.created_at).days
        age_mod = max(0.5, 1.0 - (age_days / 365) * 0.3)

        # Adjust by access count (frequently accessed = more confident)
        access_mod = min(1.0, 1.0 + memory.access_count * 0.02)

        # Combine
        final = base * status_mod * resolution_mod * age_mod * access_mod

        return max(0.0, min(1.0, final))

    def compute_retrieval_confidence(self, query: str, memories: list,
                                     recall_level: RecallLevel) -> float:
        """Compute confidence for a retrieval result."""
        if not memories:
            return 0.0

        # Average memory confidence
        memory_conf = sum(self.compute_memory_confidence(m) for m in memories) / len(memories)

        # Recall level modifier (higher level = more detail = higher confidence)
        level_modifiers = {
            RecallLevel.CURRENT_ONLY: 1.0,
            RecallLevel.LONG_TERM_SUMMARY: 0.9,
            RecallLevel.EPISODE: 0.85,
            RecallLevel.LIGHT_COMPRESSION: 0.95,
            RecallLevel.RAW: 0.98,
        }
        level_mod = level_modifiers.get(recall_level, 1.0)

        # Result count modifier (more results = more confidence)
        count_mod = min(1.0, len(memories) / 5.0)

        return max(0.0, min(1.0, memory_conf * level_mod * count_mod))

    def compute_temporal_confidence(self, memories: list,
                                    query_time: datetime | None = None) -> float:
        """Compute confidence based on temporal validity."""
        if not memories:
            return 0.0

        query_time = query_time or datetime.now()

        # Check how many memories are currently valid
        valid_count = 0
        for mem in memories:
            if mem.valid_from and mem.valid_from > query_time:
                continue
            if mem.valid_until and mem.valid_until <= query_time:
                continue
            valid_count += 1

        if len(memories) == 0:
            return 0.0

        return valid_count / len(memories)

    def compute_source_confidence(self, memories: list) -> float:
        """Compute confidence based on source quality."""
        if not memories:
            return 0.0

        # Memories with source conversation have higher confidence
        sourced = sum(1 for m in memories if m.source_conversation_id)
        return sourced / len(memories)

    def compute_overall_confidence(self, query: str, memories: list,
                                   recall_level: RecallLevel,
                                   query_time: datetime | None = None) -> ConfidenceScore:
        """Compute overall confidence for a recall response."""

        memory_conf = sum(self.compute_memory_confidence(m) for m in memories) / len(memories) if memories else 0.0
        retrieval_conf = self.compute_retrieval_confidence(query, memories, recall_level)
        temporal_conf = self.compute_temporal_confidence(memories, query_time)
        source_conf = self.compute_source_confidence(memories)

        # Weighted combination
        weights = {
            "memory": 0.4,
            "retrieval": 0.3,
            "temporal": 0.15,
            "source": 0.15,
        }

        overall = (
            memory_conf * weights["memory"] +
            retrieval_conf * weights["retrieval"] +
            temporal_conf * weights["temporal"] +
            source_conf * weights["source"]
        )

        # Determine level
        if overall >= 0.95:
            level = ConfidenceLevel.CERTAIN
        elif overall >= 0.75:
            level = ConfidenceLevel.CONFIDENT
        elif overall >= 0.5:
            level = ConfidenceLevel.UNCERTAIN
        else:
            level = ConfidenceLevel.VAGUE

        return ConfidenceScore(
            overall=overall,
            level=level,
            memory_confidence=memory_conf,
            retrieval_confidence=retrieval_conf,
            temporal_confidence=temporal_conf,
            source_confidence=source_conf,
            breakdown={
                "memory": memory_conf,
                "retrieval": retrieval_conf,
                "temporal": temporal_conf,
                "source": source_conf,
            }
        )

    def format_response_with_confidence(self, response: str,
                                         confidence: ConfidenceScore) -> str:
        """Format response with appropriate confidence language."""
        prefix = confidence.to_natural_language()
        return f"{response} {prefix}"


def create_confidence_engine(store: MemoryStore, recall_engine: RecallEngine) -> ConfidenceEngine:
    """Factory function to create confidence engine."""
    return ConfidenceEngine(store, recall_engine)
