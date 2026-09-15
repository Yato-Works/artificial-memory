from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from artificial_memory.core.interfaces import MemoryStore, RecallEngine
from artificial_memory.core.models import (
    Memory,
    MemoryType,
    RecallLevel,
    ResolutionLevel,
)
from artificial_memory.metrics.collector import get_metrics_collector


class RecallMode(StrEnum):
    """Human-like recall modes."""
    IMMEDIATE = "immediate"      # Fresh memories, high detail
    RECENT = "recent"            # Last few days, medium detail
    ESTABLISHED = "established"  # Weeks old, key points only
    DISTANT = "distant"          # Months old, gist only
    ARCHIVAL = "archival"        # Years old, vague impression


@dataclass
class HumanRecallResult:
    """Result of human-like recall with resolution info."""
    memories: list
    mode: RecallMode
    resolution_used: ResolutionLevel
    confidence: float
    reasoning: str
    resolution_adaptations: list[dict] = field(default_factory=list)


class HumanRecallEngine:
    """Human-like recall engine with adaptive resolution."""

    def __init__(self, store: MemoryStore, base_recall_engine: RecallEngine):
        self.store = store
        self.base_recall = base_recall_engine
        self.collector = get_metrics_collector()

        # Mode thresholds (days since last access/creation)
        self.mode_thresholds = {
            RecallMode.IMMEDIATE: 1,      # < 1 day
            RecallMode.RECENT: 7,         # < 1 week
            RecallMode.ESTABLISHED: 30,   # < 1 month
            RecallMode.DISTANT: 180,      # < 6 months
            RecallMode.ARCHIVAL: 365,     # < 1 year
        }

        # Resolution mapping per mode
        self.mode_resolution = {
            RecallMode.IMMEDIATE: ResolutionLevel.RAW,
            RecallMode.RECENT: ResolutionLevel.LIGHT,
            RecallMode.ESTABLISHED: ResolutionLevel.EPISODE,
            RecallMode.DISTANT: ResolutionLevel.SEMANTIC,
            RecallMode.ARCHIVAL: ResolutionLevel.LONG_TERM,
        }

        # Importance boosts for resolution
        self.importance_boost = {
            0.9: 2,  # Critical -> 2 levels higher
            0.7: 1,  # High -> 1 level higher
            0.5: 0,  # Medium -> same
            0.3: -1, # Low -> 1 level lower
        }

    def determine_mode(self, memory: Memory, query_time: datetime | None = None) -> RecallMode:
        """Determine recall mode based on memory age and importance."""
        query_time = query_time or datetime.now()

        # Use last accessed time if available, otherwise created time
        reference_time = memory.last_accessed or memory.created_at
        days_ago = (query_time - reference_time).days

        # Determine base mode
        mode = RecallMode.ARCHIVAL
        for mode_name, threshold in self.mode_thresholds.items():
            if days_ago < threshold:
                mode = mode_name
                break

        # Adjust for importance
        importance_boost = 0
        for threshold, boost in sorted(self.importance_boost.items(), reverse=True):
            if memory.importance >= threshold:
                importance_boost = boost
                break

        # Apply boost (capped at IMMEDIATE)
        modes = list(RecallMode)
        current_idx = modes.index(mode)
        new_idx = max(0, current_idx - importance_boost)

        return modes[new_idx]

    def get_target_resolution(self, mode: RecallMode, memory: Memory) -> ResolutionLevel:
        """Get target resolution for a mode, adjusted for memory type."""
        base_resolution = self.mode_resolution[mode]

        # Adjust for memory type
        type_adjustments = {
            MemoryType.DECISION: -1,      # Decisions need higher detail
            MemoryType.CURRENT: -1,       # Current state needs detail
            MemoryType.TIMELINE: 0,       # Timeline as-is
            MemoryType.EPISODE: 0,        # Episodes as-is
            MemoryType.SEMANTIC: 1,       # Semantic can be more compressed
            MemoryType.CONVERSATION_STYLE: 1,  # Style can be compressed
        }

        adjustment = type_adjustments.get(memory.memory_type, 0)
        target_value = base_resolution.value + adjustment

        # Clamp to valid range
        target_value = max(0, min(5, target_value))
        return ResolutionLevel(target_value)

    def recall(self, query: str, topic_id: int | None = None,
               max_tokens: int = 4000, query_time: datetime | None = None) -> HumanRecallResult:
        """Perform human-like recall with adaptive resolution."""
        query_time = query_time or datetime.now()

        # Get candidate memories using base recall (level 2 - episode)
        # We'll then adapt resolution per memory
        candidates, _ = self.base_recall.recall(query, topic_id, RecallLevel.EPISODE, max_tokens * 2)

        if not candidates:
            return HumanRecallResult(
                memories=[],
                mode=RecallMode.ARCHIVAL,
                resolution_used=ResolutionLevel.LONG_TERM,
                confidence=0.0,
                reasoning="No relevant memories found",
            )

        # Determine mode and target resolution for each memory
        adapted_memories = []
        adaptations = []
        total_tokens = 0

        # Sort by relevance (importance * recency)
        scored = []
        for mem in candidates:
            mode = self.determine_mode(mem, query_time)
            target_res = self.get_target_resolution(mode, mem)

            # Try to get memory at target resolution
            expanded = self.base_recall.expand_resolution(mem, target_res)
            if expanded:
                final_mem = expanded
            else:
                final_mem = mem
                target_res = mem.resolution

            # Calculate relevance score
            days_old = (datetime.now() - (mem.last_accessed or mem.created_at)).days
            recency_score = max(0, 1.0 - days_old / 365)
            relevance = mem.importance * 0.6 + recency_score * 0.4

            scored.append((final_mem, target_res, relevance, mode))

        # Sort by relevance
        scored.sort(key=lambda x: -x[2])

        # Select within token budget
        for mem, target_res, relevance, mode in scored:
            mem_tokens = len(mem.content) // 3  # Rough estimate
            if total_tokens + mem_tokens <= max_tokens:
                adapted_memories.append(mem)
                adaptations.append({
                    "memory_id": mem.id,
                    "mode": mode.value,
                    "target_resolution": target_res.name,
                    "original_resolution": mem.resolution.name,
                    "relevance": relevance,
                })
                total_tokens += mem_tokens
            else:
                break

        # Determine overall mode (most common)
        mode_counts = {}
        for a in adaptations:
            mode = RecallMode(a["mode"])
            mode_counts[mode.value] = mode_counts.get(mode.value, 0) + 1

        dominant_mode = max(mode_counts, key=mode_counts.get) if mode_counts else RecallMode.ARCHIVAL
        dominant_resolution = max(set(a["target_resolution"] for a in adaptations)) if adaptations else ResolutionLevel.LONG_TERM

        # Compute overall confidence
        avg_relevance = sum(a["relevance"] for a in adaptations) / len(adaptations) if adaptations else 0
        confidence = min(1.0, avg_relevance * (1 + len(adapted_memories) / 10))

        reasoning = f"Recalled {len(adapted_memories)} memories in {dominant_mode} mode. " \
                   f"Used adaptive resolution: {dominant_resolution}. " \
                   f"Avg relevance: {avg_relevance:.2f}"

        return HumanRecallResult(
            memories=adapted_memories,
            mode=RecallMode(dominant_mode),
            resolution_used=ResolutionLevel[dominant_resolution],
            confidence=confidence,
            reasoning=reasoning,
            resolution_adaptations=adaptations,
        )

    def simulate_human_recall(self, query: str, topic_id: int | None = None,
                              max_tokens: int = 4000) -> str:
        """Generate human-like recall response."""
        result = self.recall(query, topic_id, max_tokens)

        if not result.memories:
            return f"そのことについては、はっきりとは覚えていません。{result.reasoning}"

        # Generate response based on mode
        mode = result.mode

        if mode == RecallMode.IMMEDIATE:
            prefix = "つい最近の話ですね。"
        elif mode == RecallMode.RECENT:
            prefix = "たしか最近そんな話をしましたね。"
        elif mode == RecallMode.ESTABLISHED:
            prefix = "その件については覚えています。"
        elif mode == RecallMode.DISTANT:
            prefix = "だいぶ前の話ですが、なんとなく覚えています。"
        else:
            prefix = "ずいぶん前のことですが、そんな記憶があります。"

        # Build response from memories
        memory_summaries = []
        for mem in result.memories:
            summary = mem.content[:200].replace('\n', ' ')
            memory_summaries.append(f"- {summary}")

        response = f"{prefix}\n\n" + "\n".join(memory_summaries)

        # Add uncertainty if confidence is low
        if result.confidence < 0.6:
            response += "\n\nただ、細かいところまでは少し曖昧かもしれません。"
        elif result.confidence < 0.8:
            response += "\n\n大まかには合っていると思いますが、詳細は少し怪しいです。"

        return response

    def get_recall_explanation(self, query: str, topic_id: int | None = None) -> dict:
        """Get detailed explanation of how recall worked."""
        result = self.recall(query, topic_id)

        return {
            "query": query,
            "mode": result.mode.value,
            "resolution": result.resolution_used.name,
            "confidence": result.confidence,
            "memories_retrieved": len(result.memories),
            "reasoning": result.reasoning,
            "adaptations": result.resolution_adaptations,
        }


def create_human_recall_engine(store: MemoryStore, base_recall_engine: RecallEngine) -> HumanRecallEngine:
    """Factory function to create human recall engine."""
    return HumanRecallEngine(store, base_recall_engine)
